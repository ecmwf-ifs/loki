#!/usr/bin/env python

# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
An offline script to generate the complex control-flow patterns of the
EC-physics drivers.
"""

import click
from pathlib import Path
from codetiming import Timer

from loki import (
    Sourcefile, FindNodes, CallStatement, Transformer, info
)
from loki.analyse import dataflow_analysis_attached
from loki.expression import (
    symbols as sym, parse_expr, SubstituteExpressions
)
from loki.ir import (
    nodes as ir, FindNodes, pragmas_attached, pragma_regions_attached,
    is_loki_pragma
)
from loki.tools import as_tuple, flatten
from loki.types import BasicType

from loki.transformations.inline import inline_marked_subroutines
from loki.transformations.sanitise import transform_sequence_association_append_map
from loki.transformations.remove_code import do_remove_marked_regions
from loki.transformations.build_system import ModuleWrapTransformation


def substitute_spec_symbols(mapping, routine):
    """
    Do symbol substitution on the spec/declartion of a subroutine from
    a given string mapping via :any:`parse_expr`.

    Importantly, this will use the given :any:`Subroutine` as the scope.
    """
    symbol_map = {
        parse_expr(k, scope=routine): parse_expr(v, scope=routine)
        for k, v in mapping.items()
    }

    routine.spec = SubstituteExpressions(symbol_map).visit(routine.spec)

    return routine


def remove_openmp_regions(routine):
    """
    Remove any OpenMP annotations and replace with `!$loki parallel` pragmas
    """
    with pragma_regions_attached(routine):
        for region in FindNodes(ir.PragmaRegion).visit(routine.body):
            if region.pragma.keyword.lower() == 'omp':

                if 'PARALLEL' in region.pragma.content:
                    region._update(
                        pragma=ir.Pragma(keyword='loki', content='parallel'),
                        pragma_post=ir.Pragma(keyword='loki', content='end parallel')
                    )

    # Now remove all other pragmas
    pragma_map = {
        pragma: None for pragma in FindNodes(ir.Pragma).visit(routine.body)
        if pragma.keyword.lower() == 'omp'
    }
    routine.body = Transformer(pragma_map).visit(routine.body)

    # Note: This is slightly hacky, as some of the "OMP PARALLEL DO" regions
    # are not detected correctly! So instead we hook on the "OMP DO SCHEDULE"
    # and remove all other OMP pragmas.

    pragma_map = {
        pragma: None for pragma in FindNodes(ir.Pragma).visit(routine.body)
        if pragma.keyword == 'OMP'
    }
    routine.body = Transformer(pragma_map).visit(routine.body)


def add_openmp_pragmas(routine, global_variables={}, field_group_types={}):
    """
    Add the OpenMP directives for a parallel driver region with an
    outer block loop.
    """
    block_dim_size = 'YDGEOMETRY%YRDIM%NGPBLKS'

    # First get local variables and separate scalars and arrays
    routine_arguments = routine.arguments
    local_variables = tuple(
        v for v in routine.variables if v not in routine_arguments
    )
    local_scalars = tuple(
        v.name for v in local_variables if isinstance(v, sym.Scalar)
    )
    # Filter arrays by block-dim size, as these are global
    local_arrays = tuple(
        v.name for v in local_variables
        if isinstance(v, sym.Array) and not v.dimensions[-1] == block_dim_size
    )

    with pragma_regions_attached(routine):
        with dataflow_analysis_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if not is_loki_pragma(region.pragma, starts_with='parallel'):
                    return

                # Accumulate the set of locally used symbols and chase parents
                symbols = tuple(region.uses_symbols | region.defines_symbols)
                symbols = tuple(dict.fromkeys(flatten(
                    s.parents if s.parent else s for s in symbols
                )))

                # Start with loop variables and add local scalars and arrays
                local_vars = tuple(dict.fromkeys(flatten(
                    loop.variable for loop in FindNodes(ir.Loop).visit(region.body)
                )))

                local_vars += tuple(v for v in symbols if v.name in local_scalars)
                local_vars += tuple(v for v in symbols if v.name in local_arrays)

                # Also add used symbols that might be field groups
                local_vars += tuple(dict.fromkeys(
                    v for v in symbols
                    if isinstance(v, sym.Scalar) and str(v.type.dtype) in field_group_types
                ))

                # Filter out known global variables
                local_vars = tuple(v for v in local_vars if v not in global_variables)

                # Make field group types firstprivate
                firstprivates = tuple(dict.fromkeys(
                    v.name for v in local_vars if v.type.dtype.name in field_group_types
                ))
                # Also make values that have an initial value firstprivate
                firstprivates += tuple(v.name for v in local_vars if v.type.initial)

                # Mark all other variables as private
                privates = tuple(dict.fromkeys(
                    v.name for v in local_vars if v.name not in firstprivates
                ))

                s_fp_vars = ", ".join(str(v) for v in firstprivates)
                s_firstprivate = f'FIRSTPRIVATE({s_fp_vars})' if firstprivates else ''
                s_private = f'PRIVATE({", ".join(str(v) for v in privates)})' if privates else ''
                pragma_parallel = ir.Pragma(
                    keyword='OMP', content=f'PARALLEL {s_private} {s_firstprivate}'
                )
                region._update(
                    pragma=pragma_parallel,
                    pragma_post=ir.Pragma(keyword='OMP', content='END PARALLEL')
                )

                # And finally mark all block-dimension loops as parallel
                with pragmas_attached(routine, node_type=ir.Loop):
                    for loop in FindNodes(ir.Loop).visit(region.body):
                        # Add OpenMP DO directives onto block loops
                        if loop.variable == 'JKGLO':
                            loop._update(
                                pragma=ir.Pragma(keyword='OMP', content='DO SCHEDULE(DYNAMIC,1)'),
                                pragma_post=ir.Pragma(keyword='OMP', content='END DO'),
                            )


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', '--path', type=click.Path(), default=Path.cwd(),
              help='Path to search for initial input sources.')
@click.option('--build', '-b', '--out', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--remove-regions/--no-remove-regions', default=True,
              help='Remove pragma-marked code regions.')
@click.option('--remove-openmp/--no-remove-openmp', default=True,
              help='Flag to replace OpenMP loop annotations with Loki pragmas.')
def inline(source, build, remove_regions, remove_openmp):
    """
    Inlines EC_PHYS and CALLPAR into EC_PHYS_DRV to expose the parallel loop.
    """
    source = Path(source)
    build = Path(build)

    # Get everything set up...
    ec_phys_drv = Sourcefile.from_file(source/'ec_phys_drv.F90')['EC_PHYS_DRV']
    ec_phys = Sourcefile.from_file(source/'ec_phys.F90')['EC_PHYS']
    callpar = Sourcefile.from_file(source/'callpar.F90')['CALLPAR']

    ec_phys_drv.enrich(ec_phys)
    ec_phys.enrich(callpar)

    # Clone original and change subroutine name
    ec_phys_fc = ec_phys_drv.clone(name='EC_PHYS_FC')

    # Substitute symbols that do not exist in the caller context after inlining
    substitute_spec_symbols(routine=ec_phys, mapping = {
        'DIMS%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'DIMS%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
        'DIMS%KLEVS': 'YDSURF%YSP_SBD%NLEVS',
        }
    )
    substitute_spec_symbols(routine=callpar, mapping = {
        'KDIM%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'KDIM%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
        }
    )

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Inlined EC_PHYS in {s:.2f}s'):
        # First, get the outermost call
        ecphys_calls = [
            c for c in FindNodes(CallStatement).visit(ec_phys_fc.body) if c.name == 'EC_PHYS'
        ]

        # Ouch, this is horrible!
        call_map = {}
        transform_sequence_association_append_map(call_map, ecphys_calls[0])
        ec_phys_fc.body = Transformer(call_map).visit(ec_phys_fc.body)

        inline_marked_subroutines(ec_phys_fc)

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Inlined CALLPAR in {s:.2f}s'):
        # Now just inline CALLPAR
        inline_marked_subroutines(ec_phys_fc, allowed_aliases=('JL', 'JK', 'J2D'))

    if remove_regions:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Remove marked regions in {s:.2f}s'):
            do_remove_marked_regions(ec_phys_fc)

    if remove_openmp:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Remove OpenMP regions in {s:.2f}s'):
            # Now remove OpenMP regions, as their symbols are not remapped
            remove_openmp_regions(ec_phys_fc)

    field_group_types = [
        'FIELD_VARIABLES', 'DIMENSION_TYPE', 'STATE_TYPE',
        'PERTURB_TYPE', 'AUX_TYPE', 'AUX_RAD_TYPE', 'FLUX_TYPE',
        'AUX_DIAG_TYPE', 'AUX_DIAG_LOCAL_TYPE', 'DDH_SURF_TYPE',
        'SURF_AND_MORE_LOCAL_TYPE', 'KEYS_LOCAL_TYPE',
        'PERTURB_LOCAL_TYPE', 'GEMS_LOCAL_TYPE',
        # 'SURF_AND_MORE_TYPE', 'MODEL_STATE_TYPE',
    ]

    global_variables = [
        'PGFL', 'PGFLT1', 'YDGSGEOM', 'YDMODEL',
        'YDDIM', 'YDSTOPH', 'YDGEOMETRY',
        'YDSURF', 'YDGMV', 'SAVTEND',
        'YGFL', 'PGMV', 'PGMVT1', 'ZGFL_DYN',
        'ZCONVCTY', 'YDDIMV', 'YDPHY2',
        'PHYS_MWAVE', 'ZSPPTGFIX', 'ZSURF_SERIAL'
    ]

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Re-wrote OpenMP regions in {s:.2f}s'):
        # Now remove OpenMP regions, as their symbols are not remapped
        remove_openmp_regions(ec_phys_fc)

        # Add OpenMP pragmas around marked loops
        add_openmp_pragmas(
            routine=ec_phys_fc,
            field_group_types=field_group_types,
            global_variables=global_variables
        )

    # Replace the docstring to mark routine as auto-generated
    ec_phys_fc.docstring = """
    !**** *EC_PHYS_FC* - Standaline physics forecast-only driver

    !     Purpose.
    !     --------
    !           An automatically generated driver routine that exposes
    !           parallel regions alongside the physics control flow for
    !           using different parallelisation methods, including GPU offload.

    !     **  THIS SUBROUTINE HAS BEEN AUTO-GENERATED BY LOKI  **

    !     It is a combination of EC_PHYS_DRV, EC_PHYS and CALLPAR and be re-derived from them.

"""

    # Create source file, wrap as a module and write to file
    srcfile = Sourcefile(path=build/'ec_phys_fc_mod.F90', ir=(ec_phys_fc,))
    ModuleWrapTransformation(module_suffix='_MOD').apply(srcfile, role='kernel')

    srcfile.write()
