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

from pathlib import Path
import click
from codetiming import Timer

from loki import config as loki_config, Sourcefile, Dimension, info
from loki.ir import (
    FindNodes, SubstituteStringExpressions, Transformer, CallStatement
)

from loki.transformations.inline import inline_marked_subroutines
from loki.transformations.sanitise import (
    merge_associates, resolve_associates,
    transform_sequence_association_append_map
)
from loki.transformations.remove_code import do_remove_marked_regions
from loki.transformations.drhook import DrHookTransformation
from loki.transformations.build_system import ModuleWrapTransformation
from loki.transformations.parallel import (
    remove_openmp_regions, add_openmp_regions,
    remove_block_loops, add_block_loops,
    remove_field_api_view_updates, add_field_api_view_updates,
    remove_explicit_firstprivatisation, create_explicit_firstprivatisation
)


# List of types that we know to be FIELD API groups
field_group_types = [
    'FIELD_VARIABLES', 'STATE_TYPE', 'MODEL_STATE_TYPE',
    'PERTURB_TYPE', 'AUX_TYPE', 'AUX_RAD_TYPE', 'FLUX_TYPE',
    'AUX_DIAG_TYPE', 'AUX_DIAG_LOCAL_TYPE', 'DDH_SURF_TYPE',
    'SURF_AND_MORE_LOCAL_TYPE', 'KEYS_LOCAL_TYPE',
    'PERTURB_LOCAL_TYPE', 'GEMS_LOCAL_TYPE',
    'FIELD_3RB_ARRAY', 'FIELD_4RB_ARRAY', 'ECPHYS_OPTS_TYPE'
]

fgroup_dimension = ['DIMENSION_TYPE']
fgroup_firstprivates = ['SURF_AND_MORE_TYPE']
lcopies_firstprivates = {'ZSURF': 'ZSURFACE'}

# List of variables that we know to have global scope
global_variables = [
    'PGFL', 'PGFLT1', 'YDGSGEOM', 'YDMODEL',
    'YDDIM', 'YDSTOPH', 'YDGEOMETRY',
    'YDSURF', 'YDGMV', 'SAVTEND',
    'YGFL', 'PGMV', 'PGMVT1', 'ZGFL_DYN',
    'ZCONVCTY', 'YDDIMV', 'YDPHY2',
    'PHYS_MWAVE', 'ZSPPTGFIX', 'ZSURFACE'
]


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', '--path', type=click.Path(), default=Path.cwd(),
              help='Path to search for initial input sources.')
@click.option('--build', '-b', '--out', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--remove-openmp/--no-remove-openmp', default=True,
              help='Flag to replace OpenMP loop annotations with Loki pragmas.')
@click.option('--sanitize-assoc/--no-sanitize-assoc', default=True,
              help='Flag to trigger ASSOCIATE block sanitisation.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during processing')
def inline(source, build, remove_openmp, sanitize_assoc, log_level):
    """
    Inlines EC_PHYS and CALLPAR into EC_PHYS_DRV to expose the parallel loop.
    """
    loki_config['log-level'] = log_level

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
    ec_phys.spec = SubstituteStringExpressions({
        'DIMS%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'DIMS%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
        'DIMS%KLEVS': 'YDSURF%YSP_SBD%NLEVS',
    }, scope=ec_phys).visit(ec_phys.spec)
    callpar.spec = SubstituteStringExpressions({
        'KDIM%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'KDIM%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
    }, scope=callpar).visit(callpar.spec)

    # Before inlining, remove DR_HOOK calls from the inner routines
    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Removed inner DR_HOOK calls in {s:.2f}s'):
        DrHookTransformation(kernel_only=False, remove=True).apply(ec_phys, role='driver')
        DrHookTransformation(kernel_only=False, remove=True).apply(callpar, role='driver')

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

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Remove marked regions in {s:.2f}s'):
        do_remove_marked_regions(ec_phys_fc)

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Remove OpenMP regions in {s:.2f}s'):
        # Now remove OpenMP regions, as their symbols are not remapped
        remove_openmp_regions(ec_phys_fc, insert_loki_parallel=True)

    if not remove_openmp:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Re-wrote OpenMP regions in {s:.2f}s'):
            # Re-insert OpenMP parallel regions after inlining
            fgtypes = field_group_types + fgroup_dimension + fgroup_firstprivates
            add_openmp_regions(
                routine=ec_phys_fc,
                field_group_types=fgtypes,
                global_variables=global_variables
            )

    if sanitize_assoc:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Merged associate blocks in {s:.2f}s'):
            # First move all associatesion up to the outermost
            merge_associates(ec_phys_fc, max_parents=2)

        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Resolved associate blocks in {s:.2f}s'):
            # Then resolve all remaining inner associations
            resolve_associates(ec_phys_fc, start_depth=1)

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

    # Rename DR_HOOK calls to ensure appropriate performance logging
    DrHookTransformation(
        rename={'EC_PHYS_DRV': 'EC_PHYS_FC'}, kernel_only=False
    ).apply(ec_phys_fc, role='driver')

    # Create source file, wrap as a module adjust DR_HOOK labels and write to file
    srcfile = Sourcefile(path=build/'ec_phys_fc_mod.F90', ir=(ec_phys_fc,))
    ModuleWrapTransformation(module_suffix='_MOD').apply(srcfile, role='kernel')

    srcfile.write()


@cli.command()
@click.option('--source', '-s', '--path', type=click.Path(), default=Path.cwd(),
              help='Path to search for initial input sources.')
@click.option('--build', '-b', '--out', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--remove-block-loop/--no-remove-block-loop', default=True,
              help='Flag to replace OpenMP loop annotations with Loki pragmas.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during processing')
def parallel(source, build, remove_block_loop, log_level):
    """
    Generate parallel regions with OpenMP and OpenACC dispatch.
    """
    loki_config['log-level'] = log_level

    source = Path(source)
    build = Path(build)

    # Get everything set up...
    ec_phys_fc = Sourcefile.from_file(source/'ec_phys_fc_mod.F90')['EC_PHYS_FC']

    # Clone original and change subroutine name
    ec_phys_parallel = ec_phys_fc.clone(name='EC_PHYS_PARALLEL')

    blocking = Dimension(
        name='block', index='JKGLO', index_aliases='IBL',
        size='YDGEM%NGPTOT', aliases='YDDIM%NGPBLKS',
        bounds=('YDGEM%NGPTOT', 'YDDIM%NPROMA'),
        bounds_aliases=('ICST', 'ICEND')
    )

    if remove_block_loop:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Re-generated block loops in {s:.2f}s'):
            # Remove explicit firstprivatisation
            remove_explicit_firstprivatisation(
                ec_phys_parallel.body, fprivate_map=lcopies_firstprivates, routine=ec_phys_parallel
            )

            # Strip the outer block loop and FIELD-API boilerplate
            remove_block_loops(ec_phys_parallel)

            remove_field_api_view_updates(
                ec_phys_parallel, field_group_types=field_group_types+fgroup_firstprivates
            )

            # The add them back in according to parallel region
            add_block_loops(ec_phys_parallel, dimension=blocking)

            add_field_api_view_updates(
                ec_phys_parallel, dimension=blocking,
                field_group_types=field_group_types+fgroup_firstprivates
            )

            # Re-insert explicit firstprivate copies
            create_explicit_firstprivatisation(ec_phys_parallel, fprivate_map=lcopies_firstprivates)

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Added OpenMP regions in {s:.2f}s'):
        # Add OpenMP pragmas around marked loops
        add_openmp_regions(
            routine=ec_phys_parallel,
            field_group_types=field_group_types + fgroup_dimension,
            global_variables=global_variables
        )

    # Rename DR_HOOK calls to ensure appropriate performance logging
    DrHookTransformation(
        rename={'EC_PHYS_FC': 'EC_PHYS_PARALLEL'}, kernel_only=False
    ).apply(ec_phys_parallel, role='driver')

    # Create source file, wrap as a module and write to file
    srcfile = Sourcefile(path=build/'ec_phys_parallel_mod.F90', ir=(ec_phys_parallel,))
    ModuleWrapTransformation(module_suffix='_MOD').apply(srcfile, role='kernel')

    srcfile.write()
