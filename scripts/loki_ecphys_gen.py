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
from loki.analyse import dataflow_analysis_attached
from loki.batch import Scheduler, Pipeline
from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, SubstituteExpressions,
    Transformer, pragma_regions_attached, is_loki_pragma,
    get_pragma_parameters
)

from loki.transformations.build_system import ModuleWrapTransformation
from loki.transformations.drhook import DrHookTransformation
from loki.transformations.extract import outline_region
from loki.transformations.inline import InlineTransformation
from loki.transformations.parallel import (
    remove_openmp_regions, add_openmp_regions,
    remove_block_loops, add_block_loops,
    remove_field_api_view_updates, add_field_api_view_updates,
    remove_explicit_firstprivatisation, create_explicit_firstprivatisation
)
from loki.transformations.remove_code import (
    RemoveCodeTransformation, do_remove_unused_imports
)
from loki.transformations.sanitise import (
    SequenceAssociationTransformation, SubstituteExpressionTransformation,
    do_merge_associates, do_resolve_associates, ResolveAssociatesTransformer
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
# The FIELD VARIABLE here would trigger correct PRIVATE markers
# for YDVARS, despite firstprivatisation trick. However, this triggers
# Nvidia bug, so let's not for now... Also, once added here, need to
# remove from field_group_types above
fgroup_firstprivates = ['SURF_AND_MORE_TYPE']  #'FIELD_VARIABLES'
lcopies_firstprivates = {'ZSURF': 'ZSURFACE', 'ZDVARS': 'YDVARS'}

# List of variables that we know to have global scope
shared_variables = [
    'PGFL', 'PGFLT1', 'YDGSGEOM', 'YDMODEL',
    'YDDIM', 'YDSTOPH', 'YDGEOMETRY',
    'YDSURF', 'YDGMV', 'SAVTEND',
    'YGFL', 'PGMV', 'PGMVT1', 'ZGFL_DYN',
    'ZCONVCTY', 'YDDIMV', 'YDPHY2',
    'PHYS_MWAVE', 'ZSPPTGFIX',
    'ZSURFACE', 'YDVARS'
]


blocking_outer = Dimension(
    name='block', index=('JKGLO', 'IBL'),
    lower=('1', 'ICST'), upper=('YDGEM%NGPTOT', 'ICEND'),
    step='YDDIM%NPROMA', size='YDDIM%NGPBLKS',
)

blocking_driver = Dimension(
    name='block', index=('JKGLO', 'IBL'),
    lower=('1', 'ICST'), upper=('YDGEOMETRY%YRGEM%NGPTOT', 'ICEND'),
    step='YDGEOMETRY%YRDIM%NPROMA', size='YDGEOMETRY%YRDIM%NGPBLKS',
)


def remove_redundant_declarations(routine):
    """
    Removes all local symbol declarations that are not being used in
    the routine body.
    """
    used_symbols = FindVariables(unique=True).visit(routine.body)
    # used_symbols |= {v.parents for v in used_symbols}
    used_symbols = tuple(v.name for v in used_symbols)

    decl_map = {}
    for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
        # Filter out routine arguments; we don't want to change the signature
        if any(s.name.lower() in routine._dummies for s in decl.symbols):
            continue

        # Filter out variables that are not used in the routine body
        symbols = tuple(s for s in decl.symbols if s.name in used_symbols)
        if symbols == decl.symbols:
            continue

        # Remove if no symbols are used, otherwise strip unused ones
        decl_map[decl] = decl.clone(symbols=symbols) if symbols else None
    routine.spec = Transformer(decl_map).visit(routine.spec)


def promote_temporary_arrays(routine, horizontal, blocking):
    """
    Promote remaining block-scoped local temporary arrays to full size.
    """
    block_size = routine.resolve_typebound_var(blocking.size)
    block_idx = routine.resolve_typebound_var(blocking.index)

    arrays_to_promote = tuple(
        v.name for v in routine.variables
        if isinstance(v, sym.Array) and \
        not v.name in routine._dummies and \
        v.shape[0] == horizontal.size and \
        not v.shape[-1] == blocking.size
    )

    # First, update the body symbols (which requires the shape)
    vmap = {}
    for var in FindVariables(unique=False).visit(routine.body):
        if var.name not in arrays_to_promote:
            continue
        if var.shape and block_size in var.shape:
            continue

        if var.dimensions:
            new_dims = var.dimensions + (block_idx,)
        else:
            new_dims = tuple(sym.Range((None, None)) for _ in var.shape) + (block_idx,)
        vmap[var] = var.clone(dimensions=new_dims)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Then update the declaration and the shape with it
    for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
        if not any(s.name in arrays_to_promote for s in decl.symbols):
            continue

        symbols = tuple(
            s.clone(
                dimensions=s.dimensions+(block_size,),
                type=s.type.clone(shape=s.type.shape+(block_size,))
            )
            if s.name in arrays_to_promote else s
            for s in decl.symbols
        )
        decl._update(symbols=symbols)


def outline_driver_routines(routine):
    """
    Extracts driver routines and replaces them with an appropriate
    :any:`CallStatement`.
    """
    imports = FindNodes(ir.Import).visit(routine.spec)
    mapper = {}
    driver_routines = []
    parent_vmap = routine.variable_map
    with pragma_regions_attached(routine):
        for region in FindNodes(ir.PragmaRegion).visit(routine.body):
            if not is_loki_pragma(region.pragma, starts_with='outline'):
                continue

            # Resolve associations in the local region before processing
            ResolveAssociatesTransformer(inplace=True).visit(region)

        with dataflow_analysis_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if not is_loki_pragma(region.pragma, starts_with='outline'):
                    continue

                # Name the external routine
                parameters = get_pragma_parameters(region.pragma, starts_with='outline')
                name = parameters['name']

                intent_map = {}
                intent_map['in'] = tuple(parent_vmap[v.lower()] for v in parameters.get('in', '').split(',') if v)
                intent_map['inout'] = tuple(parent_vmap[v.lower()] for v in parameters.get('inout', '').split(',') if v)
                intent_map['out'] = tuple(parent_vmap[v.lower()] for v in parameters.get('out', '').split(',') if v)

                call, region_routine = outline_region(region, name, imports, intent_map=intent_map)

                do_remove_unused_imports(region_routine)

                driver_routines.append(region_routine)

                if 'parallel' in parameters:
                    # Propagate any "parallel" pragma marker for further processing
                    pragma = ir.Pragma(keyword='loki', content='parallel')
                    pragma_post = ir.Pragma(keyword='loki', content='end parallel')
                    region_routine.body.prepend((ir.Comment(''), ir.Comment(''), pragma))
                    region_routine.body.append((pragma_post, ir.Comment('')))

                # Replace region by call in original routine
                mapper[region] = call

            routine.body = Transformer(mapper=mapper).visit(routine.body)

    return driver_routines

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

    config =  {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': False,
            'strict': True,
            'enable_imports': False,
        },
        'routines' : {
            'ec_phys_drv' : {'role': 'driver',},
            'ec_phys' : {},
            'callpar' : {},
        }
    }

    # Parse and enrich the needed source files
    scheduler = Scheduler(config=config, paths=source, output_dir=build)

    ec_phys_fc = scheduler['#ec_phys_drv'].ir

    # Substitute symbols that do not exist in the caller context after inlining
    subs_expressions = {
        'DIMS%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'DIMS%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
        'DIMS%KLEVS': 'YDSURF%YSP_SBD%NLEVS',
        'KDIM%KLON': 'YDGEOMETRY%YRDIM%NPROMA',
        'KDIM%KLEV': 'YDGEOMETRY%YRDIMV%NFLEVG',
    }

    # Create sanitisation and inlining pipeline for control flow routines
    pipeline = Pipeline()
    pipeline += SequenceAssociationTransformation()
    pipeline += SubstituteExpressionTransformation(
        expression_map=subs_expressions, substitute_spec=True, substitute_body=False
    )
    pipeline += DrHookTransformation(kernel_only=True, remove=True)
    pipeline += InlineTransformation(
        inline_elementals=False, inline_marked=True, remove_dead_code=False,
        allowed_aliases=('JL', 'JK', 'J2D')
    )
    pipeline += RemoveCodeTransformation(
        remove_marked_regions=True, mark_with_comment=True
    )

    # Apply the inlining pipeline
    info('[Loki::EC-Physics] Applying custom pipeline from config:')
    info(pipeline)

    scheduler.process(pipeline)

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Remove OpenMP regions in {s:.2f}s'):
        # Now remove OpenMP regions, as their symbols are not remapped
        remove_openmp_regions(ec_phys_fc, insert_loki_parallel=True)

    if not remove_openmp:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Re-wrote OpenMP regions in {s:.2f}s'):
            # Re-insert OpenMP parallel regions after inlining
            fgtypes = field_group_types + fgroup_dimension + fgroup_firstprivates
            add_openmp_regions(
                routine=ec_phys_fc, dimension=blocking_outer,
                field_group_types=fgtypes,
                shared_variables=shared_variables
            )

    if sanitize_assoc:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Merged associate blocks in {s:.2f}s'):
            # First move all associatesion up to the outermost
            do_merge_associates(ec_phys_fc, max_parents=2)

        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Resolved associate blocks in {s:.2f}s'):
            # Then resolve all remaining inner associations
            do_resolve_associates(ec_phys_fc, start_depth=1)


    # Change subroutine name
    ec_phys_fc.name = 'EC_PHYS_FC'

    # Replace the docstring to mark routine as auto-generated
    ec_phys_fc.docstring = """
    !**** *EC_PHYS_FC* - Standalone physics forecast-only driver

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
@click.option('--promote-local-arrays/--no-promote-local-arrays', default=True,
              help='Flag to promote local block-scope arrays to full size')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during processing')
def parallel(source, build, remove_block_loop, promote_local_arrays, log_level):
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

    if remove_block_loop:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Re-generated block loops in {s:.2f}s'):
            # Remove explicit firstprivatisation
            remove_explicit_firstprivatisation(
                ec_phys_parallel.body, fprivate_map=lcopies_firstprivates, scope=ec_phys_parallel
            )

            # Strip the outer block loop and FIELD-API boilerplate
            remove_block_loops(ec_phys_parallel, dimension=blocking_outer)

            remove_field_api_view_updates(
                ec_phys_parallel, dim_object='IDIMS',
                field_group_types=field_group_types+fgroup_firstprivates
            )

            # The add them back in according to parallel region
            add_block_loops(ec_phys_parallel, dimension=blocking_outer)

            add_field_api_view_updates(
                ec_phys_parallel, dim_object='IDIMS', dimension=blocking_outer,
                field_group_types=field_group_types+fgroup_firstprivates
            )

            # Re-insert explicit firstprivate copies
            create_explicit_firstprivatisation(ec_phys_parallel, fprivate_map=lcopies_firstprivates)

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Extracted driver routines in {s:.2f}s'):

        driver_routines = outline_driver_routines(ec_phys_parallel)
        for driver in driver_routines:
            # Re-insert block loop with FIELD API view updates
            add_block_loops(routine=driver, dimension=blocking_driver)

            add_field_api_view_updates(
                routine=driver, dim_object='IDIMS', dimension=blocking_driver,
                field_group_types=field_group_types+fgroup_firstprivates
            )

            # Re-insert explicit firstprivate copies
            create_explicit_firstprivatisation(driver, fprivate_map=lcopies_firstprivates)

            add_openmp_regions(
                routine=driver, field_group_types=field_group_types + fgroup_dimension,
                global_variables=global_variables
            )

            # Create a new source file for the extracted routine
            filename = driver.name.lower() + '_mod.F90'
            sourcefile = Sourcefile(ir=ir.Section(driver), path=build/filename)

            ModuleWrapTransformation(module_suffix='_MOD').apply(sourcefile, role='kernel')
            sourcefile.write()

            # Add an implicit C-style import to the control-flow routine
            symbols = (sym.DeferredTypeSymbol(name=driver.name), )
            imprt = ir.Import(module=f'{driver.name}_MOD', symbols=symbols)
            ec_phys_parallel.spec.prepend(imprt)

    if promote_local_arrays:
        with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Promoted local arrays in {s:.2f}s'):
            # Bit of a hack, but easier that way
            remove_redundant_declarations(routine=ec_phys_parallel)

            promote_temporary_arrays(
                routine=ec_phys_parallel,
                horizontal=Dimension(name='horizontal', index='JL', size='YDGEOMETRY%YRDIM%NPROMA'),
                blocking=Dimension(name='blocking', index='IBL', size='YDGEOMETRY%YRDIM%NGPBLKS'),
            )

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
