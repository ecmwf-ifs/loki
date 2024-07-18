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
from loki.batch import Scheduler, Pipeline

from loki.transformations.build_system import ModuleWrapTransformation
from loki.transformations.drhook import DrHookTransformation
from loki.transformations.inline import InlineTransformation
from loki.transformations.parallel import (
    remove_openmp_regions, add_openmp_regions
)
from loki.transformations.remove_code import RemoveCodeTransformation
from loki.transformations.sanitise import (
    SequenceAssociationTransformation, SubstituteExpressionTransformation,
    do_merge_associates, do_resolve_associates
)


# List of types that we know to be FIELD API groups
field_group_types = [
    'FIELD_VARIABLES', 'DIMENSION_TYPE', 'STATE_TYPE',
    'PERTURB_TYPE', 'AUX_TYPE', 'AUX_RAD_TYPE', 'FLUX_TYPE',
    'AUX_DIAG_TYPE', 'AUX_DIAG_LOCAL_TYPE', 'DDH_SURF_TYPE',
    'SURF_AND_MORE_LOCAL_TYPE', 'KEYS_LOCAL_TYPE',
    'PERTURB_LOCAL_TYPE', 'GEMS_LOCAL_TYPE',
    # 'SURF_AND_MORE_TYPE', 'MODEL_STATE_TYPE',
]

# List of variables that we know to have global scope
shared_variables = [
    'PGFL', 'PGFLT1', 'YDGSGEOM', 'YDMODEL',
    'YDDIM', 'YDSTOPH', 'YDGEOMETRY',
    'YDSURF', 'YDGMV', 'SAVTEND',
    'YGFL', 'PGMV', 'PGMVT1', 'ZGFL_DYN',
    'ZCONVCTY', 'YDDIMV', 'YDPHY2',
    'PHYS_MWAVE', 'ZSPPTGFIX', 'ZSURFACE'
]


blocking = Dimension(
    name='block', index=('JKGLO', 'IBL'), size='YDDIM%NGPBLKS'
)


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
            add_openmp_regions(
                routine=ec_phys_fc, dimension=blocking,
                field_group_types=field_group_types,
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
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during processing')
def parallel(source, build, log_level):
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

    with Timer(logger=info, text=lambda s: f'[Loki::EC-Physics] Added OpenMP regions in {s:.2f}s'):
        # Add OpenMP pragmas around marked loops
        add_openmp_regions(
            routine=ec_phys_parallel,
            field_group_types=field_group_types,
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
