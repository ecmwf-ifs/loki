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

from loki import config as loki_config, Sourcefile, info
from loki.batch import Scheduler, Pipeline

from loki.transformations.build_system import ModuleWrapTransformation
from loki.transformations.inline import InlineTransformation
from loki.transformations.remove_code import RemoveCodeTransformation
from loki.transformations.sanitise import SequenceAssociationTransformation


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', '--path', type=click.Path(), default=Path.cwd(),
              help='Path to search for initial input sources.')
@click.option('--build', '-b', '--out', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during processing')
def inline(source, build, log_level):
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

    # Create sanitisation and inlining pipeline for control flow routines
    pipeline = Pipeline()
    pipeline += SequenceAssociationTransformation()
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

    # Create source file, wrap as a module and write to file
    srcfile = Sourcefile(path=build/'ec_phys_fc_mod.F90', ir=(ec_phys_fc,))
    ModuleWrapTransformation(module_suffix='_MOD').apply(srcfile, role='kernel')

    srcfile.write()
