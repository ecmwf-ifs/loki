#!/usr/bin/env python

# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Loki head script for source-to-source transformations for batch processing
via the :any:`Scheduler`.
"""

from pathlib import Path
import sys
import click

from loki import config as loki_config, Sourcefile, as_tuple, info
from loki.batch import Scheduler, SchedulerConfig, ProcessingStrategy
from loki.cli.common import cli, frontend_options, scheduler_options

from loki.transformations.build_system import FileWriteTransformation


@cli.command()
@frontend_options
@scheduler_options
@click.option('--mode', '-m', default='idem', type=click.STRING,
              help='Transformation mode, selecting which code transformations to apply.')
@click.option('--config', default=None, type=click.Path(),
              help='Path to custom scheduler configuration file')
@click.option('--plan-file', type=click.Path(), default=None,
              help='Process pipeline in planning mode and generate CMake "plan" file.')
@click.option('--callgraph', '-g', type=click.Path(), default=None,
              help='Generate and display the subroutine callgraph.')
@click.option('--root', type=click.Path(), default=None,
              help='Root path to which all paths are relative to.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during batch processing')
def convert(
        frontend_opts, scheduler_opts, mode, config, plan_file, callgraph, root, log_level
):
    """
    Batch-processing mode for Fortran-to-Fortran transformations that
    employs a :class:`Scheduler` to process large numbers of source
    files.

    Based on the given "mode" string, configuration file, source file
    paths and build arguments the :any:`Scheduler` will perform
    automatic call-tree exploration and apply a set of
    :any:`Transformation` objects to this call tree.
    """

    loki_config['log-level'] = log_level

    if plan_file is not None:
        processing_strategy = ProcessingStrategy.PLAN
        info(f'[Loki] Creating CMake plan file from config: {config}')
    else:
        processing_strategy = ProcessingStrategy.DEFAULT
        info(f'[Loki] Batch-processing source files using config: {config} ')

    config = SchedulerConfig.from_file(config)

    # set default transformation mode in Scheduler config
    config.default['mode'] = mode

    # Note, in order to get function inlinig correct, we need full knowledge
    # of any imported symbols and functions. Since we cannot yet retro-fit that
    # after creation, we need to make sure that the order of definitions can
    # be used to create a coherent stack of type definitions.
    # definitions with new scheduler not necessary anymore. However, "source" need to be adjusted
    #  in order to allow the scheduler to find the dependencies
    definitions = []
    for h in scheduler_opts.header:
        sfile = Sourcefile.from_file(filename=h, definitions=definitions, **frontend_opts.asdict)
        definitions = definitions + list(sfile.definitions)

    # Create a scheduler to bulk-apply source transformations
    paths = [Path(p) for p in as_tuple(scheduler_opts.source)]
    paths += [Path(h).parent for h in as_tuple(scheduler_opts.header)]
    # Skip full source parse for planning mode
    full_parse = processing_strategy == ProcessingStrategy.DEFAULT
    scheduler = Scheduler(
        paths=paths, config=config, full_parse=full_parse,
        definitions=definitions, output_dir=scheduler_opts.build, **frontend_opts.asdict
    )

    # If requested, apply a custom pipeline from the scheduler config
    # Note that this new entry point will bypass all other default
    # behaviour and exit immediately after.
    if mode not in config.pipelines:
        msg = f'[Loki] ERROR: Pipeline or transformation mode {mode} not found in config file.\n'
        msg += '[Loki] Please provide a config file with configured transformation or pipelines instead.\n'
        sys.exit(msg)

    info(f'[Loki-transform] Applying custom pipeline {mode} from config:')
    info(str(config.pipelines[mode]))

    # scheduler.process(config.pipelines[mode], proc_strategy=processing_strategy)
    scheduler.process_config(proc_strategy=processing_strategy)

    mode = mode.replace('-', '_')  # Sanitize mode string

    # Write out all modified source files into the build directory
    file_write_trafo = scheduler.config.transformations.get('FileWriteTransformation', None)
    if not file_write_trafo:
        file_write_trafo = FileWriteTransformation(cuf='cuf' in mode)
    scheduler.process(transformation=file_write_trafo, proc_strategy=processing_strategy)

    if plan_file is not None:
        scheduler.write_cmake_plan(plan_file, rootpath=root)

    if callgraph:
        scheduler.callgraph(callgraph)


@cli.command('plan')
@frontend_options
@scheduler_options
@click.option('--mode', '-m', default='idem', type=click.STRING,
              help='Transformation mode, selecting which code transformations to apply.')
@click.option('--config', '-c', type=click.Path(),
              help='Path to configuration file.')
@click.option('--root', type=click.Path(), default=None,
              help='Root path to which all paths are relative to.')
@click.option('--callgraph', '-g', type=click.Path(), default=None,
              help='Generate and display the subroutine callgraph.')
@click.option('--plan-file', type=click.Path(),
              help='CMake "plan" file to generate.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during batch processing')
@click.pass_context
def plan(ctx, *_args, **_kwargs):
    """
    Create a "plan", a schedule of files to inject and transform for a
    given configuration.
    """
    return ctx.forward(convert)
