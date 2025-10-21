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

from loki import (
    config as loki_config, Sourcefile, Frontend, as_tuple,
    set_excepthook, auto_post_mortem_debugger, info
)
from loki.batch import Scheduler, SchedulerConfig, ProcessingStrategy

from loki.transformations.build_system import FileWriteTransformation


@click.group()
@click.option('--debug/--no-debug', default=False, show_default=True,
              help=('Enable / disable debug mode. This automatically attaches '
                    'a debugger when exceptions occur'))
def cli(debug):
    if debug:
        set_excepthook(hook=auto_post_mortem_debugger)


@cli.command()
@click.option('--mode', '-m', default='idem', type=click.STRING,
              help='Transformation mode, selecting which code transformations to apply.')
@click.option('--config', default=None, type=click.Path(),
              help='Path to custom scheduler configuration file')
@click.option('--build', '-b', '--out-path', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--source', '-s', '--path', type=click.Path(), multiple=True,
              help='Path to search during source exploration.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--cpp/--no-cpp', default=False,
              help='Trigger C-preprocessing of source files.')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--define', '-D', multiple=True,
              help='Additional symbol definitions for the C-preprocessor')
@click.option('--omni-include', type=click.Path(), multiple=True,
              help='Additional path for header files, specifically for OMNI')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional .xmod file(s) for OMNI')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
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
        mode, config, build, source, header, cpp, include, define,
        omni_include, xmod, frontend, plan_file, callgraph, root,
        log_level
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

    build_args = {
        'preprocess': cpp,
        'includes': include,
        'defines': define,
        'xmods': xmod,
        'omni_includes': omni_include,
    }

    frontend = Frontend[frontend.upper()]
    frontend_type = Frontend.FP if frontend == Frontend.OMNI else frontend

    # Note, in order to get function inlinig correct, we need full knowledge
    # of any imported symbols and functions. Since we cannot yet retro-fit that
    # after creation, we need to make sure that the order of definitions can
    # be used to create a coherent stack of type definitions.
    # definitions with new scheduler not necessary anymore. However, "source" need to be adjusted
    #  in order to allow the scheduler to find the dependencies
    definitions = []
    for h in header:
        sfile = Sourcefile.from_file(filename=h, frontend=frontend_type, definitions=definitions,
                **build_args)
        definitions = definitions + list(sfile.definitions)

    # Create a scheduler to bulk-apply source transformations
    paths = [Path(p) for p in as_tuple(source)]
    paths += [Path(h).parent for h in as_tuple(header)]
    # Skip full source parse for planning mode
    full_parse = processing_strategy == ProcessingStrategy.DEFAULT
    scheduler = Scheduler(
        paths=paths, config=config, frontend=frontend,
        full_parse=full_parse, definitions=definitions,
        output_dir=build, **build_args
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

    scheduler.process(config.pipelines[mode], proc_strategy=processing_strategy)

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
@click.option('--mode', '-m', default='idem', type=click.STRING,
              help='Transformation mode, selecting which code transformations to apply.')
@click.option('--config', '-c', type=click.Path(),
              help='Path to configuration file.')
@click.option('--header', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--source', '-s', type=click.Path(), multiple=True,
              help='Path to source files to transform.')
@click.option('--build', '-b', type=click.Path(), default=None,
              help='Path to build directory for source generation.')
@click.option('--root', type=click.Path(), default=None,
              help='Root path to which all paths are relative to.')
@click.option('--include', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--cpp/--no-cpp', default=False,
              help='Trigger C-preprocessing of source files.')
@click.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
              help='Frontend parser to use (default FP)')
@click.option('--callgraph', '-g', type=click.Path(), default=None,
              help='Generate and display the subroutine callgraph.')
@click.option('--plan-file', type=click.Path(),
              help='CMake "plan" file to generate.')
@click.option('--log-level', '-l', default='info', envvar='LOKI_LOGGING',
              type=click.Choice(['debug', 'detail', 'perf', 'info', 'warning', 'error']),
              help='Log level to output during batch processing')
@click.pass_context
def plan(
        ctx, mode, config, header, source, build, root, include,
        cpp, frontend, callgraph, plan_file, log_level
):
    """
    Create a "plan", a schedule of files to inject and transform for a
    given configuration.
    """
    return ctx.forward(convert)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
