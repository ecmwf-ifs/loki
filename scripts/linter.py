#!/usr/bin/env python3

import click
from concurrent.futures import as_completed
from itertools import chain
from logging import FileHandler
from pathlib import Path

from fparser.two.utils import FortranSyntaxError

from loki.logging import logger, DEBUG, warning, info, error, debug
from loki.sourcefile import SourceFile
from loki.frontend import FP
from loki.build import workqueue


def get_relative_path_and_anchor(path, anchor):
    """
    If the given path is an absolute path, it is converted into a
    relative path and returns the corresponding anchor.
    """
    p = Path(path)
    if p.is_absolute():
        anchor = p.anchor
        p = p.relative_to(anchor)
        return anchor, str(p)
    return anchor, path


def get_file_list(includes, excludes, basedir):
    """
    From given lists of include and exclude file names or patterns
    this routine builds the actual lists of file names to be
    included and excluded.
    """
    # Transform absolute paths to relative paths
    includes = [get_relative_path_and_anchor(p, basedir) for p in includes]
    excludes = [get_relative_path_and_anchor(p, basedir) for p in excludes]
    # Building include and exclude lists first...
    incl = [Path(p[0]).glob(p[1]) for p in includes]
    incl = sorted(chain(*incl))
    excl = [Path(p[0]).glob(p[1]) for p in excludes]
    excl = sorted(chain(*excl))
    # ...and sanitising them afterwards.
    excl = [f for f in excl if f in incl]
    incl = [f for f in incl if f not in excl]
    return incl, excl


def parse_file(filename, frontend=FP):
    debug('[%s] Parsing...', filename)
    try:
        _ = SourceFile.from_file(filename, frontend=frontend)
    except FortranSyntaxError as excinfo:
        error('[%s] Parsing failed: %s\n', filename, excinfo)
        return False
    except Exception as excinfo:
        error('[%s] Parsing failed: %s\n', filename, excinfo)
        return False
    info('[%s] Parsing completed without error.', filename)
    return True


@click.group()
@click.option('--debug/--no-debug', default=False,
              help='Enable / disable debug mode. Includes more verbose output.')
@click.option('--log', type=click.Path(),
              help='Write more detailed information to a log file.')
@click.pass_context
def cli(ctx, debug, log):
    ctx.obj['DEBUG'] = debug
    if debug:
        logger.setLevel(DEBUG)
    if log:
        file_handler = FileHandler(log, mode='w')
        file_handler.setLevel(DEBUG)
        logger.addHandler(file_handler)


@cli.command(help='Check for syntax errors and compliance to coding rules.')
@click.option('--include', '-I', type=str, multiple=True,
              help='File name or pattern for file names to be checked.')
@click.option('--exclude', '-X', type=str, multiple=True,
              help=('File name or pattern for file names to be excluded. '
                    'This allows to exclude files that were included by '
                    '--include.'))
@click.option('--basedir', type=click.Path(exists=True, file_okay=False),
              help=('(Default: current working directory) Base directory '
                    'relative to which relative --include/--exclude patterns '
                    'are interpreted.'))
@click.option('--workers', type=int, default=4,
              help=('(Default: 4) Number of worker processes to use. With '
                    '--debug enabled this option is ignored and only one '
                    'process is used.'))
@click.pass_context
def check(ctx, include, exclude, basedir, workers):
    info('Base directory: %s', basedir)
    info('Include patterns:')
    for p in include:
        info('  - %s', p)
    info('Exclude patterns:')
    for p in exclude:
        info('  - %s', p)
    info('')

    debug('Searching for files using specified patterns...')
    files, excludes = get_file_list(include, exclude, basedir)
    info('%d files selected for checking (%d files excludes).',
         len(files), len(excludes))
    info('Using %d workers.', workers)
    info('')

    success_count = 0
    if workers == 1:
        for f in files:
            success_count += parse_file(f)
    else:
        with workqueue(workers, logger=logger) as q:
            q_tasks = [q.call(parse_file, f, log_queue=q.log_queue)
                       for f in files]
            for t in as_completed(q_tasks):
                success_count += t.result()

    info('')
    info('%d files parsed successfully', success_count)

    fail_count = len(files) - success_count
    if fail_count > 0:
        warning('%d files failed to parse', fail_count)


if __name__ == "__main__":
    cli(obj={})
