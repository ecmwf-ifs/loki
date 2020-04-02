#!/usr/bin/env python3

import click
from concurrent.futures import as_completed
from itertools import chain
from logging import FileHandler
from pathlib import Path
from multiprocessing import Manager
import yaml

from loki.logging import logger, DEBUG, warning, info, debug
from loki.sourcefile import SourceFile
from loki.frontend import FP
from loki.build import workqueue
from loki.lint import Linter, Reporter, DefaultHandler, JunitXmlHandler


class OutputFile(object):
    '''Helper class to encapsulate opening and writing to a file.
    This exists because opening the file immediately and then passing
    its ``write`` function to a handler makes it impossible to pickle
    it afterwards, which would make parallel execution infeasible.
    Instead of creating a more complicated interface for the handlers
    we opted for this way of a just-in-time file handler.
    '''

    def __init__(self, filename):
        self.file_name = filename
        self.file_handle = None

    def _check_open(self):
        if not self.file_handle:
            self.file_handle = open(self.file_name, 'w')

    def __del__(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def write(self, msg):
        self._check_open()
        self.file_handle.write(msg)


def get_relative_path_and_anchor(path, anchor):
    """
    If the given path is an absolute path, it is converted into a
    relative path and returned together with the corresponding anchor.
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


def check_file(filename, linter, frontend=FP, preprocess=False):
    debug('[%s] Parsing...', filename)
    try:
        source = SourceFile.from_file(filename, frontend=frontend, preprocess=preprocess)
    except Exception as excinfo:
        linter.reporter.add_file_error(filename, type(excinfo), str(excinfo))
        return False
    debug('[%s] Parsing completed without error.', filename)
    linter.check(source)
    linter.fix(source)
    return True


@click.group()
@click.option('--debug/--no-debug', default=False, show_default=True,
              help='Enable / disable debug mode. Includes more verbose output.')
@click.option('--log', type=click.Path(writable=True),
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


@cli.command(help='Get default configuration of linter and rules.')
@click.option('--output-file', '-o', type=click.File(mode='w'),
              help='Write default configuration to file.')
@click.pass_context
def default_config(ctx, output_file):
    config = Linter.default_config()
    config_str = yaml.dump(config, default_flow_style=False)

    if output_file:
        output_file.write(config_str)
    else:
        logger.info(config_str)


@cli.command(help='List all available rules.')
@click.option('--with-title/--without-title', default=False, show_default=True,
              help='With / without title and id from each rule\'s docs.')
@click.pass_context
def rules(ctx, with_title):
    rule_list = Linter.lookup_rules()
    rule_names = [rule.__name__ for rule in rule_list]
    max_width_name = max(len(name) for name in rule_names)

    if with_title:
        rule_ids = [rule.docs.get('id', '') for rule in rule_list]
        max_width_id = max(len(id_) for id_ in rule_ids)
        rule_titles = [rule.docs.get('title', '').format(**rule.config)
                       if rule.config else rule.docs.get('title', '')
                       for rule in rule_list]

        fmt_string = '{name:<{name_width}}  {id:^{id_width}}  {title}'
        output_string = '\n'.join(
            fmt_string.format(name=name, name_width=max_width_name,
                              id=id_, id_width=max_width_id, title=title)
            for name, id_, title in zip(rule_names, rule_ids, rule_titles))

    else:
        output_string = '\n'.join(rule_names)

    logger.info(output_string)


@cli.command(help='Check for syntax errors and compliance to coding rules.')
@click.option('--include', '-I', type=str, multiple=True,
              help=('File name or pattern for file names to be checked. '
                    'Allows for relative and absolute paths/glob patterns.'))
@click.option('--exclude', '-X', type=str, multiple=True,
              help=('File name or pattern for file names to be excluded. '
                    'This allows to exclude files that were included by '
                    '--include.'))
@click.option('--basedir', type=click.Path(exists=True, file_okay=False),
              default=Path.cwd(), show_default=True,
              help=('Base directory relative to which --include/--exclude '
                    'patterns are interpreted.'))
@click.option('--config', '-c', type=click.File(),
              help='Configuration file for behaviour of linter and rules.')
@click.option('--worker', type=int, default=4, show_default=True,
              help=('Number of worker processes to use. With --debug enabled '
                    'this option is ignored and only one worker is used.'))
@click.option('--preprocess/--no-preprocess', default=False, show_default=True,
              help=('Enable preprocessing of files before parsing them. '
                    'This applies some string replacements to work around '
                    'frontend deficiencies and saves a copy of the modified '
                    'source in a separate file.'))
@click.option('--junitxml', type=click.Path(dir_okay=False, writable=True),
              help='Enable output in JUnit XML format to the given file.')
@click.pass_context
def check(ctx, include, exclude, basedir, config, worker, preprocess, junitxml):
    info('Base directory: %s', basedir)
    info('Include patterns:')
    for p in include:
        info('  - %s', p)
    info('Exclude patterns:')
    for p in exclude:
        info('  - %s', p)
    info('')

    config_values = yaml.safe_load(config) if config else {}

    debug('Searching for files using specified patterns...')
    files, excludes = get_file_list(include, exclude, basedir)
    info('%d files selected for checking (%d files excluded).',
         len(files), len(excludes))
    info('Using %d worker.', worker)
    info('')

    handlers = [DefaultHandler(basedir=basedir)]
    if junitxml:
        junitxml_file = OutputFile(junitxml)
        handlers.append(JunitXmlHandler(target=junitxml_file.write, basedir=basedir))

    linter = Linter(reporter=Reporter(handlers), config=config_values)

    success_count = 0
    if worker == 1:
        for f in files:
            success_count += check_file(f, linter, preprocess=preprocess)
    else:
        manager = Manager()
        linter.reporter.init_parallel(manager)

        with workqueue(workers=worker, logger=logger, manager=manager) as q:
            q_tasks = [q.call(check_file, f, linter, log_queue=q.log_queue,
                              preprocess=preprocess) for f in files]
            for t in as_completed(q_tasks):
                success_count += t.result()

    linter.reporter.output()

    info('')
    info('%d files parsed successfully', success_count)

    fail_count = len(files) - success_count
    if fail_count > 0:
        warning('%d files failed to parse', fail_count)


if __name__ == "__main__":
    cli(obj={})
