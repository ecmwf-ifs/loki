#!/usr/bin/env python3

import sys
import importlib
from concurrent.futures import as_completed
from itertools import chain
from logging import FileHandler
from pathlib import Path
from multiprocessing import Manager
import click
import yaml

from loki.logging import logger, DEBUG, warning, info, debug
from loki.sourcefile import Sourcefile
from loki.frontend import FP
from loki.build import workqueue
from loki.lint import Linter, Reporter, DefaultHandler, JunitXmlHandler, ViolationFileHandler
from loki.tools import yaml_include_constructor, auto_post_mortem_debugger, as_tuple


class OutputFile:
    """
    Helper class to encapsulate opening and writing to a file.
    This exists because opening the file immediately and then passing
    its ``write`` function to a handler makes it impossible to pickle
    it afterwards, which would make parallel execution infeasible.
    Instead of creating a more complicated interface for the handlers
    we opted for this way of a just-in-time file handler.
    """

    def __init__(self, filename):
        self.file_name = filename
        self.file_handle = None

    def _check_open(self):
        if not self.file_handle:
            self.file_handle = open(self.file_name, 'w')  # pylint: disable=consider-using-with

    def __del__(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def write(self, msg):
        self._check_open()
        self.file_handle.write(msg)


def get_rules(module):
    """
    Return the list of all available rules in the named module.
    """
    rules_module = importlib.import_module(f'lint_rules.{module}')
    return Linter.lookup_rules(rules_module)


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


def check_and_fix_file(filename, linter, frontend=FP, preprocess=False, fix=False,
                       backup_suffix=None, ctx=None):
    debug('[%s] Parsing...', filename)
    try:
        source = Sourcefile.from_file(filename, frontend=frontend, preprocess=preprocess)
        debug('[%s] Parsing completed without error.', filename)
        report = linter.check(source)
        if fix:
            linter.fix(source, report, backup_suffix=backup_suffix)
    except Exception as excinfo:  # pylint: disable=broad-except
        linter.reporter.add_file_error(filename, type(excinfo), str(excinfo))
        if ctx and ctx.obj.get('DEBUG'):
            raise excinfo
        return False
    return True


@click.group()
@click.option('--debug/--no-debug', default=False, show_default=True,
              help=('Enable / disable debug mode. This incures more verbose '
                    'output and automatically attaches a debugger on exceptions.'))
@click.option('--log', type=click.Path(writable=True),
              help='Write more detailed information to a log file.')
@click.option('--rules-module', default='ifs_coding_standards_2011', show_default=True,
              help='Select Python module with rules in lint_rules.')
@click.pass_context
def cli(ctx, debug, log, rules_module):  # pylint:disable=redefined-outer-name
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj['DEBUG'] = debug
    ctx.obj['rules_module'] = rules_module
    if debug:
        logger.setLevel(DEBUG)
        sys.excepthook = auto_post_mortem_debugger
    if log:
        file_handler = FileHandler(log, mode='w')
        file_handler.setLevel(DEBUG)
        logger.addHandler(file_handler)


@cli.command(help='Get default configuration of linter and rules.')
@click.option('--output-file', '-o', type=click.File(mode='w'),
              help='Write default configuration to file.')
@click.pass_context
def default_config(ctx, output_file):  # pylint: disable=unused-argument
    config = Linter.default_config(rules=get_rules(ctx.obj['rules_module']))
    # Eliminate empty config dicts
    config = {key: val for key, val in config.items() if val}
    config_str = yaml.dump(config, default_flow_style=False)

    if output_file:
        output_file.write(config_str)
    else:
        logger.info(config_str)


@cli.command(help='List all available rules.')
@click.option('--with-title/--without-title', default=False, show_default=True,
              help='With / without title and id from each rule\'s docs.')
@click.option('--sort-by', type=click.Choice(['title', 'id']), default='title',
              show_default=True, help='Sort rules by a specific criterion.')
@click.pass_context
def rules(ctx, with_title, sort_by):  # pylint: disable=unused-argument
    rule_list = get_rules(ctx.obj['rules_module'])
    sort_keys = {'title': lambda rule: rule.__name__.lower(),
                 'id': lambda rule: list(map(int, rule.docs.get('id').split('.')))}
    rule_list.sort(key=sort_keys[sort_by])

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
              help=('Base directory relative to which --include/--exclude '
                    'patterns are interpreted.'))
@click.option('--config', '-c', type=click.File(),
              help='Configuration file for behaviour of linter and rules.')
@click.option('--fix/--no-fix', default=False, show_default=True,
              help='Attempt to fix problems where possible.')
@click.option('--backup-suffix', type=str,
              help=('When fixing, create a backup of the original file with '
                    'the given suffix.'))
@click.option('--worker', type=int, default=4, show_default=True,
              help=('Number of worker processes to use. With --debug enabled '
                    'this option is ignored and only one worker is used.'))
@click.option('--write-violation-file', is_flag=False,
              flag_value='violations.yml', default=None,
              help=('Write a YAML file that lists for every file the rules '
                    'violated. Can be included into a config to disable them in '
                    'future linting runs.'))
# @click.option('--preprocess/--no-preprocess', default=False, show_default=True,
#               help='Enable C-preprocessing of files before parsing them.')
@click.option('--junitxml', type=click.Path(dir_okay=False, writable=True),
              help='Enable output in JUnit XML format to the given file.')
@click.pass_context
def check(ctx, include, exclude, basedir, config, fix, backup_suffix, worker, write_violation_file, junitxml):
    yaml.add_constructor('!include', yaml_include_constructor, yaml.SafeLoader)
    config_values = yaml.safe_load(config) if config else {}
    if ctx.obj['DEBUG']:
        worker = 1

    if 'include' in config_values:
        include += as_tuple(config_values['include'])
    if 'exclude' in config_values:
        exclude += as_tuple(config_values['exclude'])

    if basedir and 'basedir' in config_values:
        warning('basedir given as explicit argument and in the config file. Ignoring the config file value.')
    if not basedir:
        basedir = config_values.get('basedir', Path.cwd())

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
    info('%d files selected for checking (%d files excluded).',
         len(files), len(excludes))

    info('')
    info('Using %d worker.', worker)

    rule_list = get_rules(ctx.obj['rules_module'])
    info('%d rules available.', len(rule_list))

    handlers = [DefaultHandler(basedir=basedir)]
    if junitxml:
        junitxml_file = OutputFile(junitxml)
        handlers.append(JunitXmlHandler(target=junitxml_file.write, basedir=basedir))
    if write_violation_file:
        violation_file = OutputFile(write_violation_file)
        handlers.append(ViolationFileHandler(target=violation_file.write, basedir=basedir))

    linter = Linter(reporter=Reporter(handlers), rules=rule_list, config=config_values)
    info('Checking against %d rules.', len(linter.rules))
    info('')

    if backup_suffix and not backup_suffix.startswith('.'):
        backup_suffix = '.' + backup_suffix

    success_count = 0
    if worker == 1:
        for f in files:
            success_count += check_and_fix_file(f, linter, fix=fix, backup_suffix=backup_suffix, ctx=ctx)
    else:
        manager = Manager()
        linter.reporter.init_parallel(manager)

        with workqueue(workers=worker, logger=logger, manager=manager) as q:
            log_queue = q.log_queue if hasattr(q, 'log_queue') else None  # pylint: disable=no-member
            q_tasks = [q.call(check_and_fix_file, f, linter, log_queue=log_queue, fix=fix, backup_suffix=backup_suffix)
                       for f in files]
            for t in as_completed(q_tasks):
                success_count += t.result()

    linter.reporter.output()

    info('')
    info('%d files parsed successfully', success_count)

    fail_count = len(files) - success_count
    if fail_count > 0:
        warning('%d files failed to parse', fail_count)


if __name__ == "__main__":
    cli(obj={})  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
