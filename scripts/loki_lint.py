#!/usr/bin/env python3

# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys
import importlib
from logging import FileHandler
from pathlib import Path
import click
import yaml

from loki.logging import logger, DEBUG, warning, info, debug, error
from loki.lint import Linter, lint_files
from loki.tools import yaml_include_constructor, auto_post_mortem_debugger, as_tuple


def get_rules(module):
    """
    Return the list of all available rules in the named module.
    """
    rules_module = importlib.import_module(f'lint_rules.{module}')
    return Linter.lookup_rules(rules_module)


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
@click.option('--write-violations-file', is_flag=False, flag_value='violations.yml', default=None,
              help=('Write a YAML file that lists for every file the violated rules. '
                    'The file can be included into a config file to disable reporting '
                    'these violations in subsequent linting runs.'))
@click.option('--scheduler/--no-scheduler', default=False, show_default=True,
              help='Use a Scheduler to plan source file traversal.')
@click.option('--junitxml', type=click.Path(dir_okay=False, writable=True),
              help='Enable output in JUnit XML format to the given file.')
@click.pass_context
def check(ctx, include, exclude, basedir, config, fix, backup_suffix, worker,
          write_violations_file, scheduler, junitxml):
    yaml.add_constructor('!include', yaml_include_constructor, yaml.SafeLoader)
    config_values = yaml.safe_load(config) if config else {}
    if ctx.obj['DEBUG']:
        worker = 1

    if include:
        if 'include' in config_values:
            info('Merging include patterns from config and command line')
            config_values['include'] = as_tuple(config_values['include']) + as_tuple(include)
        else:
            config_values['include'] = as_tuple(include)
        include += as_tuple(config_values['include'])

    if 'include' not in config_values:
        error('No include pattern given')
        return

    if exclude:
        if 'exclude' in config_values:
            info('Merging exclude patterns from config and command line')
            config_values['exclude'] = as_tuple(config_values['exclude']) + as_tuple(exclude)
        else:
            config_values['exclude'] = as_tuple(exclude)

    if basedir:
        if 'basedir' in config_values:
            warning('Overwriting `basedir` value in the config file with command line argument')
        config_values['basedir'] = basedir
    elif 'basedir' not in config_values:
        config_values['basedir'] = Path.cwd()

    debug('Base directory: %s', basedir)

    if scheduler:
        config_values.setdefault('scheduler', {
            'default': {
                'mode': 'lint',
                'role': 'kernel',
                'expand': True,
                'strict': False,
            }
        })
    else:
        debug('Include patterns:')
        for p in include:
            debug('  - %s', p)
        debug('Exclude patterns:')
        for p in exclude:
            debug('  - %s', p)
        debug('')

    rule_list = get_rules(ctx.obj['rules_module'])
    debug('%d rules available.', len(rule_list))

    config_values['fix'] = fix
    if backup_suffix:
        if not backup_suffix.startswith('.'):
            backup_suffix = '.' + backup_suffix
        config_values['backup_suffix'] = backup_suffix

    config_values['max_workers'] = worker

    if write_violations_file:
        config_values['violations_file'] = write_violations_file
    if junitxml:
        config_values['junitxml_files'] = junitxml

    checked_count = lint_files(rule_list, config_values)
    info('%d files checked', checked_count)


if __name__ == "__main__":
    cli(obj={})  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
