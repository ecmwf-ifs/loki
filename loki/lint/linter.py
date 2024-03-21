# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
:any:`Linter` operator class definition to drive rule checking for
:any:`Sourcefile` objects
"""
from concurrent.futures import as_completed
import inspect
from multiprocessing import Manager
from pathlib import Path
import shutil
from codetiming import Timer

from loki.build import workqueue
from loki.batch import Scheduler, SchedulerConfig, Item
from loki.config import config as loki_config
from loki.lint.reporter import (
    FileReport, RuleReport, Reporter, LazyTextfile,
    DefaultHandler, JunitXmlHandler, ViolationFileHandler
)
from loki.lint.utils import Fixer
from loki.logging import logger
from loki.sourcefile import Sourcefile
from loki.tools import filehash, find_paths, CaseInsensitiveDict
from loki.transform import Transformation


__all__ = ['Linter', 'LinterTransformation', 'lint_files']


class Linter:
    """
    The operator class for Loki's linter functionality

    It allows to check :any:`Sourcefile` objects for compliance to rules
    specified as subclasses of :any:`GenericRule`.

    Parameters
    ----------
    reporter : :any:`Reporter`
        The reporter instance to be used for problem reporting.
    rules : list of :any:`GenericRule` or a Python module
        List of rules to check files against or a module that contains the rules.
    config : dict, optional
        Configuration (e.g., from config file) to change behaviour of rules.
    """
    def __init__(self, reporter, rules, config=None):
        self.reporter = reporter
        if inspect.ismodule(rules):
            rule_names = config.get('rules') if config else None
            self.rules = Linter.lookup_rules(rules, rule_names=rule_names)
        elif config and config.get('rules') is not None:
            self.rules = [rule for rule in rules if rule.__name__ in config.get('rules')]
        else:
            self.rules = rules
        self.config = self.default_config(self.rules)
        self.update_config(config)

    @staticmethod
    def lookup_rules(rules_module, rule_names=None):
        """
        Obtain all available rule classes in a module

        Parameters
        ----------
        rules_module : Python module
            The module in which rules are implemented.
        rule_names : list of str, optional
            Only look for rules with a name that is in this list.

        Returns
        -------
        list
            A list of rule classes.
        """
        rule_list = inspect.getmembers(
            rules_module, lambda obj: inspect.isclass(obj) and obj.__name__ in rules_module.__all__)
        if rule_names is not None:
            rule_list = [r for r in rule_list if r[0] in rule_names]
        return [r[1] for r in rule_list]

    @staticmethod
    def default_config(rules):
        """
        Return default configuration for a list of rules

        Parameters
        ----------
        rules : list
            List of rules for which to compile the default config.

        Returns
        -------
        dict
            Mapping of rule names to the dict of default configuration
            values for each rule.
        """
        # List of rules
        config = {'rules': [rule.__name__ for rule in rules]}
        # Default options for rules
        for rule in rules:
            config[rule.__name__] = rule.config
        return config

    def update_config(self, config):
        """
        Update the stored configuration using the given :data:`config` dict
        """
        if config is None:
            return
        for key, val in config.items():
            # If we have a dict, update that entry
            if isinstance(val, dict) and key in self.config:
                self.config[key].update(val)
            else:
                self.config[key] = val

    def check(self, sourcefile, overwrite_rules=None, overwrite_config=None, **kwargs):
        """
        Check the given :data:`sourcefile` and compile a :any:`FileReport`.

        The file report is then stored in the :any:`Reporter` given while
        creating the :any:`Linter`. Additionally, the file report is returned,
        e.g., to use it wiht :meth:`fix`.

        Parameters
        ----------
        sourcefile : :any:`Sourcefile`
            The source file to check.
        overwrite_rules : list of rules, optional
            List of rules to check. This overwrites the stored list of rules.
        overwrite_config : dict, optional
            Configuration that is used to update the stored configuration.

        Returns
        -------
        :any:`FileReport`
            The report for this file containing any discovered violations.
        """
        if not isinstance(sourcefile, Sourcefile):
            raise TypeError(f'{type(sourcefile)} given, {Sourcefile} expected')

        # Prepare config
        config = self.config
        if overwrite_config:
            config.update(overwrite_config)
        disable_config = config.get('disable')
        if not isinstance(disable_config, dict):
            disable_config = {}

        # Initialize report for this file
        filename = str(sourcefile.path) if sourcefile.path else None
        file_report = FileReport(filename, hash=filehash(sourcefile.source.string))

        # Check "disable" config section for an entry matching the file name and, if given, filehash
        disabled_rules = CaseInsensitiveDict()
        disable_file_key = next((key for key in disable_config if sourcefile.path.match(key)), None)
        if disable_file_key:
            disable_file = disable_config[disable_file_key]
            if 'filehash' not in disable_file or disable_file['filehash'] == file_report.hash:
                for rule in disable_file.get('rules', []):
                    if isinstance(rule, dict):
                        for name, line_hashes in rule.items():
                            disabled_rules[name] = line_hashes
                    else:
                        disabled_rules[rule] = True

        # Prepare list of rules
        rules = overwrite_rules if overwrite_rules is not None else self.rules
        rules = [rule for rule in rules if disabled_rules.get(rule.__name__) is not True]

        timer = Timer(logger=None)

        # Run all the rules on that file
        for rule in rules:
            timer.start()
            rule_report = RuleReport(rule, disabled=disabled_rules.get(rule.__name__))
            rule.check(sourcefile, rule_report, config[rule.__name__], **kwargs)
            rule_report.elapsed_sec = timer.stop()
            file_report.add(rule_report)

        # Store the file report
        self.reporter.add_file_report(file_report)
        return file_report

    def fix(self, sourcefile, file_report, backup_suffix=None, overwrite_config=None):
        """
        Fix all rule violations in :data:`file_report` that were reported by
        fixable rules and write them into the original file

        Parameters
        ----------
        sourcefile : :any:`Sourcefile`
            The source file to fix.
        file_report : :any:`FileReport`
            The report created by :meth:`check` for that file.
        backup_suffix : str, optional
            Create a copy of the original file using this file name suffix.
        overwrite_config : dict, optional
            Configuration that is used to update the stored configuration.
        """
        if not isinstance(sourcefile, Sourcefile):
            raise TypeError(f'{type(sourcefile)} given, {Sourcefile} expected')
        file_path = Path(sourcefile.path)
        assert file_path == Path(file_report.filename)

        # Nothing to do if there are no fixable reports
        if not file_report.fixable_reports:
            return

        # Make a backup copy if requested
        if backup_suffix:
            backup_path = file_path.with_suffix(backup_suffix + file_path.suffix)
            shutil.copy(file_path, backup_path)

        # Extract configuration
        config = self.config
        if overwrite_config:
            config.update(overwrite_config)

        # Apply the fixes
        sourcefile = Fixer.fix(sourcefile, file_report.fixable_reports, config)

        # Create the the source string for the output
        sourcefile.write(conservative=True)


class LinterTransformation(Transformation):
    """
    Apply :class:`Linter` as a :any:`Transformation` to :any:`Sourcefile`

    The :any:`FileReport` is stored in the ``trafo_data` in an :any:`Item`
    object, if it is provided to :meth:`transform_file`, e.g., during a
    :any:`Scheduler` traversal.

    Parameters
    ----------
    linter : :class:`Linter`
        The linter instance to use
    key : str, optional
        Lookup key overwrite for stored reports in the ``trafo_data`` of :any:`Item`
    """

    _key = 'LinterTransformation'

    # This transformation is applied over the file graph
    traverse_file_graph = True

    item_filter = Item  # Include everything in the dependency tree

    def __init__(self, linter, key=None, **kwargs):
        self.linter = linter
        self.counter = 0
        if key:
            self._key = key
        super().__init__(**kwargs)

    def transform_file(self, sourcefile, **kwargs):
        item = kwargs.get('item')
        report = self.linter.check(sourcefile, **kwargs)
        self.counter += 1
        if item:
            item.trafo_data[self._key] = report
        if self.linter.config.get('fix'):
            self.linter.fix(sourcefile, report, backup_suffix=self.linter.config.get('backup_suffix'))


def lint_files_scheduler(linter, basedir, config):
    """
    Discover files relative to :data:`basedir` using :any:`SchedulerConfig`
    from :data:`config`, and apply :data:`linter` on each of them.
    """
    scheduler = Scheduler(paths=[basedir], config=SchedulerConfig.from_dict(config))
    transformation = LinterTransformation(linter=linter)
    scheduler.process(transformation=transformation)
    return transformation.counter


def check_and_fix_file(path, linter, fix=False, backup_suffix=None):
    """
    Check the file at :data:`path` with :data:`linter` and, optionally,
    fix it
    """
    try:
        source = Sourcefile.from_file(path)
        report = linter.check(source)
        if fix:
            linter.fix(source, report, backup_suffix=backup_suffix)
    except Exception as exc:  # pylint: disable=broad-except
        linter.reporter.add_file_error(path, type(exc), str(exc))
        if loki_config['debug']:
            raise exc
        return False
    return True


def lint_files_glob(linter, basedir, include, exclude=None, max_workers=1, fix=False, backup_suffix=None):
    """
    Discover files relative to :data:`basedir` using patterns in :data:`include`
    and apply :data:`linter` on each of them.
    """
    files = find_paths(basedir, include, ignore=exclude)
    checked_count = 0
    if max_workers == 1 or loki_config['debug']:
        for path in files:
            checked_count += check_and_fix_file(path, linter, fix=fix, backup_suffix=backup_suffix)
    else:
        manager = Manager()
        linter.reporter.init_parallel(manager)

        with workqueue(workers=max_workers, logger=logger, manager=manager) as q:
            log_queue = getattr(q, 'log_queue', None)
            q_tasks = [
                q.call(check_and_fix_file, f, linter, fix=fix, backup_suffix=backup_suffix, log_queue=log_queue)
                for f in files
            ]
            for t in as_completed(q_tasks):
                checked_count += t.result()

    return checked_count


def lint_files(rules, config, handlers=None):
    """
    Construct a :any:`Linter` according to :data:`config` and
    check the rules in :data:`rules`

    Depending on the given config values, this will use a :any:`Scheduler`
    to discover files and drive the linting, or apply glob-based file
    discovery and apply linting to each of them.

    Common config options include:

    .. code-block::

       {
           'basedir': <some file path>,
           'max_workers': <n>, # Optional: use multiple workers
           'fix': <True|False>, # Optional: attempt automatic fixing of rule violations
           'backup_suffix': <suffix>, # Optional: Backup original file with given suffix
           'junitxml_file': <some file path>,  # Optional: write JunitXML-output of lint results
           'violations_file': <some file path>,  # Optional: write a YAML file containing violations
           'rules': ['SomeRule', 'AnotherRule', ...],  # Optional: select only these rules
           'SomeRule': <rule options>, # Optional: configuration values for individual rules
        }

    The ``basedir`` option is given as the discovery path to the :any:`Scheduler`.
    See :any:`SchedulerConfig` for more details on the available config options.

    See :any:`JunitXmlHandler` and :any:`ViolationFileHandler` for more details
    on the output file options.

    The ``rules`` option in the config allows selecting only certain rules out of
    the provided :data:`rules` argument.

    In addition, :data:`config` takes for scheduler the following options:

    .. code-block::

       {
           'scheduler': <SchedulerConfig values>
       }

    If the ``scheduler`` key is found in :data:`config`, the scheduler-based
    linting is automatically enabled.

    For glob-based file discovery, the config takes the following options:

    .. code-block::

       {
           'include': [<some pattern>, <another pattern>, ...]
           'exclude': [<some pattern>] # Optional
       }

    The ``include`` and ``exclude`` options are provided to :any:`find_paths` to
    discover files that should be linted.

    Parameters
    ----------
    rules : list of :any:`GenericRule` or a Python module
        List of rules to check files against or a module that contains the rules.
    config : dict
        Configuration for file discovery/scheduler and linting rules
    handlers : list, optional
        Additional instances of :any:`GenericHandler` to use during linting

    Returns
    -------
    int :
        The number of checked files
    """
    basedir = config['basedir']

    if not handlers:
        handlers = []
    handlers += [DefaultHandler(basedir=basedir)]
    if 'junitxml_file' in config:
        junitxml_file = LazyTextfile(config['junitxml_file'])
        handlers.append(JunitXmlHandler(target=junitxml_file.write, basedir=basedir))
    if 'violations_file' in config:
        violations_file = LazyTextfile(config['violations_file'])
        handlers.append(ViolationFileHandler(
            target=violations_file.write, basedir=basedir,
            use_line_hashes=config.get('use_violations_file_line_hashes', True)
        ))

    linter = Linter(reporter=Reporter(handlers), rules=rules, config=config)
    if 'scheduler' in config:
        checked_count = lint_files_scheduler(linter, basedir, config['scheduler'])
    else:
        checked_count = lint_files_glob(
            linter, basedir, config['include'],
            exclude=config.get('exclude'), max_workers=config.get('max_workers', 1),
            fix=config.get('fix', False), backup_suffix=config.get('backup_suffix')
        )

    linter.reporter.output()
    return checked_count
