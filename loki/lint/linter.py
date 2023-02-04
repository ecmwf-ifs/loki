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
import inspect
import shutil
import time
from pathlib import Path

from loki.lint.reporter import FileReport, RuleReport
from loki.lint.utils import Fixer
from loki.sourcefile import Sourcefile
from loki.tools import filehash
from loki.transform import Transformation


__all__ = ['Linter', 'LinterTransformation']


class Linter:
    """
    The operator class for Loki's linter functionality

    It allows to check :any:`Sourcefile` objects for compliance to rules
    specified as subclasses of :any:`GenericRule`.

    Parameters
    ----------
    reporter : :any:`Reporter`
        The reporter instance to be used for problem reporting.
    Rules : list of :any:`GenericRule` or a Python module
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

    def check(self, sourcefile, overwrite_rules=None, overwrite_config=None):
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
        disabled_rules = []
        disable_file_key = next((key for key in disable_config if sourcefile.path.match(key)), None)
        if disable_file_key:
            disable_file = disable_config[disable_file_key]
            if 'filehash' not in disable_file or disable_file['filehash'] == file_report.hash:
                disabled_rules = disable_file.get('rules', [])

        # Prepare list of rules
        rules = overwrite_rules if overwrite_rules is not None else self.rules
        rules = [rule for rule in rules if not rule.__name__ in disabled_rules]

        # Run all the rules on that file
        for rule in rules:
            start_time = time.time()
            rule_report = RuleReport(rule)
            rule.check(sourcefile, rule_report, config[rule.__name__])
            end_time = time.time()
            rule_report.elapsed_sec = end_time - start_time
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
        # TODO: this does not necessarily preserve the order of things in the file
        # as it will first generate all modules and then all subroutines
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

    def __init__(self, linter, key=None, **kwargs):
        self.linter = linter
        if key:
            self._key = key
        super().__init__(**kwargs)

    def transform_file(self, sourcefile, **kwargs):
        item = kwargs.get('item')
        report = self.linter.check(sourcefile)
        if item:
            item.trafo_data[self._key] = report
