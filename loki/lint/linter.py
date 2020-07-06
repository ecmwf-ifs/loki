import importlib
import inspect
import shutil
import time
from pathlib import Path

from loki.lint.reporter import FileReport, RuleReport
from loki.lint.utils import Fixer
from loki.sourcefile import SourceFile


class Linter:
    """
    The operator class for Loki's linter functionality.

    It allows to check `SourceFile` objects for compliance to the rules
    implemented in `loki.lint.rules`.

    :param `loki.lint.reporter.Reporter` reporter: the reporter instance to be
        used for problem reporting.
    :param list rules: (optional) list of rules to check files against. Defaults
        to all rules.
    :param dict config: (optional) config (e.g., from config file) to change
        behaviour of rules.
    """
    def __init__(self, reporter, rules=None, config=None):
        self.reporter = reporter
        rule_names = None
        if rules is not None:
            rule_names = [rule.__name__ for rule in rules]
        elif config is not None:
            rule_names = config.get('rules') if config is not None else None
        self.rules = rules if rules is not None else Linter.lookup_rules(rule_names)
        self.config = self.default_config(self.rules)
        self.update_config(config)

    @staticmethod
    def lookup_rules(rule_names=None):
        """
        Return list of all rules available or all rules contained in the given list of names.
        """
        rules = importlib.import_module('loki.lint.rules')
        rule_list = inspect.getmembers(
            rules, lambda obj: inspect.isclass(obj) and obj.__name__ in rules.__all__)
        if rule_names is not None:
            rule_list = [r for r in rule_list if r[0] in rule_names]
        return [r[1] for r in rule_list]

    @staticmethod
    def default_config(rules=None):
        """
        Return default configuration for all or the given rules.
        """
        config = {}
        if rules is None:
            rules = Linter.lookup_rules()
        # List of rules
        config['rules'] = [rule.__name__ for rule in rules]
        # Default options for rules
        for rule in rules:
            config[rule.__name__] = rule.config
        return config

    def update_config(self, config):
        """
        Update the configuration using the given `config` dict.
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
        Check the given `sourcefile` and compile the `FileReport`.
        The report is then stored in the `reporter` and returned (e.g., to use it for `fix()`).

        :param SourceFile sourcefile: the source file to check.
        :param list overwrite_rules: (optional) list of rules to check.
        :param dict overwrite_config: (optional) configuration that is used to update the
            stored configuration.

        :return: the `FileReport`.
        """
        if not isinstance(sourcefile, SourceFile):
            raise TypeError('{} given, {} expected'.format(type(sourcefile), SourceFile))
        # Prepare list of rules and configuration
        rules = overwrite_rules if overwrite_rules is not None else self.rules
        config = self.config
        if overwrite_config:
            config.update(overwrite_config)
        # Initialize report for this file
        file_report = FileReport(str(sourcefile.path))
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
        Fix all problems reported by fixable rules.

        :param SourceFile sourcefile: the source file to fix.
        :param FileReport file_report: the report created by `check()` for that file.
        :param str backup_suffix: (optional) suffix to use for a copy of the original file.
        :param dict overwrite_config: (optional) configuration that is used to update the
            stored configuration.
        """
        if not isinstance(sourcefile, SourceFile):
            raise TypeError('{} given, {} expected'.format(type(sourcefile), SourceFile))
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
