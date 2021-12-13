import inspect
import shutil
import time
from pathlib import Path

from loki.lint.reporter import FileReport, RuleReport
from loki.lint.utils import Fixer
from loki.sourcefile import Sourcefile


class Linter:
    """
    The operator class for Loki's linter functionality.

    It allows to check `Sourcefile` objects for compliance to the rules
    implemented in `loki.lint.rules`.

    :param `loki.lint.reporter.Reporter` reporter: the reporter instance to be
        used for problem reporting.
    :param rules: list of rules to check files against or module that contains
        the rules.
    :param dict config: (optional) config (e.g., from config file) to change
        behaviour of rules.
    """
    def __init__(self, reporter, rules, config=None):
        self.reporter = reporter
        if inspect.ismodule(rules):
            rule_names = config.get('rules') if config else None
            self.rules = Linter.lookup_rules(rules, rule_names=rule_names)
        elif config and config.get('rules'):
            self.rules = [rule for rule in rules if rule.__name__ in config.get('rules')]
        else:
            self.rules = rules
        self.config = self.default_config(self.rules)
        self.update_config(config)

    @staticmethod
    def lookup_rules(rules_module, rule_names=None):
        """
        Return list of all rules available or all rules contained in the given list of names.

        :param rules_module: the module in which rules are implemented.
        :param list rule_names: (optional) list of rule names to look for.

        :return: list of rules.
        """
        rule_list = inspect.getmembers(
            rules_module, lambda obj: inspect.isclass(obj) and obj.__name__ in rules_module.__all__)
        if rule_names is not None:
            rule_list = [r for r in rule_list if r[0] in rule_names]
        return [r[1] for r in rule_list]

    @staticmethod
    def default_config(rules):
        """
        Return default configuration for all rules.

        :param rules: list of rules or module in which rules are implemented.
        :return: `dict` with the list of rule names and the default
            configuration values for each rule.
        """
        # List of rules
        config = {'rules': [rule.__name__ for rule in rules]}
        # Default options for rules
        for rule in rules:
            config[rule.__name__] = rule.config
        return config

    def update_config(self, config):
        """
        Update the stored configuration using the given `config` dict.
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

        :param Sourcefile sourcefile: the source file to check.
        :param list overwrite_rules: (optional) list of rules to check.
        :param dict overwrite_config: (optional) configuration that is used to update the
            stored configuration.

        :return: the `FileReport`.
        """
        if not isinstance(sourcefile, Sourcefile):
            raise TypeError(f'{type(sourcefile)} given, {Sourcefile} expected')
        # Prepare list of rules and configuration
        rules = overwrite_rules if overwrite_rules is not None else self.rules
        config = self.config
        if overwrite_config:
            config.update(overwrite_config)
        # Initialize report for this file
        filename = str(sourcefile.path) if sourcefile.path else None
        file_report = FileReport(filename)
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

        :param Sourcefile sourcefile: the source file to fix.
        :param FileReport file_report: the report created by `check()` for that file.
        :param str backup_suffix: (optional) suffix to use for a copy of the original file.
        :param dict overwrite_config: (optional) configuration that is used to update the
            stored configuration.
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
