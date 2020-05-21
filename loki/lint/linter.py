import importlib
import inspect
import time
from pathlib import Path

from loki.lint.reporter import FileReport, RuleReport
from loki.sourcefile import SourceFile
from loki.backend import fgen


class Linter:

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
        rules = importlib.import_module('loki.lint.rules')
        rule_list = inspect.getmembers(
            rules, lambda obj: inspect.isclass(obj) and obj.__name__ in rules.__all__)
        if rule_names is not None:
            rule_list = [r for r in rule_list if r[0] in rule_names]
        return [r[1] for r in rule_list]

    @staticmethod
    def default_config(rules=None):
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
        if config is None:
            return
        for key, val in config.items():
            # If we have a dict, update that entry
            if isinstance(val, dict) and key in self.config:
                self.config[key].update(val)
            else:
                self.config[key] = val

    def check(self, sourcefile, overwrite_rules=None, overwrite_config=None):
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

    def fix(self, sourcefile):
        if not isinstance(sourcefile, SourceFile):
            raise TypeError('{} given, {} expected'.format(type(sourcefile), SourceFile))
        # TODO: Actually do any fixing
        # ...
        # Determine path for the fixed file
        file_path = Path(sourcefile.path)
        fixed_path = file_path.with_suffix('.fix' + file_path.suffix)

        # Build the source string
        # Note: this does not necessarily preserve the order of things in the file
        # as it will first generate all modules and then all subroutines
        source = [fgen(mod, conservative=True) for mod in sourcefile.modules]
        source += [fgen(routine, conservative=True) for routine in sourcefile.subroutines]
        SourceFile.to_file(source='\n\n'.join(source), path=fixed_path)
