import inspect

from loki.lint.reporter import FileReport, RuleReport
from loki.sourcefile import SourceFile


class Linter(object):

    def __init__(self, reporter, config=None):
        self.reporter = reporter
        self.config = config or {}

    @staticmethod
    def _lookup_rules():
        import loki.lint.rules as rules
        rule_list = inspect.getmembers(
            rules, lambda obj: inspect.isclass(obj) and obj.__name__.endswith('Rule'))
        rule_list = [r[1] for r in rule_list if r[0] != 'GenericRule']
        return rule_list

    def check(self, sourcefile, rules=None):
        if not isinstance(sourcefile, SourceFile):
            raise TypeError('{} given, {} expected'.format(type(sourcefile), SourceFile))
        # Get the list of rules
        if rules is None:
            rules = Linter._lookup_rules()
        # Initialize report for this file
        file_report = FileReport(str(sourcefile.path))
        # Run all the rules on that file
        for rule in rules:
            config = rule.config
            config.update(self.config.get(rule.__name__, {}))
            rule_report = RuleReport(rule)
            rule.check(sourcefile, rule_report, config)
            file_report.add(rule_report)
        # Store the file report
        self.reporter.add_file_report(file_report)

    def fix(self, ast):
        pass
