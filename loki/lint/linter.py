import inspect


class Linter(object):

    def __init__(self, config=None):
        self.config = config or {}

    @staticmethod
    def _lookup_rules():
        import loki.lint.rules as rules
        rule_list = inspect.getmembers(
            rules, lambda obj: inspect.isclass(obj) and obj.__name__.endswith('Rule'))
        rule_list = [r[1] for r in rule_list if r[0] != 'GenericRule']
        return rule_list

    def check(self, ast, reporter, rules=None):
        if rules is None:
            rules = Linter._lookup_rules()
        for rule in rules:
            config = rule.config
            config.update(self.config.get(rule.__name__, {}))
            rule.check(ast, reporter, config)

    def fix(self, ast, reporter):
        pass
