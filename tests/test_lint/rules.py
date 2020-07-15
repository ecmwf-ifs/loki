from loki.lint import GenericRule, RuleType

__all__ = ['DummyRule']


class DummyRule(GenericRule):

    type = RuleType.WARN

    docs = {'title': 'A dummy rule for the sake of testing the Linter'}

    config = {'dummy_key': 'dummy value'}
