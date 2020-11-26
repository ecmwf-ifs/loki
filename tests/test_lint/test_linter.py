import pytest

from loki.lint import GenericHandler, Reporter, Linter, GenericRule
from loki.sourcefile import Sourcefile

import rules


@pytest.mark.parametrize('rule_names, num_rules', [
    (None, 1),
    (['FooRule'], 0),
    (['DummyRule'], 1)
])
def test_linter_lookup_rules(rule_names, num_rules):
    '''Make sure that linter picks up all rules by default.'''
    rule_list = Linter.lookup_rules(rules, rule_names=rule_names)
    assert len(rule_list) == num_rules


def test_linter_fail():
    '''Make sure that linter fails if it is not given a source file.'''
    with pytest.raises(TypeError, match=r'.*Sourcefile.*expected.*'):
        Linter(None, rules).check(None)


def test_linter_check():
    '''Make sure that linter runs through all given and hands them
    the right config.'''
    class TestRule(GenericRule):
        config = {'key': 'default_value'}

        @classmethod
        def check(cls, ast, rule_report, config):
            assert len(config) == 1
            assert 'key' in config
            assert config['key'] == 'default_value'
            rule_report.add('TestRule', 'Location')

    class TestRule2(GenericRule):
        config = {'key': 'default_value'}

        @classmethod
        def check(cls, ast, rule_report, config):
            assert len(config) == 2
            assert 'key' in config
            assert config['key'] == 'non_default_value'
            assert 'other_key' in config
            assert config['other_key'] == 'other_value'
            rule_report.add('TestRule2', 'Location2')

    class TestHandler(GenericHandler):

        def handle(self, file_report):
            assert len(file_report.reports) == 2
            assert len(file_report.reports[0].problem_reports) == 1
            assert file_report.reports[0].problem_reports[0].msg == 'TestRule2'
            assert file_report.reports[0].problem_reports[0].location == 'Location2'
            assert file_report.reports[0].rule == TestRule2
            assert file_report.reports[1].problem_reports[0].msg == 'TestRule'
            assert file_report.reports[1].problem_reports[0].location == 'Location'
            assert file_report.reports[1].rule == TestRule

        def output(self, handler_reports):
            pass

    config = {
        'TestRule2': {
            'other_key': 'other_value',
            'key': 'non_default_value'
        }
    }
    reporter = Reporter(handlers=[TestHandler()])
    rule_list = [TestRule2, TestRule]
    linter = Linter(reporter, rule_list, config=config)
    linter.check(Sourcefile('test_file'))
