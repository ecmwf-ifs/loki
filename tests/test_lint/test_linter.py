import pytest

from loki.lint.linter import Linter
from loki.lint.reporter import Reporter, GenericHandler
import loki.lint.rules as rules
from loki.sourcefile import SourceFile


def test_linter_lookup_rules():
    '''Make sure that linter picks up all rules by default.'''
    rule_list = Linter._lookup_rules()
    rule_names = [r.__name__ for r in rule_list]
    all_rules = [r for r in rules.__dict__ if r.endswith('Rule')]
    diff = set(all_rules) - set(rule_names)
    assert diff == {'GenericRule'}


def test_linter_fail():
    '''Make sure that linter fails if it is not given a source file.'''
    with pytest.raises(TypeError, match=r'.*SourceFile.*expected.*'):
        Linter(None).check(None)


def test_linter_check():
    '''Make sure that linter runs through all given rules and hands them
    the right config.'''
    class TestRule(rules.GenericRule):
        config = {'key': 'default_value'}

        @classmethod
        def check(cls, ast, rule_report, config):
            assert len(config) == 1
            assert 'key' in config
            assert config['key'] == 'default_value'
            rule_report.add('TestRule', 'Location')

    class TestRule2(rules.GenericRule):
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

    config = {
        'TestRule2': {
            'other_key': 'other_value',
            'key': 'non_default_value'
        }
    }
    reporter = Reporter(handlers=[TestHandler()])
    rule_list = [TestRule2, TestRule]
    linter = Linter(reporter, config=config)
    linter.check(SourceFile('test_file'), rules=rule_list)
