import importlib
import pytest

import rules

from loki import Sourcefile, Assignment, FindNodes, FindVariables
from loki.lint import GenericHandler, Reporter, Linter, GenericRule

@pytest.fixture(scope='module', name='rules')
def fixture_rules():
    rules = importlib.import_module('rules')
    return rules


@pytest.mark.parametrize('rule_names, num_rules', [
    (None, 1),
    (['FooRule'], 0),
    (['DummyRule'], 1)
])
def test_linter_lookup_rules(rules, rule_names, num_rules):
    '''Make sure that linter picks up all rules by default.'''
    rule_list = Linter.lookup_rules(rules, rule_names=rule_names)
    assert len(rule_list) == num_rules


def test_linter_fail(rules):
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


@pytest.mark.parametrize('match,report_counts', [
    ('xxx', 2),  #3),
    ('$$$', 1),  #2),
    ('###', 1),  #2),
    ('$$', 0),  #1),
    ('##', 0),  #1),
    pytest.param('$', 0, marks=pytest.mark.xfail()),  # Sourcefile-level comments are not preserved in Loki
    pytest.param('#', 0, marks=pytest.mark.xfail()),  # Sourcefile-level comments are not preserved in Loki
])
def test_linter_disable_per_scope(match, report_counts):
    class AlwaysComplainRule(GenericRule):
        docs = {'id': '13.37'}

        @classmethod
        def check_sourcefile(cls, ast, rule_report, config):  # pylint: disable=unused-argument
            rule_report.add(cls.__name__, ast)

        check_module = check_sourcefile
        check_subroutine = check_sourcefile

    class TestHandler(GenericHandler):
        def handle(self, file_report):
            return len(file_report.reports[0].problem_reports)

        def output(self, handler_reports):
            pass


    fcode = """
! $loki-lint$: disable=13.37
! #loki-lint#: disable=AlwaysComplainRule

module linter_mod
! $$loki-lint$$  : disable=13.37
! ##loki-lint##:disable=AlwaysComplainRule

contains

subroutine linter_routine
! $$$loki-lint$$$  :disable=13.37
! ###loki-lint###: redherring=abc disable=AlwaysComplainRule

end subroutine linter_routine
end module linter_mod
    """.strip()

    fcode = fcode.replace('{match}loki-lint{match}'.format(match=match), 'loki-lint')
    sourcefile = Sourcefile.from_source(fcode)

    handler = TestHandler()
    reporter = Reporter(handlers=[handler])
    rule_list = [AlwaysComplainRule]
    linter = Linter(reporter, rule_list)
    linter.check(sourcefile)

    assert reporter.handlers_reports[handler] == [report_counts]


@pytest.mark.parametrize('rule_list,count', [
    ('', 8),
    ('NonExistentRule', 8),
    ('13.37', 5),
    ('AssignmentComplainRule', 5),
    ('NonExistentRule,AssignmentComplainRule', 5),
    ('23.42', 3),
    ('VariableComplainRule', 3),
    ('23.42,NonExistentRule', 3),
    ('13.37,23.42', 0),
    ('VariableComplainRule,13.37', 0),
    ('23.42,VariableComplainRule,AssignmentComplainRule', 0),
])
def test_linter_disable_inline(rule_list, count):
    class AssignmentComplainRule(GenericRule):
        docs = {'id': '13.37'}

        @classmethod
        def check_subroutine(cls, subroutine, rule_report, config):  # pylint: disable=unused-argument
            for node in FindNodes(Assignment).visit(subroutine.ir):
                rule_report.add(cls.__name__ + '_' + str(node.source.lines[0]), node)

    class VariableComplainRule(GenericRule):
        docs = {'id': '23.42'}

        @classmethod
        def check_subroutine(cls, subroutine, rule_report, config):  # pylint: disable=unused-argument
            for node, variables in FindVariables(with_ir_node=True).visit(subroutine.body):
                for var in variables:
                    rule_report.add(cls.__name__ + '_' + str(var), node)

    class TestHandler(GenericHandler):
        def handle(self, file_report):
            return sum(len(report.problem_reports) for report in file_report.reports)

        def output(self, handler_reports):
            pass

    fcode = """
subroutine linter_disable_inline
integer :: a, b, c

a = 1  ! loki-lint: disable=###
b = 2  !loki-lint:disable=###
c = a + b!     loki-lint       :      disable=###
end subroutine linter_disable_inline
    """.strip()

    fcode = fcode.replace('###', rule_list)
    sourcefile = Sourcefile.from_source(fcode)

    handler = TestHandler()
    reporter = Reporter(handlers=[handler])
    rule_list = [AssignmentComplainRule, VariableComplainRule]
    linter = Linter(reporter, rule_list)
    linter.check(sourcefile)

    assert reporter.handlers_reports[handler] == [count]
