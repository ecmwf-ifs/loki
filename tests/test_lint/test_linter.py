from pathlib import Path
import importlib
import pytest

from loki import Sourcefile, Assignment, FindNodes, FindVariables
from loki.lint import GenericHandler, Reporter, Linter, GenericRule

@pytest.fixture(scope='module', name='rules')
def fixture_rules():
    rules = importlib.import_module('rules')
    return rules


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='dummy_file')
def dummy_file_fixture(here):
    file_path = here/'test_linter_dummy_file.F90'
    fcode = """
! dummy file for linter tests
subroutine dummy
end subroutine dummy
    """.strip()
    file_path.write_text(fcode)
    yield file_path
    file_path.unlink()


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


def test_linter_check(dummy_file):
    '''Make sure that linter runs through all given and hands them
    the right config.'''
    class TestRule(GenericRule):
        config = {'key': 'default_value'}

        @classmethod
        def check(cls, ast, rule_report, config):
            assert len(config) == 1
            assert 'key' in config
            assert config['key'] == 'default_value'
            rule_report.add('TestRule', ast)

    class TestRule2(GenericRule):
        config = {'key': 'default_value'}

        @classmethod
        def check(cls, ast, rule_report, config):
            assert len(config) == 2
            assert 'key' in config
            assert config['key'] == 'non_default_value'
            assert 'other_key' in config
            assert config['other_key'] == 'other_value'
            rule_report.add('TestRule2', ast)

    class TestHandler(GenericHandler):

        def handle(self, file_report):
            assert len(file_report.reports) == 2
            assert len(file_report.reports[0].problem_reports) == 1
            assert file_report.reports[0].problem_reports[0].msg == 'TestRule2'
            assert file_report.reports[0].problem_reports[0].location.path == dummy_file
            assert file_report.reports[0].rule == TestRule2
            assert file_report.reports[1].problem_reports[0].msg == 'TestRule'
            assert file_report.reports[1].problem_reports[0].location.path == dummy_file
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
    linter.check(Sourcefile.from_file(dummy_file))


@pytest.mark.parametrize('file_rule,module_rule,subroutine_rule,assignment_rule,report_counts', [
    ('', '', '', '', 3),
    ('', '', '', '13.37', 3),
    ('', '', '13.37', '', 2),
    pytest.param('', '13.37', '', '', 1, marks=pytest.mark.xfail()),
    ('BlubRule', 'FooRule', 'BarRule', 'BazRule', 3),
    ('', '', '', 'AlwaysComplainRule', 3),
    ('', '', 'AlwaysComplainRule', '', 2),
    pytest.param('', 'AlwaysComplainRule', '', '', 1, marks=pytest.mark.xfail()),
    pytest.param('AlwaysComplainRule', '', '', '', 0, marks=pytest.mark.xfail()),
    pytest.param('13.37', '', '', '', 0, marks=pytest.mark.xfail()),
    # Note: Failed tests are due to the fact that rule disable lookup currently works
    # the wrong way around, see LOKI-64 for details
])
def test_linter_disable_per_scope(file_rule, module_rule, subroutine_rule, assignment_rule, report_counts):
    class AlwaysComplainRule(GenericRule):
        docs = {'id': '13.37'}

        @classmethod
        def check_file(cls, sourcefile, rule_report, config):  # pylint: disable=unused-argument
            rule_report.add(cls.__name__, sourcefile)

        check_module = check_file
        check_subroutine = check_file

    class TestHandler(GenericHandler):
        def handle(self, file_report):
            return len(file_report.reports[0].problem_reports)

        def output(self, handler_reports):
            pass


    fcode = f"""
! loki-lint: disable={file_rule}

module linter_mod
! loki-lint:disable={module_rule}

contains

subroutine linter_routine
! loki-lint: redherring=abc disable={subroutine_rule}
  integer :: i

  i = 1  ! loki-lint  : disable={assignment_rule}
end subroutine linter_routine
end module linter_mod
    """.strip()
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


@pytest.mark.parametrize('disable_config,count', [
    ({}, 8),  # Empty 'disable' section in config should work
    ({'file.F90': {'rules': ['MyMadeUpRule']}}, 8),  # Disables non-existent rule, no effect
    ({'file.F90': {'rules': ['AssignmentComplainRule']}}, 5),  # Disables one rule
    ({'file.f90': {'rules': ['AssignmentComplainRule']}}, 8),  # Filename spelled wrong, no effect
    ({'file.F90': {'rules': ['VariableComplainRule']}}, 3),  # Disables another rule
    ({'file.F90': {'rules': ['AssignmentComplainRule', 'VariableComplainRule']}}, 0),  # Disables all rules
    ({'file.F90': {  # Disables rule with correct filehash
        'filehash': 'd0d8dd935d0e98a951cbd6c703847bac',
        'rules': ['AssignmentComplainRule']
    }}, 5),
    ({'file.F90': {  # Wrong filehash, no effect
        'filehash': 'd0d8dd935d0e98a951cbd6c703847baa',
        'rules': ['AssignmentComplainRule']
    }}, 8)
])
def test_linter_disable_config(disable_config, count):
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
module linter_disable_config_mod
    implicit none

    integer :: modvar

contains

    subroutine linter_disable_inline
        integer :: a, b, c

        a = 1
        b = 2
        c = a + b
    end subroutine linter_disable_inline
end module linter_disable_config_mod
    """.strip()

    sourcefile = Sourcefile.from_source(fcode)
    sourcefile.path = Path('file.F90')  # specify a dummy filename
    rule_list = [AssignmentComplainRule, VariableComplainRule]

    config = Linter.default_config(rules=rule_list)
    config['disable'] = disable_config

    handler = TestHandler()
    reporter = Reporter(handlers=[handler])
    linter = Linter(reporter, rule_list, config=config)
    linter.check(sourcefile)

    assert reporter.handlers_reports[handler] == [count]
