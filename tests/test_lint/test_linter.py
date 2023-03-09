# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
from pathlib import Path
import pytest

from loki import Sourcefile, Assignment, FindNodes, FindVariables, gettempdir
from loki.lint import (
    GenericHandler, Reporter, Linter, GenericRule,
    LinterTransformation, lint_files, LazyTextfile
)

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


@pytest.fixture(scope='module', name='dummy_rules')
def dummy_rules_fixture():
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

    yield [TestRule2, TestRule]


@pytest.fixture(scope='module', name='dummy_handler')
def dummy_handler_fixture(dummy_file, dummy_rules):
    class TestHandler(GenericHandler):

        def handle(self, file_report):
            assert len(file_report.reports) == 2
            assert len(file_report.reports[0].problem_reports) == 1
            assert file_report.reports[0].problem_reports[0].msg == 'TestRule2'
            assert file_report.reports[0].problem_reports[0].location.path == dummy_file
            assert file_report.reports[0].rule == dummy_rules[0]
            assert file_report.reports[1].problem_reports[0].msg == 'TestRule'
            assert file_report.reports[1].problem_reports[0].location.path == dummy_file
            assert file_report.reports[1].rule == dummy_rules[1]

        def output(self, handler_reports):
            pass

    yield TestHandler()


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


def test_linter_check(dummy_file, dummy_rules, dummy_handler):
    '''Make sure that linter runs through all given rules and hands them
    the right config.'''

    config = {
        'TestRule2': {
            'other_key': 'other_value',
            'key': 'non_default_value'
        }
    }
    reporter = Reporter(handlers=[dummy_handler])
    linter = Linter(reporter, dummy_rules, config=config)
    linter.check(Sourcefile.from_file(dummy_file))


def test_linter_transformation(dummy_file, dummy_rules, dummy_handler):
    '''Make sure that linter runs through all given rules and hands them
    the right config when called via Transformation.'''

    config = {
        'TestRule2': {
            'other_key': 'other_value',
            'key': 'non_default_value'
        }
    }
    reporter = Reporter(handlers=[dummy_handler])
    linter = Linter(reporter, dummy_rules, config=config)
    transformation = LinterTransformation(linter=linter)
    transformation.apply(Sourcefile.from_file(dummy_file))


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

class PicklableTestHandler(GenericHandler):

    def __init__(self, basedir, target):
        super().__init__(basedir)
        self.target = target

    def handle(self, file_report):
        return str(self.get_relative_filename(file_report.filename))

    def output(self, handler_reports):
        self.target('\n'.join(handler_reports))


@pytest.mark.parametrize('max_workers', [None, 1, 4])
@pytest.mark.parametrize('counter,exclude,files', [
    (13, None, [
        'projA/module/compute_l1_mod.f90',
        'projA/module/compute_l2_mod.f90',
        'projA/module/driverA_mod.f90',
        'projA/module/driverB_mod.f90',
        'projA/module/driverC_mod.f90',
        'projA/module/driverD_mod.f90',
        'projA/module/header_mod.f90',
        'projA/module/kernelA_mod.F90',
        'projA/module/kernelB_mod.F90',
        'projA/module/kernelC_mod.f90',
        'projA/module/kernelD_mod.f90',
        'projA/source/another_l1.F90',
        'projA/source/another_l2.F90'
    ]),
    (13, [], [
        'projA/module/compute_l1_mod.f90',
        'projA/module/compute_l2_mod.f90',
        'projA/module/driverA_mod.f90',
        'projA/module/driverB_mod.f90',
        'projA/module/driverC_mod.f90',
        'projA/module/driverD_mod.f90',
        'projA/module/header_mod.f90',
        'projA/module/kernelA_mod.F90',
        'projA/module/kernelB_mod.F90',
        'projA/module/kernelC_mod.f90',
        'projA/module/kernelD_mod.f90',
        'projA/source/another_l1.F90',
        'projA/source/another_l2.F90'
    ]),
    (5, ['**/kernel*', '**/driver*'], [
        'projA/module/compute_l1_mod.f90',
        'projA/module/compute_l2_mod.f90',
        'projA/module/header_mod.f90',
        'projA/source/another_l1.F90',
        'projA/source/another_l2.F90'
    ]),
    (4, ['*.f90'], [
        'projA/module/kernelA_mod.F90',
        'projA/module/kernelB_mod.F90',
        'projA/source/another_l1.F90',
        'projA/source/another_l2.F90'
    ])
])
def test_linter_lint_files_glob(here, rules, counter, exclude, files, max_workers):
    basedir = here.parent/'sources'
    config = {
        'basedir': str(basedir),
        'include': ['projA/**/*.f90', 'projA/**/*.F90'],
    }
    if exclude is not None:
        config['exclude'] = exclude
    if max_workers is not None:
        config['max_workers'] = max_workers

    target_file_name = gettempdir()/'linter_lint_files_glob.log'
    if max_workers and max_workers > 1:
        target = LazyTextfile(target_file_name)
    else:
        target = target_file_name.open('w')
    handler = PicklableTestHandler(basedir=basedir, target=target.write)
    checked = lint_files(rules, config, handlers=[handler])

    assert checked == counter

    if not max_workers or max_workers == 1:
        target.close()

    checked_files = Path(target_file_name).read_text().splitlines()
    assert len(checked_files) == counter
    if max_workers and max_workers > 1:
        # Cannot guarantee order anymore
        assert set(checked_files) == set(files)
    else:
        assert checked_files == files

    target_file_name.unlink(missing_ok=True)


@pytest.mark.parametrize('counter,routines,files', [
    (5, [{'name': 'driverA', 'role': 'driver'}], [
        'module/driverA_mod.f90',
        'module/kernelA_mod.F90',
        'module/compute_l1_mod.f90',
        'source/another_l1.F90',
        'source/another_l2.F90'
    ]),
    (3, [
        {'name': 'another_l1', 'role': 'driver'},
        {'name': 'compute_l1', 'role': 'driver'}
    ], [
        'source/another_l1.F90',
        'module/compute_l1_mod.f90',
        'source/another_l2.F90',
    ]),
    (2, [
        {'name': 'another_l1', 'role': 'driver'}
    ], [
        'source/another_l1.F90',
        'source/another_l2.F90'
    ]),
])
def test_linter_lint_files_scheduler(here, rules, counter, routines, files):
    basedir = here.parent/'sources/projA'

    class TestHandler(GenericHandler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.counter = 0
            self.files = []

        def handle(self, file_report):
            self.counter += 1
            self.files += [str(Path(file_report.filename).relative_to(basedir))]

        def output(self, handler_reports):
            pass

    config = {
        'basedir': str(basedir),
        'scheduler': {
            'default': {
                'mode': 'lint',
                'role': 'kernel',
                'expand': True,
                'strict': False,
                'block': ['compute_l2']
            },
            'routine': routines
        }
    }

    handler = TestHandler()
    checked = lint_files(rules, config, handlers=[handler])

    assert checked == counter
    assert handler.counter == counter
    assert handler.files == files


@pytest.mark.parametrize('config', [
    {'scheduler': {
        'default': {
            'mode': 'lint',
            'role': 'kernel',
            'expand': True,
            'strict': True,
        },
        'routine': [{
            'name': 'other_routine',
        }]
    }},
    {'include': ['linter_lint_files_fix.F90']}
])
def test_linter_lint_files_fix(config):

    class TestRule(GenericRule):

        fixable = True

        @classmethod
        def check_subroutine(cls, subroutine, rule_report, config):
            if not subroutine.name.isupper():
                rule_report.add(f'Subroutine name "{subroutine.name}" is not upper case', subroutine)

        @classmethod
        def fix_sourcefile(cls, sourcefile, rule_report, config):
            if rule_report.problem_reports:
                sourcefile._source = None

        @classmethod
        def fix_subroutine(cls, subroutine, rule_report, config):
            assert len(rule_report.problem_reports) == 1
            if rule_report.problem_reports[0].location is subroutine:
                subroutine.name = subroutine.name.upper()
                return {None: None}
            return {}

    fcode = """
subroutine some_routine
implicit none
end subroutine some_routine

subroutine OTHER_ROUTINE
implicit none
call some_routine
end subroutine OTHER_ROUTINE
    """.strip()
    assert fcode.count('some_routine') == 3
    assert fcode.count('SOME_ROUTINE') == 0

    basedir = gettempdir()
    filename = basedir/'linter_lint_files_fix.F90'
    filename.write_text(fcode)

    config.update({
        'basedir': basedir,
        'fix': True,
    })

    checked_files = lint_files([TestRule], config)
    assert checked_files == 1

    fixed_fcode = filename.read_text()
    assert fixed_fcode.count('some_routine') == 1  # call statement
    assert fixed_fcode.count('SOME_ROUTINE') == 2

    filename.unlink(missing_ok=True)
