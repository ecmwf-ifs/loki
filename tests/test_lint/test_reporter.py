import pytest

from loki import Intrinsic
from loki.lint.reporter import ProblemReport, RuleReport, FileReport, DefaultHandler
from loki.lint.rules import GenericRule


@pytest.fixture(scope='module', name='dummy_file_report')
def fixture_dummy_file_report():
    file_report = FileReport('file.f90')
    rule_report = RuleReport(GenericRule)
    rule_report.add('Some message', Intrinsic('foobar'))
    rule_report.add('Other message', Intrinsic('baz'))
    file_report.add(rule_report)
    return file_report


class DummyLogger:

    def __init__(self):
        self.messages = []

    def write(self, msg):
        self.messages += [msg]


def test_reports():
    file_report = FileReport('file.f90')
    assert not file_report.reports and file_report.reports is not None

    class SomeRule(GenericRule):
        pass

    rule_report = RuleReport(SomeRule)
    assert not rule_report.problem_reports and rule_report.problem_reports is not None
    rule_report.add('Some message', Intrinsic('foobar'))
    rule_report.add('Other message', Intrinsic('baz'))
    assert len(rule_report.problem_reports) == 2
    assert isinstance(rule_report.problem_reports[0], ProblemReport)
    assert rule_report.problem_reports[0].msg == 'Some message'

    file_report.add(rule_report)
    assert len(file_report.reports) == 1


def test_default_handler_immediate(dummy_file_report):
    logger_target = DummyLogger()
    handler = DefaultHandler(target=logger_target.write)
    reports = handler.handle(dummy_file_report)
    assert len(logger_target.messages) == 2
    handler.output([reports])
    assert len(logger_target.messages) == 2


def test_default_handler_not_immediate(dummy_file_report):
    logger_target = DummyLogger()
    handler = DefaultHandler(target=logger_target.write, immediate_output=False)
    reports = handler.handle(dummy_file_report)
    assert len(logger_target.messages) == 0
    handler.output([reports])
    assert len(logger_target.messages) == 2


@pytest.mark.skip()
def test_junit_xml_handler():
    # TODO
    pass
