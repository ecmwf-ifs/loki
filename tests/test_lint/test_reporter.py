from pathlib import Path
import pytest

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from loki import Intrinsic
from loki.lint.reporter import ProblemReport, RuleReport, FileReport, DefaultHandler, ViolationFileHandler
from loki.lint.rules import GenericRule


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='dummy_file')
def dummy_file_fixture(here):
    file_path = here/'test_reporter_dummy_file.F90'
    fcode = "! dummy file for reporter tests"
    file_path.write_text(fcode)
    yield file_path
    file_path.unlink()


@pytest.fixture(scope='module', name='dummy_file_report')
def fixture_dummy_file_report(dummy_file):
    file_report = FileReport(str(dummy_file))
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


def test_reports(dummy_file):
    file_report = FileReport(str(dummy_file))
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


@pytest.mark.skipif(not HAVE_YAML, reason='Pyyaml not installed')
def test_violation_file_handler(dummy_file, dummy_file_report):
    logger_target = DummyLogger()
    handler = ViolationFileHandler(target=logger_target.write)
    reports = handler.handle(dummy_file_report)
    handler.output([reports])
    assert len(logger_target.messages) == 1
    yaml_report = yaml.safe_load(logger_target.messages[0])
    assert len(yaml_report) == 1
    assert str(dummy_file) in yaml_report
    file_report = yaml_report[str(dummy_file)]
    assert file_report['filehash'] == dummy_file_report.hash
    assert len(file_report['rules']) == 1
    assert 'GenericRule' in file_report['rules']


@pytest.mark.skip()
def test_junit_xml_handler():
    # TODO
    pass
