# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from loki.lint.linter import lint_files
from loki.lint.reporter import (
    ProblemReport, RuleReport, FileReport,
    DefaultHandler, ViolationFileHandler,
    LazyTextfile
)
from loki.lint.rules import GenericRule, RuleType


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


@pytest.fixture(scope='module', name='rules')
def fixture_rules():
    rules = importlib.import_module('rules')
    return rules


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


def test_lazy_textfile(tmp_path):
    # Choose the output file and make sure it doesn't exist
    filename = tmp_path/'lazytextfile.log'

    # Instantiating the object should _not_ create the file
    f = LazyTextfile(filename)
    assert not filename.exists()

    # Writing to the object should open (and therefore create) the file
    f.write('s0me TEXT')
    assert filename.exists()

    # Writing more to the object should append text
    f.write(' AAAAND other Th1ngs!!!')

    # Deleting the object should (hopefully) trigger __del__,
    # which should flush the buffers to disk and allow us to read
    # (and check) the content
    del f
    assert filename.read_text() == 's0me TEXT AAAAND other Th1ngs!!!'


@pytest.mark.parametrize('max_workers', [None, 1])
@pytest.mark.parametrize('fail_on,failures', [(None,0), ('kernel',4)])
def test_linter_junitxml(tmp_path, testdir, max_workers, fail_on, failures):
    class RandomFailingRule(GenericRule):
        type = RuleType.WARN
        docs = {'title': 'A dummy rule for the sake of testing the Linter'}
        config = {'dummy_key': 'dummy value'}

        @classmethod
        def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
            if fail_on and fail_on in subroutine.name:
                rule_report.add(cls.__name__, subroutine)

    basedir = testdir/'sources'
    junitxml_file = tmp_path/'linter_junitxml_outputfile.xml'
    config = {
        'basedir': str(basedir),
        'include': ['projA/**/*.f90', 'projA/**/*.F90'],
        'junitxml_file': str(junitxml_file)
    }
    if max_workers is not None:
        config['max_workers'] = max_workers

    checked = lint_files([RandomFailingRule], config)

    assert checked == 15

    # Just a few sanity checks on the XML
    xml = ET.parse(junitxml_file).getroot()
    assert xml.tag == 'testsuites'
    assert xml.attrib['tests'] == '15'
    assert xml.attrib['failures'] == str(failures)


@pytest.mark.skipif(not HAVE_YAML, reason='Pyyaml not installed')
@pytest.mark.parametrize('max_workers', [None, 1])
@pytest.mark.parametrize('fail_on,failures', [(None,0), ('kernel',4)])
@pytest.mark.parametrize('use_line_hashes', [None, False, True])
def test_linter_violation_file(tmp_path, testdir, rules, max_workers, fail_on, failures, use_line_hashes):
    class RandomFailingRule(GenericRule):
        type = RuleType.WARN
        docs = {'title': 'A dummy rule for the sake of testing the Linter'}
        config = {'dummy_key': 'dummy value'}

        @classmethod
        def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
            if fail_on and fail_on in subroutine.name:
                rule_report.add(cls.__name__, subroutine)

    basedir = testdir/'sources'
    violations_file = tmp_path/'linter_violations_file.yml'
    config = {
        'basedir': str(basedir),
        'include': ['projA/**/*.f90', 'projA/**/*.F90'],
        'violations_file': str(violations_file),
    }
    if use_line_hashes is not None:
        config['use_violations_file_line_hashes'] = use_line_hashes
    if max_workers is not None:
        config['max_workers'] = max_workers

    checked = lint_files([RandomFailingRule, rules.DummyRule], config)

    assert checked == 15

    # Just a few sanity checks on the yaml
    yaml_report = yaml.safe_load(violations_file.read_text())
    if not failures:
        assert yaml_report is None
    else:
        assert len(yaml_report) == failures

        for file, report in yaml_report.items():
            assert fail_on in file
            if use_line_hashes is False:
                assert 'filehash' in report
                assert report['rules'] == ['RandomFailingRule']
            else:
                assert 'filehash' not in report
                assert len(report['rules']) == 1
                assert 'RandomFailingRule' in report['rules'][0]
                if file.endswith('kernelE_mod.f90'):
                    assert len(report['rules'][0]['RandomFailingRule']) == 2
                else:
                    assert len(report['rules'][0]['RandomFailingRule']) == 1

    # Plug the violations file into the config and see if we don't have
    # violations in another linter pass
    config['disable'] = yaml_report
    checked = lint_files([RandomFailingRule, rules.DummyRule], config)
    assert checked == 15
    assert yaml.safe_load(violations_file.read_text()) is None
