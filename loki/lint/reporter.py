from pathlib import Path
from junit_xml import TestSuite, TestCase

from loki.subroutine import Subroutine
from loki.module import Module
from loki.logging import logger
from loki.lint.utils import get_filename_from_parent


class ProblemReport:

    def __init__(self, msg, location):
        self.msg = msg
        self.location = location


class RuleReport:

    def __init__(self, rule, problem_reports=None):
        self.rule = rule
        self.problem_reports = problem_reports or []
        self.elapsed_sec = 0.

    def add(self, msg, location):
        self.problem_reports.append(ProblemReport(msg, location))


class FileReport:

    def __init__(self, filename, reports=None):
        self.filename = filename
        self.reports = reports or []

    def add(self, rule_report):
        if not isinstance(rule_report, RuleReport):
            raise TypeError('{} given, {} expected'.format(type(rule_report), RuleReport))
        self.reports.append(rule_report)


class Reporter:

    def __init__(self, handlers=None):
        if not handlers:
            handlers = [DefaultHandler()]
        self.handlers_reports = {handler: [] for handler in handlers}

    def init_parallel(self, manager):
        parallel_reports = manager.dict()
        for handler, reports in self.handlers_reports.items():
            parallel_reports[handler] = manager.list(reports)
        self.handlers_reports = parallel_reports

    def add_file_report(self, file_report):
        if not isinstance(file_report, FileReport):
            raise TypeError('{} given, {} expected'.format(type(file_report), FileReport))
        for handler, reports in self.handlers_reports.items():
            reports.append(handler.handle(file_report))

    def add_file_error(self, filename, rule, msg):
        problem_report = ProblemReport(msg, None)
        rule_report = RuleReport(rule, [problem_report])
        file_report = FileReport(filename, [rule_report])
        self.add_file_report(file_report)

    def output(self):
        for handler, reports in self.handlers_reports.items():
            handler.output(reports)


class GenericHandler:

    def __init__(self, basedir=None):
        self.basedir = basedir

    def format_location(self, filename, location):
        if not filename:
            filename = get_filename_from_parent(location) or ''
        if filename and self.basedir:
            try:
                filename = Path(filename).relative_to(self.basedir)
            except ValueError:
                pass
        line = ''
        source = getattr(location, '_source', getattr(location, 'source', None))
        if source is not None:
            if source.lines[0] == source.lines[1]:
                line = ' (l. {})'.format(source.lines[0])
            else:
                line = ' (ll. {}-{})'.format(*source.lines)
        scope = ''
        if isinstance(location, Subroutine):
            scope = ' in routine "{}"'.format(location.name)
        if isinstance(location, Module):
            scope = ' in module "{}"'.format(location.name)
        return '{}{}{}'.format(filename, line, scope)

    def handle(self, file_report):
        raise NotImplementedError()

    def output(self, handler_reports):
        raise NotImplementedError()


class DefaultHandler(GenericHandler):

    fmt_string = '{rule}: {location} - {msg}'

    def __init__(self, target=logger.warning, immediate_output=True, basedir=None):
        super().__init__(basedir)
        self.target = target
        self.immediate_output = immediate_output

    def handle(self, file_report):
        filename = file_report.filename
        reports_list = []
        for rule_report in file_report.reports:
            rule = rule_report.rule.__name__
            if hasattr(rule_report.rule, 'docs') and rule_report.rule.docs:
                if 'id' in rule_report.rule.docs:
                    rule = '[{}] '.format(rule_report.rule.docs['id']) + rule
            for problem in rule_report.problem_reports:
                location = self.format_location(filename, problem.location)
                msg = self.fmt_string.format(rule=rule, location=location, msg=problem.msg)
                if self.immediate_output:
                    self.target(msg)
                reports_list.append(msg)
        return reports_list

    def output(self, handler_reports):
        if not self.immediate_output:
            for reports in handler_reports:
                for report in reports:
                    self.target(report)


class JunitXmlHandler(GenericHandler):

    fmt_string = '{location} - {msg}'

    def __init__(self, target=logger.warning, basedir=None):
        super().__init__(basedir)
        self.target = target

    def handle(self, file_report):
        filename = file_report.filename
        classname = str(Path(filename).with_suffix(''))
        test_cases = []
        for rule_report in file_report.reports:
            kwargs = {'name': rule_report.rule.__name__, 'classname': classname,
                      'allow_multiple_subelements': True, 'elapsed_sec': rule_report.elapsed_sec}
            messages = []
            for problem in rule_report.problem_reports:
                location = self.format_location(filename, problem.location)
                msg = self.fmt_string.format(location=location, msg=problem.msg)
                messages.append(msg)
            test_cases.append((kwargs, messages))
        return (filename, test_cases)

    def output(self, handler_reports):
        testsuites = []
        for filename, tc_args in handler_reports:
            testcases = []
            for kwargs, messages in tc_args:
                testcase = TestCase(**kwargs)
                for msg in messages:
                    testcase.add_failure_info(msg)
                testcases.append(testcase)
            testsuites.append(TestSuite(filename, testcases))
        xml_string = TestSuite.to_xml_string(testsuites)
        self.target(xml_string)
