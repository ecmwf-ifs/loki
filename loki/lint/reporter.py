import weakref

from junit_xml import TestSuite, TestCase

from loki.subroutine import Subroutine
from loki.module import Module
from loki.logging import logger


class Report(object):

    def __init__(self, rule, filename, location, msg):
        self.rule = rule
        self.filename = filename
        self._location = weakref.ref(location)
        self.msg = msg

    @property
    def location(self):
        return self._location()


class Reporter(object):

    def __init__(self, handlers=None):
        self.handlers = handlers or [DefaultHandler()]
        self.reports = [{} for _ in self.handlers]
        self._file_list_type = list

    def init_parallel(self, manager):
        self._file_list_type = manager.list
        parallel_reports = manager.list()
        for handler_reports in self.reports:
            file_reports = manager.dict()
            for filename, reports in handler_reports.items():
                file_reports[filename] = self._file_list_type(reports)
            parallel_reports.append(file_reports)
        self.reports = parallel_reports

    @staticmethod
    def get_filename_from_parent(location):
        scope = location
        while hasattr(scope, 'parent') and scope.parent:
            # Go up until we are at SourceFile level
            scope = scope.parent
        if hasattr(scope, 'path'):
            return scope.path
        return None

    def add(self, rule, location, msg):
        filename = self.get_filename_from_parent(location)
        report = Report(rule, filename, location, msg)
        self.handle(report)

    def handle(self, report):
        if not isinstance(report, Report):
            raise TypeError(
                'Object of type "{}" given, must be of type "{}"'.format(
                    type(report), Report))
        for handler, reports in zip(self.handlers, self.reports):
            if report.filename not in reports:
                reports[report.filename] = self._file_list_type()
            reports[report.filename].append(handler.handle(report))

    def output(self):
        for handler, handler_reports in zip(self.handlers, self.reports):
            handler.output(handler_reports)


class GenericHandler(object):

    def __init__(self):
        pass

    @staticmethod
    def format_location(filename, location):
        if not filename:
            filename = Reporter.get_filename_from_parent(location) or ''
        line = ''
        if hasattr(location, '_source') and location._source:
            if location._source.lines[0] == location._source.lines[1]:
                line = ' (l. {})'.format(location._source.lines[0])
            else:
                line = ' (ll. {}-{})'.format(*location._source.lines)
        routine = ''
        if isinstance(location, Subroutine):
            routine = ' in routine "{}"'.format(location.name)
        return '{}{}{}'.format(filename, line, routine)

    def handle(self, report):
        raise NotImplementedError()

    def output(self, handler_reports):
        raise NotImplementedError()


class DefaultHandler(GenericHandler):

    fmt_string = '{rule}: {location} - {msg}'

    def __init__(self, target=logger.warning, immediate_output=True):
        super().__init__()
        self.target = target
        self.immediate_output = immediate_output

    def handle(self, report):
        rule = report.rule.__name__
        location = self.format_location(report.filename, report.location)
        msg = self.fmt_string.format(rule=rule, location=location, msg=report.msg)
        if self.immediate_output:
            self.target(msg)
        return msg

    def output(self, handler_reports):
        if not self.immediate_output:
            for reports in handler_reports.values():
                for report in reports:
                    self.target(report)


class JunitXmlHandler(GenericHandler):

    def __init__(self, target=logger.warning):
        super().__init__()
        self.target = target

    def handle(self, report):
        if isinstance(report.location, (Subroutine, Module)):
            classname = report.location.name
        else:
            classname = None
        testcase = TestCase(report.rule.__name__, classname=classname,
                            file=report.filename)
        testcase.add_failure_info(report.msg)
        return testcase

    def output(self, handler_reports):
        test_suites = []
        for filename, reports in handler_reports.items():
            test_suites.append(TestSuite(filename, reports))
        xml_string = TestSuite.to_xml_string(test_suites)
        self.target(xml_string)
