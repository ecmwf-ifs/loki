import weakref

from loki.subroutine import Subroutine
from loki.logging import logger


class Report(object):

    def __init__(self, rule, location, msg):
        self.rule = rule
        self._location = weakref.ref(location)
        self.msg = msg

    @property
    def location(self):
        return self._location()


class Reporter(object):

    def __init__(self, formatters=None):
        self.formatters = formatters or [DefaultFormatter()]
        self.reports = []

    def add(self, rule, location, msg):
        report = Report(rule, location, msg)
        self.add_report(report)

    def add_report(self, report):
        if not isinstance(report, Report):
            raise TypeError(
                'Object of type "{}" given, must be of type "{}"'.format(
                    type(report), Report))
        self.reports.append(report)
        self.print_report(report)

    def print_report(self, report):
        for formatter in self.formatters:
            formatter.print(report)


class DefaultFormatter(object):

    fmt_string = '{rule}: {location} - {msg}'

    def __init__(self):
        pass

    def get_filename_from_parent(location):
        scope = location
        while hasattr(scope, 'parent') and scope.parent:
            # Go up until we are at SourceFile level
            scope = scope.parent
        if hasattr(scope, 'path'):
            return scope.path
        return None

    @classmethod
    def format_location(cls, location):
        filename = cls.get_filename_from_parent(location) or ''
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

    def print(self, report):
        rule = report.rule.__name__
        location = self.format_location(report.location)
        msg = self.fmt_string.format(rule=rule, location=location, msg=report.msg)
        logger.warning(msg)
