from pathlib import Path

try:
    from junit_xml import TestSuite, TestCase
    HAVE_JUNIT_XML = True
except ImportError:
    HAVE_JUNIT_XML = False

from loki import Sourcefile, Module, Subroutine, Node
from loki.logging import logger, error
from loki.lint.utils import get_filename_from_parent, is_rule_disabled


class ProblemReport:
    """
    Data type to represent a problem reported for a node in the IR

    Parameters
    ----------
    msg : str
        The message describing the problem.
    location : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine` or :any:`Node`
        The IR component in which the problem exists.
    """

    def __init__(self, msg, location):
        self.msg = msg
        self.location = location


class RuleReport:
    """
    Container type to collect all individual problems reported by a rule

    All :class:`RuleReport` instances that belong to a file are
    collected in a :class:`FileReport`.

    Parameters
    ----------
    :param rule: the rule that generated the report.
    :type rule: subclass of :py:class:`GenericRule`
    :param list problem_reports: (optional) list of :py:class:`ProblemReport`.
    """

    def __init__(self, rule, problem_reports=None):
        self.rule = rule
        self.problem_reports = problem_reports or []
        self.elapsed_sec = 0.

    def add(self, msg, location):
        """
        Convenience function to append a problem report to the list of problems
        reported by the rule.

        :param str msg: the message describing the problem.
        :param location: the IR node or expression node in which the problem exists.
        """
        if not isinstance(location, (Sourcefile, Module, Subroutine, Node)):
            raise TypeError('Invalid type for report location: {}'.format(type(location).__name__))
        if not is_rule_disabled(location, self.rule.identifiers()):
            self.problem_reports.append(ProblemReport(msg, location))


class FileReport:
    """
    Container type to collect all rule reports for a file.

    :param str filename: the filename of the file the reports belong to.
    :param list reports: (optional) list of :py:class:`RuleReport`.
    """

    def __init__(self, filename, reports=None):
        self.filename = filename
        self.reports = reports or []

    def add(self, rule_report):
        """
        Append a rule report to the list of reports.

        :param :py:class:`RuleReport` rule_report: the report to be stored.
        """
        if not isinstance(rule_report, RuleReport):
            raise TypeError(f'{type(rule_report)} given, {RuleReport} expected')
        self.reports.append(rule_report)

    @property
    def fixable_reports(self):
        """
        Yield only those rule reports that belong to a rule that can be fixed.
        """
        fixable_reports = [report for report in self.reports
                           if report.rule.fixable and report.problem_reports]
        return fixable_reports


class Reporter:
    """
    Manager for problem reports and their handler.

    It collects file reports and feeds them to all available handlers to generate
    their individual reporting pieces.
    Note that this processing of reports happens immediately when adding a new file
    report for two reasons:
        1. Enable immediate output functionality (i.e., being able to print problems
           as soon as they are detected and not only at the very end of a (lengthy)
           multi file parser run.
        2. To allow parallel processing. The location of problem reports is not
           pickable and thus they need to be processed into a pickable form.

    The class maintains a `dict` in which a list of reports is stored for each handler.
    In a parallel setting, this needs to be initialized explicitly to enable thread
    safe data structures by calling `init_parallel()`.

    :param list handlers: (optional) list of enabled handlers. If none given,
        :py:class:`DefaultHandler` will be used.
    """

    def __init__(self, handlers=None):
        if not handlers:
            handlers = [DefaultHandler()]
        self.handlers_reports = {handler: [] for handler in handlers}

    def init_parallel(self, manager):
        """
        Additional initialization step when using the reporter in a parallel setting.

        :param :py:class:`multiprocessing.Manager` manager: the multiprocessing manager
            that can be used to create thread safe data structures.
        """
        parallel_reports = manager.dict()
        for handler, reports in self.handlers_reports.items():
            parallel_reports[handler] = manager.list(reports)
        self.handlers_reports = parallel_reports

    def add_file_report(self, file_report):
        """
        Process a file report in all handlers and store the results.

        :param :py:class:`FileReport` file_report: the file report to be processed.
        """
        if not isinstance(file_report, FileReport):
            raise TypeError(f'{type(file_report)} given, {FileReport} expected')
        for handler, reports in self.handlers_reports.items():
            reports.append(handler.handle(file_report))

    def add_file_error(self, filename, rule, msg):
        """
        Create a file report with a single problem reported and add it.

        This is a convenience function that can be used, e.g., to report a failing rule
        or other problems with a certain file.

        :param str filename: the file name of the corresponding file.
        :param rule: the rule that exposed the problem or `None`.
        :param str msg: the description of the problem.
        """
        problem_report = ProblemReport(msg, None)
        rule_report = RuleReport(rule, [problem_report])
        file_report = FileReport(filename, [rule_report])
        self.add_file_report(file_report)

    def output(self):
        """
        Call the `output` function for all reports on every handler.
        """
        for handler, reports in self.handlers_reports.items():
            handler.output(reports)


class GenericHandler:
    """
    Base class for report handler.

    :param str basedir: (optional) basedir relative to which file paths are given.
    """

    def __init__(self, basedir=None):
        self.basedir = basedir

    def format_location(self, filename, location):
        """
        Create a string representation of the location given in a `ProblemReport`.

        For a given location it tries to determine:
            - the file name (if not given)
            - the source line(s)
            - the name of the scope (i.e., enclosing subroutine or module)

        :param str filename: the file name of the source file.
        :param location: the AST node that triggered the problem report.
        :type location: an IR or expression node, or a Subroutine, Sourcefile or Module
            object.

        :return: the formatted string in the form
            "<filename> (l. <line(s)>) [in routine/module ...]"
        :rtype: str
        """
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
                line = f' (l. {source.lines[0]})'
            else:
                line = f' (ll. {source.lines[0]}-{source.lines[1]})'
        scope = ''
        if isinstance(location, Subroutine):
            scope = f' in routine "{location.name}"'
        if isinstance(location, Module):
            scope = f' in module "{location.name}"'
        return f'{filename}{line}{scope}'

    def handle(self, file_report):
        """
        Handle the given `file_report`.

        This routine has to be implemented by the handler class.
        It should either print/save the report immediately or return a picklable
        object that is later to be printed/saved via `output()`.
        """
        raise NotImplementedError()

    def output(self, handler_reports):
        """
        Output the list of report objects created by `handle()`.
        """
        raise NotImplementedError()


class DefaultHandler(GenericHandler):
    """
    The default report handler for command line output of problems.

    :param target: the output destination.
    :param bool immediate_output: print problems immediately if True, otherwise
        collect messages and print when calling `output()`.
    :param str basedir: (optional) basedir relative to which file paths are given.
    """

    fmt_string = '{rule}: {location} - {msg}'

    def __init__(self, target=logger.warning, immediate_output=True, basedir=None):
        super().__init__(basedir)
        self.target = target
        self.immediate_output = immediate_output

    def handle(self, file_report):
        """
        Creates a string output of all problem reports and (by default) prints them
        immediately to `target`.

        :param FileReport file_report: the file report to be processed.
        :return: the list of problem report strings.
        :rtype: list of str
        """
        filename = file_report.filename
        reports_list = []
        for rule_report in file_report.reports:
            rule = rule_report.rule.__name__
            if hasattr(rule_report.rule, 'docs') and rule_report.rule.docs:
                if 'id' in rule_report.rule.docs:
                    rule = f'[{rule_report.rule.docs["id"]}] ' + rule
            for problem in rule_report.problem_reports:
                location = self.format_location(filename, problem.location)
                msg = self.fmt_string.format(rule=rule, location=location, msg=problem.msg)
                if self.immediate_output:
                    self.target(msg)
                reports_list.append(msg)
        return reports_list

    def output(self, handler_reports):
        """
        Print all reports to `target` if `immediate_output` is disabled.

        :param handler_reports: the list of lists of reports.
        :type handler_reports: list of list of str
        """
        if not self.immediate_output:
            for reports in handler_reports:
                for report in reports:
                    self.target(report)


class JunitXmlHandler(GenericHandler):
    """
    Report handler class that generates JUnit-compatible XML output that can be understood
    by CI platforms such as Jenkins or Bamboo.

    :param target: the output destination.
    :param str basedir: (optional) basedir relative to which file paths are given.
    """

    fmt_string = '{location} - {msg}'

    def __init__(self, target=logger.warning, basedir=None):
        if not HAVE_JUNIT_XML:
            error('junit_xml is not available')
            raise RuntimeError

        super().__init__(basedir)
        self.target = target

    def handle(self, file_report):
        """
        Creates tuples of string arguments for :py:class:`junit_xml.TestCase`

        :param FileReport file_report: the file report to be processed.
        """
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
        """
        Generate the XML output from the list of reports.
        """
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
