# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

try:
    from junit_xml import TestSuite, TestCase, to_xml_report_string
    HAVE_JUNIT_XML = True
except ImportError:
    HAVE_JUNIT_XML = False

from loki.ir import Node
from loki.lint.utils import get_filename_from_parent, is_rule_disabled, get_location_hash
from loki.logging import logger, error
from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine
from loki.tools import filehash

__all__ = [
    'ProblemReport', 'RuleReport', 'FileReport', 'Reporter',
    'GenericHandler', 'DefaultHandler', 'ViolationFileHandler',
    'JunitXmlHandler', 'LazyTextfile'
]


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
    rule : :any:`GenericRule`
        The rule that generated the report
    reports : list of :any:`ProblemReport`, optional
        List of problem reports for this rule
    disabled : bool or list, optional
        Flag to disable reporting of this rule. If set to `True`, no violations will
        be reported. If a list of hashes is provided, the first line of a violation
        will be compared against these hashes and not reported if found.
    """

    def __init__(self, rule, reports=None, disabled=None):
        self.rule = rule
        self.problem_reports = reports or []
        self.disabled = disabled or []
        self.elapsed_sec = 0.

    def add(self, msg, location):
        """
        Convenience function to append a problem report to the list of problems
        reported by the rule.

        Parameters
        ----------
        msg : str
            The message describing the problem.
        location : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine` or :any:`Node`
            The IR node or expression node in which the problem exists.
        """
        if self.disabled is True:
            return
        if not isinstance(location, (Sourcefile, Module, Subroutine, Node)):
            raise TypeError(f'Invalid type for report location: {type(location).__name__}')
        if not is_rule_disabled(location, self.rule.identifiers(), self.disabled):
            self.problem_reports.append(ProblemReport(msg, location))


class FileReport:
    """
    Container type to collect all rule reports for a file

    Parameters
    ----------
    filename : str
        The filename of the file the report is for
    hash : str, optional
        Provide a hash for the file's content to identify the file version
    reports : list, optional
        List of :py:class:`RuleReport`.
    """

    def __init__(self, filename, hash=None, reports=None):  # pylint: disable=redefined-builtin
        self.filename = filename
        self.hash = hash or filehash(Path(filename).read_text())
        self.reports = reports or []

    def add(self, rule_report):
        """
        Append a rule report to the list of reports.

        Parameters
        -----------
        rule_report : :any:`RuleReport`
            The report to be stored.
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

    #. Enable immediate output functionality (i.e., being able to print problems
       as soon as they are detected and not only at the very end of a (lengthy)
       multi file parser run.
    #. To allow parallel processing. The location of problem reports is not
       pickable and thus they need to be processed into a pickable form.

    The class maintains a `dict` in which a list of reports is stored for each handler.
    In a parallel setting, this needs to be initialized explicitly to enable thread
    safe data structures by calling `init_parallel()`.

    Parameters
    ----------
    list handlers : list of :any:`GenericHandler`, optional
        The enabled handlers. If none given, :any:`DefaultHandler` will be used.
    """

    def __init__(self, handlers=None):
        if not handlers:
            handlers = [DefaultHandler()]
        self.handlers_reports = {handler: [] for handler in handlers}

    def init_parallel(self, manager):
        """
        Additional initialization step when using the reporter in a parallel setting.

        Parameters
        ----------
        manager : :any:`multiprocessing.Manager`
            The multiprocessing manager that should be used to create thread safe data structures.
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

        Parameters
        ----------
        filename : str
            The file name of the corresponding file.
        rule : :any:`GenericRule`
            The rule that exposed the problem or `None`.
        msg : str
            A description of the problem.
        """
        problem_report = ProblemReport(msg, None)
        rule_report = RuleReport(rule, reports=[problem_report])
        file_report = FileReport(filename, reports=[rule_report])
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

    Parameters
    ----------
    basedir : str, optional
        Base directory path relative to which file paths are given.
    """

    def __init__(self, basedir=None):
        self.basedir = basedir

    def get_relative_filename(self, filename):
        if filename and self.basedir:
            try:
                filename = Path(filename).relative_to(self.basedir)
            except ValueError:
                pass
        return filename


    def format_location(self, filename, location):
        """
        Create a string representation of the location given in a `ProblemReport`.

        For a given location it tries to determine:
            - the file name (if not given)
            - the source line
            - the name of the scope (i.e., enclosing subroutine or module)

        Parameters
        ----------
        filename : str
            The file name of the source file.
        location : :any:`Node` or :any:`Subroutine` or :any:`Sourcefile` or :any:`Module`
            The AST node that triggered the problem report.

        Returns
        -------
        str
            The formatted string in the form
            "<filename> (l. <line(s)>) [in routine/module ...]"
        """
        if not filename:
            filename = get_filename_from_parent(location) or ''
        filename = self.get_relative_filename(filename)

        source = getattr(location, '_source', getattr(location, 'source', None))
        if source is not None:
            line = f' (l. {source.lines[0]})'
        else:
            line = ''

        if isinstance(location, Subroutine):
            scope = f' in routine "{location.name}"'
        elif isinstance(location, Module):
            scope = f' in module "{location.name}"'
        else:
            scope = ''
        return f'{filename}{line}{scope}'

    def handle(self, file_report):  # pylint: disable=unused-argument
        """
        Handle the given :attr:`file_report`.

        This routine has to be implemented by the handler class.
        It should either print/save the report immediately or return a picklable
        object that is later to be printed/saved via :meth:`output`.

        Note that the only requirement is that :meth:`handle` and
        :meth:`output` are compatible in the sense that a list of objects
        returned by :meth:`handle` can be processed by :meth:`output`.
        """
        raise NotImplementedError()

    def output(self, handler_reports):
        """
        Output the list of report objects created by :meth:`handle`.
        """
        raise NotImplementedError()


class DefaultHandler(GenericHandler):
    """
    The default report handler for command line output of problems.

    Parameters
    ----------
    target : optional
        The output destination as a callback. Will be called with a string.
        Defaults to :attr:`loki.logging.logger.warning`
    immediate_output : bool, optional
        Print problems immediately if `True`, otherwise
        collect messages and print when calling `output()`. Defaults to `True`
    basedir : str, optional
        Base directory path relative to which file paths are given.
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

        Parameters
        ----------
        file_report : :any:`FileReport`
            The file report to be processed.

        Returns
        -------
        list of str
            The list of problem report strings.
        """
        filename = file_report.filename
        reports_list = []
        for rule_report in file_report.reports:
            rule = rule_report.rule.__name__
            if hasattr(rule_report.rule, 'docs') and rule_report.rule.docs:
                if 'id' in rule_report.rule.docs:
                    rule = f'[{rule_report.rule.docs["id"]}] {rule}'
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

        Parameters
        ----------
        handler_reports : list of list of str
            The list of lists of reports.
        """
        if not self.immediate_output:
            for reports in handler_reports:
                for report in reports:
                    self.target(report)


class ViolationFileHandler(GenericHandler):
    """
    Report handler class that writes a YAML file with rules violated per file

    The content of the YAML file has the form

    .. code-block:: yaml

        'path/to/my/file.F90':
            filehash: 'abc123'
            rules:
            - SomeRule
            - SomeOtherRule

    This YAML file can be included into a linter config file to disable reporting
    of these rules on that version of the file (source modifications identified by
    :any:`filehash`) in future runs.

    An alternative format of this file is the following

    .. code-block:: yaml

        'path/to/my/file.F90':
            rules:
            - Some Rule:
              - <line hash>
              - <line hash>
            - SomeOtherRule:
              - <line hash>

    This will disable reporting violations of given rules for the specified file,
    as long as the line hash matches one of the listed line hashes. This file format
    can be generated by specifying :data:`use_line_hashes`.

    Parameters
    ----------
    target :
        The output destination
    basedir : str, optional
        Base directory path relative to which file paths are given.
    use_line_hashes : bool, optional
        Disable rule violations per line
    """
    def __init__(self, target=logger.warning, basedir=None, use_line_hashes=False):
        if not HAVE_YAML:
            error('Pyyaml is not available')
            raise RuntimeError

        super().__init__(basedir)
        self.target = target
        self.use_line_hashes = use_line_hashes

    def handle(self, file_report):
        """
        Create YAML block for this file

        Parameters
        ----------
        file_report : :any:`FileReport`
            The file report to be processed

        Returns
        -------
        str
            YAML block for this file
        """
        if self.use_line_hashes:
            violated_rules = [
                {
                    rule_report.rule.__name__: [
                        line_hash for problem_report in rule_report.problem_reports
                        if (line_hash := get_location_hash(problem_report.location))
                    ]
                }
                for rule_report in file_report.reports
                if rule_report.problem_reports
            ]
        else:
            violated_rules = [
                rule_report.rule.__name__ for rule_report in file_report.reports
                if rule_report.problem_reports
            ]

        if violated_rules:
            violations_report = {}
            violations_report['rules'] = violated_rules
            if not self.use_line_hashes:
                violations_report['filehash'] = file_report.hash
            return yaml.dump({
                str(self.get_relative_filename(file_report.filename)): violations_report
            })
        return ''

    def output(self, handler_reports):
        """
        Generate the YAML output from the list of reports.
        """
        self.target('\n'.join(handler_reports))


class JunitXmlHandler(GenericHandler):
    """
    Report handler class that generates JUnit-compatible XML output that can be understood
    by CI platforms such as Jenkins or Bamboo

    Parameters
    ----------
    target :
        The output destination
    basedir : str, optional
        Base directory path relative to which file paths are given.
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
        Creates tuples of string arguments for `junit_xml.TestCase`

        Parameters
        ----------
        file_report : :any:`FileReport`
            The file report to be processed

        Returns
        -------
        tuple(str, list)
            Tuples of the form ``(filename, [(kwargs, messages)])`` with :attr:`kwargs`
            being the constructor arguments for `junit_xml.TestCase` and
            :attr:`messages` a list of strings.
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
        xml_string = to_xml_report_string(testsuites)
        self.target(xml_string)


class LazyTextfile:
    """
    Helper class to encapsulate opening and writing to a file.

    This exists because opening the file immediately and then passing
    its ``write`` function to a :any:`GenericHandler` makes it
    impossible to pickle it afterwards, which would make parallel
    execution infeasible.

    Instead of creating a more complicated interface for the handlers
    we opted for this way of a just-in-time/lazy file handler.

    The file is opened automatically for writing when calling :meth:`write`
    for the first time, and closed when the object is garbage collected.

    Parameters
    ----------
    filename : str
        The filename of the output file
    """

    def __init__(self, filename):
        self.file_name = Path(filename)
        self.file_handle = None

    def _check_open(self):
        """
        Check if the file is open already, otherwise open it
        """
        if not self.file_handle:
            self.file_handle = self.file_name.open(mode='w')  # pylint: disable=consider-using-with

    def __del__(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def write(self, msg):
        """
        Write the given :data:`msg` to the file

        The file is opened first, unless it is open already
        """
        self._check_open()
        self.file_handle.write(msg)
