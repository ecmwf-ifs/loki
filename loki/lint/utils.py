from enum import Enum

from loki import SourceFile, Module, Subroutine, Transformer

__all__ = ['RuleType', 'GenericRule', 'Fixer', 'get_filename_from_parent']


class RuleType(Enum):
    """
    Available types for rules with increasing severity.
    """

    INFO = 1
    WARN = 2
    SERIOUS = 3
    ERROR = 4


class GenericRule:
    '''
    Generic interface for linter rules providing default values and the
    general `check` routine that calls the specific entry points to rules
    (subroutines, modules, and the source file).

    When adding a new rule, it must inherit from :py:class:`GenericRule`
    and define `type` and provide `title` (and `id`, if applicable) in `docs`.
    Optional configuration values can be defined in `config` together with
    the default value for this option. Only the relevant entry points to a
    rule must be implemented.

    '''
    type = None

    docs = None

    config = {}

    fixable = False

    deprecated = False

    replaced_by = ()

    @classmethod
    def check_module(cls, module, rule_report, config):
        '''
        Perform rule checks on module level. Must be implemented by
        a rule if applicable.
        '''

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''
        Perform rule checks on subroutine level. Must be implemented by
        a rule if applicable.
        '''

    @classmethod
    def check_file(cls, sourcefile, rule_report, config):
        '''
        Perform rule checks on file level. Must be implemented by
        a rule if applicable.
        '''

    @classmethod
    def check(cls, ast, rule_report, config):
        '''
        Perform checks on all entities in the given IR object.

        This routine calls `check_module`, `check_subroutine` and `check_file`
        as applicable for all entities in the given IR object.

        :param ast: the IR object to be checked.
        :type ast: :py:class:`SourceFile`, :py:class:`Module`, or
                   :py:class:`Subroutine`
        :param rule_report: the reporter object for the rule.
        :type rule_report: :py:class:`RuleReport`
        :param dict config: a `dict` with the config values.

        '''
        # Perform checks on source file level
        if isinstance(ast, SourceFile):
            cls.check_file(ast, rule_report, config)

            # Then recurse for all modules and subroutines in that file
            if hasattr(ast, 'modules') and ast.modules is not None:
                for module in ast.modules:
                    cls.check(module, rule_report, config)
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                for subroutine in ast.subroutines:
                    cls.check(subroutine, rule_report, config)

        # Perform checks on module level
        elif isinstance(ast, Module):
            cls.check_module(ast, rule_report, config)

            # Then recurse for all subroutines in that module
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                for subroutine in ast.subroutines:
                    cls.check(subroutine, rule_report, config)

        # Peform checks on subroutine level
        elif isinstance(ast, Subroutine):
            cls.check_subroutine(ast, rule_report, config)

            # Recurse for any procedures contained in a subroutine
            if hasattr(ast, 'members') and ast.members is not None:
                for member in ast.members:
                    cls.check(member, rule_report, config)

    @classmethod
    def fix_module(cls, module, rule_report, config):
        '''
        Fix rule on module level. Must be implemented by a rule if applicable.
        '''

    @classmethod
    def fix_subroutine(cls, subroutine, rule_report, config):
        '''
        Fix rule on subroutine level. Must be implemented by a rule if applicable.
        '''

    @classmethod
    def fix_sourcefile(cls, sourcefile, rule_report, config):
        '''
        Fix rule on sourcefile level. Must be implemented by a rule if applicable.
        '''


class Fixer:
    """
    Operater class to fix problems reported by fixable rules.
    """

    @classmethod
    def fix_module(cls, module, reports, config):
        """
        Call `fix_module` for all rules and apply the transformations.
        """
        # TODO: implement this!
        return module

    @classmethod
    def fix_subroutine(cls, subroutine, reports, config):
        """
        Call `fix_subroutine` for all rules and apply the transformations.
        """
        mapper = {}
        for report in reports:
            rule_config = config[report.rule.__name__]
            mapper.update(report.rule.fix_subroutine(subroutine, report, rule_config) or {})

        if mapper:
            # Apply the changes and invalidate source objects
            subroutine.spec = Transformer(mapper).visit(subroutine.spec)
            subroutine.body = Transformer(mapper).visit(subroutine.body)
            subroutine._source = None
            parent = subroutine.parent
            while parent is not None:
                parent._source = None
                parent = getattr(parent, 'parent', None)

        return subroutine

    @classmethod
    def fix_sourcefile(cls, sourcefile, reports, config):
        """
        Call `fix_sourcefile` for all rules and apply the transformations.
        """
        # TODO: implement this!
        return sourcefile

    @classmethod
    def fix(cls, ast, reports, config):
        """
        Attempt to fix problems flagged by fixable rules in the given IR object.

        This routine calls `fix_module`, `fix_subroutine` and `fix_file`
        as applicable for all rules on all entities in the given IR object.

        :param ast: the IR object to be fixed.
        :type ast: :py:class:`SourceFile`, :py:class:`Module`, or
                   :py:class:`Subroutine`
        :param list reports: the fixable :py:class:`RuleReport` reports.
        :type rule_report: :py:class:`FileReport`
        :param dict config: a `dict` with the config values.

        :return: the modified AST object.
        """

        # Fix on source file level
        if isinstance(ast, SourceFile):
            # Depth-first traversal
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                ast._routines = [cls.fix(routine, reports, config) for routine in ast.subroutines]
            if hasattr(ast, 'modules') and ast.modules is not None:
                ast._modules = [cls.fix(module, reports, config) for module in ast.modules]

            ast = cls.fix_sourcefile(ast, reports, config)

        # Fix on module level
        elif isinstance(ast, Module):
            # Depth-first traversal
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                ast.routines = [cls.fix(routine, reports, config) for routine in ast.subroutines]

            ast = ast.fix_module(ast, reports, config)

        # Fix on subroutine level
        elif isinstance(ast, Subroutine):
            # Depth-first traversal
            if hasattr(ast, 'members') and ast.members is not None:
                ast._members = [cls.fix(member, reports, config) for member in ast.members]

            ast = cls.fix_subroutine(ast, reports, config)

        return ast


def get_filename_from_parent(obj):
    """
    Try to determine filename by following ``parent`` attributes
    until :py:class:``loki.sourcefile.SourceFile`` is encountered.

    :param obj: A source file, module or subroutine object.
    :return: The filename or ``None``
    :rtype: str or NoneType
    """
    scope = obj
    while hasattr(scope, 'parent') and scope.parent:
        # Go up until we are at SourceFile level
        scope = scope.parent
    if hasattr(scope, 'path'):
        return scope.path
    return None
