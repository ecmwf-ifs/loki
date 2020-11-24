from enum import Enum

from loki import Sourcefile, Module, Subroutine


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
        :type ast: :py:class:`Sourcefile`, :py:class:`Module`, or
                   :py:class:`Subroutine`
        :param rule_report: the reporter object for the rule.
        :type rule_report: :py:class:`RuleReport`
        :param dict config: a `dict` with the config values.

        '''
        # Perform checks on source file level
        if isinstance(ast, Sourcefile):
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
