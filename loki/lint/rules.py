import re

from loki import FindNodes
from loki.lint import RuleType, GenericRule
import loki.ir as ir


class MaxDummyArgsRule(GenericRule):

    type = RuleType.INFO

    docs = {
        'title': 'Routines should have no more than {max_num_arguments} dummy arguments.',
    }

    config = {
        'max_num_arguments': 10
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        """
        Count the number of dummy arguments and report if given
        maximum number exceeded.
        """
        num_arguments = len(subroutine.arguments)
        if num_arguments > config['max_num_arguments']:
            fmt_string = 'Subroutine has {} dummy arguments (should not have more than {})'
            msg = fmt_string.format(num_arguments, config['max_num_arguments'])
            rule_report.add(msg, subroutine)


class ImplicitNoneRule(GenericRule):

    type = RuleType.SERIOUS

    docs = {
        'title': '"IMPLICIT NONE" is mandatory in all routines.',
    }

    _regex = re.compile(r'implicit\s+none\b', re.I)

    @staticmethod
    def check_for_implicit_none(ast):
        """
        Check for intrinsic nodes that match the regex.
        """
        for intr in FindNodes(ir.Intrinsic).visit(ast):
            if ImplicitNoneRule._regex.match(intr.text):
                break
        else:
            return False
        return True

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        """
        Check for IMPLICIT NONE in the subroutine's spec or any enclosing
        scope.
        """
        found_implicit_none = cls.check_for_implicit_none(subroutine.ir)

        # Check if enclosing scopes contain implicit none
        scope = subroutine.parent
        while scope and not found_implicit_none:
            if hasattr(scope, 'spec') and scope.spec:
                found_implicit_none = cls.check_for_implicit_none(scope.spec)
            scope = scope.parent if hasattr(scope, 'parent') else None

        if not found_implicit_none:
            # No 'IMPLICIT NONE' intrinsic node was found
            rule_report.add('No "IMPLICIT NONE" found', subroutine)


class BannedStatementsRule(GenericRule):

    type = RuleType.WARN

    docs = {
        'title': 'Banned statements.',
    }

    config = {
        'banned': ['GO TO'],
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Check for banned statements in intrinsic nodes.'''
        for intr in FindNodes(ir.Intrinsic).visit(subroutine.ir):
            for keyword in config['banned']:
                if keyword.lower() in intr.text.lower():
                    msg = 'Banned keyword "{}"'.format(keyword)
                    rule_report.add(msg, intr)


# Create the __all__ property of the module to contain only the rule names
__all__ = tuple(name for name in dir() if name.endswith('Rule') and name != 'GenericRule')
