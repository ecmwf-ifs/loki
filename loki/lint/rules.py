import re

from loki import SourceFile, Module, Subroutine, DataType
from loki.visitors import FindNodes
from loki.expression import FindLiterals, IntLiteral, FloatLiteral
import loki.ir as ir


__all__ = ['GenericRule', 'ImplicitNoneRule']


def get_filename_from_parent(obj):
    parent = obj
    while hasattr(parent, 'parent') and parent.parent:
        # Go up until we are at SourceFile level
        parent = parent.parent
    if hasattr(parent, 'path'):
        return parent.path
    return 'Unknown file'


def get_line_from_source(source):
    if not source:
        return '???'
    if source.lines[0] == source.lines[1]:
        return str(source.lines[0])
    return '{}-{}'.format(*source.lines)


class GenericRule(object):

    type = None

    docs = None

    config = {}

    fixable = False

    deprecated = False

    replaced_by = ()

    @classmethod
    def check_module(cls, module, rule_report, config):
        pass

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        pass

    @classmethod
    def check_file(cls, sourcefile, rule_report, config):
        pass

    @classmethod
    def check(cls, ast, rule_report, config):
        # Perform checks on source file level
        if isinstance(ast, SourceFile):
            cls.check_file(ast, rule_report, config)

        # Perform checks on module level
        if isinstance(ast, SourceFile):
            # If we have a source file, we call the routine for each module
            for module in ast.modules:
                cls.check_module(module, rule_report, config)
        elif isinstance(ast, Module):
            cls.check_module(ast, rule_report, config)

        # Perform checks on subroutine level
        if isinstance(ast, (SourceFile, Module)):
            # If we have a source file or module, we call the routine for
            # each module and subroutine
            if hasattr(ast, 'routines') and ast.routines is not None:
                for subroutine in ast.routines:
                    cls.check_subroutine(subroutine, rule_report, config)
            if hasattr(ast, 'modules') and ast.modules is not None:
                for module in ast.modules:
                    for subroutine in module.routines or []:
                        cls.check_subroutine(subroutine, rule_report, config)
        elif isinstance(ast, Subroutine):
            cls.check_subroutine(ast, rule_report, config)

            # Recurse for any procedures contained in a subroutine
            if hasattr(ast, 'members') and ast.members is not None:
                for member in ast.members:
                    cls.check_subroutine(member, rule_report, config)


class SubroutineLengthRule(GenericRule):  # Coding standards 2.2

    type = 'suggestion'

    docs = {
        'name': 'Subroutine length shall be limited'
    }

    config = {
        'max_num_statements': 300
    }

    # List of nodes that are considered executable statements
    exec_nodes = (
        ir.Statement, ir.MaskedStatement, ir.Intrinsic, ir.Allocation,
        ir.Deallocation, ir.Nullify, ir.CallStatement
    )

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Count the number of nodes in the subroutine and check if they exceed
        a given maximum number.
        '''
        num_nodes = len(FindNodes(SubroutineLengthRule.exec_nodes).visit(subroutine.ir))
        if num_nodes > config['max_num_statements']:
            fmt_string = 'Found {} executable statements (maximum allowed: {})'
            msg = fmt_string.format(num_nodes, config['max_num_statements'])
            rule_report.add(msg, subroutine)


class ArgumentNumberRule(GenericRule):  # Coding standards 3.6

    type = 'problem'

    docs = {
        'name': 'Number of dummy arguments should be small'
    }

    config = {
        'max_num_arguments': 50
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Count the number of dummy arguments and report if given
        maximum number exceeded.
        '''
        num_arguments = len(subroutine.arguments)
        if num_arguments > config['max_num_arguments']:
            fmt_string = 'Found {} dummy arguments (maximum allowed: {})'
            msg = fmt_string.format(num_arguments, config['max_num_arguments'])
            rule_report.add(msg, subroutine)


class ImplicitNoneRule(GenericRule):  # Coding standards 4.4

    type = 'problem'

    docs = {
        'name': 'Implicit None is mandatory'
    }

    _regex = re.compile(r'implicit\s+none\b', re.I)

    @staticmethod
    def check_for_implicit_none(ast):
        # Check for intrinsic nodes with 'implicit none'
        for intr in FindNodes(ir.Intrinsic).visit(ast):
            if ImplicitNoneRule._regex.match(intr.text):
                break
        else:
            return False
        return True

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Check for IMPLICIT NONE in the subroutine's spec or any enclosing
        scope.
        '''
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


class ExplicitKindRule(GenericRule):  # Coding standards 4.7

    type = 'problem'

    docs = {
        'name': 'Variables and constants must be declared with explicit kind'
    }

    config = {
        'allowed_type_kinds': {
            'INTEGER': ['JPIM', 'JPIT', 'JPIB', 'JPIA', 'JPIS', 'JPIH'],
            'REAL': ['JPRB', 'JPRM', 'JPRS', 'JPRT', 'JPRH', 'JPRD']
        }
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Check for explicit kind information in constants and
        variable declarations.
        '''
        int_kinds = config['allowed_type_kinds']['INTEGER']
        real_kinds = config['allowed_type_kinds']['REAL']

        for node, exprs in FindLiterals(
                unique=False, with_expression_root=True).visit(subroutine.ir):
            for lit in exprs:
                if isinstance(lit, (IntLiteral, FloatLiteral)):
                    if not lit.kind:
                        rule_report.add('"{}" without explicit KIND declared.'.format(lit), node)
                    elif (isinstance(lit, IntLiteral) and lit.kind.upper() not in int_kinds) or \
                            (isinstance(lit, FloatLiteral) and lit.kind.upper() not in real_kinds):
                        rule_report.add('"{}" is not an allowed KIND value.'.format(lit.kind), node)

        for var in subroutine.variables:
            if var.type.dtype in (DataType.INTEGER, DataType.REAL):
                if not var.type.kind:
                    rule_report.add('"{}" without explicit KIND declared.'.format(var), subroutine)
                elif (var.type.dtype == DataType.INTEGER and var.type.kind.upper() not in int_kinds) or \
                        (var.type.dtype == DataType.REAL and var.type.kind.upper() not in real_kinds):
                    rule_report.add(
                        '"{}" is not an allowed KIND value for "{}".'.format(var.type.kind, var),
                        subroutine)


class BannedStatementsRule(GenericRule):  # Coding standards 4.11

    type = 'problem'

    docs = {
        'name': 'Some statements are banned'
    }

    config = {
        'banned': ['STOP', 'PRINT', 'RETURN', 'ENTRY', 'DIMENSION',
                   'DOUBLE PRECISION', 'COMPLEX', 'GO TO', 'CONTINUE',
                   'FORMAT', 'COMMON', 'EQUIVALENCE'],
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Check for banned statements in intrinsic nodes.'''
        for intr in FindNodes(ir.Intrinsic).visit(subroutine.ir):
            for keyword in config['banned']:
                if keyword.lower() in intr.text.lower():
                    msg = 'Banned keyword "{}"'.format(keyword)
                    rule_report.add(msg, intr)
