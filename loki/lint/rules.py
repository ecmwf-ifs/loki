import re

from loki import SourceFile, Module, Subroutine
from loki.visitors import FindNodes
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
    def check(cls, ast, reporter, config):
        # Perform checks on module level
        if hasattr(cls, 'check_module'):
            if isinstance(ast, SourceFile):
                # If we have a source file, we call the routine for each module
                for module in ast.modules:
                    cls.check_module(module, reporter, config)
            elif isinstance(ast, Module):
                cls.check_module(ast, reporter, config)

        # Perform checks on subroutine level
        if hasattr(cls, 'check_subroutine'):
            if isinstance(ast, (SourceFile, Module)):
                # If we have a source file or module, we call the routine for
                # each module and subroutine
                if hasattr(ast, 'routines') and ast.routines is not None:
                    for subroutine in ast.routines:
                        cls.check_subroutine(subroutine, reporter, config)
                if hasattr(ast, 'modules') and ast.modules is not None:
                    for module in ast.modules:
                        for subroutine in module.routines:
                            cls.check_subroutine(subroutine, reporter, config)
            elif isinstance(ast, Subroutine):
                cls.check_subroutine(ast, reporter, config)

                # Recurse for any procedures contained in a subroutine
                if hasattr(ast, 'members') and ast.members is not None:
                    for member in ast.members:
                        cls.check_subroutine(member, reporter, config)

        if hasattr(cls, 'check_file'):
            if isinstance(ast, SourceFile):
                cls.check_file(ast, reporter, config)


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

    @staticmethod
    def check_subroutine(ast, reporter, config):
        # count number of executable statements
        num_nodes = len(FindNodes(SubroutineLengthRule.exec_nodes).visit(ast.ir))
        if num_nodes > config['max_num_statements']:
            print(('{}: Number of executable statements ({}) in routine "{}" '
                   'exceeds maximum number allowed ({})').format(
                       get_filename_from_parent(ast), num_nodes, ast.name,
                       config['max_num_statements']))


class ArgumentNumberRule(GenericRule):  # Coding standards 3.6

    type = 'problem'

    docs = {
        'name': 'Number of dummy arguments should be small'
    }

    config = {
        'max_num_arguments': 50
    }

    @staticmethod
    def check_subroutine(ast, reporter, config):
        # check number of arguments
        num_arguments = len(ast.arguments)
        if num_arguments > config['max_num_arguments']:
            print(('{}: Number of dummy arguments ({}) in routine {} '
                   'exceeds maximum number allowed ({})').format(
                       get_filename_from_parent(ast), num_arguments,
                       ast.name, config['max_num_arguments']))


class ImplicitNoneRule(GenericRule):  # Coding standards 4.4

    type = 'problem'

    docs = {
        'name': 'Implicit None is mandatory'
    }

    _regex = re.compile(r'implicit\s+none\b', re.I)

    @staticmethod
    def check_subroutine(ast, reporter, config):
        # Check for intrinsic nodes with 'implicit none'
        for intr in FindNodes(ir.Intrinsic).visit(ast.ir):
            if ImplicitNoneRule._regex.match(intr.text):
                break
        else:
            # No 'IMPLICIT NONE' intrinsic node was found
            print('{}: No "IMPLICIT NONE" in routine {}'.format(
                get_filename_from_parent(ast), ast.name))


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

    @staticmethod
    def check_subroutine(ast, reporter, config):
        # Check for intrinsic nodes containing the banned statements
        for intr in FindNodes(ir.Intrinsic).visit(ast.ir):
            for keyword in config['banned']:
                if keyword.lower() in intr.text.lower():
                    print('{}: Line {} - Banned keyword "{}"'.format(
                        get_filename_from_parent(ast),
                        get_line_from_source(intr._source), keyword))
