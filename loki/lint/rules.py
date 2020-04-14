import re
from pymbolic.primitives import is_zero

from loki import DataType
from loki.visitors import FindNodes
from loki.expression import (IntLiteral, FloatLiteral, StringLiteral, LogicLiteral,
                             Array, RangeIndex, ExpressionFinder, ExpressionRetriever)
from loki.lint.utils import GenericRule, RuleType
import loki.ir as ir


class CodeBodyRule(GenericRule):  # Coding standards 1.3

    type = RuleType.WARN

    docs = {
        'id': '1.3',
        'title': ('Rules for Code Body: '
                  'Nesting of conditional blocks should not be more than {max_nesting_depth} '
                  'levels deep;',),
    }

    config = {
        'max_nesting_depth': 3,
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Check the code body: Nesting of conditional blocks.'''
        # Determine all conditionals inside each body of a conditional
        # while max_nesting_depth is not yet reached
        visitor = FindNodes((ir.Conditional, ir.MultiConditional), greedy=True)
        bodies = list(subroutine.ir)
        for _ in range(config['max_nesting_depth']):
            level_bodies, bodies = bodies, []
            for body in level_bodies:
                if body:
                    for cond in visitor.visit(body):
                        bodies += list(cond.bodies)
                        if hasattr(cond, 'else_body'):
                            bodies += [cond.else_body]

        # If there are still conditionals inside the list of bodies, they are
        # too deeply nested
        fmt_string = 'Nesting of conditionals exceeds limit of {}'
        msg = fmt_string.format(config['max_nesting_depth'])
        visitor = FindNodes((ir.Conditional, ir.MultiConditional), greedy=False)
        for body in bodies:
            for cond in visitor.visit(body):
                rule_report.add(msg, cond)


class LimitSubroutineStatementsRule(GenericRule):  # Coding standards 2.2

    type = RuleType.WARN

    docs = {
        'id': '2.2',
        'title': 'Subroutines should have no more than {max_num_statements} executable statements.',
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
        num_nodes = len(FindNodes(cls.exec_nodes).visit(subroutine.ir))
        if num_nodes > config['max_num_statements']:
            fmt_string = 'Found {} executable statements (maximum allowed: {})'
            msg = fmt_string.format(num_nodes, config['max_num_statements'])
            rule_report.add(msg, subroutine)


class MaxDummyArgsRule(GenericRule):  # Coding standards 3.6

    type = RuleType.INFO

    docs = {
        'id': '3.6',
        'title': 'Routines should have no more than {max_num_arguments} dummy arguments.',
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


class MplCdstringRule(GenericRule):  # Coding standards 3.12

    type = RuleType.SERIOUS

    docs = {
        'id': '3.12',
        'title': 'Calls to MPL subroutines should provide a "CDSTRING" identifying the caller.',
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Check all calls to MPL subroutines for a CDSTRING.'''
        for call in FindNodes(ir.CallStatement).visit(subroutine.ir):
            if call.name.upper().startswith('MPL_'):
                for kw, _ in call.kwarguments:
                    if kw.upper() == 'CDSTRING':
                        break
                else:
                    fmt_string = 'No "CDSTRING" provided in call to {}'
                    msg = fmt_string.format(call.name)
                    rule_report.add(msg, call)


class ImplicitNoneRule(GenericRule):  # Coding standards 4.4

    type = RuleType.SERIOUS

    docs = {
        'id': '4.4',
        'title': '"IMPLICIT NONE" is mandatory in all routines.',
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

    type = RuleType.SERIOUS

    docs = {
        'id': '4.7',
        'title': ('Variables and constants must be declared with explicit kind, using the kinds '
                  'defined in "PARKIND1" and "PARKIND2".'),
    }

    config = {
        'declaration_types': ['INTEGER', 'REAL'],
        'constant_types': ['INTEGER', 'REAL'],
        'allowed_type_kinds': {
            'INTEGER': ['JPIM', 'JPIT', 'JPIB', 'JPIA', 'JPIS', 'JPIH'],
            'REAL': ['JPRB', 'JPRM', 'JPRS', 'JPRT', 'JPRH', 'JPRD']
        }
    }

    @staticmethod
    def check_kind_declarations(subroutine, types, allowed_type_kinds, rule_report):
        '''Helper function that carries out the check for explicit kind specification
        on all declarations.
        '''
        # TODO: Include actual declarations in reporting (instead of just the routine)
        for var in subroutine.variables:
            if var.type.dtype in types:
                if not var.type.kind:
                    rule_report.add('"{}" without explicit KIND declared.'.format(var), subroutine)
                elif allowed_type_kinds.get(var.type.dtype) and \
                        var.type.kind.upper() not in allowed_type_kinds[var.type.dtype]:
                    rule_report.add(
                        '"{}" is not an allowed KIND value for "{}".'.format(var.type.kind, var),
                        subroutine)

    @staticmethod
    def check_kind_literals(subroutine, types, allowed_type_kinds, rule_report):
        '''Helper function that carries out the check for explicit kind specification
        on all literals.
        '''
        def retrieve(expr):
            # Custom retriever that yields the literal types specified in config and stops
            # recursion on arrays and array subscripts (to avoid warnings about integer
            # constants in array subscripts)
            retriever = ExpressionRetriever(
                lambda e: isinstance(e, types),
                recurse_query=lambda e: not isinstance(e, (Array, RangeIndex)))
            retriever(expr)
            return retriever.exprs

        finder = ExpressionFinder(unique=False, retrieve=retrieve, with_expression_root=True)

        for node, exprs in finder.visit(subroutine.ir):
            for literal in exprs:
                if is_zero(literal) or str(literal) == '0':
                    continue
                if not literal.kind:
                    rule_report.add('"{}" without explicit KIND declared.'.format(literal), node)
                elif allowed_type_kinds.get(literal.__class__) and \
                        literal.kind.upper() not in allowed_type_kinds[literal.__class__]:
                    rule_report.add('"{}" is not an allowed KIND value.'.format(literal.kind), node)

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        '''Check for explicit kind information in constants and
        variable declarations.
        '''
        # 1. Check variable declarations for explicit KIND
        # When we check variable type information, we have instances of DataType to identify
        # whether a variable is REAL, INTEGER, ... Therefore, we create a map that uses
        # the corresponding DataType values as keys to look up allowed kinds for each type.
        # Since the case does not matter, we convert all allowed type kinds to upper case.
        types = tuple(DataType.from_str(name) for name in config['declaration_types'])
        allowed_type_kinds = {}
        if config.get('allowed_type_kinds'):
            allowed_type_kinds = {DataType.from_str(name): [kind.upper() for kind in kinds]
                                  for name, kinds in config['allowed_type_kinds'].items()}

        cls.check_kind_declarations(subroutine, types, allowed_type_kinds, rule_report)

        # 2. Check constants for explicit KIND
        # Mapping from data type names to internal classes
        type_map = {'INTEGER': IntLiteral, 'REAL': FloatLiteral,
                    'LOGICAL': LogicLiteral, 'CHARACTER': StringLiteral}

        # Constants are represented by an instance of some Literal class, which directly
        # gives us their type. Therefore, we create a map that uses the corresponding
        # Literal types as keys to look up allowed kinds for each type. Again, we
        # convert all allowed type kinds to upper case.
        types = tuple(type_map[name] for name in config['constant_types'])
        if config.get('allowed_type_kinds'):
            allowed_type_kinds = {type_map[name]: [kind.upper() for kind in kinds]
                                  for name, kinds in config['allowed_type_kinds'].items()}

        cls.check_kind_literals(subroutine, types, allowed_type_kinds, rule_report)


class BannedStatementsRule(GenericRule):  # Coding standards 4.11

    type = RuleType.WARN

    docs = {
        'id': '4.11',
        'title': 'Banned statements.',
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
