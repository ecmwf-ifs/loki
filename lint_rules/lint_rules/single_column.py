from pathlib import Path
from collections import defaultdict
import re
from dataclasses import dataclass

from loki import (
    Visitor, fgen, Node, CallStatement, Loop, FindNodes, ExpressionFinder, FindExpressionRoot, ExpressionRetriever,
    flatten, as_tuple, strip_inline_comments, Module, Subroutine, BasicType, ir, FindVariables, Array, Assignment,
    Scalar, IntLiteral, StatementFunction, Intrinsic, InlineCall
)
from loki.lint import GenericRule, RuleType
from loki.expression import symbols as sym

intrinsics = ['DIM', 'SQRT', 'ADJUSTR', 'DATAN2', 'IEEE_SUPPORT_FLAG', 'MAXVAL', 'MAXVAL', 'IEEE_SUPPORT_HALTING',
              'DDIM', 'DMAX1', 'TAN', 'IEEE_SUPPORT_INF', 'CLOG', 'ASIN', 'AMAX1', 'IEEE_LOGB', 'ALLOCATED', 'MIN',
              'IEEE_SUPPORT_DATATYPE', 'IEEE_RINT', 'RRSPACING', 'MAXLOC', 'DINT', 'AIMAG', 'LEN_TRIM', 'UNPACK',
              'CPU_TIME', 'CEXP', 'RANDOM_SEED', 'SIZE', 'MINLOC', 'IEEE_NEXT_AFTER', 'LLE', 'HUGE', 'MATMUL', 'CHAR',
              'ISIGN', 'DATE_AND_TIME', 'IEEE_SELECTED_REAL_KIND', 'SELECTED_REAL_KIND', 'IEEE_SUPPORT_IO', 'NULL',
              'COS', 'ISHFT', 'CSIN', 'BIT_SIZE', 'IEEE_GET_HALTING_MODE', 'DIGITS', 'CEILING', 'ALOG10', 'MINEXPONENT',
              'EXP', 'SUM', 'LOG10', 'IEEE_CLASS', 'DABS', 'SUM', 'RESHAPE', 'IEEE_IS_NEGATIVE', 'MINVAL', 'MAXLOC',
              'REAL', 'SIGN', 'IEEE_SUPPORT_DENORMAL', 'IEEE_SET_ROUNDING_MODE', 'AMIN1', 'MOD', 'SPREAD', 'DEXP',
              'CMPLX', 'SCALE', 'COUNT', 'SHAPE', 'TINY', 'SELECTED_INT_KIND', 'MODULO', 'NEAREST', 'AMOD', 'DNINT',
              'CCOS', 'MIN1', 'DMIN1', 'IBITS', 'COSH', 'DSIGN', 'MAXEXPONENT', 'MAX0', 'IEEE_SET_HALTING_MODE',
              'CSHIFT', 'DASIN', 'ALOG', 'ACHAR', 'IEEE_SET_STATUS', 'SYSTEM_CLOCK', 'MINVAL', 'SIN', 'IEOR', 'DMOD',
              'MALLOC', 'DCOSH', 'IEEE_IS_NORMAL', 'IEEE_SCALB', 'TRIM', 'MPI_SIZEOF', 'IEEE_SUPPORT_STANDARD',
              'IEEE_IS_NAN', 'PACK', 'SNGL', 'DATAN', 'LLT', 'IFIX', 'SCAN', 'KIND', 'RANGE', 'IEEE_IS_FINITE', 'NINT',
              'TRANSFER', 'ABS', 'ACOS', 'ICHAR', 'MIN0', 'AINT', 'RANDOM_NUMBER', 'REPEAT', 'LOG', 'ADJUSTL', 'UBOUND',
              'IEEE_COPY_SIGN', 'IEEE_SUPPORT_SQRT', 'GET_COMMAND', 'TRANSPOSE', 'ANINT', 'DSIN', 'LBOUND', 'EXPONENT',
              'SET_EXPONENT', 'ALL', 'ASSOCIATED', 'IEEE_ARITHMETIC', 'IEEE_GET_FLAG', 'SINH', 'IEEE_GET_STATUS',
              'ISHFTC', 'LEN', 'DPROD', 'NOT', 'DBLE', 'DSQRT', 'MINLOC', 'IOR', 'IEEE_UNORDERED', 'IDIM', 'INDEX',
              'DTANH', 'CMPLX', 'IDINT', 'IAND', 'C_F_POINTER', 'AMIN0', 'INT', 'FRACTION', 'DLOG10', 'ANY',
              'IEEE_SUPPORT_ROUNDING', 'C_ASSOCIATED', 'EOSHIFT', 'DLOG', 'AMAX0', 'DACOS', 'PRECISION', 'SPACING',
              'IDNINT', 'C_LOC', 'CABS', 'COMMAND_ARGUMENT_COUNT', 'IEEE_SUPPORT_NAN', 'EPSILON', 'ATAN2',
              'PRODUCT', 'IBCLR', 'DCOS', 'ATAN', 'IEEE_SET_FLAG', 'DSINH', 'DTAN', 'IEEE_VALUE', 'IBSET', 'MAX1',
              'MERGE', 'BTEST', 'DOT_PRODUCT', 'IACHAR', 'IEEE_SUPPORT_DIVIDE', 'CONJG', 'VERIFY', 'FLOOR', 'MAX',
              'PRODUCT', 'FLOAT', 'LGT', 'LOGICAL', 'MVBITS', 'IABS', 'RADIX', 'CSQRT', 'IEEE_GET_ROUNDING_MODE',
              'IEEE_REM', 'LGE', 'TANH']


def is_intrinsic(routine_name):
    if routine_name in intrinsics:
        return True
    else:
        return False


def is_pure_elemental(routine):
    if "pure" in routine.prefix and "elemental" in routine.prefix:
        return True
    else:
        return False


@dataclass
class DepthNode:
    """Store node object and depth in c-style struct."""

    node: Node
    depth: int


class VectorLoopRule(GenericRule):
    type = RuleType.WARN

    docs = {
        'title': ('...'
                  '...'),
    }

    config = {
        'induction_variable': 'k'
    }

    class FindNodesDepth(FindNodes):
        """Visitor that computes node-depth relative to subroutine body. Returns list of DepthNode objects."""

        def __init__(self, match, greedy=False):
            super().__init__(match, mode='type', greedy=greedy)

        def visit_Node(self, o, **kwargs):
            """
            Add the node to the returned list if it matches the criteria and increment depth
            before visiting all children.
            """

            ret = kwargs.pop('ret', self.default_retval())
            depth = kwargs.pop('depth', 0)
            if self.rule(self.match, o):
                ret.append(DepthNode(o, depth))
                if self.greedy:
                    return ret
            for i in o.children:
                ret = self.visit(i, depth=depth + 1, ret=ret, **kwargs)
            return ret or self.default_retval()

    class NestingVisitor(Visitor):

        @classmethod
        def default_retval(cls):
            return []

        def visit(self, o, *args, **kwargs):
            # print("visit: {}".format(o))
            return flatten(super().visit(o, *args, **kwargs))

        def visit_Loop(self, o, **kwargs):
            # print("visit loop: {}".format(o))
            nested_loop = [o]
            nested_loop += self.visit(o.body, **kwargs)
            return nested_loop

        def visit_CallStatement(self, o, **kwargs):
            nested_subroutine = [o]
            # nested_subroutine += self.visit(o.body, **kwargs)
            return nested_subroutine

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):

        # print("\nsubroutine: {}".format(subroutine))
        # print("subroutine.ir: {}".format(subroutine.ir))

        loops = cls.FindNodesDepth(Loop).visit(subroutine.body)
        # print("loops: {}".format(loops))

        # counting loop variable occurrences
        #######################################################
        # Either loop variable name
        # loop_variables = [loop.node.variable.name for loop in loops]
        # or loop variable name and depth (as tuple)
        # loop_variables = [(loop.node.variable.name, loop.depth) for loop in loops]
        #
        # unique_loop_variables = list(set(loop_variables))
        # loop_variables_occurrences = {}
        # for i, elem in enumerate(unique_loop_variables):
        #     loop_variables_occurrences[elem] = loop_variables.count(elem)
        # for loop_variable in loop_variables_occurrences:
        #     print("variable: {} - occurrences: {}".format(loop_variable, loop_variables_occurrences[loop_variable]))
        #######################################################

        # Working version
        #######################################################
        # vector_loops = [loop for loop in loops if loop.node.variable == config["induction_variable"]]
        # for k, loop in enumerate(vector_loops):  # loops):
        #     # print("{} - variable: {} | bounds: {} | depth : {}".format(k, loop.node.variable, loop.node.bounds,
        #     #                                                            loop.depth))
        #
        #     # print(fgen(loop.node.body))
        #     nested = cls.NestingVisitor().visit(loop.node.body)
        #     # nested loops and subroutines
        #     # for _nested in nested:
        #     #     print("nested: {}".format(_nested))
        #     # get nested loops (within vector loops)
        #     nested_loops = [_nested for _nested in nested if isinstance(_nested, Loop)]
        #     for nested_loop in nested_loops:
        #         # print("nested loop within vector loop: {}".format(nested_loop))
        #         rule_report.add("nested loop within vector loop: {}".format(nested_loop), subroutine)
        #     # get nested subroutines (within vector loops)
        #     nested_subroutines = [_nested for _nested in nested if isinstance(_nested, CallStatement)]
        #     for nested_subroutine in nested_subroutines:
        #         # print("nested loop within vector loop: {}".format(nested_loop))
        #         rule_report.add("nested loop within vector loop: {}".format(nested_loop), subroutine)
        #######################################################

        # Another (shorter) working version
        #######################################################
        # vector_loops = [loop for loop in loops if loop.node.variable == config["induction_variable"]]
        # vector_loops = [loop for loop in FindNodes(Loop).visit(subroutine.body) if
        #                 loop.variable == config["induction_variable"]]
        # for loop in vector_loops:
        #     # just check the rule (no matter how many nested loops/subroutines) ...
        #     if FindNodes((Loop, CallStatement)).visit(loop.body):
        #         rule_report.add("nested loop/subroutine within {}".format(loop), subroutine)
        #     # or check the rule more precisely
        #     # for node in FindNodes((Loop, CallStatement)).visit(loop.body):
        #     #     rule_report.add("nested loop/subroutine within: {}".format(node), subroutine)
        #######################################################

        # Another working and EXTENDED version, allowing for nested loops with fixed bounds
        #######################################################
        vector_loops = [loop for loop in FindNodes(Loop).visit(subroutine.body) if
                        loop.variable == config["induction_variable"]]

        for loop in vector_loops:
            for node in FindNodes((Loop, CallStatement)).visit(loop.body):
                if isinstance(node, Loop):

                    if cls.allowed_loop_bounds(node.bounds.lower) and cls.allowed_loop_bounds(node.bounds.upper):
                        continue

                    rule_report.add("nested loop within: {}".format(node), subroutine)
                else:
                    rule_report.add("nested subroutine within: {}".format(node), subroutine)
        #######################################################

    @classmethod
    def allowed_loop_bounds(cls, var):
        # TODO: allow for Sum/Product ... if elements are Parameter Scalars or IntLiterals ?!
        if isinstance(var, Scalar):
            if var.type.parameter:
                return True
        elif isinstance(var, IntLiteral):
            return True

        return False


class NoIndirectArrayIndexingRule(GenericRule):
    type = RuleType.WARN

    docs = {
        'title': ('...'
                  '...'),
    }

    # config = {
    #     'induction_variable': 'k'
    # }

    # relevant loki.ir classes
    # Assignment, ConditionalAssignment (?), DataDeclaration(?), MaskedStatement (?), ProcedureDeclaration (?)

    # idea 1: Visitor class

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        print("...")
        # print("subroutine: {}".format(subroutine))
        # print('vars::', ', '.join([str(v) for v in subroutine.variables]))

        # sufficient for array1(array2(i,k),i) but NOT for array1(j,i) with j = array2(i,k)
        #########################################################################
        # for var in FindVariables().visit(subroutine.body):
        #     if isinstance(var, Array):  # and isinstance(var.dimensions.variable, Array):
        #         # print("array variable: {} | dimensions: {}".format(var, var.dimensions))
        #         for dimension in var.dimensions:
        #             if isinstance(dimension, Array):
        #                 print("var: {} with at least one indirect array access: {}!".format(var, dimension))
        #                 break
        #########################################################################

        # sufficient for array1(array2(i,k),i) AND for array1(j,i) with j = array2(i,k) as long as
        # j is not a loop variable or a conditional assignment
        # TODO: necessary to include loop variable assignment and conditional assignment?
        #  - via ConditionalAssignment (e.g. FindNodes(ConditionalAssignment).visit ... ?
        #  - via Loop.bounds ?
        #########################################################################
        assignments = {}
        for assignment in FindNodes(Assignment).visit(subroutine.body):
            lhs = assignment.lhs
            if isinstance(lhs, Array):
                lhs = lhs.symbol
            # print("assignment: {}".format(lhs))
            if lhs not in assignments:
                assignments[lhs] = []
            assignments[lhs].append(assignment.rhs)

        # for assignment in assignments:
        # print("assignments for {}: {}".format(assignment, assignments[assignment]))
        # print("assignment: {}".format(assignment))

        for var in FindVariables().visit(subroutine.body):
            indirect_access = False
            if isinstance(var, Array):  # and isinstance(var.dimensions.variable, Array):
                # print("array variable: {} | dimensions: {}".format(var, var.dimensions))
                for dimension in var.dimensions:
                    if isinstance(dimension, Array):
                        indirect_access = True
                        break
                    if dimension in assignments and any(
                            [isinstance(assignment, Array) for assignment in assignments[dimension]]):
                        indirect_access = True
                        break
                if indirect_access:
                    # print("var: {} with at least one indirect array access: {}!".format(var, dimension))
                    rule_report.add(f"var: {var} with at least one indirect array access: {dimension}!", subroutine)
        #########################################################################

    # def check_module(cls, module, rule_report, config):
    #    ...

    # def check_file(cls, sourcefile, rule_report, config):
    #    ...


class HorizontalIndexingRule(GenericRule):
    type = RuleType.WARN

    docs = {
        'title': ('...'
                  '...'),
    }

    # config = {
    #     'induction_variable': 'k'
    # }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        for var in FindVariables().visit(subroutine.body):
            if isinstance(var, Array) and "KLON" in FindVariables().visit(var.dimensions):
                if var.dimensions[0] != 'jl':
                    rule_report.add(f"var: {var} with wrong horizontal index {var.dimensions[0]}!", subroutine)


class ImplicitLoopsRule(GenericRule):
    type = RuleType.WARN

    docs = {
        'title': ('...'
                  '...'),
    }

    # Implicit/Implied loops
    # * arr1 = arr2
    # * arr1(:) = 1
    # * ... ????

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        # loops = FindNodes(Loop).visit(subroutine.body)
        # print(f"loops: {loops}")
        arrays = [_ for _ in FindVariables().visit(subroutine.body) if isinstance(_, Array)]
        for array in arrays:
            # print(f"{array.shape} vs {array.dimensions}")
            # print(f"{array.dimensions}")
            if array.shape and (len(array.dimensions) < len(array.shape) or
                                sym.RangeIndex((None, None, None)) in array.dimensions):
                rule_report.add(f"implicit loop for array: {array}!", subroutine)


class IntrinsicFunctionsRule(GenericRule):
    type = RuleType.WARN

    docs = {
        'title': ('...'
                  '...'),
    }

    # Implicit/Implied loops
    # * arr1 = arr2
    # * arr1(:) = 1
    # * ... ????

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        # calls = [_ for _ in FindNodes(CallStatement).visit(subroutine.body)] #  if is_intrinsic(_.routine.name.upper())]
        # statement_function = [_ for _ in FindNodes(StatementFunction).visit(subroutine.body)]
        # intrinsics = [_ for _ in FindNodes(Intrinsic).visit(subroutine.body)]
        # inline_calls = [_ for _ in FindNodes(InlineCall).visit(subroutine.body)]
        # nodes = [FindVariables().visit(_) for _ in FindNodes(Node).visit(subroutine.body)]
        # variables = [_ for _ in FindVariables().visit(subroutine.body) if is_intrinsic(_)]
        intrinsic_functions = [_ for _ in FindVariables().visit(subroutine.body) if is_intrinsic(_)]

        # print(f"calls: {calls}")
        # print(f"statement functions: {statement_function}")
        # print(f"intrinsics: {intrinsics}")
        # print(f"inline calls: {inline_calls}")
        # print(f"nodes: {nodes}")
        # print(f"variables: {variables}")
        # for call in calls:
        #     print("call: {call.routine.name}")
        #     rule_report(f"intrinsic function: {call}!", call)
        for intrinsic_function in intrinsic_functions:
            rule_report.add(f"intrinsic function: {intrinsic_function}", subroutine)


# Create the __all__ property of the module to contain only the rule names
__all__ = tuple(name for name in dir() if name.endswith('Rule') and name != 'GenericRule')
