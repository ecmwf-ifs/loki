from pathlib import Path
from collections import defaultdict
import re
from dataclasses import dataclass

from loki import (
    Visitor, fgen, Node, CallStatement, Loop, FindNodes, ExpressionFinder, FindExpressionRoot, ExpressionRetriever,
    flatten, as_tuple, strip_inline_comments, Module, Subroutine, BasicType, ir, FindVariables, Array, Assignment,
    Scalar, IntLiteral, StatementFunction, Intrinsic, InlineCall, is_iterable
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
    node: Node
    depth: int

###################


class SC1v1Rule(GenericRule):
    """
    SC1: horizontal indexing rule.

    Version 1: really simple/basic implementation

    * checking for innermost (0th index) of arrays

    TODO:
     * are there any conditions to include/exclude arrays?
    """

    type = RuleType.WARN

    docs = {
        'title': ('SC1: horizontal indexing rule.'
                  'Variables referring to horizontal indices should be named consistently!'),
    }

    config = {
        'horizontal_var': ['jl'],
        'horizontal_shape': ['klon', 'nproma', 'kproma'],
        'horizontal_index': 0
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):

        # arrays = [var for var in FindVariables().visit(subroutine.body) if isinstance(var, sym.Array)]
        # for array in arrays:
        #     print(f"array: {array}, shape: {array.shape}, dim: {array.dimensions}")
        #     if array.shape[0] not in config['horizontal_shape'] or
        #     array.dimensions[0] not in config['horizontal_var']:
        #         msg = "[v1] horizontal indexing violation, variables referring to " \
        #               "horizontal indices should be named consistently!"
        #         rule_report.add(msg=msg, location=subroutine)

        retriever = ExpressionRetriever(lambda e: isinstance(e, sym.Array))
        finder = ExpressionFinder(unique=False, retrieve=retriever.retrieve, with_ir_node=True)
        # spec: checking for shape
        for i_node, (node, expr_list) in enumerate(finder.visit(subroutine.spec)):
            for array in expr_list:
                if array.shape and array.shape[config['horizontal_index']] not in config['horizontal_shape']:
                    msg = f"[v1] horizontal indexing violation for {array.name}, variables referring to horizontal " \
                          f"indices should be named consistently - shape: {array.shape[config['horizontal_index']]}!"
                    rule_report.add(msg=msg, location=node)
        # body: checking for dimensions
        for i_node, (node, expr_list) in enumerate(finder.visit(subroutine.body)):
            for array in expr_list:
                if array.dimensions and array.dimensions[config['horizontal_index']] not in config['horizontal_var']:
                    msg = f"[v1] horizontal indexing violation for {array.name}, variables referring to horizontal " \
                          f"indices should be named consistently - dim: " \
                          f"{array.dimensions[config['horizontal_index']]})!"
                    rule_report.add(msg=msg, location=node)


class SC2v1Rule(GenericRule):
    """
    SC2: horizontal looping rule.

    TODO:
     * are arr(:, ...) (under some conditions) acceptable?
     * necessary/reasonable to check for the innermost loop (to be the horizontal loop)?
    """

    type = RuleType.WARN

    docs = {
        'title': ('SC2: horizontal looping rule.'
                  'All loops over the innermost, horizontal array dimension should be explicit!'),
    }

    config = {
        'horizontal_var': ['jl'],
        'horizontal_shape': ['klon', 'nproma', 'kproma'],
        'horizontal_index': 0
    }

    class FindNodesDepth(FindNodes):

        def __init__(self, match, greedy=False):
            super().__init__(match, mode='type', greedy=greedy)

        def visit_Node(self, o, **kwargs):
            ret = kwargs.pop('ret', self.default_retval())
            depth = kwargs.pop('depth', 0)
            if self.rule(self.match, o):
                ret.append(DepthNode(o, depth))
                if self.greedy:
                    return ret
            for i in o.children:
                ret = self.visit(i, depth=depth + 1, ret=ret, **kwargs)
            return ret or self.default_retval()

    @classmethod
    def innermost_loops(cls, loops):
        innermost = []
        if loops:
            for i in range(1, len(loops) - 1):
                if loops[i - 1].depth <= loops[i].depth >= loops[i + 1].depth:
                    innermost.append(loops[i])
            if len(loops) > 1:
                if loops[-1].depth > loops[-2].depth:
                    innermost.append(loops[-1])
        return innermost

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):

        retriever = ExpressionRetriever(lambda e: isinstance(e, sym.Array))
        finder = ExpressionFinder(unique=False, retrieve=retriever.retrieve, with_ir_node=True)
        for i_node, (node, expr_list) in enumerate(finder.visit(subroutine.body)):
            for array in expr_list:
                if array.dimensions and isinstance(array.dimensions[config['horizontal_index']], sym.RangeIndex):
                    msg = f"[v1]: loop over the innermost, horizontal array dimension not explicit for array " \
                          f"{array.name}, referring to '{array.dimensions[config['horizontal_index']]}'"
                    rule_report.add(msg=msg, location=node)

        # further checking for innermost loop to be the horizontal loop ...
        # TODO: allow for more deeply nested loops (if loop bounds are e.g. compile time constants?)
        innermost = cls.innermost_loops(cls.FindNodesDepth(ir.Loop).visit(subroutine.body))
        for loop in innermost:
            if loop.node.variable not in config['horizontal_var']:
                msg = f"[v1] innermost loop with depth: {loop.depth} not the horizontal loop : {loop.node}!"
                rule_report.add(msg=msg, location=loop.node)


class SC3v1Rule(GenericRule):
    """
    SC3: function calls from inside `KPROMA` loops
    """

    type = RuleType.WARN

    docs = {
        'title': ('SC3: function calls from inside `KPROMA` loops.'
                  'Inside tight horizontal loops of type DO `JL=KIDIA,KFDIA`, '
                  'calls should be restricted to intrinsics!'),
    }

    config = {
        'horizontal_var': ['jl'],
        'horizontal_shape': ['klon', 'nproma', 'kproma'],
        'horizontal_index': 0
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):

        retriever = ExpressionRetriever(lambda e: isinstance(e, sym.InlineCall))  # or isinstance(e, ir.Assignment))
        finder = ExpressionFinder(unique=False, retrieve=retriever.retrieve, with_ir_node=True)

        horizontal_loops = [loop for loop in FindNodes(Loop).visit(subroutine.body) if
                            loop.variable in config["horizontal_var"]]

        for loop in horizontal_loops:
            # check for nested loops and call statements
            for node in FindNodes((Loop, CallStatement, Intrinsic)).visit(loop.body):
                if isinstance(node, Loop):
                    # if cls.allowed_loop_bounds(node.bounds.lower) and cls.allowed_loop_bounds(node.bounds.upper):
                    #     continue
                    msg = f"[v1] nested loop within loop {loop}"
                    rule_report.add(msg=msg, location=node)
                # TODO: are there any to report?
                elif isinstance(node, Intrinsic):
                    msg = f"[v1] nested intrinsic {node}"
                    rule_report.add(msg=msg, location=node)
                else:
                    msg = f"nested call to {node.name} within {loop}"
                    rule_report.add(msg=msg, location=node)
            # check for inline calls/intrinsics
            # TODO: are there any to report?
            for node, expr_list in finder.visit(loop.body):
                for expr in expr_list:
                    msg = f"[v1] nested intrinsic: {expr}"
                    rule_report.add(msg=msg, location=node)


class SC4v1Rule(GenericRule):
    """
    SC4: no horizontal indirection

    TODO: what to cover regarding indirect array accesses?
     - "direct" indirect access: arr1(arr2(...),...)
     - "indirect" indirect access: j = arr2(...), arr1(j, ...)
     - "indirect, indirect" indirect access: k = arr2(...), j=k, arr1(j, ...)
     - ...
    """

    type = RuleType.WARN

    docs = {
        'title': ('SC4: no horizontal indirection.'
                  'Where relevant (Single Column), indirect addressing on the innermost, '
                  'horizontal array index shall not be used!'),
    }

    config = {
        'horizontal_var': ['jl'],
        'horizontal_shape': ['klon', 'nproma', 'kproma'],
        'horizontal_index': 0
    }

    # @classmethod
    # def indirect_addressing(cls, expr):
    #     return isinstance(expr, sym.Array)

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        retriever = ExpressionRetriever(lambda e: isinstance(e, sym.Array))
        finder = ExpressionFinder(unique=False, retrieve=retriever.retrieve, with_ir_node=True)
        findings = finder.visit(subroutine.ir)
        for i_node, (node, expr_list) in enumerate(findings):
            for expr in expr_list:
                array_in_dims = [isinstance(_, sym.Array) for _ in expr.dimensions]
                if array_in_dims and array_in_dims[config['horizontal_index']]:  # or any(array_in_dims):
                    msg = f"[v1] at least one indirect array access for {expr} - " \
                          f"{[(i, expr.dimensions[i].name) for i, val in enumerate(array_in_dims) if val]} "
                    rule_report.add(msg=msg, location=node)


class SC4v2Rule(GenericRule):
    """
    SC4: no horizontal indirection

    TODO: what to cover regarding indirect array accesses?
     - "direct" indirect access: arr1(arr2(...),...)
     - "indirect" indirect access: j = arr2(...), arr1(j, ...)
     - "indirect, indirect" indirect access: k = arr2(...), j=k, arr1(j, ...)
     - ...
    """

    type = RuleType.WARN

    docs = {
        'title': ('SC4: no horizontal indirection.'
                  'Where relevant (Single Column), indirect addressing on the innermost, '
                  'horizontal array index shall not be used!'),
    }

    config = {
        'horizontal_var': ['jl'],
        'horizontal_shape': ['klon', 'nproma', 'kproma'],
        'horizontal_index': 0
    }

    # @classmethod
    # def indirect_addressing(cls, expr):
    #     return isinstance(expr, sym.Array)

    class FindValues(FindNodes):

        def __init__(self, match, lhs=None, greedy=False):
            super().__init__(match, mode='type', greedy=greedy)
            self.lhs = lhs

        def visit_Node(self, o, **kwargs):
            ret = kwargs.pop('ret', self.default_retval())
            depth = kwargs.pop('depth', 0)
            if self.rule(self.match, o):
                if o.lhs == self.lhs:
                    ret.append(DepthNode(o, depth))
                if self.greedy:
                    return ret
            for i in o.children:
                ret = self.visit(i, depth=depth + 1, ret=ret, **kwargs)
            return ret or self.default_retval()

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        retriever = ExpressionRetriever(lambda e: isinstance(e, sym.Array))
        finder = ExpressionFinder(unique=False, retrieve=retriever.retrieve, with_ir_node=True)
        findings = finder.visit(subroutine.body)
        potential_indirect = []
        for i_node, (node, expr_list) in enumerate(findings):
            for expr in expr_list:
                if expr.dimensions:
                    for var in FindVariables().visit(expr.dimensions[config['horizontal_index']]):
                        potential_indirect.append((node, expr, var))
        # TODO: take assignment order into consideration!?
        #   (a variable is possibly assigned to an array only after a specific node ...)
        for potential in potential_indirect:
            _rhs = []
            for assigned_value in cls.FindValues(ir.Assignment, potential[2]).visit(subroutine.body):
                _rhs.extend(FindVariables().visit(assigned_value.node.rhs))
            if any(isinstance(_, sym.Array) for _ in _rhs) or \
                    any(isinstance(_, sym.Array) for _ in FindVariables().visit(potential[2])):
                msg = f"[v2] at least one possible indirect array access for {potential[1]}"
                rule_report.add(msg=msg, location=potential[0])


class SC5v1potentialRule(GenericRule):
    """
    SC5 (potential rule): no horizontal reduction across vector loop
    """

    type = RuleType.WARN

    docs = {
        'title': ('SC5 (potential rule): no horizontal reduction across vector loop.'
                  'No reduction operations (reducing the elements of an array into a single value) across '
                  'the vector loop/horizontal index `JL`!'),
    }

    config = {
        'horizontal_var': ['jl'],
        'horizontal_shape': ['klon', 'nproma', 'kproma'],
        'horizontal_index': 0
    }

    reduction_functions = ['all', 'any', 'count', 'maxval', 'minval', 'product', 'sum']

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        loops = [loop for loop in FindNodes(ir.Loop).visit(subroutine.body) if
                 loop.variable == config["horizontal_var"][0]]

        for loop in loops:
            retriever = ExpressionRetriever(lambda e: isinstance(e, sym.Array))
            finder = ExpressionFinder(unique=False, retrieve=retriever.retrieve, with_ir_node=True)
            findings = finder.visit(loop.body)
            for finding, expr_list in findings:
                if isinstance(finding, ir.Assignment) and \
                        not isinstance(finding.rhs, sym.InlineCall) and finding.lhs in finding.rhs and \
                        isinstance(finding.lhs, sym.Scalar):
                    rule_report.add(f"horizontal reduction at {finding}", finding)

        retriever = ExpressionRetriever(lambda e: isinstance(e, sym.InlineCall))
        finder = ExpressionFinder(unique=False, retrieve=retriever.retrieve, with_ir_node=True)
        for node, expr_list in finder.visit(subroutine.body):
            for expr in expr_list:
                if expr.name.lower() in cls.reduction_functions:
                    # TODO: check whether horizontal dimension is involved
                    rule_report.add(f"potential horizontal reduction at {expr}", node)


class SC6v1potentialRule(GenericRule):
    """
    SC6 (potential rule): no horizontal index array accessing with an offset
    """

    type = RuleType.WARN

    docs = {
        'title': ('SC6 (potential rule): no horizontal index array accessing with an offset.'
                  'Arrays accessed in the horizontal dimension via the horizontal index `JL` shall only be accessed '
                  'via the horizontal index, thus without any offset like e.g. `JL + n` '
                  '(with `n`  as an arbitrary integer).!'),
    }

    config = {
        'horizontal_var': ['jl'],
        'horizontal_shape': ['klon', 'nproma', 'kproma'],
        'horizontal_index': 0
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        retriever = ExpressionRetriever(lambda e: isinstance(e, sym.Array))
        finder = ExpressionFinder(unique=False, retrieve=retriever.retrieve, with_ir_node=True)
        arrays = finder.visit(subroutine.body)
        for node, expr_list in arrays:
            for array in expr_list:
                for dim in array.dimensions:
                    if config["horizontal_var"][0] in FindVariables().visit(dim):
                        if isinstance(dim, sym.Sum):
                            rule_report.add(f"horizontal array access with offset for {array} - {dim}", node)


# Create the __all__ property of the module to contain only the rule names
__all__ = tuple(name for name in dir() if name.endswith('Rule') and name != 'GenericRule')
