"""
Collection of utility routines that provide loop transformations.

"""
from collections import defaultdict
import operator as op
import numpy as np
from pymbolic.primitives import Variable

from loki.expression import (
    symbols as sym, SubstituteExpressions, FindVariables,
    accumulate_polynomial_terms, simplify, is_constant, symbolic_op)
from loki.ir import Loop, Conditional, Comment, Pragma
from loki.logging import info
from loki.tools import is_loki_pragma, get_pragma_parameters, flatten, as_tuple
from loki.visitors import FindNodes, Transformer

__all__ = ['loop_fusion', 'loop_fission', 'Polyhedron']


class Polyhedron:
    """
    Halfspace representation of a (convex) polyhedron.

    A polyhedron `P c R^d` is described by a set of inequalities, in matrix form
    ```
    P = { x=[x1,...,xd]^T c R^d | Ax <= b }
    ```
    with n-by-d matrix `A` and d-dimensional right hand side `b`.

    In loop transformations, polyhedrons are used to represent iteration spaces of
    d-deep loop nests.

    :param np.array A: the representation matrix A.
    :param np.array b: the right hand-side vector b.
    :param list variables: list of variables representing the dimensions in the polyhedron.
    """

    def __init__(self, A, b, variables=None):
        A = np.array(A, dtype=np.dtype(int))
        b = np.array(b, dtype=np.dtype(int))
        assert A.ndim == 2 and b.ndim == 1
        assert A.shape[0] == b.shape[0]
        self.A = A
        self.b = b

        self.variables = None
        self.variable_names = None
        if variables is not None:
            assert len(variables) == A.shape[1]
            self.variables = variables
            self.variable_names = [v.name.lower() for v in self.variables]

    def variable_to_index(self, variable):
        if self.variable_names is None:
            raise RuntimeError('No variables list associated with polyhedron.')
        if isinstance(variable, (sym.Array, sym.Scalar)):
            variable = variable.name.lower()
        assert isinstance(variable, str)
        return self.variable_names.index(variable)

    @staticmethod
    def _to_literal(value):
        if value < 0:
            return sym.Product((-1, sym.IntLiteral(abs(value))))
        return sym.IntLiteral(value)

    def lower_bounds(self, index_or_variable):
        """
        Return all lower bounds imposed on a variable.

        Lower bounds for variable `j` are given by the index set
        ```
        L = {i in {0,...,d-1} | A_ij < 0}
        ```

        :param index_or_variable: the index, name, or expression symbol for which the
                    lower bounds are produced.
        :type index_or_variable: int or str or sym.Array or sym.Scalar

        :returns list: the bounds for that variable.
        """
        if isinstance(index_or_variable, int):
            j = index_or_variable
        else:
            j = self.variable_to_index(index_or_variable)

        bounds = []
        for i in range(self.A.shape[0]):
            if self.A[i,j] < 0:
                components = [self._to_literal(self.A[i,k]) * self.variables[k]
                              for k in range(self.A.shape[1]) if k != j and self.A[i,k] != 0]
                if not components:
                    lhs = sym.IntLiteral(0)
                elif len(components) == 1:
                    lhs = components[0]
                else:
                    lhs = sym.Sum(as_tuple(components))
                bounds += [simplify(sym.Quotient(self._to_literal(self.b[i]) - lhs,
                                                 self._to_literal(self.A[i,j])))]
        return bounds

    def upper_bounds(self, index_or_variable):
        """
        Return all upper bounds imposed on a variable.

        Upper bounds for variable `j` are given by the index set
        ```
        U = {i in {0,...,d-1} | A_ij > 0}
        ```

        :param index_or_variable: the index, name, or expression symbol for which the
                    upper bounds are produced.
        :type index_or_variable: int or str or sym.Array or sym.Scalar

        :returns list: the bounds for that variable.
        """
        if isinstance(index_or_variable, int):
            j = index_or_variable
        else:
            j = self.variable_to_index(index_or_variable)

        bounds = []
        for i in range(self.A.shape[0]):
            if self.A[i,j] > 0:
                components = [self._to_literal(self.A[i,k]) * self.variables[k]
                              for k in range(self.A.shape[1]) if k != j and self.A[i,k] != 0]
                if not components:
                    lhs = sym.IntLiteral(0)
                elif len(components) == 1:
                    lhs = components[0]
                else:
                    lhs = sym.Sum(as_tuple(components))
                bounds += [simplify(sym.Quotient(self._to_literal(self.b[i]) - lhs,
                                                 self._to_literal(self.A[i,j])))]
        return bounds

    @classmethod
    def from_loop_ranges(cls, loop_variables, loop_ranges):
        """
        Create polyhedron from a list of loop ranges and associated variables.
        """
        assert len(loop_ranges) == len(loop_variables)

        # Add any variables that are not loop variables to the vector of variables
        variables = list(loop_variables)
        variable_names = [v.name.lower() for v in variables]
        for v in sorted(FindVariables().visit(loop_ranges), key=lambda v: v.name.lower()):
            if v.name.lower() not in variable_names:
                variables += [v]
                variable_names += [v.name.lower()]

        n = 2 * len(loop_ranges)
        d = len(variables)
        A = np.zeros([n, d], dtype=np.dtype(int))
        b = np.zeros([n], dtype=np.dtype(int))

        for i, (loop_variable, loop_range) in enumerate(zip(loop_variables, loop_ranges)):
            assert loop_range.step is None or loop_range.step == '1'
            j = variables.index(loop_variable.name.lower())

            # Create inequality from lower bound
            lower_bound = simplify(loop_range.start)
            if not (is_constant(lower_bound) or
                    isinstance(lower_bound, (Variable, sym.Sum, sym.Product))):
                raise ValueError('Cannot derive inequality from bound {}'.format(str(lower_bound)))

            summands = accumulate_polynomial_terms(lower_bound)
            b[2*i] = -summands.pop(1, 0)
            A[2*i, j] = -1
            for base, coef in summands.items():
                if not len(base) == 1:
                    raise ValueError('Non-affine lower bound {}'.format(str(lower_bound)))
                A[2*i, variables.index(base[0].name.lower())] = coef

            # Create inequality from upper bound
            upper_bound = simplify(loop_range.stop)
            if not (is_constant(upper_bound) or
                    isinstance(upper_bound, (Variable, sym.Sum, sym.Product))):
                raise ValueError('Cannot derive inequality from bound {}'.format(str(upper_bound)))

            summands = accumulate_polynomial_terms(upper_bound)
            b[2*i+1] = summands.pop(1, 0)
            A[2*i+1, j] = 1
            for base, coef in summands.items():
                if not len(base) == 1:
                    raise ValueError('Non-affine upper bound {}'.format(str(upper_bound)))
                A[2*i+1, variable_names.index(base[0].name.lower())] = -coef

        return cls(A, b, variables)


def loop_fusion(routine):
    """
    Search for loops annotated with the `loki loop-fusion` pragma and attempt
    to fuse them into a single loop.
    """
    # Extract all annotated loops and sort them into fusion groups
    fusion_groups = defaultdict(list)
    for loop in FindNodes(Loop).visit(routine.body):
        if is_loki_pragma(loop.pragma, starts_with='loop-fusion'):
            parameters = get_pragma_parameters(loop.pragma, starts_with='loop-fusion')
            group = parameters.get('group', 'default')
            fusion_groups[group] += [loop]

    if not fusion_groups:
        return

    def _pragma_range_to_loop_range(parameters):
        if 'range' not in parameters:
            return None

        # TODO: This would be easier and more powerful with a `parse_expression` routine
        bounds = []
        for bound in parameters['range'].split(':'):
            if bound.isnumeric():
                bounds += [sym.IntLiteral(bound)]
            # TODO: parse more complex expressions
            else:
                bounds += [sym.Variable(name=bound, scope=routine.scope)]
        return sym.LoopRange(as_tuple(bounds))

    # Merge loops in each group and put them in the position of the group's first loop
    loop_map = {}
    for group, loop_list in fusion_groups.items():

        # First, extract user-annotated loop ranges from pragmas
        loop_ranges = [_pragma_range_to_loop_range(get_pragma_parameters(loop.pragma, starts_with='loop-fusion'))
                       for loop in loop_list]

        # If we have a pragma somewhere with an explicit loop range, we use that for the fused loop
        range_set = {r for r in loop_ranges if r is not None}
        if len(range_set) not in (0, 1):
            raise RuntimeError('Pragma-specified loop ranges in group "{}" do not match'.format(group))

        fusion_range = None
        if range_set:
            fusion_range = range_set.pop()

        # Next, extract loop ranges for remaining loops in group
        loop_ranges = [loop_range if loop_range is not None else loop.bounds
                       for loop_range, loop in zip(loop_ranges, loop_list)]

        # Convert loop ranges to iteration space polyhedrons for easier alignment
        iteration_spaces = [Polyhedron.from_loop_ranges([loop.variable], [loop.bounds]) for loop in loop_list]

        # Find the fused iteration space (if not given by a pragma)
        if fusion_range is None:
            lower_bounds, upper_bounds = [], []
            for p in iteration_spaces:
                for bound in p.lower_bounds(0):
                    # Decide if we learn something new from this bound, which could be because:
                    # (1) we don't have any bounds, yet
                    # (2) bound is smaller than existing lower bounds (i.e. diff < 0)
                    # (3) bound is not constant and none of the existing bounds are lower (i.e. diff >= 0)
                    diff = [simplify(bound - b) for b in lower_bounds]
                    is_any_negative = any(is_constant(d) and symbolic_op(d, op.lt, 0) for d in diff)
                    is_any_not_negative = any(is_constant(d) and symbolic_op(d, op.ge, 0) for d in diff)
                    is_new_bound = (not lower_bounds or is_any_negative or
                                    (not is_constant(bound) and not is_any_not_negative))
                    if is_new_bound:
                        # Remove any lower bounds made redundant by bound:
                        lower_bounds = [b for b, d in zip(lower_bounds, diff)
                                        if not (is_constant(d) and symbolic_op(d, op.lt, 0))]
                        lower_bounds += [bound]

                for bound in p.upper_bounds(0):
                    # Decide if we learn something new from this bound, which could be because:
                    # (1) we don't have any bounds, yet
                    # (2) bound is larger than existing upper bounds (i.e. diff > 0)
                    # (3) bound is not constant and none of the existing bounds are larger (i.e. diff <= 0)
                    diff = [simplify(bound - b) for b in upper_bounds]
                    is_any_positive = any(is_constant(d) and symbolic_op(d, op.gt, 0) for d in diff)
                    is_any_not_positive = any(is_constant(d) and symbolic_op(d, op.le, 0) for d in diff)
                    is_new_bound = (not upper_bounds or is_any_positive or
                                    (not is_constant(bound) and not is_any_not_positive))
                    if is_new_bound:
                        # Remove any lower bounds made redundant by bound:
                        upper_bounds = [b for b, d in zip(upper_bounds, diff)
                                        if not (is_constant(d) and symbolic_op(d, op.gt, 0))]
                        upper_bounds += [bound]

            if len(lower_bounds) == 1:
                lower_bounds = lower_bounds[0]
            else:
                fct_symbol = sym.ProcedureSymbol('min', scope=routine.scope)
                lower_bounds = sym.InlineCall(fct_symbol, parameters=as_tuple(lower_bounds))

            if len(upper_bounds) == 1:
                upper_bounds = upper_bounds[0]
            else:
                fct_symbol = sym.ProcedureSymbol('max', scope=routine.scope)
                upper_bounds = sym.InlineCall(fct_symbol, parameters=as_tuple(lower_bounds))

            fusion_range = sym.LoopRange((lower_bounds, upper_bounds))

        # Align loop ranges and collect bodies
        loop_bodies = []
        fusion_variable = loop_list[0].variable
        for (loop, p) in zip(loop_list, iteration_spaces):
            body = loop.body

            # Replace loop variable if necessary
            if loop.variable != fusion_variable:
                body = SubstituteExpressions({loop.variable: fusion_variable}).visit(body)

            # Wrap in conditional if loop length is different
            conditions = []
            if symbolic_op(loop.bounds.start, op.ne, fusion_range.start):
                conditions += [sym.Comparison(fusion_variable, '>=', loop.bounds.start)]
            if symbolic_op(loop.bounds.stop, op.ne, fusion_range.stop):
                conditions += [sym.Comparison(fusion_variable, '<=', loop.bounds.stop)]
            if conditions:
                if len(conditions) == 1:
                    conditions = conditions[0]
                else:
                    conditions = sym.LogicalAnd(as_tuple(conditions))
                body = Conditional(conditions=[conditions], bodies=[body], else_body=())

            loop_bodies += [body]

        loop_map[loop_list[0]] = (
            Comment('! Loki transformation loop-fusion group({})'.format(group)),
            Loop(variable=fusion_variable, body=flatten(loop_bodies), bounds=fusion_range))
        loop_map.update({loop: None for loop in loop_list[1:]})

    # Apply transformation
    routine.body = Transformer(loop_map).visit(routine.body)

    info('%s: fused %d loops in %d groups.', routine.name,
         sum(len(loop_list) for loop_list in fusion_groups.values()), len(fusion_groups))


def loop_fission(routine):
    """
    Search for `loki loop-fission` pragmas inside loops and attempt to split them into
    multiple loops.
    """
    comment = Comment('! Loki transformation loop-fission')
    loop_map = {}

    for loop in FindNodes(Loop).visit(routine.body):
        pragmas = [ch for ch in loop.body
                   if isinstance(ch, Pragma) and is_loki_pragma(ch, starts_with='loop-fission')]
        if not pragmas:
            continue

        pragma_indices = [-1] + sorted([loop.body.index(p) for p in pragmas]) + [len(loop.body)]
        bodies = [loop.body[start+1:stop] for start, stop in zip(pragma_indices[:-1], pragma_indices[1:])]
        loop_map[loop] = [(comment, Loop(variable=loop.variable, bounds=loop.bounds, body=body))
                          for body in bodies]

    if loop_map:
        routine.body = Transformer(loop_map).visit(routine.body)
        info('%s: split %d loops into %d loops.', routine.name, len(loop_map),
             sum(len(loop_list) for loop_list in loop_map.values()))
