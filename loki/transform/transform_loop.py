"""
Collection of utility routines that provide loop transformations.

"""
from collections import defaultdict
import numpy as np
from pymbolic import primitives as pmbl

from loki.expression import (
    symbols as sym, SubstituteExpressions, FindVariables,
    accumulate_polynomial_terms, simplify)
from loki.ir import Loop, Conditional, Comment
from loki.logging import info
from loki.tools import is_loki_pragma, get_pragma_parameters, flatten, as_tuple
from loki.visitors import FindNodes, Transformer

__all__ = ['loop_fusion', 'Polyhedron']


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
            if not (pmbl.is_constant(lower_bound) or
                    isinstance(lower_bound, (pmbl.Variable, sym.Sum, sym.Product))):
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
            if not (pmbl.is_constant(upper_bound) or
                    isinstance(upper_bound, (pmbl.Variable, sym.Sum, sym.Product))):
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
    fusion_variables = {}
    fusion_ranges = {}
    for loop in FindNodes(Loop).visit(routine.body):
        if is_loki_pragma(loop.pragma, starts_with='loop-fusion'):
            parameters = get_pragma_parameters(loop.pragma, starts_with='loop-fusion')
            group = parameters.get('group', 'default')
            fusion_groups[group] += [loop]
            fusion_variables.setdefault(group, loop.variable)

            if 'range' in parameters:
                bounds = []
                for bound in parameters['range'].split(':'):
                    if bound.isnumeric():
                        bounds += [sym.IntLiteral(bound)]
                    # TODO: parse more complex expressions
                    else:
                        bounds += [sym.Variable(name=bound, scope=routine.symbols)]
                loop_range = sym.LoopRange(as_tuple(bounds))
            else:
                loop_range = loop.bounds
            fusion_ranges.setdefault(group, loop_range)
            # TODO: Check step (None vs 1)
            if fusion_ranges[group].start != loop_range.start or fusion_ranges[group].stop != loop_range.stop:
                raise RuntimeError('Loop ranges in group "{}" do not match'.format(group))

    if not fusion_groups:
        return

    missing_ranges = set(fusion_groups.keys()) - set(fusion_ranges.keys())
    if missing_ranges:
        raise RuntimeError('No loop ranges given for group(s) "{}".'.format('", "'.join(missing_ranges)))

    # Built merged loops
    loop_map = {}
    for group, loop_list in fusion_groups.items():
        loop_range = fusion_ranges[group]
        loop_variable = fusion_variables[group]

        bodies = []
        for loop in loop_list:
            # Replace loop variable if it differs
            if loop.variable != loop_variable:
                body = SubstituteExpressions({loop.variable: loop_variable}).visit(loop.body)
            else:
                body = loop.body

            # Enclose in a conditional block if loop range is smaller
            conditions = []
            if loop.bounds.start != loop_range.start:
                conditions += [sym.Comparison(loop_variable, '>=', loop.bounds.start)]
            if loop.bounds.stop != loop_range.stop:
                conditions += [sym.Comparison(loop_variable, '<=', loop.bounds.stop)]
            if loop.bounds.step not in (None, loop_range.step or 1):
                shifted_variable = sym.Sum(loop_variable, sym.Product(-1, loop.bounds.start))
                call = sym.InlineCall('mod', parameters=(shifted_variable, loop.bounds.step))
                conditions += [sym.Comparison(call, '==', sym.Literal(0))]
            if conditions:
                if len(conditions) == 1:
                    condition = conditions[0]
                else:
                    condition = sym.LogicalAnd(conditions)
                bodies += [Conditional(conditions=[condition], bodies=[body], else_body=())]
            else:
                bodies += [body]

        loop_map[loop_list[0]] = (
            Comment('! Loki transformation loop-fusion group({})'.format(group)),
            Loop(variable=loop_variable, body=flatten(bodies), bounds=loop_range))
        loop_map.update({loop: None for loop in loop_list[1:]})

    # Apply transformation
    routine.body = Transformer(loop_map).visit(routine.body)

    info('%s: fused %d loops in %d groups.', routine.name,
         sum(len(loop_list) for loop_list in fusion_groups.values()), len(fusion_groups))
