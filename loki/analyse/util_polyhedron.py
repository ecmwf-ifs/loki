# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from loki import TypedSymbol, Loop

from loki.expression import (
    symbols as sym,
    FindVariables,
    simplify,
    is_constant,
    accumulate_polynomial_terms,
)
from loki.tools import as_tuple

__all__ = ["Polyhedron"]


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
            raise RuntimeError("No variables list associated with polyhedron.")
        if isinstance(variable, TypedSymbol):
            variable = variable.name.lower()
        assert isinstance(variable, str)
        return self.variable_names.index(variable)

    @staticmethod
    def _to_literal(value):
        if value < 0:
            return sym.Product((-1, sym.IntLiteral(abs(value))))
        return sym.IntLiteral(value)

    def lower_bounds(self, index_or_variable, ignore_variables=None):
        """
        Return all lower bounds imposed on a variable.

        Lower bounds for variable `j` are given by the index set
        ```
        L = {i in {0,...,d-1} | A_ij < 0}
        ```

        :param index_or_variable: the index, name, or expression symbol for which the
                    lower bounds are produced.
        :type index_or_variable: int or str or sym.Array or sym.Scalar
        :param ignore_variables: optional list of variable names, indices or symbols
                    for which constraints should be ignored if they depend on one of them.
        :type ignore_variables: list or None

        :returns list: the bounds for that variable.
        """
        if isinstance(index_or_variable, int):
            j = index_or_variable
        else:
            j = self.variable_to_index(index_or_variable)

        if ignore_variables:
            ignore_variables = [
                i if isinstance(i, int) else self.variable_to_index(i)
                for i in ignore_variables
            ]

        bounds = []
        for i in range(self.A.shape[0]):
            if self.A[i, j] < 0:
                if ignore_variables and any(
                    self.A[i, k] != 0 for k in ignore_variables
                ):
                    # Skip constraint that depends on any of the ignored variables
                    continue

                components = [
                    self._to_literal(self.A[i, k]) * self.variables[k]
                    for k in range(self.A.shape[1])
                    if k != j and self.A[i, k] != 0
                ]
                if not components:
                    lhs = sym.IntLiteral(0)
                elif len(components) == 1:
                    lhs = components[0]
                else:
                    lhs = sym.Sum(as_tuple(components))
                bounds += [
                    simplify(
                        sym.Quotient(
                            self._to_literal(self.b[i]) - lhs,
                            self._to_literal(self.A[i, j]),
                        )
                    )
                ]
        return bounds

    def upper_bounds(self, index_or_variable, ignore_variables=None):
        """
        Return all upper bounds imposed on a variable.

        Upper bounds for variable `j` are given by the index set
        ```
        U = {i in {0,...,d-1} | A_ij > 0}
        ```

        :param index_or_variable: the index, name, or expression symbol for which the
                    upper bounds are produced.
        :type index_or_variable: int or str or sym.Array or sym.Scalar
        :param ignore_variables: optional list of variable names, indices or symbols
                    for which constraints should be ignored if they depend on one of them.
        :type ignore_variables: list or None

        :returns list: the bounds for that variable.
        """
        if isinstance(index_or_variable, int):
            j = index_or_variable
        else:
            j = self.variable_to_index(index_or_variable)

        if ignore_variables:
            ignore_variables = [
                i if isinstance(i, int) else self.variable_to_index(i)
                for i in ignore_variables
            ]

        bounds = []
        for i in range(self.A.shape[0]):
            if self.A[i, j] > 0:
                if ignore_variables and any(
                    self.A[i, k] != 0 for k in ignore_variables
                ):
                    # Skip constraint that depends on any of the ignored variables
                    continue

                components = [
                    self._to_literal(self.A[i, k]) * self.variables[k]
                    for k in range(self.A.shape[1])
                    if k != j and self.A[i, k] != 0
                ]
                if not components:
                    lhs = sym.IntLiteral(0)
                elif len(components) == 1:
                    lhs = components[0]
                else:
                    lhs = sym.Sum(as_tuple(components))
                bounds += [
                    simplify(
                        sym.Quotient(
                            self._to_literal(self.b[i]) - lhs,
                            self._to_literal(self.A[i, j]),
                        )
                    )
                ]
        return bounds

    @staticmethod
    def generate_entries_for_lower_bound(bound, variables, index):
        """
        Helper routine to generate matrix and right-hand side entries for a
        given lower bound.

        NB: This can only deal with affine bounds, i.e. expressions that are
            constant or can be reduced to a linear polynomial.

        Upper bounds can be derived from this by multiplying left-hand side and
        right-hand side with -1.

        :param bound: the expression representing the lower bound.
        :param list variables: the list of variable names.
        :param int index: the index of the variable constrained by this bound.

        :return: the pair ``(lhs, rhs)`` of the row in the matrix inequality.
        :rtype: tuple(np.array, np.array)
        """
        supported_types = (sym.TypedSymbol, sym.MetaSymbol, sym.Sum, sym.Product)
        if not (is_constant(bound) or isinstance(bound, supported_types)):
            raise ValueError(f"Cannot derive inequality from bound {str(bound)}")
        summands = accumulate_polynomial_terms(bound)
        b = -summands.pop(1, 0)  # Constant term or 0
        A = np.zeros([1, len(variables)], dtype=np.dtype(int))
        A[0, index] = -1
        for base, coef in summands.items():
            if not len(base) == 1:
                raise ValueError(f"Non-affine bound {str(bound)}")
            A[0, variables.index(base[0].name.lower())] = coef
        return A, b

    @classmethod
    def from_loop_ranges(cls, loop_variables, loop_ranges):
        """
        Create polyhedron from a list of loop ranges and associated variables.
        """
        assert len(loop_ranges) == len(loop_variables)

        # Add any variables that are not loop variables to the vector of variables
        variables = list(loop_variables)
        variable_names = [v.name.lower() for v in variables]
        for v in sorted(
            FindVariables().visit(loop_ranges), key=lambda v: v.name.lower()
        ):
            if v.name.lower() not in variable_names:
                variables += [v]
                variable_names += [v.name.lower()]

        n = 2 * len(loop_ranges)
        d = len(variables)
        A = np.zeros([n, d], dtype=np.dtype(int))
        b = np.zeros([n], dtype=np.dtype(int))

        for i, (loop_variable, loop_range) in enumerate(
            zip(loop_variables, loop_ranges)
        ):
            assert loop_range.step is None or loop_range.step == "1"
            j = variables.index(loop_variable.name.lower())

            # Create inequality from lower bound
            lhs, rhs = cls.generate_entries_for_lower_bound(
                loop_range.start, variable_names, j
            )
            A[2 * i, :] = lhs
            b[2 * i] = rhs

            # Create inequality from upper bound
            lhs, rhs = cls.generate_entries_for_lower_bound(
                loop_range.stop, variable_names, j
            )
            A[2 * i + 1, :] = -lhs
            b[2 * i + 1] = -rhs

        return cls(A, b, variables)

    @classmethod
    def from_nested_loops(cls, nested_loops: list[Loop]):
        """
        Helper function, for creating a polyhedron from a list of loops.
        """
        return cls.from_loop_ranges(
            [l.variable for l in nested_loops], [l.bounds for l in nested_loops]
        )

    def get_B_b_representation(self):
        """
        Return the matrix and vector constructing the polyhedron in the B-b notation
        as used in the Compilers: Principles, Techniques, and Tools book
        """

        return -self.A, np.reshape(self.b, (-1, 1))
