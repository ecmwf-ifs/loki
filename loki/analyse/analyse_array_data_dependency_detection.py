# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from math import gcd
from warnings import warn
from dataclasses import dataclass
from numpy.typing import NDArray as npt_NDArray
from numpy import int_ as np_int_
import numpy as np
from ortools.linear_solver import pywraplp

from loki.analyse.util_linear_algebra import (
    generate_reduced_row_echelon_form,
    back_substitution,
    is_independent_system,
    yield_one_d_systems,
    bounds_of_one_d_system,
)

__all__ = ["has_data_dependency"]

NDArrayInt = npt_NDArray[np_int_]


@dataclass
class IterationSpaceRepresentation:
    """
    Represents an iteration space as an inequality system. The matrix A, vector b and unkowns x
    represent the inequality system Ax <= b, where the inequality is to be interpreted element wise.
    """

    matrix: NDArrayInt
    vector: NDArrayInt


@dataclass
class AffineArrayAccessFunctionRepresentation:
    """
    Represents an affine array access function as a linear system. The matrix F and vector f
    represent the linear system Fx + f, where x is the iteration space vector of unkowns.
    """

    matrix: NDArrayInt
    vector: NDArrayInt


@dataclass
class StaticArrayAccessInstanceRepresentation:
    """
    Represents a static array access instance using an iteration space representation and
    an affine array access function representation.
    """

    iteration_space: IterationSpaceRepresentation
    access_function: AffineArrayAccessFunctionRepresentation

    def __init__(
        self,
        iteration_space: IterationSpaceRepresentation,
        access_function: AffineArrayAccessFunctionRepresentation,
    ):
        self.iteration_space = IterationSpaceRepresentation(*iteration_space)
        self.access_function = AffineArrayAccessFunctionRepresentation(*access_function)


def _assert_correct_dimensions(
    first_array_access: StaticArrayAccessInstanceRepresentation,
    second_array_access: StaticArrayAccessInstanceRepresentation,
) -> None:
    def assert_compatibility_of_iteration_space(
        first: IterationSpaceRepresentation, second: IterationSpaceRepresentation
    ):
        (number_inequalties_1, number_variables_1) = first.matrix.shape
        (rhs_number_inequalities_1, column_count) = first.vector.shape
        assert column_count == 1, "Right hand side is a single column vector!"
        assert (
            number_inequalties_1 == rhs_number_inequalities_1
        ), "System matrix and right hand side vector require same amount of rows!"

        (number_inequalties_2, number_variables_2) = second.matrix.shape
        (rhs_number_inequalities_2, column_count) = first.vector.shape
        assert column_count == 1, "Right hand side is a single column vector!"
        assert (
            number_inequalties_2 == rhs_number_inequalities_2
        ), "System matrix and right hand side vector require same amount of rows!"

        assert number_variables_1 == number_variables_2, (
            "Same number of variables per iteration space is assumed with "
            "each variable from the first iteration space having its equivalent "
            "at the same place in the second iteration space"
        )

    def assert_compatibility_of_access_function(
        first: AffineArrayAccessFunctionRepresentation,
        second: AffineArrayAccessFunctionRepresentation,
    ):
        (number_rows_1, number_columns_1) = first.matrix.shape
        (rhs_number_rows_1, column_count) = first.vector.shape
        assert column_count == 1, "Right hand side is a single column vector!"
        assert (
            number_rows_1 == rhs_number_rows_1
        ), "System matrix and right hand side vector require same amount of rows!"

        (number_rows_2, number_columns_2) = second.matrix.shape
        (rhs_number_rows_2, column_count) = first.vector.shape
        assert column_count == 1, "Right hand side is a single column vector!"
        assert (
            number_rows_2 == rhs_number_rows_2
        ), "System matrix and right hand side vector require same amount of rows!"

        assert number_columns_1 == number_columns_2, (
            "Same number of variables per access function is assumed with "
            "each variable from the first access function having its equivalent "
            "at the same place in the second access function"
        )

    def assert_compatibility_of_access_and_iteration_unkowns(
        function: AffineArrayAccessFunctionRepresentation,
        space: IterationSpaceRepresentation,
    ):
        (_, number_variables_space) = space.matrix.shape
        (_, number_variables_function) = function.matrix.shape

        assert (
            number_variables_space == number_variables_function
        ), "Same number of variables per access function and iteration space is assumed"

    assert_compatibility_of_iteration_space(
        first_array_access.iteration_space, second_array_access.iteration_space
    )
    assert_compatibility_of_access_function(
        first_array_access.access_function, second_array_access.access_function
    )
    assert_compatibility_of_access_and_iteration_unkowns(
        first_array_access.access_function, second_array_access.iteration_space
    )


def _safe_integer_division(x, y):
    result = x // y
    if (x % y != 0).all():
        raise ValueError("Division does not result in an integer.")
    return result


def _gaussian_eliminiation_for_diophantine_equations(
    augmented_matrix: NDArrayInt,
) -> (bool, NDArrayInt):
    """
    Calculate the Row Echelon Form (REF) of an augmented matrix while ensuring integer solutions
    to linear Diophantine equations.

    Args:
        augmented_matrix (numpy.ndarray): The input augmented system matrix.

    Returns:
        (HasIntegerSolution, numpy.ndarray): The tuple of if the solution exists and the REF of the input matrix.

    This function computes the Row Echelon Form (REF) of a given matrix of integers while enforcing that
    each linear Diophantine equation in the system has integer solutions. It follows Theorem 11.32
    in "Compilers: Principles, Techniques, and Tools" by Aho, Lam, Sethi, and Ullman (2015).
    """

    def gcd_condition(A):
        """Check that gcd condition of linear Diophantine equation is satisfied"""
        if A[0, -1] % gcd(*A[0, :-1]) != 0:
            raise ValueError

    status = True
    solution = None
    try:
        solution = generate_reduced_row_echelon_form(
            augmented_matrix,
            conditional_check=gcd_condition,
            division_operator=_safe_integer_division,
        )
    except ValueError:
        status = False

    return (status, solution)


def _does_independent_system_violate_bounds(
    matrix: NDArrayInt, vector: NDArrayInt
) -> bool:
    """
    Check if a system of inequalities which is seperable into N one dimensional problems represented by a
    matrix and a vector violates its bounds. The following system is considered
    A x <= b, where A is the matrix, b the vector and x the vector of unkowns.


    Parameters:
    - matrix (NDArrayInt): A 2D NumPy array representing the matrix of the independent system.
    - vector (NDArrayInt): A 1D NumPy array representing the vector of the independent system.

    Returns:
    - bool: True if any of the 1d systems violate their bounds, False otherwise.
    """
    for one_d_system in yield_one_d_systems(matrix, vector):
        lower_bounds, upper_bounds = bounds_of_one_d_system(
            one_d_system[0].astype(float), -one_d_system[1].astype(float)
        )
        if np.amax(lower_bounds) > np.amin(upper_bounds):
            return True
    return False


def _solve_inequalties_with_ortools(
    constraint_coeffs: NDArrayInt, right_hand_side: NDArrayInt
):
    """
    Solve a system of inequalities using ortools. With A the constraint coefficient matrix, b the right
    hand side and x the unkowns, the system is of the form A x <= b.
    """
    number_constraints, number_variables = constraint_coeffs.shape
    assert (
        number_constraints,
        1,
    ) == right_hand_side.shape, (
        "number of number_constraints, "
        "i.e. the number of rows in matrix must match number of columns in right hand side vector"
    )

    solver = pywraplp.Solver.CreateSolver("SCIP")
    assert solver, "Solver could not be created."

    inf = solver.infinity()
    unkowns = [solver.IntVar(-inf, inf, "x_" + str(i)) for i in range(number_variables)]

    for i in range(number_constraints):
        constraint_expr = [
            constraint_coeffs[i][j] * unkowns[j] for j in range(number_variables)
        ]
        solver.Add(sum(constraint_expr) <= right_hand_side[i, 0])

    solver.Maximize(0)  # arbitrary simple objective function

    return solver.Solve()


def has_data_dependency(
    first_array_access: StaticArrayAccessInstanceRepresentation,
    second_array_access: StaticArrayAccessInstanceRepresentation,
) -> bool:
    if not isinstance(first_array_access, StaticArrayAccessInstanceRepresentation):
        first_array_access = StaticArrayAccessInstanceRepresentation(
            *first_array_access
        )

    if not isinstance(second_array_access, StaticArrayAccessInstanceRepresentation):
        second_array_access = StaticArrayAccessInstanceRepresentation(
            *second_array_access
        )

    _assert_correct_dimensions(first_array_access, second_array_access)

    F, f = (
        first_array_access.access_function.matrix,
        first_array_access.access_function.vector,
    )
    F_dash, f_dash = (
        second_array_access.access_function.matrix,
        second_array_access.access_function.vector,
    )

    (has_solution, solved_system) = _gaussian_eliminiation_for_diophantine_equations(
        np.hstack(
            (
                F,
                -F_dash,
                f - f_dash,
            )
        )
    )

    if not has_solution:
        return False

    n, m = solved_system.shape

    assert n < m, "The augmented system is assumed to be underdetermined."

    R, S, g_hat = solved_system[:, :n], solved_system[:, n:-1], solved_system[:, -1:]

    B, b = (
        first_array_access.iteration_space.matrix,
        first_array_access.iteration_space.vector,
    )
    B_dash, b_dash = (
        second_array_access.iteration_space.matrix,
        second_array_access.iteration_space.vector,
    )

    K = np.block([[B, np.zeros_like(B)], [np.zeros_like(B_dash), B_dash]])
    k = np.vstack((b, b_dash))

    n, m = R.shape
    K_R = K[0:n, 0:m]

    U = K[n:, 0:m]
    V = K[0:n, m:]
    Z = K[n:, m:]

    tmp1 = back_substitution(R, S, _safe_integer_division)
    tmp2 = back_substitution(R, g_hat, _safe_integer_division)

    matrix_final_polygon = np.vstack((-K_R.dot(tmp1) + V, -U.dot(tmp1) + Z))
    vector_final_polygon = k + np.vstack((K_R.dot(tmp2), U.dot(tmp2)))

    if is_independent_system(
        matrix_final_polygon
    ) and _does_independent_system_violate_bounds(
        matrix_final_polygon, vector_final_polygon
    ):
        return False

    status = _solve_inequalties_with_ortools(matrix_final_polygon, vector_final_polygon)
    match status:
        case pywraplp.Solver.OPTIMAL:
            return True
        case pywraplp.Solver.FEASIBLE:
            return True
        case pywraplp.Solver.INFEASIBLE:
            return False
        case _:
            warn(
                "ortools produced a not considered status --> assuming data dependency."
            )

    return True  # assume data dependency if all tests are inconclusive
