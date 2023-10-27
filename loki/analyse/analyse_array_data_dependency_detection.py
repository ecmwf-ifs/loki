# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from math import gcd
from warnings import warn
from dataclasses import dataclass
from typing import Any, List
from numpy.typing import NDArray as npt_NDArray
from numpy import int_ as np_int_
import numpy as np
from loki.expression import (
    FindVariables,
    accumulate_polynomial_terms,
    simplify,
    is_constant,
    symbols as sym,
)
from loki.analyse.util_linear_algebra import (
    generate_row_echelon_form,
    back_substitution,
    is_independent_system,
    yield_one_d_systems,
)

try:
    from ortools.linear_solver import pywraplp

    HAVE_ORTOOLS = True
except ImportError:
    HAVE_ORTOOLS = False

__all__ = [
    "has_data_dependency",
    "HAVE_ORTOOLS",
    "construct_affine_array_access_function_representation",
]

NDArrayInt = npt_NDArray[np_int_]


@dataclass
class IterationSpaceRepresentation:
    """
    Represents an iteration space as an inequality system. The matrix A, vector b and unkowns x
    represent the inequality system Ax <= b, where the inequality is to be interpreted element wise.
    """

    matrix: NDArrayInt
    vector: NDArrayInt


# abbrivation for readability
IterSpacRepr = IterationSpaceRepresentation


@dataclass
class AffineArrayAccessFunctionRepresentation:
    """
    Represents an affine array access function as a linear system. The matrix F and vector f
    represent the linear system Fx + f, where x is the iteration space vector of unkowns.
    """

    matrix: NDArrayInt
    vector: NDArrayInt


# abbrivation for readability
AffiAcceRepr = AffineArrayAccessFunctionRepresentation


@dataclass
class StaticArrayAccessInstanceRepresentation:
    """
    Represents a static array access instance using an iteration space representation and
    an affine array access function representation.
    """

    iteration_space: IterSpacRepr
    access_function: AffiAcceRepr


# abbrivation for readability
StatInstRepr = StaticArrayAccessInstanceRepresentation


def _assert_correct_dimensions(
    first_array_access: StatInstRepr,
    second_array_access: StatInstRepr,
) -> None:
    def assert_compatibility_of_iteration_space(
        first: IterSpacRepr, second: IterSpacRepr
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
        first: AffiAcceRepr,
        second: AffiAcceRepr,
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
        function: AffiAcceRepr,
        space: IterSpacRepr,
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
        solution = generate_row_echelon_form(
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

    def get_bounds(matrix, vector):
        """Gets x <= c and x >= d conditions"""
        larger_zero = matrix > 0
        upper_bounds = vector[larger_zero] / matrix[larger_zero]

        smaller_zero = matrix < 0
        lower_bounds = vector[smaller_zero] / matrix[smaller_zero]

        return np.unique(lower_bounds), np.unique(upper_bounds)

    def get_zero_lhs_conditions(matrix, vector):
        """Gets 0 <= e conditions"""
        zero_lhs = matrix == 0
        return vector[zero_lhs]

    for one_d_system in yield_one_d_systems(matrix, vector):
        if not np.all(0 <= get_zero_lhs_conditions(one_d_system[0], one_d_system[1])):
            return True

        lower_bounds, upper_bounds = get_bounds(
            one_d_system[0].astype(float), one_d_system[1].astype(float)
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


def has_data_dependency_impl(
    first_array_access: StatInstRepr,
    second_array_access: StatInstRepr,
) -> bool:
    # fmt: off
    (has_solution, solved_system) = _gaussian_eliminiation_for_diophantine_equations(
        np.hstack(
            (
                first_array_access.access_function.matrix,
                -second_array_access.access_function.matrix,
                first_array_access.access_function.vector - second_array_access.access_function.vector,
            )
        )
    )
    # fmt: on
    if not has_solution:
        return False

    n, m = solved_system.shape

    assert n < m, "The augmented system is assumed to be underdetermined."

    R, S, g_hat = solved_system[:, :n], solved_system[:, n:-1], solved_system[:, -1:]

    # fmt: off
    K = np.block(
        [
            [
                first_array_access.iteration_space.matrix, np.zeros_like(first_array_access.iteration_space.matrix),
            ],
            [
                np.zeros_like(second_array_access.iteration_space.matrix), second_array_access.iteration_space.matrix,
            ],
        ]
    )
    # fmt: on
    k = np.vstack(
        (
            first_array_access.iteration_space.vector,
            second_array_access.iteration_space.vector,
        )
    )

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

    if not HAVE_ORTOOLS:
        warn(
            "ortools is not installed --> assuming data dependency --> will lead to alot of false positives"
        )
        return True

    status = _solve_inequalties_with_ortools(matrix_final_polygon, vector_final_polygon)

    known_status = {
        pywraplp.Solver.OPTIMAL: True,
        pywraplp.Solver.FEASIBLE: True,
        pywraplp.Solver.INFEASIBLE: False,
    }

    if status in known_status:
        return known_status[status]
    warn("ortools produced a not considered status --> assuming data dependency.")

    return True  # assume data dependency if all tests are inconclusive


def _ensure_correct_StatInstRepr(array_access: Any) -> StatInstRepr:
    if isinstance(array_access, StatInstRepr):
        return array_access

    if len(array_access) == 2:
        iteration_space = array_access[0]
        access_function = array_access[1]

        if isinstance(iteration_space, IterSpacRepr) and isinstance(
            access_function, AffiAcceRepr
        ):
            return StatInstRepr(iteration_space, access_function)

        if isinstance(iteration_space, IterSpacRepr) and len(access_function) == 2:
            return StatInstRepr(iteration_space, AffiAcceRepr(*access_function))

        if len(iteration_space) == 2 and isinstance(access_function, AffiAcceRepr):
            return StatInstRepr(IterSpacRepr(*iteration_space), access_function)

        if len(iteration_space) == 2 and len(access_function) == 2:
            return StatInstRepr(
                IterSpacRepr(*iteration_space), AffiAcceRepr(*access_function)
            )

    raise ValueError(f"Cannot convert {str(array_access)} to StatInstRepr")


def has_data_dependency(
    first_array_access: StatInstRepr, second_array_access: StatInstRepr
):
    first_array_access = _ensure_correct_StatInstRepr(first_array_access)
    second_array_access = _ensure_correct_StatInstRepr(second_array_access)

    _assert_correct_dimensions(first_array_access, second_array_access)

    return has_data_dependency_impl(first_array_access, second_array_access)


def construct_affine_array_access_function_representation(
    array_dimensions_expr: tuple(), additional_variables: List[str] = None
):
    """
    Construct a matrix, vector representation of the access function of an array.
    E.g. z[i], where the expression ("i", ) should be passed to this function,
         y[1+3,4-j], where ("1+3", "4-j") should be passed to this function,
         if var=Array(...), then var.dimensions should be passed to this function.
    Returns: matrix, vector: F,f mapping a vecector i within the bounds Bi+b>=0 to the
    array location Fi+f
    """

    def generate_row(expr, variables):
        supported_types = (sym.TypedSymbol, sym.MetaSymbol, sym.Sum, sym.Product)
        if not (is_constant(expr) or isinstance(expr, supported_types)):
            raise ValueError(f"Cannot derive inequality from expr {str(expr)}")
        simplified_expr = simplify(expr)
        terms = accumulate_polynomial_terms(simplified_expr)
        const_term = terms.pop(1, 0)  # Constant term or 0
        row = np.zeros(len(variables), dtype=np.dtype(int))

        for base, coef in terms.items():
            if not len(base) == 1:
                raise ValueError(f"Non-affine bound {str(simplified_expr)}")
            row[variables.index(base[0].name.lower())] = coef

        return row, const_term

    def unique_order_preserving(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    if additional_variables is None:
        additional_variables = []

    for variable in additional_variables:
        assert variable.lower() == variable

    variables = additional_variables.copy()
    variables += list(
        {v.name.lower() for v in FindVariables().visit(array_dimensions_expr)}
    )
    variables = unique_order_preserving(variables)

    n = len(array_dimensions_expr)
    d = len(variables)

    F = np.zeros([n, d], dtype=np.dtype(int))
    f = np.zeros([n, 1], dtype=np.dtype(int))

    for dimension_index, sub_expr in enumerate(array_dimensions_expr):
        row, constant = generate_row(sub_expr, variables)
        F[dimension_index] = row
        f[dimension_index, 0] = constant
    return AffiAcceRepr(F, f), variables
