# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import pytest
import numpy as np

from loki.analyse.util_linear_algebra import (
    back_substitution,
    row_echelon_form_under_gcd_condition,
    NoIntegerSolution,
    bounds_of_one_d_system,
)


@pytest.mark.parametrize(
    "upper_triangular_square_matrix, right_hand_side, expected, divison_operation",
    [
        (
            [[2, 1, -1], [0, 0.5, 0.5], [0, 0, -1]],
            [[8], [1], [1]],
            [[2], [3], [-1]],
            lambda x, y: x / y,
        ),
        (
            [[2, 0], [0, 1]],
            [[10], [11]],
            [[5], [11]],
            lambda x, y: x // y,
        ),
    ],
)
def test_backsubstitution(
    upper_triangular_square_matrix, right_hand_side, expected, divison_operation
):
    assert np.allclose(
        back_substitution(
            np.array(upper_triangular_square_matrix),
            np.array(right_hand_side),
            divison_operation,
        ),
        np.array(expected),
    )


@pytest.mark.parametrize(
    "matrix, should_fail, result",
    [
        ([[2, 0, 1], [0, 2, 0]], True, None),
        ([[1, -2, 1, 0], [3, 2, 1, 5]], True, None),
        ([[1, -1, -10]], False, [[1, -1, -10]]),
    ],
)
def test_row_echelon_form_under_gcd_condition(matrix, should_fail, result):
    matrix = np.array(matrix, dtype=np.dtype(int))
    if should_fail:
        with pytest.raises(NoIntegerSolution):
            _ = row_echelon_form_under_gcd_condition(matrix)
    else:
        result = np.array(result, dtype=np.dtype(int))
        assert np.array_equal(row_echelon_form_under_gcd_condition(matrix), result)


@pytest.mark.parametrize(
    "matrix, rhs, expected_lower, expected_upper",
    [
        (  # test no upper bound only lower bound
            np.array([[1], [2], [3], [4], [5]]),
            np.array([[10], [20], [30], [40], [50]]),
            [10],
            [],
        ),
        (  # test no lower bound only upper bound
            np.array([[-1], [-2], [-3], [-4], [-5]]),
            np.array([[-10], [-20], [-30], [-40], [-50]]),
            [],
            [10],
        ),
        (  # test no lower bound only upper bound
            -np.array([[1], [2], [3], [4], [5]]),
            -np.array([[10], [20], [30], [40], [50]]),
            [],
            [10],
        ),
        (  # test no upper bound only lower bounds
            np.array([[1], [2], [3], [4], [5]]),
            np.array([[10], [20], [30], [40], [50]]),
            [10],
            [],
        ),
        (  # test both upper and lower bounds
            np.array([[1], [-1]]),
            np.array([[0], [-10]]),
            [0],
            [10],
        ),
        (np.array([[0]]), np.array([[1]]), [1], []),
        (
            np.array([[0], [0], [1], [-1], [1], [-1], [0], [0]]),
            np.array([[-10], [0], [-11], [1], [0], [-10], [0], [-10]]),
            [-11, 0],
            [-1, 10],
        ),
    ],
)
def test_bounds_of_one_d_system(matrix, rhs, expected_lower, expected_upper):
    lower_bounds, upper_bounds = bounds_of_one_d_system(matrix, rhs)
    assert np.allclose(lower_bounds, expected_lower)
    assert np.allclose(upper_bounds, expected_upper)
