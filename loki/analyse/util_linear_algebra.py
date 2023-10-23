# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from math import gcd
from numpy import zeros_like, dot
from numpy import vstack, hstack

__all__ = [
    "back_substitution",
    "generate_reduced_row_echelon_form",
    "NoIntegerSolution",
    "row_echelon_form_under_gcd_condition",
]


def back_substitution(
    upper_triangular_square_matrix,
    right_hand_side,
    divison_operation=lambda x, y: x / y,
):
    """
    Solve a linear system of equations via back substitution Rx = y where R is an upper triangular
    square matrix and y is a vector.
    """
    R = upper_triangular_square_matrix
    y = right_hand_side

    x = zeros_like(y)

    assert R[-1, -1] != 0

    x[-1] = divison_operation(y[-1], R[-1, -1])

    for i in range(len(y) - 2, -1, -1):
        x[i] = divison_operation((y[i] - dot(R[i, i + 1 :], x[i + 1 :])), R[i, i])

    return x


def generate_reduced_row_echelon_form(A, conditional_check=lambda: None):
    """
    Calculate reduced row echelon form of matrix A
    """

    # Credit: https://math.stackexchange.com/questions/3073083/how-to-reduce-matrix-into-row-echelon-form-in-numpy
    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i, 0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon_form_under_gcd_condition(A[:, 1:])
        # and then add the first zero-column back
        return hstack([A[:, :1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        A[[i, 0]] = A[[0, i]]

    # check condition
    conditional_check(A)

    # we divide first row by first element in it
    A[0] = A[0] // A[0, 0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:, 0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon_form_under_gcd_condition(A[1:, 1:])

    # we add first row and first (zero) column, and return
    return vstack([A[:1], hstack([A[1:, :1], B])])


class NoIntegerSolution(Exception):
    pass


def row_echelon_form_under_gcd_condition(A):
    """Return Row Echelon Form of matrix A enforcing that each linear equation has an integer solution following
    Theorem 11.32 in Aho, A. V., Lam, M. S., Sethi, R., &amp; Ullman, J. D. (2015). Compilers: Principles,
    techniques, and Tools. Pearson India Education Services.
    """

    def gcd_condition(A):
        """Check that gcd condition of linear Diophantine equation is satisfied"""
        if A[0, -1] % gcd(*A[0, :-1]) != 0:
            raise NoIntegerSolution()

    return generate_reduced_row_echelon_form(A, conditional_check=gcd_condition)
