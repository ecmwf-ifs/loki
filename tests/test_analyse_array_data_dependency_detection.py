# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np
from loki import Scope, parse_fparser_expression
from loki.analyse.analyse_array_data_dependency_detection import (
    has_data_dependency,
    HAVE_ORTOOLS,
    construct_affine_array_access_function_representation,
)

array = np.array

@pytest.mark.parametrize(
    "first_access_represenetation, second_access_represenetation, error_type",
    [
        # totaly empty tuple
        (tuple(), tuple(), ValueError),
        # tuple with two empty tuple --> correct form
        ((tuple(), tuple()), (tuple(), tuple()), ValueError),
        #
    ],
)
def test_has_data_dependency_false_inputs(
    first_access_represenetation, second_access_represenetation, error_type
):
    with pytest.raises(error_type):
        _ = has_data_dependency(
            first_access_represenetation, second_access_represenetation
        )

@pytest.mark.parametrize(
    "array_dimensions_expr, expected",
    [
        ("i-1", ([[1, 0]], [[-1]])),
        ("i,j", ([[1, 0], [0, 1]], [[0], [0]])),
        ("j,j+1", ([[0, 1], [0, 1]], [[0], [1]])),
        ("1,2", ([[0, 0], [0, 0]], [[1], [2]])),
        ("1,i,2*i+j", ([[0, 0], [1, 0], [2, 1]], [[1], [0], [0]])),
    ],
)
def test_access_function_creation(array_dimensions_expr, expected):
    scope = Scope()
    first = parse_fparser_expression(f"z({array_dimensions_expr})", scope)

    use_these_variables = ["i", "j"]

    access, variables = construct_affine_array_access_function_representation(
        first.dimensions, use_these_variables
    )

    assert np.array_equal(access.matrix, np.array(expected[0], dtype=np.dtype(int)))
    assert np.array_equal(access.vector, np.array(expected[1], dtype=np.dtype(int)))
    assert np.array_equal(variables, np.array(["i", "j"], dtype=np.dtype(object)))

expected_result = {
    "Example 11.29.1. from Compilers: Principles, Techniques, and Tools": True,
    "Example 11.30 from Compilers: Principles, Techniques, and Tools": False,
    "Example 11.35 from Compilers: Principles, Techniques, and Tools": not HAVE_ORTOOLS,
    "Example Anti Dependency": True,
    "Exercise 11.6.5 a) from Compilers: Principles, Techniques, and Tools": not HAVE_ORTOOLS,
    "Exercise 11.6.5 b) from Compilers: Principles, Techniques, and Tools": not HAVE_ORTOOLS,
    "Exercise 11.6.5 c) from Compilers: Principles, Techniques, and Tools": True,
    "Exercise 11.6.5 d) from Compilers: Principles, Techniques, and Tools": not HAVE_ORTOOLS,
}


examples = [
    (
        "Example 11.29.1. from Compilers: Principles, Techniques, and Tools",
        # Comparing Z[i] and Z[i-1]
        # for (i = 1; i <= 10; i++) {
        #  Z[i] = Z[i-1];
        # }
        #
        # first access
        (
            # iteration space representation
            (array([[1], [-1]]), array([[10], [-1]])),
            # function access representation
            (array([[1]]), array([[-1]])),
        ),
        # second access
        (
            # iteration space representation
            (array([[1], [-1]]), array([[10], [-1]])),
            # function access representation
            (array([[1]]), array([[0]])),
        ),
    ),
    # Example 11.29.2. from Compilers: Principles, Techniques, and Tools
    # is ignored, since it shows a comparison between an access with itself which does not make any sense
    #
    (
        "Example 11.30 from Compilers: Principles, Techniques, and Tools",
        # Comparing Z[2*i] and Z[2*i+1]
        # for (i = 1; i < 10; i++) {
        #  Z[2*i] = 10;
        # }
        # for (j = 1; j < 10; j++) {
        #  Z[2*j+1] = 20;
        # }
        #
        # first access
        (
            # iteration space representation
            (array([[1, 0], [-1, 0]]), array([[10], [-1]])),
            # function access representation
            (array([[2, 0]]), array([[0]])),
        ),
        # second access
        (
            # iteration space representation
            (array([[0, 1], [0, -1]]), array([[10], [-1]])),
            # function access representation
            (array([[0, 2]]), array([[1]])),
        ),
    ),
    (
        "Example 11.35 from Compilers: Principles, Techniques, and Tools",
        # Comparing Z[i,j] and Z[j+10,i+11]
        # for (i = 0; i <= 10; i++)
        #   for (j = 0; j <= 10; j++)
        #       Z[i,j] = Z[j+10,i+11];
        #
        # first access
        (
            # iteration space representation
            (
                array([[-1, 0], [0, -1], [1, 0], [0, 1]]),
                array([[0], [0], [10], [10]]),
            ),
            # function access representation
            (array([[1, 0], [0, 1]]), array([[0], [0]])),
        ),
        # second access
        (
            # iteration space representation
            (
                array([[-1, 0], [0, -1], [1, 0], [0, 1]]),
                array([[0], [0], [10], [10]]),
            ),
            # function access representation
            (array([[0, 1], [1, 0]]), array([[10], [11]])),
        ),
    ),
    (
        "Example Anti Dependency",
        # Comparing Z[i,j] and Z[j+10,i+11]
        # for (i = 2; i <= N; i++)
        #   Z[i] = Z[i-1] + 1;
        #
        # first access
        (
            # iteration space representation
            (
                array([[1, -1], [-1, 0]]),
                array([[0], [-2]]),
            ),
            # function access representation
            (array([[1, 0]]), array([[0]])),
        ),
        # second access
        (
            # iteration space representation
            (
                array([[1, -1], [-1, 0]]),
                array([[0], [-2]]),
            ),
            # function access representation
            (array([[1, 0]]), array([[-1]])),
        ),
    ),
    (
        "Exercise 11.6.5 a) from Compilers: Principles, Techniques, and Tools",
        # Comparing Z[i,j,k] = Z[i+100,j+100,k+100];
        # for (i=0; i<100; i++)
        # for (j=0; j<100; j++)
        #    for (k=0; k<100; k++)
        #        Z[i,j,k] = Z[i+100,j+100,k+100];
        #
        # first access
        (
            # iteration space representation
            (
                array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ),
                array([[99], [99], [99], [0], [0], [0]]),
            ),
            # function access representation
            (
                array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                array([[0], [0], [0]]),
            ),
        ),
        # second access
        (
            # iteration space representation
            (
                array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ),
                array([[99], [99], [99], [0], [0], [0]]),
            ),
            # function access representation
            (
                array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                array([[100], [100], [100]]),
            ),
        ),
    ),
    (
        "Exercise 11.6.5 b) from Compilers: Principles, Techniques, and Tools",
        # Comparing Z[i,j,k] = Z[j+100,k+100,i+100];
        # for (i=0; i<100; i++)
        # for (j=0; j<100; j++)
        #    for (k=0; k<100; k++)
        #        Z[i,j,k] = Z[j+100,k+100,i+100];
        #
        # first access
        (
            # iteration space representation
            (
                array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ),
                array([[99], [99], [99], [0], [0], [0]]),
            ),
            # function access representation
            (
                array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                array([[0], [0], [0]]),
            ),
        ),
        # second access
        (
            # iteration space representation
            (
                array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ),
                array([[99], [99], [99], [0], [0], [0]]),
            ),
            # function access representation
            (
                array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                array([[100], [100], [100]]),
            ),
        ),
    ),
    (
        "Exercise 11.6.5 c) from Compilers: Principles, Techniques, and Tools",
        # Comparing Z[i,j,k] = Z[j-50,k-50,i-50];
        # for (i=0; i<100; i++)
        #    for (j=0; j<100; j++)
        #        for (k=0; k<100; k++)
        #            Z[i,j,k] = Z[j-50,k-50,i-50];
        #
        # first access
        (
            # iteration space representation
            (
                array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ),
                array([[99], [99], [99], [0], [0], [0]]),
            ),
            # function access representation
            (
                array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                array([[0], [0], [0]]),
            ),
        ),
        # second access
        (
            # iteration space representation
            (
                array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ),
                array([[99], [99], [99], [0], [0], [0]]),
            ),
            # function access representation
            (
                array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                array([[-50], [-50], [-50]]),
            ),
        ),
    ),
    (
        "Exercise 11.6.5 d) from Compilers: Principles, Techniques, and Tools",
        # Comparing Z[i,j,k] = Z[i+99,k+100,j];
        # for (i=0; i<100; i++)
        #    for (j=0; j<100; j++)
        #        for (k=0; k<100; k++)
        #            Z[i,j,k] = Z[i+99,k+100,j];
        #
        # first access
        (
            # iteration space representation
            (
                array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ),
                array([[99], [99], [99], [0], [0], [0]]),
            ),
            # function access representation
            (
                array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                array([[0], [0], [0]]),
            ),
        ),
        # second access
        (
            # iteration space representation
            (
                array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                ),
                array([[99], [99], [99], [0], [0], [0]]),
            ),
            # function access representation
            (
                array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                array([[99], [100], [0]]),
            ),
        ),
    ),
]


@pytest.mark.parametrize(
    "name, first_access_represenetation, second_access_represenetation",
    examples,
)
def test_has_data_dependency(
    name, first_access_represenetation, second_access_represenetation
):
    assert expected_result[name] == has_data_dependency(
        first_access_represenetation, second_access_represenetation
    ), f"Test {name} failed!"
