# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np

from loki import Subroutine
from loki.jit_build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends

from loki.transformations.array_indexing.demote import demote_variables


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_demote_variables(tmp_path, frontend):
    """
    Apply variable demotion to a range of array variables.
    """
    fcode = """
subroutine transform_demote_variables(scalar, vector, matrix, n, m)
  implicit none
  integer, intent(in) :: n, m
  integer, intent(inout) :: scalar, vector(n), matrix(n, n)
  integer :: tmp_scalar, tmp_vector(n, m), tmp_matrix(n, m, n)
  integer :: jl, jk, jm

  do jl=1,n
    do jm=1,m
      tmp_vector(jl, jm) = scalar + jl
    end do
  end do

  do jm=1,m
    do jl=1,n
      scalar = jl
      vector(jl) = tmp_vector(jl, jm) + tmp_vector(jl, jm)

      do jk=1,n
        tmp_matrix(jk, jm, jl) = vector(jl) + jk
      end do
    end do
  end do

  do jk=1,n
    do jm=1,m
      do jl=1,n
        matrix(jk, jl) = tmp_matrix(jk, jm, jl)
      end do
    end do
  end do
end subroutine transform_demote_variables
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    n = 3
    m = 2
    scalar = np.array(0)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    function(scalar, vector, matrix, n, m)

    assert scalar == 3
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))

    # Do the variable demotion for all relevant array variables
    demote_variables(routine, ['tmp_vector', 'tmp_matrix'], ['m'])

    assert isinstance(routine.variable_map['scalar'], sym.Scalar)
    assert isinstance(routine.variable_map['vector'], sym.Array)
    assert routine.variable_map['vector'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['tmp_vector'], sym.Array)
    assert routine.variable_map['tmp_vector'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['matrix'], sym.Array)
    assert routine.variable_map['matrix'].shape == (routine.variable_map['n'], routine.variable_map['n'])
    assert isinstance(routine.variable_map['tmp_matrix'], sym.Array)
    assert routine.variable_map['tmp_matrix'].shape == (routine.variable_map['n'], routine.variable_map['n'])

    # Test promoted routine
    demoted_filepath = tmp_path/(f'{routine.name}_demoted_{frontend}.f90')
    demoted_function = jit_compile(routine, filepath=demoted_filepath, objname=routine.name)

    n = 3
    m = 2
    scalar = np.array(0)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    demoted_function(scalar, vector, matrix, n, m)

    assert scalar == 3
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))

    # Test that the transformation doesn't fail for scalar arguments and leaves the
    # IR unchanged
    demoted_fcode = routine.to_fortran()
    demote_variables(routine, ['jl'], ['m'])
    assert routine.to_fortran() == demoted_fcode


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_demote_dimension_arguments(tmp_path, frontend):
    """
    Apply variable demotion to array arguments defined with DIMENSION
    keywords.
    """
    fcode = """
subroutine transform_demote_dimension_arguments(vec1, vec2, matrix, n, m)
    implicit none
    integer, intent(in) :: n, m
    integer, dimension(n), intent(inout) :: vec1, vec2
    integer, dimension(n, m), intent(inout) :: matrix
    integer, dimension(n) :: vec_tmp
    integer :: i, j

    do i=1,n
        do j=1,m
        vec_tmp(i) = vec1(i) + vec2(i)
        matrix(i, j) = matrix(i, j) + vec_tmp(i)
        end do
    end do
end subroutine transform_demote_dimension_arguments
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    assert isinstance(routine.variable_map['vec1'], sym.Array)
    assert routine.variable_map['vec1'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['vec2'], sym.Array)
    assert routine.variable_map['vec2'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['matrix'], sym.Array)
    assert routine.variable_map['matrix'].shape == (routine.variable_map['n'], routine.variable_map['m'])

    n = 3
    m = 2
    vec1 = np.zeros(shape=(n,), order='F', dtype=np.int32) + 3
    vec2 = np.zeros(shape=(n,), order='F', dtype=np.int32) + 2
    matrix = np.zeros(shape=(n, m), order='F', dtype=np.int32) + 1
    function(vec1, vec2, matrix, n, m)

    assert np.all(vec1 == 3) and np.sum(vec1) == 9
    assert np.all(vec2 == 2) and np.sum(vec2) == 6
    assert np.all(matrix == 6) and np.sum(matrix) == 36

    demote_variables(routine, ['vec1', 'vec_tmp', 'matrix'], ['n'])

    assert isinstance(routine.variable_map['vec1'], sym.Scalar)
    assert isinstance(routine.variable_map['vec2'], sym.Array)
    assert routine.variable_map['vec2'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['matrix'], sym.Array)
    assert routine.variable_map['matrix'].shape == (routine.variable_map['m'],)

    # Test promoted routine
    demoted_filepath = tmp_path/(f'{routine.name}_demoted_{frontend}.f90')
    demoted_function = jit_compile(routine, filepath=demoted_filepath, objname=routine.name)

    n = 3
    m = 2
    vec1 = np.array(3)
    vec2 = np.zeros(shape=(n,), order='F', dtype=np.int32) + 2
    matrix = np.zeros(shape=(m, ), order='F', dtype=np.int32) + 1
    demoted_function(vec1, vec2, matrix, n, m)

    assert vec1 == 3
    assert np.all(vec2 == 2) and np.sum(vec2) == 6
    assert np.all(matrix == 16) and np.sum(matrix) == 32
