# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np

from loki import Subroutine, FindNodes, Loop
from loki.build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends

from loki.transformations.transform_region import region_hoist


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist(tmp_path, frontend):
    """
    A very simple hoisting example
    """
    fcode = """
subroutine transform_region_hoist(a, b, c)
  integer, intent(out) :: a, b, c

  a = 5

!$loki region-hoist target

  a = 1

!$loki region-hoist
  b = a
!$loki end region-hoist

  c = a + b
end subroutine transform_region_hoist
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 1 and b == 1 and c == 2

    # Apply transformation
    region_hoist(routine)
    hoisted_filepath = tmp_path/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    a, b, c = hoisted_function()
    assert a == 1 and b == 5 and c == 6


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist_inlined_pragma(tmp_path, frontend):
    """
    Hoisting when pragmas are potentially inlined into other nodes.
    """
    fcode = """
subroutine transform_region_hoist_inlined_pragma(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev), b(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl

!$loki region-hoist target

  do jl=1,klon
    a(jl, 1) = jl
  end do

  do jk=2,klev
    do jl=1,klon
      a(jl, jk) = a(jl, jk-1)
    end do
  end do

!$loki region-hoist
  do jk=1,klev
    b(1, jk) = jk
  end do
!$loki end region-hoist

  do jk=1,klev
    do jl=2,klon
      b(jl, jk) = b(jl-1, jk)
    end do
  end do
end subroutine transform_region_hoist_inlined_pragma
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    klon, klev = 32, 100
    ref_a = np.array([[jl + 1] * klev for jl in range(klon)], order='F')
    ref_b = np.array([[jk + 1 for jk in range(klev)] for _ in range(klon)], order='F')

    # Test the reference solution
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert [str(loop.variable) for loop in loops] == ['jl', 'jk', 'jl', 'jk', 'jk', 'jl']

    # Apply transformation
    region_hoist(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert [str(loop.variable) for loop in loops] == ['jk', 'jl', 'jk', 'jl', 'jk', 'jl']

    hoisted_filepath = tmp_path/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    hoisted_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist_multiple(tmp_path, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    """
    fcode = """
subroutine transform_region_hoist_multiple(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1

!$loki region-hoist target
!$loki region-hoist target group(some-group)

  a = a + 1
  a = a + 1
!$loki region-hoist group(some-group)
  a = a + 1
!$loki end region-hoist
  a = a + 1

!$loki region-hoist
  b = a
!$loki end region-hoist

!$loki region-hoist group(some-group)
  c = a + b
!$loki end region-hoist
end subroutine transform_region_hoist_multiple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 5 and b == 5 and c == 10

    # Apply transformation
    region_hoist(routine)
    hoisted_filepath = tmp_path/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    a, b, c = hoisted_function()
    assert a == 5 and b == 1 and c == 3


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist_collapse(tmp_path, frontend):
    """
    Use collapse with region-hoist.
    """
    fcode = """
subroutine transform_region_hoist_collapse(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev), b(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl

!$loki region-hoist target

  do jl=1,klon
    a(jl, 1) = jl
  end do

  do jk=2,klev
    do jl=1,klon
      a(jl, jk) = a(jl, jk-1)
    end do
  end do

  do jk=1,klev
!$loki region-hoist collapse(1)
    b(1, jk) = jk
!$loki end region-hoist
  end do

  do jk=1,klev
    do jl=2,klon
      b(jl, jk) = b(jl-1, jk)
    end do
  end do
end subroutine transform_region_hoist_collapse
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    klon, klev = 32, 100
    ref_a = np.array([[jl + 1] * klev for jl in range(klon)], order='F')
    ref_b = np.array([[jk + 1 for jk in range(klev)] for _ in range(klon)], order='F')

    # Test the reference solution
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert [str(loop.variable) for loop in loops] == ['jl', 'jk', 'jl', 'jk', 'jk', 'jl']

    # Apply transformation
    region_hoist(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 7
    assert [str(loop.variable) for loop in loops] == ['jk', 'jl', 'jk', 'jl', 'jk', 'jk', 'jl']

    hoisted_filepath = tmp_path/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    hoisted_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist_promote(tmp_path, frontend):
    """
    Use collapse with region-hoist.
    """
    fcode = """
subroutine transform_region_hoist_promote(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev), b(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl, b_tmp

!$loki region-hoist target

  do jl=1,klon
    a(jl, 1) = jl
  end do

  do jk=2,klev
    do jl=1,klon
      a(jl, jk) = a(jl, jk-1)
    end do
  end do

  do jk=1,4
    b(1, jk) = jk
  end do

  do jk=5,klev
!$loki region-hoist collapse(1) promote(b_tmp)
    b_tmp = jk + 1
!$loki end region-hoist
    b(1, jk) = b_tmp - 1
  end do

  do jk=1,klev
    do jl=2,klon
      b(jl, jk) = b(jl-1, jk)
    end do
  end do
end subroutine transform_region_hoist_promote
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    klon, klev = 32, 100
    ref_a = np.array([[jl + 1] * klev for jl in range(klon)], order='F')
    ref_b = np.array([[jk + 1 for jk in range(klev)] for _ in range(klon)], order='F')

    # Test the reference solution
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 7
    assert [str(loop.variable) for loop in loops] == ['jl', 'jk', 'jl', 'jk', 'jk', 'jk', 'jl']

    assert isinstance(routine.variable_map['b_tmp'], sym.Scalar)

    # Apply transformation
    region_hoist(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 8
    assert [str(loop.variable) for loop in loops] == ['jk', 'jl', 'jk', 'jl', 'jk', 'jk', 'jk', 'jl']

    b_tmp = routine.variable_map['b_tmp']
    assert isinstance(b_tmp, sym.Array) and len(b_tmp.type.shape) == 1
    assert str(b_tmp.type.shape[0]) == 'klev'

    hoisted_filepath = tmp_path/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    hoisted_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)
