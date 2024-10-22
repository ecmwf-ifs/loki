# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Dimension, FindNodes, Loop
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.scope import Scope, SymbolAttributes
from loki.types import BasicType


def test_dimension_properties():
    """
    Test that :any:`Dimension` objects store the correct strings.
    """
    scope = Scope()
    type_int = SymbolAttributes(dtype=BasicType.INTEGER)
    i = sym.Variable(name='i', type=type_int, scope=scope)
    n = sym.Variable(name='n', type=type_int, scope=scope)
    z = sym.Variable(name='z', type=type_int, scope=scope)
    one = sym.IntLiteral(1)
    two = sym.IntLiteral(2)

    simple = Dimension('simple', index='i', upper='n', size='z')
    assert simple.index == i
    assert simple.upper == n
    assert simple.size == z

    detail = Dimension(index='i', lower='1', upper='n', step='2', size='z')
    assert detail.index == i
    assert detail.lower == one
    assert detail.upper == n
    assert detail.step == two
    assert detail.size == z
    # Check derived properties
    assert detail.bounds == (one, n)
    assert detail.range == sym.LoopRange((1, n))

    multi = Dimension(
        index=('i', 'idx'), lower=('1', 'start'), upper=('n', 'end'), size='z'
    )
    assert multi.index == i
    assert multi.indices == (i, sym.Variable(name='idx', type=type_int, scope=scope))
    assert multi.lower == (one, sym.Variable(name='start', type=type_int, scope=scope))
    assert multi.upper == (n, sym.Variable(name='end', type=type_int, scope=scope))
    assert multi.size == z
    # Check derived properties
    assert multi.bounds ==  (one, n)
    assert multi.range == sym.LoopRange((1, n))


@pytest.mark.parametrize('frontend', available_frontends())
def test_dimension_size(frontend):
    """
    Test that :any:`Dimension` objects match size expressions.
    """
    fcode = """
subroutine test_dimension_size(nlon, start, end, arr)
  integer, intent(in) :: NLON, START, END
  real, intent(inout) :: arr(nlon)
  real :: local_arr(1:nlon)
  real :: range_arr(end-start+1)

  arr(start:end) = 1.
end subroutine test_dimension_size
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Create the dimension object and make sure we match all array sizes
    dim = Dimension(name='test_dim', size='nlon', bounds=('start', 'end'))
    assert routine.variable_map['nlon'] == dim.size
    assert routine.variable_map['arr'].dimensions[0] == dim.size

    # Ensure that aliased size expressions laos trigger right
    assert routine.variable_map['nlon'] in dim.size_expressions
    assert routine.variable_map['local_arr'].dimensions[0] in dim.size_expressions
    assert routine.variable_map['range_arr'].dimensions[0] in dim.size_expressions


@pytest.mark.parametrize('frontend', available_frontends())
def test_dimension_index_range(frontend):
    """
    Test that :any:`Dimension` objects match index and range expressions.
    """
    fcode = """
subroutine test_dimension_index(nlon, start, end, arr)
  integer, intent(in) :: NLON, START, END
  real, intent(inout) :: arr(nlon)
  integer :: I

  do i=start, end
    arr(I) = 1.
  end do
end subroutine test_dimension_index
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Create the dimension object and make sure we match all array sizes
    dim = Dimension(name='test_dim', index='i', bounds=('start', 'end'))
    assert routine.variable_map['i'] == dim.index

    assert FindNodes(Loop).visit(routine.body)[0].bounds == dim.range
    assert FindNodes(Loop).visit(routine.body)[0].bounds.lower == dim.bounds[0]
    assert FindNodes(Loop).visit(routine.body)[0].bounds.upper == dim.bounds[1]

    # Test the correct creation of horizontal dim with aliased bounds vars
    _ = Dimension('test_dim_alias', bounds_aliases=('bnds%start', 'bnds%end'))
