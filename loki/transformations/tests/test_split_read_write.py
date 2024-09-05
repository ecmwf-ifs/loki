# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Dimension, Subroutine
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, pragma_regions_attached,
    is_loki_pragma
)
from loki.frontend import available_frontends
from loki.transformations import SplitReadWriteTransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'), aliases=('nproma',))

@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk', aliases=('nlev',))


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_read_write(frontend, horizontal, vertical):
    """
    Test pragma-assisted splitting of reads and writes.
    """

    fcode = """
subroutine kernel(nlon, nz, start, end, n1, n2, n3, var0, var1, var2, nfre)
  implicit none

  integer, intent(in) :: nlon, nz, n1, n2, n3, start, end, nfre
  real, intent(inout) :: var0(nlon,nfre,6), var1(nlon, nz, 6), var2(nlon,nz)
  integer :: jl, jk, m

  !$loki split-read-write
  do jk = 1,nz
    do jl = start,end
       var1(jl, jk, n1) = var1(jl, jk, n1) + 1.
       var1(jl, jk, n1) = var1(jl, jk, n1) * 2.
       var1(jl, jk, n2) = var1(jl, jk, n2) + var1(jl, jk, n1)
       var2(jl, jk    ) = 0.
    end do
  end do
  print *, "a leaf node that shouldn't be copied"
  !$loki end split-read-write

  !.....should be transformed to........
  !!$loki split-read-write
  !  do jk=1,nz
  !    do jl=start,end
  !      loki_temp_0(jl, jk) = var1(jl, jk, n1) + 1.
  !      loki_temp_0(jl, jk) = loki_temp_0(jl, jk)*2.
  !      loki_temp_1(jl, jk) = var1(jl, jk, n2) + loki_temp_0(jl, jk)
  !      var2(jl, jk) = 0.
  !    end do
  !  end do
  !  print *, 'a leaf node that shouldn''t be copied'
  !  do jk=1,nz
  !    do jl=start,end
  !      var1(jl, jk, n1) = loki_temp_0(jl, jk)
  !      var1(jl, jk, n2) = loki_temp_1(jl, jk)
  !    end do
  !  end do
  !!$loki end split-read-write

  do m = 1,nfre
  !$loki split-read-write
     if( m < nfre/2 )then
        do jl = start,end
           var0(jl, m, n3) = var0(jl, m, n3) + 1.
        end do
     endif
  !$loki end split-read-write
  !.....should be transformed to........
  !!$loki split-read-write
  !  if (m < nfre / 2) then
  !   do jl=start,end
  !     loki_temp_2(jl) = var0(jl, m, n3) + 1.
  !   end do
  !  end if
  !  if (m < nfre / 2) then
  !    do jl=start,end
  !      var0(jl, m, n3) = loki_temp_2(jl)
  !    end do
  !  end if
  !!$loki end split-read-write
  end do

end subroutine kernel
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    SplitReadWriteTransformation(dimensions=(horizontal, vertical)).apply(routine)

    with pragma_regions_attached(routine):

        pragma_regions = FindNodes(ir.PragmaRegion).visit(routine.body)
        assert len(pragma_regions) == 2

        #=========== check first pragma region ==============#
        region = pragma_regions[0]
        assert is_loki_pragma(region.pragma, starts_with='split-read-write')

        # check that temporaries were declared
        assert 'loki_temp_0(nlon,nz)' in routine.variables
        assert 'loki_temp_1(nlon,nz)' in routine.variables

        # check correctly nested loops
        outer_loops = [l for l in FindNodes(ir.Loop).visit(region.body) if l.variable == 'jk']
        assert len(outer_loops) == 2
        for loop in outer_loops:
            _loops = FindNodes(ir.Loop).visit(loop.body)
            assert len(_loops) == 1
            assert _loops[0].variable == 'jl'

        # check simple assignment is only in first copy of region
        assert 'var2(jl,jk)' in FindVariables().visit(outer_loops[0])
        assert not 'var2(jl,jk)' in FindVariables().visit(outer_loops[1])

        # check print statement is only present in first copy of region
        assert len(FindNodes(ir.Intrinsic).visit(region)) == 1

        # check correctness of split reads
        assigns = FindNodes(ir.Assignment).visit(outer_loops[0].body)
        assert len(assigns) == 4
        assert assigns[0].lhs == assigns[1].lhs
        assert assigns[1].rhs == f'{assigns[0].lhs}*2.'
        assert assigns[2].lhs != assigns[0].lhs
        assert assigns[2].lhs.dimensions == assigns[0].lhs.dimensions
        assert f'{assigns[0].lhs}' in assigns[2].rhs

        # check correctness of split writes
        _assigns = FindNodes(ir.Assignment).visit(outer_loops[1].body)
        assert len(_assigns) == 2
        assert _assigns[0].lhs == 'var1(jl, jk, n1)'
        assert _assigns[1].lhs == 'var1(jl, jk, n2)'
        assert _assigns[0].rhs == assigns[0].lhs
        assert _assigns[1].rhs == assigns[2].lhs


        #=========== check second pragma region ==============#
        region = pragma_regions[1]
        assert is_loki_pragma(region.pragma, starts_with='split-read-write')

        conds = FindNodes(ir.Conditional).visit(region.body)
        assert len(conds) == 2

        # check that temporaries were declared
        assert 'loki_temp_2(nlon)' in routine.variables

        # check correctness of split reads
        assigns = FindNodes(ir.Assignment).visit(conds[0])
        assert len(assigns) == 1
        assert assigns[0].lhs == 'loki_temp_2(jl)'
        assert 'var0(jl, m, n3)' in assigns[0].rhs

        # check correctness of split writes
        assigns = FindNodes(ir.Assignment).visit(conds[1])
        assert len(assigns) == 1
        assert assigns[0].lhs == 'var0(jl, m, n3)'
        assert assigns[0].rhs == 'loki_temp_2(jl)'
