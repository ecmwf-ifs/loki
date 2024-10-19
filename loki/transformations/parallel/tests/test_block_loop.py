# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Module, Dimension
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes

from loki.transformations.parallel import (
    remove_block_loops, add_block_loops
)


@pytest.mark.parametrize('frontend', available_frontends())
def test_remove_block_loops(tmp_path, frontend):
    """
    A simple test for :any:`remove_block_loops`
    """
    fcode_type = """
module geom_mod
  type geom_type
    integer :: nproma, ngptot
  end type geom_type
end module geom_mod
"""

    fcode = """
subroutine test_remove_block_loop(ydgeom, npoints, nlev, arr)
  use geom_mod, only: geom_type
  implicit none
  type(geom_type), intent(in) :: ydgeom
  integer(kind=4), intent(in) :: npoints, nlev
  real(kind=8), intent(inout) :: arr(:,:,:)
  integer :: JKGLO, IBL, ICEND, JK, JL

  DO JKGLO=1,YDGEOM%NGPTOT,YDGEOM%NPROMA
    ICEND = MIN(YDGEOM%NPROMA, YDGEOM%NGPTOT - JKGLO + 1)
    IBL = (JKGLO - 1) / YDGEOM%NPROMA + 1

    CALL MY_KERNEL(ARR(:,:,IBL))
  END DO


  DO JKGLO=1,YDGEOM%NGPTOT,YDGEOM%NPROMA
    ICEND = MIN(YDGEOM%NPROMA, YDGEOM%NGPTOT - JKGLO + 1)
    IBL = (JKGLO - 1) / YDGEOM%NPROMA + 1

    DO JK=1, NLEV
      DO JL=1, NPOINTS
        ARR(JL, JK, IBL) = 42.0
      END DO
    END DO
  END DO
end subroutine test_remove_block_loop
"""
    _ = Module.from_source(fcode_type, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 4
    assert loops[0].variable == 'jkglo'
    assert loops[1].variable == 'jkglo'
    assert loops[2].variable == 'jk'
    assert loops[3].variable == 'jl'
    assert len(FindNodes(ir.Assignment).visit(loops[0].body)) == 2
    assert len(FindNodes(ir.Assignment).visit(loops[1].body)) == 3

    block = Dimension(
        'block', index=('jkglo', 'ibl'), step='YDGEOM%NPROMA',
        lower=('1', 'ICST'), upper=('YDGEOM%NGPTOT', 'ICEND')
    )
    remove_block_loops(routine, dimension=block)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 2
    assert loops[0].variable == 'jk'
    assert loops[1].variable == 'jl'
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_add_block_loops(frontend):
    """
    A simple test for :any:`add_block_loops`
    """
    fcode = """
subroutine test_add_block_loop(ydgeom, npoints, nlev, arr)
  integer(kind=4), intent(in) :: npoints, nlev
  real(kind=8), intent(inout) :: arr(:,:,:)
  integer :: JKGLO, IBL, ICEND

!$loki parallel
  call my_kernel(arr(:,:,ibl))
!$loki end parallel

!$loki parallel
  do jk=1, nlev
    do jl=1, npoints
      arr(jl, jk, ibl) = 42.0
    end do
  end do
!$loki end parallel
end subroutine test_add_block_loop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 2
    assert loops[0].variable == 'jk'
    assert loops[1].variable == 'jl'
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 1

    blocking = Dimension(
        name='block', index=('JKGLO', 'IBL'),
        lower=('1', 'ICST'),
        upper=('YDGEOM%NGPTOT', 'ICEND'),
        step='YDGEOM%NPROMA',
        size='YDGEOM%NGPBLKS',
    )

    add_block_loops(routine, dimension=blocking)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 4
    assert loops[0].variable == 'jkglo'
    assert loops[1].variable == 'jkglo'
    assert loops[2].variable == 'jk'
    assert loops[3].variable == 'jl'

    assigns1 = FindNodes(ir.Assignment).visit(loops[0].body)
    assert len(assigns1) == 2
    assert assigns1[0].lhs == 'ICEND'
    assert str(assigns1[0].rhs) == 'MIN(YDGEOM%NPROMA, YDGEOM%NGPTOT - JKGLO + 1)'
    assert assigns1[1].lhs == 'IBL'
    assert assigns1[1].rhs == '(JKGLO - 1) / YDGEOM%NPROMA + 1'

    assigns2 = FindNodes(ir.Assignment).visit(loops[1].body)
    assert len(assigns2) == 3
    assert assigns2[0].lhs == 'ICEND'
    assert str(assigns2[0].rhs) == 'MIN(YDGEOM%NPROMA, YDGEOM%NGPTOT - JKGLO + 1)'
    assert assigns2[1].lhs == 'IBL'
    assert assigns2[1].rhs == '(JKGLO - 1) / YDGEOM%NPROMA + 1'
    assert assigns2[2].lhs == 'arr(jl, jk, ibl)'
    assert assigns2[2].rhs == 42.0
