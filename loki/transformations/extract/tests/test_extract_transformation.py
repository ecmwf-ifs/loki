# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Module, Sourcefile
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes

from loki.transformations.extract import ExtractTransformation


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('outline_regions', [False, True])
@pytest.mark.parametrize('extract_internals', [False, True])
def test_extract_transformation_module(extract_internals, outline_regions, frontend):
    """
    Test basic subroutine extraction from marker pragmas in modules.
    """
    fcode = """
module test_extract_mod
implicit none
contains

subroutine outer(n, a, b)
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a, b(n)
  real(kind=8) :: x(n), y(n, n+1)
  integer :: i, j

  x(:) = a
  do i=1, n
    y(i,:) = b(i)
  end do

  !$loki outline name(test1)
  do i=1, n
    do j=1, n+1
      x(i) = x(i)  + y(i, j)
    end do
  end do
  !$loki end outline

  do i=1, n
    call plus_one(x, i=i)
  end do

contains
  subroutine plus_one(f, i)
    real(kind=8), intent(inout) :: f(:)
    integer, intent(in) :: i

    f(i) = f(i) + 1.0
  end subroutine plus_one
end subroutine outer
end module test_extract_mod
"""
    module = Module.from_source(fcode, frontend=frontend)

    ExtractTransformation(
        extract_internals=extract_internals, outline_regions=outline_regions
    ).apply(module)

    routines = tuple(r for r in module.contains.body if isinstance(r, Subroutine))
    assert len(routines) == 1 + (1 if extract_internals else 0) + (1 if outline_regions else 0)
    assert ('plus_one' in module) ==  extract_internals
    assert ('test1' in module) ==  outline_regions

    outer = module['outer']
    assert len(FindNodes(ir.CallStatement).visit(outer.body)) == (2 if outline_regions else 1)
    outer_internals = tuple(r for r in outer.contains.body if isinstance(r, Subroutine))
    assert len(outer_internals) == (0 if extract_internals else 1)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('outline_regions', [False, True])
@pytest.mark.parametrize('extract_internals', [False, True])
def test_extract_transformation_sourcefile(extract_internals, outline_regions, frontend):
    """
    Test internal procedure extraction and region outlining from subroutines.
    """
    fcode = """
subroutine outer(n, a, b)
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a, b(n)
  real(kind=8) :: x(n), y(n, n+1)
  integer :: i, j

  x(:) = a
  do i=1, n
    y(i,:) = b(i)
  end do

  !$loki outline name(test1)
  do i=1, n
    do j=1, n+1
      x(i) = x(i)  + y(i, j)
    end do
  end do
  !$loki end outline

  do i=1, n
    call plus_one(x, i=i)
  end do

contains
  subroutine plus_one(f, i)
    real(kind=8), intent(inout) :: f(:)
    integer, intent(in) :: i

    f(i) = f(i) + 1.0
  end subroutine plus_one
end subroutine outer
"""
    sourcefile = Sourcefile.from_source(fcode, frontend=frontend)

    ExtractTransformation(
        extract_internals=extract_internals, outline_regions=outline_regions
    ).apply(sourcefile)

    routines = tuple(r for r in sourcefile.ir.body if isinstance(r, Subroutine))
    assert len(routines) == 1 + (1 if extract_internals else 0) + (1 if outline_regions else 0)
    assert ('plus_one' in sourcefile) ==  extract_internals
    assert ('test1' in sourcefile) ==  outline_regions

    outer = sourcefile['outer']
    assert len(FindNodes(ir.CallStatement).visit(outer.body)) == (2 if outline_regions else 1)
    outer_internals = tuple(r for r in outer.contains.body if isinstance(r, Subroutine))
    assert len(outer_internals) == (0 if extract_internals else 1)
