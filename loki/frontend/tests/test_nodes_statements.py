# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for statement-like IR nodes.
"""

import pytest

from loki import Module, Subroutine
from loki.jit_build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
def test_check_alloc_opts(tmp_path, frontend):
    """
    Test the use of SOURCE and STAT in allocate
    """

    fcode = """
module alloc_mod
    integer, parameter :: jprb = selected_real_kind(13,300)

    type explicit
        real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
        real(kind=jprb) :: red_herring
    end type explicit

    type deferred
        real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
        real(kind=jprb), allocatable :: red_herring
    end type deferred

contains

    subroutine alloc_deferred(item)
        type(deferred), intent(inout) :: item
        integer :: stat
        allocate(item%vector(3), stat=stat)
        allocate(item%matrix(3, 3))
    end subroutine alloc_deferred

    subroutine free_deferred(item)
        type(deferred), intent(inout) :: item
        integer :: stat
        deallocate(item%vector, stat=stat)
        deallocate(item%matrix)
    end subroutine free_deferred

    subroutine check_alloc_source(item, item2)
        type(explicit), intent(inout) :: item
        type(deferred), intent(inout) :: item2
        real(kind=jprb), allocatable :: vector(:), vector2(:)

        allocate(vector, source=item%vector)
        vector(:) = vector(:) + item%scalar
        item%vector(:) = vector(:)

        allocate(vector2, source=item2%vector)  ! Try mold here when supported by fparser
        vector2(:) = item2%scalar
        item2%vector(:) = vector2(:)
    end subroutine check_alloc_source
end module alloc_mod
    """.strip()

    # Parse the source and validate the IR
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    allocations = FindNodes(ir.Allocation).visit(module['check_alloc_source'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is not None for alloc in allocations)
    assert all(alloc.status_var is None for alloc in allocations)

    allocations = FindNodes(ir.Allocation).visit(module['alloc_deferred'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is None for alloc in allocations)
    assert allocations[0].status_var is not None
    assert allocations[1].status_var is None

    deallocs = FindNodes(ir.Deallocation).visit(module['free_deferred'].body)
    assert len(deallocs) == 2
    assert deallocs[0].status_var is not None
    assert deallocs[1].status_var is None

    # Sanity check for the backend
    assert module.to_fortran().lower().count(', stat=stat') == 2

    # Generate Fortran and test it
    filepath = tmp_path/(f'frontends_check_alloc_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='alloc_mod')

    item = mod.explicit()
    item.scalar = 1.
    item.vector[:] = 1.

    item2 = mod.deferred()
    mod.alloc_deferred(item2)
    item2.scalar = 2.
    item2.vector[:] = -1.

    mod.check_alloc_source(item, item2)
    assert (item.vector == 2.).all()
    assert (item2.vector == 2.).all()
    mod.free_deferred(item2)


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does not like intrinsic shading for member functions!')]
))
def test_intrinsic_shadowing(tmp_path, frontend):
    """
    Test that locally defined functions that shadow intrinsics are handled.
    """
    fcode_algebra = """
module algebra_mod
implicit none
contains
  function dot_product(a, b) result(c)
    real(kind=8), intent(inout) :: a(:), b(:)
    real(kind=8) :: c
  end function dot_product

  function min(x, y)
    real(kind=8), intent(in) :: x, y
    real(kind=8) :: min

    min = y
    if (x < y) min = x
  end function min
end module algebra_mod
"""

    fcode = """
module test_intrinsics_mod
use algebra_mod, only: dot_product
implicit none

contains

  subroutine test_intrinsics(a, b, c, d)
    use algebra_mod, only: min
    implicit none
    real(kind=8), intent(inout) :: a(:), b(:)
    real(kind=8) :: c, d, e

    c = dot_product(a, b)
    d = max(c, a(1))
    e = min(c, a(1))

  contains

    function max(x, y)
      real(kind=8), intent(in) :: x, y
      real(kind=8) :: max

      max = y
      if (x > y) max = x
    end function max
  end subroutine test_intrinsics
end module test_intrinsics_mod
"""
    algebra = Module.from_source(fcode_algebra, frontend=frontend, xmods=[tmp_path])
    module = Module.from_source(
        fcode, definitions=algebra, frontend=frontend, xmods=[tmp_path]
    )
    routine = module['test_intrinsics']

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3

    assert isinstance(assigns[0].rhs.function, sym.ProcedureSymbol)
    assert not assigns[0].rhs.function.type.is_intrinsic
    assert assigns[0].rhs.function.type.dtype.procedure == algebra['dot_product']

    assert isinstance(assigns[1].rhs.function, sym.ProcedureSymbol)
    assert not assigns[1].rhs.function.type.is_intrinsic
    assert assigns[1].rhs.function.type.dtype.procedure == routine.members[0]

    assert isinstance(assigns[2].rhs.function, sym.ProcedureSymbol)
    assert not assigns[2].rhs.function.type.is_intrinsic
    assert assigns[2].rhs.function.type.dtype.procedure == algebra['min']


@pytest.mark.parametrize('frontend', available_frontends())
def test_empty_print_statement(frontend):
    """
    Test if an empty print statement (PRINT *) is parsed correctly.
    """
    fcode = """
SUBROUTINE test_routine()
    IMPLICIT NONE
    print *
    ! Using single quotes to simplify the test comparison (see below)
    print *, 'test_text'
END SUBROUTINE test_routine
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    print_stmts = [
        intr for intr in FindNodes(ir.Intrinsic).visit(routine.ir)
        if 'print' in intr.text.lower()
    ]
    assert print_stmts[0].text.lower() == 'print *'
    # NOTE: OMNI always uses single quotes ('') to represent string data in PRINT statements
    #       while fparser will mimic the quotes used in the parsed source code
    assert print_stmts[1].text.lower() == "print *, 'test_text'"
