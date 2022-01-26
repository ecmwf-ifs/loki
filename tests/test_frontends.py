from pathlib import Path
import pytest

from conftest import jit_compile, clean_test, available_frontends
from loki import Module, FindNodes, Allocation, Deallocation


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends())
def test_check_alloc_opts(here, frontend):
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
"""

    # Parse the source and validate the IR
    module = Module.from_source(fcode, frontend=frontend)

    allocations = FindNodes(Allocation).visit(module['check_alloc_source'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is not None for alloc in allocations)
    assert all(alloc.status_var is None for alloc in allocations)

    allocations = FindNodes(Allocation).visit(module['alloc_deferred'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is None for alloc in allocations)
    assert allocations[0].status_var is not None
    assert allocations[1].status_var is None

    deallocs = FindNodes(Deallocation).visit(module['free_deferred'].body)
    assert len(deallocs) == 2
    assert deallocs[0].status_var is not None
    assert deallocs[1].status_var is None

    # Sanity check for the backend
    assert module.to_fortran().lower().count(', stat=stat') == 2

    # Generate Fortran and test it
    filepath = here/(f'frontends_check_alloc_{frontend}.f90')
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

    clean_test(filepath)
