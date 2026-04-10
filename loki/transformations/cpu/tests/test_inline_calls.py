# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`InlineCallSiteForVectorisation` (T5).
"""

import pytest

from loki import Dimension, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, Loop, CallStatement, Assignment, Intrinsic
from loki.backend import fgen

from loki.transformations.cpu.inline_calls import InlineCallSiteForVectorisation


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


# -----------------------------------------------------------------
# Basic inlining of a member procedure
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_in_horizontal_loop(frontend, horizontal):
    """
    A small member subroutine called inside a horizontal loop is
    inlined at the call site.
    """
    fcode = """
subroutine test_inline(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    call small_helper(za(jl), zb(jl))
  end do

contains

  subroutine small_helper(a, b)
    real, intent(in) :: a
    real, intent(inout) :: b
    b = a * 2.0 + 1.0
  end subroutine

end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InlineCallSiteForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # The call should be gone
    loops = FindNodes(Loop).visit(routine.body)
    calls_in_loop = FindNodes(CallStatement).visit(loops[0].body)
    assert len(calls_in_loop) == 0

    # The callee's assignment should now be in the loop body
    code = fgen(routine)
    # The inlined code should contain the computation
    assigns = FindNodes(Assignment).visit(loops[0].body)
    assert len(assigns) >= 1


# -----------------------------------------------------------------
# Call outside horizontal loop (should NOT be inlined)
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_call_outside_loop_not_inlined(frontend, horizontal):
    """
    A call to a member procedure that is NOT inside a horizontal
    loop should be left untouched.
    """
    fcode = """
subroutine test_no_inline(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)

  call small_helper(za(1), zb(1))

contains

  subroutine small_helper(a, b)
    real, intent(in) :: a
    real, intent(inout) :: b
    b = a * 2.0
  end subroutine

end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InlineCallSiteForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # The call should still be there
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1


# -----------------------------------------------------------------
# Callee with I/O (should NOT be inlined)
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_callee_with_io_not_inlined(frontend, horizontal):
    """
    A member subroutine containing WRITE should NOT be inlined.
    """
    fcode = """
subroutine test_io_callee(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    call helper_with_io(za(jl), zb(jl))
  end do

contains

  subroutine helper_with_io(a, b)
    real, intent(in) :: a
    real, intent(inout) :: b
    b = a * 2.0
    write(*, *) 'Debug: ', b
  end subroutine

end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InlineCallSiteForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Call should still be there (not inlined due to I/O)
    loops = FindNodes(Loop).visit(routine.body)
    calls_in_loop = FindNodes(CallStatement).visit(loops[0].body)
    assert len(calls_in_loop) == 1


# -----------------------------------------------------------------
# Callee with nested calls (should NOT be inlined)
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_callee_with_nested_call_not_inlined(frontend, horizontal):
    """
    A member subroutine that itself contains a CALL should NOT be
    inlined (the nested call would remain opaque).
    """
    fcode = """
subroutine test_nested_call(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    call helper_with_call(za(jl), zb(jl))
  end do

contains

  subroutine helper_with_call(a, b)
    real, intent(in) :: a
    real, intent(inout) :: b
    b = a * 2.0
    call external_sub(b)
  end subroutine

end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InlineCallSiteForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Call should still be there (not inlined due to nested call)
    loops = FindNodes(Loop).visit(routine.body)
    calls_in_loop = FindNodes(CallStatement).visit(loops[0].body)
    assert len(calls_in_loop) == 1


# -----------------------------------------------------------------
# Driver role skip
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_driver_role_skipped(frontend, horizontal):
    """
    When role='driver', no transformation is applied.
    """
    fcode = """
subroutine test_driver_inline(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    call small_helper(za(jl), zb(jl))
  end do

contains

  subroutine small_helper(a, b)
    real, intent(in) :: a
    real, intent(inout) :: b
    b = a * 2.0
  end subroutine

end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InlineCallSiteForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='driver')

    # Call should still be there
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1


# -----------------------------------------------------------------
# Multiple calls in same loop
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_multiple_calls_inlined(frontend, horizontal):
    """
    Multiple calls to different member procedures in the same loop
    are all inlined.
    """
    fcode = """
subroutine test_multi_inline(kidia, kfdia, klon, za, zb, zc)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon), zc(klon)
  integer :: jl

  do jl = kidia, kfdia
    call helper_a(za(jl), zb(jl))
    call helper_b(za(jl), zc(jl))
  end do

contains

  subroutine helper_a(a, b)
    real, intent(in) :: a
    real, intent(inout) :: b
    b = a * 2.0
  end subroutine

  subroutine helper_b(a, c)
    real, intent(in) :: a
    real, intent(inout) :: c
    c = a + 1.0
  end subroutine

end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InlineCallSiteForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # All calls should be gone
    loops = FindNodes(Loop).visit(routine.body)
    calls_in_loop = FindNodes(CallStatement).visit(loops[0].body)
    assert len(calls_in_loop) == 0
