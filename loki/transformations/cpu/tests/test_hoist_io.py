# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`HoistWriteFromLoop` (T3).
"""

import pytest

from loki import Dimension, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    FindNodes, Loop, Intrinsic, Assignment, Conditional
)
from loki.backend import fgen

from loki.transformations.cpu.hoist_io import HoistWriteFromLoop


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


# -----------------------------------------------------------------
# Basic WRITE hoisting
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_write_hoisted_from_loop(frontend, horizontal):
    """
    A WRITE inside a horizontal loop is replaced with a boolean flag,
    and the diagnostic WRITE is emitted after the loop.
    """
    fcode = """
subroutine test_write(kidia, kfdia, klon, zdiff, ztol)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdiff(klon), ztol
  integer :: jl

  do jl = kidia, kfdia
    if (zdiff(jl) > ztol) then
      write(*, *) 'WARNING: large diff at JL=', jl, zdiff(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = HoistWriteFromLoop(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # The WRITE should no longer be inside the loop
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    io_in_loop = [n for n in FindNodes(Intrinsic).visit(loops[0].body)
                  if 'WRITE' in n.text.upper() or 'PRINT' in n.text.upper()]
    assert len(io_in_loop) == 0

    # There should be a flag assignment inside the loop
    assigns_in_loop = FindNodes(Assignment).visit(loops[0].body)
    flag_assigns = [a for a in assigns_in_loop
                    if 'LLHOIST' in str(a.lhs).upper()]
    assert len(flag_assigns) >= 1

    # There should be a flag initialisation before the loop
    # and a conditional WRITE after the loop
    code = fgen(routine)
    assert 'LLHOIST_WARN_1' in code.upper()
    assert '.FALSE.' in code.upper()
    assert 'HOISTED' in code.upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_print_hoisted_from_loop(frontend, horizontal):
    """
    A PRINT statement inside a horizontal loop is also hoisted.
    """
    fcode = """
subroutine test_print(kidia, kfdia, klon, zval)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) < 0.0) then
      print *, 'Negative value at ', jl
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = HoistWriteFromLoop(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    io_in_loop = [n for n in FindNodes(Intrinsic).visit(loops[0].body)
                  if 'PRINT' in n.text.upper()]
    assert len(io_in_loop) == 0

    code = fgen(routine)
    assert 'LLHOIST_WARN_1' in code.upper()
    assert 'HOISTED' in code.upper()


# -----------------------------------------------------------------
# Loop without I/O (should NOT be modified)
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_no_io_loop_unchanged(frontend, horizontal):
    """
    A horizontal loop with no WRITE/PRINT is left untouched.
    """
    fcode = """
subroutine test_no_io(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    zb(jl) = za(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    code_before = fgen(routine)

    trafo = HoistWriteFromLoop(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    code_after = fgen(routine)
    # No flag variables should be introduced
    assert 'LLHOIST' not in code_after.upper()


# -----------------------------------------------------------------
# Multiple I/O statements in one loop
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_multiple_writes_in_loop(frontend, horizontal):
    """
    Multiple WRITE statements in a single loop are all replaced by
    a single flag, and a summary diagnostic is emitted after the loop.
    """
    fcode = """
subroutine test_multi_write(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (za(jl) < 0.0) then
      write(*, *) 'Negative za at ', jl
    end if
    if (zb(jl) < 0.0) then
      write(*, *) 'Negative zb at ', jl
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = HoistWriteFromLoop(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    io_in_loop = [n for n in FindNodes(Intrinsic).visit(loops[0].body)
                  if 'WRITE' in n.text.upper()]
    assert len(io_in_loop) == 0

    code = fgen(routine)
    assert 'LLHOIST_WARN_1' in code.upper()


# -----------------------------------------------------------------
# Non-horizontal loop I/O is not touched
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_non_horizontal_loop_io_untouched(frontend, horizontal):
    """
    A WRITE in a non-horizontal loop should NOT be hoisted.
    """
    fcode = """
subroutine test_vert_io(klev)
  implicit none
  integer, intent(in) :: klev
  integer :: jk

  do jk = 1, klev
    write(*, *) 'Level ', jk
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = HoistWriteFromLoop(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # I/O should still be in the loop
    loops = FindNodes(Loop).visit(routine.body)
    io_in_loop = [n for n in FindNodes(Intrinsic).visit(loops[0].body)
                  if 'WRITE' in n.text.upper()]
    assert len(io_in_loop) == 1
    assert 'LLHOIST' not in fgen(routine).upper()


# -----------------------------------------------------------------
# Driver role skip
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_driver_role_skipped(frontend, horizontal):
    """
    When role='driver', no transformation is applied.
    """
    fcode = """
subroutine test_driver_io(kidia, kfdia, klon, za)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    write(*, *) 'Value: ', za(jl)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = HoistWriteFromLoop(horizontal=horizontal)
    trafo.apply(routine, role='driver')

    # I/O should still be in the loop
    loops = FindNodes(Loop).visit(routine.body)
    io_in_loop = [n for n in FindNodes(Intrinsic).visit(loops[0].body)
                  if 'WRITE' in n.text.upper()]
    assert len(io_in_loop) == 1


# -----------------------------------------------------------------
# Write preserved alongside computation
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_write_with_computation(frontend, horizontal):
    """
    A loop with both I/O and computation: the computation stays,
    the I/O is hoisted, and the flag assignment replaces the WRITE.
    """
    fcode = """
subroutine test_mixed(kidia, kfdia, klon, zdiff, zresult, ztol)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdiff(klon), ztol
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zdiff(jl) > ztol) then
      write(*, *) 'WARNING: large diff at JL=', jl, zdiff(jl)
    end if
    zresult(jl) = zdiff(jl) * 0.5
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = HoistWriteFromLoop(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # I/O is gone from the loop
    io_in_loop = [n for n in FindNodes(Intrinsic).visit(loops[0].body)
                  if 'WRITE' in n.text.upper()]
    assert len(io_in_loop) == 0

    # Computation assignment is still in the loop
    assigns = FindNodes(Assignment).visit(loops[0].body)
    comp_assigns = [a for a in assigns
                    if 'LLHOIST' not in str(a.lhs).upper()]
    assert len(comp_assigns) >= 1

    # Post-loop diagnostic exists
    code = fgen(routine)
    assert 'HOISTED' in code.upper()
    # The computation is preserved
    assert 'ZRESULT' in code.upper()
