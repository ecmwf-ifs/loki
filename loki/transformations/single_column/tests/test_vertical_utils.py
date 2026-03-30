# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :mod:`loki.transformations.single_column.vertical_utils`.
"""

import pytest

from loki import Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.ir import FindNodes, Loop, Conditional

from loki.transformations.single_column.vertical_utils import (
    _is_klev_plus_n,
    _extract_plus_n,
    _is_jk_eq_1,
    _is_zero_literal,
    _loop_upper_bound_expr,
    _loop_upper_bound_str,
    _collect_vertical_loops,
    _build_bounds_guard,
    _find_demotable_arrays,
    _merge_vertical_loops,
)


# --------------------------------------------------------------------------
# _is_klev_plus_n
# --------------------------------------------------------------------------


class TestIsKlevPlusN:
    """Tests for the _is_klev_plus_n helper."""

    def test_expression_klev_plus_1(self):
        """KLEV + 1 should return True."""
        klev = sym.Variable(name='klev')
        expr = sym.Sum((klev, sym.IntLiteral(1)))
        assert _is_klev_plus_n(expr, 'klev') is True

    def test_expression_klev_plus_3(self):
        """KLEV + 3 should return True."""
        klev = sym.Variable(name='KLEV')
        expr = sym.Sum((klev, sym.IntLiteral(3)))
        assert _is_klev_plus_n(expr, 'klev') is True

    def test_expression_plain_klev(self):
        """Plain KLEV (no +N) should return False."""
        klev = sym.Variable(name='klev')
        assert _is_klev_plus_n(klev, 'klev') is False

    def test_expression_different_var(self):
        """NFOO + 1 should return False when vertical_size is klev."""
        nfoo = sym.Variable(name='nfoo')
        expr = sym.Sum((nfoo, sym.IntLiteral(1)))
        assert _is_klev_plus_n(expr, 'klev') is False

    def test_string_klev_plus_1(self):
        """String fallback: 'klev+1' should return True."""
        assert _is_klev_plus_n('klev+1', 'klev') is True

    def test_string_plain_klev(self):
        """String fallback: 'klev' should return False."""
        assert _is_klev_plus_n('klev', 'klev') is False

    def test_none_returns_false(self):
        """None should return False."""
        assert _is_klev_plus_n(None, 'klev') is False


# --------------------------------------------------------------------------
# _extract_plus_n
# --------------------------------------------------------------------------


class TestExtractPlusN:
    """Tests for the _extract_plus_n helper."""

    def test_expression_klev_plus_1(self):
        klev = sym.Variable(name='klev')
        expr = sym.Sum((klev, sym.IntLiteral(1)))
        assert _extract_plus_n(expr, 'klev') == 1

    def test_expression_klev_plus_5(self):
        klev = sym.Variable(name='KLEV')
        expr = sym.Sum((klev, sym.IntLiteral(5)))
        assert _extract_plus_n(expr, 'KLEV') == 5

    def test_expression_plain_klev(self):
        klev = sym.Variable(name='klev')
        assert _extract_plus_n(klev, 'klev') == 0

    def test_string_klev_plus_2(self):
        assert _extract_plus_n('klev+2', 'klev') == 2

    def test_string_plain_klev(self):
        assert _extract_plus_n('klev', 'klev') == 0


# --------------------------------------------------------------------------
# _is_jk_eq_1
# --------------------------------------------------------------------------


class TestIsJkEq1:
    """Tests for the _is_jk_eq_1 helper."""

    def test_jk_eq_1(self):
        """JK == 1 should return True."""
        jk = sym.Variable(name='JK')
        expr = sym.Comparison(operator='==', left=jk, right=sym.IntLiteral(1))
        assert _is_jk_eq_1(expr, 'jk') is True

    def test_1_eq_jk(self):
        """1 == JK should return True (commutative)."""
        jk = sym.Variable(name='JK')
        expr = sym.Comparison(operator='==', left=sym.IntLiteral(1), right=jk)
        assert _is_jk_eq_1(expr, 'jk') is True

    def test_jk_eq_2(self):
        """JK == 2 should return False."""
        jk = sym.Variable(name='JK')
        expr = sym.Comparison(operator='==', left=jk, right=sym.IntLiteral(2))
        assert _is_jk_eq_1(expr, 'jk') is False

    def test_jk_gt_1(self):
        """JK > 1 should return False (wrong operator)."""
        jk = sym.Variable(name='JK')
        expr = sym.Comparison(operator='>', left=jk, right=sym.IntLiteral(1))
        assert _is_jk_eq_1(expr, 'jk') is False

    def test_different_var(self):
        """JM == 1 should return False when loop_var is JK."""
        jm = sym.Variable(name='JM')
        expr = sym.Comparison(operator='==', left=jm, right=sym.IntLiteral(1))
        assert _is_jk_eq_1(expr, 'jk') is False

    def test_case_insensitive(self):
        """jk == 1 with loop_var 'JK' should return True."""
        jk = sym.Variable(name='jk')
        expr = sym.Comparison(operator='==', left=jk, right=sym.IntLiteral(1))
        assert _is_jk_eq_1(expr, 'JK') is True


# --------------------------------------------------------------------------
# _is_zero_literal
# --------------------------------------------------------------------------


class TestIsZeroLiteral:
    """Tests for the _is_zero_literal helper."""

    def test_int_zero(self):
        assert _is_zero_literal(sym.IntLiteral(0)) is True

    def test_int_nonzero(self):
        assert _is_zero_literal(sym.IntLiteral(1)) is False

    def test_float_zero(self):
        assert _is_zero_literal(sym.FloatLiteral(0.0)) is True

    def test_float_nonzero(self):
        assert _is_zero_literal(sym.FloatLiteral(1.5)) is False

    def test_float_zero_string(self):
        """FloatLiteral stores value as string."""
        assert _is_zero_literal(sym.FloatLiteral('0.0')) is True

    def test_variable_returns_false(self):
        """A variable is not a zero literal."""
        v = sym.Variable(name='x')
        assert _is_zero_literal(v) is False


# --------------------------------------------------------------------------
# _build_bounds_guard
# --------------------------------------------------------------------------


class TestBuildBoundsGuard:
    """Tests for _build_bounds_guard."""

    def test_basic_guard(self):
        """Should produce JK >= lower .AND. JK <= upper."""
        jk = sym.Variable(name='JK')
        lower = sym.IntLiteral(1)
        upper = sym.Variable(name='KLEV')
        guard = _build_bounds_guard(jk, lower, upper)
        assert isinstance(guard, sym.LogicalAnd)
        guard_str = str(guard).lower()
        assert '>=' in guard_str
        assert '<=' in guard_str


# --------------------------------------------------------------------------
# _loop_upper_bound_expr / _loop_upper_bound_str (via parsed Fortran)
# --------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_upper_bound(frontend):
    """_loop_upper_bound_expr/str should return the upper bound of a loop."""
    fcode = """
    subroutine test_upper(klev)
      implicit none
      integer, intent(in) :: klev
      integer :: jk
      real :: x

      do jk = 1, klev
        x = real(jk)
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = [l for l in FindNodes(Loop).visit(routine.body)
             if l.variable.name.lower() == 'jk']
    assert len(loops) == 1

    expr = _loop_upper_bound_expr(loops[0])
    assert expr is not None
    assert str(expr).strip().lower() == 'klev'

    s = _loop_upper_bound_str(loops[0])
    assert s == 'klev'


# --------------------------------------------------------------------------
# _collect_vertical_loops (via parsed Fortran)
# --------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends())
def test_collect_vertical_loops_basic(frontend):
    """Should find top-level JK loops in source order."""
    fcode = """
    subroutine test_collect(klev)
      implicit none
      integer, intent(in) :: klev
      integer :: jk
      real :: a, b

      do jk = 1, klev
        a = real(jk)
      end do

      b = 0.0

      do jk = 1, klev
        b = b + real(jk)
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    result = _collect_vertical_loops(routine.body, 'jk')
    assert len(result) == 2
    # Each entry is (loop, cond_wrapper)
    for loop, cond in result:
        assert isinstance(loop, Loop)
        assert loop.variable.name.lower() == 'jk'
        assert cond is None


@pytest.mark.parametrize('frontend', available_frontends())
def test_collect_vertical_loops_conditional_wrapper(frontend):
    """JK loop inside a Conditional should be returned with the wrapper."""
    fcode = """
    subroutine test_collect_cond(klev, flag)
      implicit none
      integer, intent(in) :: klev
      logical, intent(in) :: flag
      integer :: jk
      real :: a

      if (flag) then
        do jk = 1, klev
          a = real(jk)
        end do
      end if
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    result = _collect_vertical_loops(routine.body, 'jk')
    assert len(result) == 1
    loop, cond = result[0]
    assert isinstance(loop, Loop)
    assert cond is not None
    assert isinstance(cond, Conditional)


@pytest.mark.parametrize('frontend', available_frontends())
def test_collect_vertical_loops_skips_non_jk(frontend):
    """A non-JK loop should not be returned."""
    fcode = """
    subroutine test_non_jk(n)
      implicit none
      integer, intent(in) :: n
      integer :: i
      real :: a

      do i = 1, n
        a = real(i)
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    result = _collect_vertical_loops(routine.body, 'jk')
    assert len(result) == 0


# --------------------------------------------------------------------------
# _find_demotable_arrays (via parsed Fortran)
# --------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends())
def test_find_demotable_simple(frontend):
    """A local KLEV array accessed only at offset 0 inside a JK loop
    should be demotable."""
    fcode = """
    subroutine test_demote(nlon, klev, result_out)
      implicit none
      integer, intent(in) :: nlon, klev
      real, intent(out) :: result_out(nlon, klev)
      integer :: jl, jk
      real :: tmp(nlon, klev)

      do jk = 1, klev
        do jl = 1, nlon
          tmp(jl, jk) = real(jk)
          result_out(jl, jk) = tmp(jl, jk) * 2.0
        end do
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = [l for l in FindNodes(Loop).visit(routine.body)
             if l.variable.name.lower() == 'jk']
    assert len(loops) == 1

    demotable = _find_demotable_arrays(routine, 'jk', 'klev', loops[0])
    assert 'tmp' in demotable


@pytest.mark.parametrize('frontend', available_frontends())
def test_find_demotable_argument_not_demoted(frontend):
    """An argument array should never be demotable."""
    fcode = """
    subroutine test_no_demote(nlon, klev, arr)
      implicit none
      integer, intent(in) :: nlon, klev
      real, intent(inout) :: arr(nlon, klev)
      integer :: jl, jk

      do jk = 1, klev
        do jl = 1, nlon
          arr(jl, jk) = arr(jl, jk) + 1.0
        end do
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = [l for l in FindNodes(Loop).visit(routine.body)
             if l.variable.name.lower() == 'jk']
    demotable = _find_demotable_arrays(routine, 'jk', 'klev', loops[0])
    assert 'arr' not in demotable


@pytest.mark.parametrize('frontend', available_frontends())
def test_find_demotable_nonzero_offset_excluded(frontend):
    """A local array accessed at JK-1 should NOT be demotable."""
    fcode = """
    subroutine test_no_demote_offset(nlon, klev, out)
      implicit none
      integer, intent(in) :: nlon, klev
      real, intent(out) :: out(nlon, klev)
      integer :: jl, jk
      real :: tmp(nlon, klev)

      do jk = 1, klev
        do jl = 1, nlon
          tmp(jl, jk) = real(jk)
        end do
      end do

      do jk = 2, klev
        do jl = 1, nlon
          out(jl, jk) = tmp(jl, jk - 1)
        end do
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = [l for l in FindNodes(Loop).visit(routine.body)
             if l.variable.name.lower() == 'jk']
    demotable = _find_demotable_arrays(routine, 'jk', 'klev', loops[0])
    assert 'tmp' not in demotable


# --------------------------------------------------------------------------
# _merge_vertical_loops (via parsed Fortran)
# --------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_vertical_loops_basic(frontend):
    """Two JK loops should merge into one with IF guards."""
    fcode = """
    subroutine test_merge(klev)
      implicit none
      integer, intent(in) :: klev
      integer :: jk
      real :: a, b

      do jk = 1, klev
        a = real(jk)
      end do

      do jk = 1, klev
        b = real(jk) * 2.0
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Before: 2 JK loops
    loops_before = [l for l in FindNodes(Loop).visit(routine.body)
                    if l.variable.name.lower() == 'jk']
    assert len(loops_before) == 2

    merged = _merge_vertical_loops(routine, 'jk', 'klev')
    assert merged is not None

    # After: 1 JK loop
    loops_after = [l for l in FindNodes(Loop).visit(routine.body)
                   if l.variable.name.lower() == 'jk']
    assert len(loops_after) == 1

    # The merged loop should contain IF guards
    conds = FindNodes(Conditional).visit(loops_after[0].body)
    assert len(conds) >= 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_vertical_loops_different_bounds(frontend):
    """Merging loops with KLEV and KLEV+1 bounds should use KLEV+1."""
    fcode = """
    subroutine test_merge_bounds(klev)
      implicit none
      integer, intent(in) :: klev
      integer :: jk
      real :: a, b

      do jk = 1, klev
        a = real(jk)
      end do

      do jk = 1, klev + 1
        b = real(jk)
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    merged = _merge_vertical_loops(routine, 'jk', 'klev')
    assert merged is not None

    # The merged loop upper bound should be KLEV + 1
    upper_str = str(merged.bounds.children[1]).strip().lower()
    assert 'klev' in upper_str
    assert '1' in upper_str


@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_no_loops(frontend):
    """If there are no JK loops, merge should return None."""
    fcode = """
    subroutine test_no_loops(n)
      implicit none
      integer, intent(in) :: n
      integer :: i
      real :: a

      do i = 1, n
        a = real(i)
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    result = _merge_vertical_loops(routine, 'jk', 'klev')
    assert result is None
