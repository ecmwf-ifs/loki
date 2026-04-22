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
from loki.ir import FindNodes, Loop, Conditional, Assignment
from loki.backend import fgen

from loki.types import BasicType, SymbolAttributes
from loki.transformations.single_column.vertical_utils import (
    _is_klev_plus_n,
    _extract_plus_n,
    _is_jk_eq_1,
    _is_zero_literal,
    _loop_upper_bound_expr,
    _loop_upper_bound_str,
    _collect_vertical_loops,
    _collect_call_arg_names,
    _collect_refs_outside_loops,
    _collect_outside_refs_for_array,
    _build_bounds_guard,
    _find_demotable_arrays,
    _make_zero_literal,
    _merge_vertical_loops,
    _remove_self_assignments,
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
        # Check structural properties of the guard: (JK >= 1) .AND. (JK <= KLEV)
        left, right = guard.children
        assert left.operator == '>='
        assert left.left == 'jk'
        assert left.right == 1
        assert right.operator == '<='
        assert right.left == 'jk'
        assert right.right == 'klev'


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
    assert expr == 'klev'

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

    demotable = _find_demotable_arrays(routine, 'jk', 'klev')
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
    demotable = _find_demotable_arrays(routine, 'jk', 'klev')
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
    demotable = _find_demotable_arrays(routine, 'jk', 'klev')
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
    assert merged.bounds.children[1] == 'klev + 1'


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


@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_vertical_loops_cond_else_body(frontend):
    """A JK loop wrapped in IF/ELSE should preserve the else branch
    after merging."""
    fcode = """
    subroutine test_merge_cond_else(klev, flag)
      implicit none
      integer, intent(in) :: klev
      logical, intent(in) :: flag
      integer :: jk
      real :: a, b

      if (flag) then
        do jk = 1, klev
          a = real(jk)
        end do
      else
        b = 0.0
      end if

      do jk = 1, klev
        b = real(jk) * 2.0
      end do
    end subroutine
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Before: 2 JK loops (one wrapped in IF/ELSE)
    loops_before = [l for l in FindNodes(Loop).visit(routine.body)
                    if l.variable.name.lower() == 'jk']
    assert len(loops_before) == 2

    merged = _merge_vertical_loops(routine, 'jk', 'klev')
    assert merged is not None

    # After: 1 JK loop
    loops_after = [l for l in FindNodes(Loop).visit(routine.body)
                   if l.variable.name.lower() == 'jk']
    assert len(loops_after) == 1

    # The merged loop body should contain a conditional with an
    # else branch that preserves the original ``b = 0.0`` statement.
    code = fgen(routine).lower()
    assert 'else' in code
    assert 'b = 0.0' in code or 'b = 0' in code


# --------------------------------------------------------------------------
# _make_zero_literal
# --------------------------------------------------------------------------


class TestMakeZeroLiteral:
    """Tests for the _make_zero_literal helper."""

    def test_integer_type(self):
        """INTEGER type should produce IntLiteral(0)."""
        stype = SymbolAttributes(BasicType.INTEGER)
        result = _make_zero_literal(stype)
        assert isinstance(result, sym.IntLiteral)
        assert result.value == 0

    def test_logical_type(self):
        """LOGICAL type should produce LogicLiteral('.FALSE.')."""
        stype = SymbolAttributes(BasicType.LOGICAL)
        result = _make_zero_literal(stype)
        assert isinstance(result, sym.LogicLiteral)
        assert fgen(result).upper() == '.FALSE.'

    def test_real_with_named_kind(self):
        """REAL with a named kind (e.g. JPRB) should keep the kind."""
        kind_var = sym.Variable(name='JPRB')
        stype = SymbolAttributes(BasicType.REAL, kind=kind_var)
        result = _make_zero_literal(stype)
        assert isinstance(result, sym.FloatLiteral)
        assert result.kind is kind_var
        assert fgen(result) == '0.0_JPRB'

    def test_real_with_inline_call_kind(self):
        """REAL with an InlineCall kind (OMNI-resolved) should omit the kind.

        When the OMNI frontend resolves ``JPRB`` to
        ``selected_real_kind(13, 300)``, the kind is an :any:`InlineCall`.
        The literal must not embed it (``0.0_selected_real_kind(...)`` is
        invalid Fortran).
        """
        call_kind = sym.InlineCall(
            function=sym.ProcedureSymbol(name='selected_real_kind'),
            parameters=(sym.IntLiteral(13), sym.IntLiteral(300)),
        )
        stype = SymbolAttributes(BasicType.REAL, kind=call_kind)
        result = _make_zero_literal(stype)
        assert isinstance(result, sym.FloatLiteral)
        assert result.kind is None
        assert fgen(result) == '0.0'

    def test_real_without_kind(self):
        """REAL without an explicit kind should produce a plain 0.0."""
        stype = SymbolAttributes(BasicType.REAL)
        result = _make_zero_literal(stype)
        assert isinstance(result, sym.FloatLiteral)
        assert result.kind is None
        assert fgen(result) == '0.0'

    def test_deferred_type(self):
        """DEFERRED dtype should fall through to the REAL branch."""
        stype = SymbolAttributes(BasicType.DEFERRED)
        result = _make_zero_literal(stype)
        assert isinstance(result, sym.FloatLiteral)
        assert fgen(result) == '0.0'


# --------------------------------------------------------------------------
# _collect_call_arg_names
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_collect_call_arg_names(frontend):
    """
    _collect_call_arg_names should return lowercase names of all
    variables passed as positional or keyword arguments to calls.
    """
    fcode = """
  SUBROUTINE test_callargs(nlon, nz, a, b, c, d)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL :: a(nlon), b(nlon), c(nlon), d(nlon)
    REAL :: local_var
    INTEGER :: jl

    local_var = 1.0
    CALL foo(a, b, key=c)
    DO jl = 1, nlon
      d(jl) = local_var
    END DO
  END SUBROUTINE test_callargs
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    result = _collect_call_arg_names(routine)

    # Positional and keyword args should be collected
    assert 'a' in result
    assert 'b' in result
    assert 'c' in result

    # Variables not passed to any call should be absent
    assert 'local_var' not in result
    assert 'd' not in result


# --------------------------------------------------------------------------
# _collect_refs_outside_loops — visit_Loop on non-skipped loops
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_collect_refs_outside_loops_loop_bounds(frontend):
    """
    _RefCollector.visit_Loop should collect variable references from
    the bounds of non-skipped loops (e.g. DO JM = 1, NCLV).
    """
    fcode = """
  SUBROUTINE test_ref_loop(nlon, nz, nclv, a)
    INTEGER, INTENT(IN) :: nlon, nz, nclv
    REAL :: a(nlon, nz, nclv)
    INTEGER :: jl, jk, jm

    DO jm = 1, nclv
      a(1, 1, jm) = 0.0
    END DO

    DO jk = 1, nz
      DO jl = 1, nlon
        a(jl, jk, 1) = 1.0
      END DO
    END DO
  END SUBROUTINE test_ref_loop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # The JK loop is in the skip set; the JM loop is not
    jk_loops = [l for l in FindNodes(Loop).visit(routine.body)
                if l.variable.name.lower() == 'jk']
    assert len(jk_loops) == 1
    loop_nodes = set(jk_loops)

    result = _collect_refs_outside_loops(routine.body, loop_nodes)

    # JM loop bounds should be collected (visit_Loop recurses)
    assert 'jm' in result
    assert 'nclv' in result
    # Array referenced in non-skipped loop body
    assert 'a' in result
    # JL only appears inside the skipped JK loop
    assert 'jl' not in result


# --------------------------------------------------------------------------
# _collect_outside_refs_for_array — visit_Conditional (condition check)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_collect_outside_refs_for_array_conditional(frontend):
    """
    _ArrayRefCollector.visit_Conditional should detect array references
    in a conditional's condition expression, not only in its body.
    """
    fcode = """
  SUBROUTINE test_arrref_cond(nlon, nz, arr, out)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN) :: arr(nlon)
    REAL, INTENT(OUT) :: out(nlon, nz)
    INTEGER :: jl, jk

    IF (arr(1) > 0.0) THEN
      out(1, 1) = 1.0
    END IF

    DO jk = 1, nz
      DO jl = 1, nlon
        out(jl, jk) = arr(jl)
      END DO
    END DO
  END SUBROUTINE test_arrref_cond
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    jk_loops = [l for l in FindNodes(Loop).visit(routine.body)
                if l.variable.name.lower() == 'jk']
    assert len(jk_loops) == 1

    result = _collect_outside_refs_for_array(
        routine.body, set(jk_loops), 'arr'
    )

    # The IF condition references 'arr', so a Conditional should be collected
    assert len(result) >= 1
    assert any(isinstance(node, Conditional) for node in result)


# --------------------------------------------------------------------------
# _merge_vertical_loops — Both KLEV+N, compare N values
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_vertical_loops_both_klev_plus_n(frontend):
    """
    When two loops both have KLEV+N upper bounds with different N values,
    the merged loop should take the larger N as its upper bound.
    """
    fcode = """
  SUBROUTINE test_klev_n(nlon, nz, a, b)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(OUT) :: a(nlon, nz + 1), b(nlon, nz + 2)
    INTEGER :: jl, jk

    DO jk = 1, nz + 1
      DO jl = 1, nlon
        a(jl, jk) = 1.0
      END DO
    END DO

    DO jk = 1, nz + 2
      DO jl = 1, nlon
        b(jl, jk) = 2.0
      END DO
    END DO
  END SUBROUTINE test_klev_n
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    vloops = _collect_vertical_loops(routine.body, 'jk')
    assert len(vloops) == 2

    merged = _merge_vertical_loops(routine, 'jk', 'nz')

    # Should produce a single merged loop
    jk_loops = [l for l in FindNodes(Loop).visit(routine.body)
                if l.variable.name.lower() == 'jk']
    assert len(jk_loops) == 1

    # The upper bound should be NZ + 2 (the larger N)
    assert merged.bounds.children[1] == 'nz + 2'

    # Should have IF guards
    conds = FindNodes(Conditional).visit(merged.body)
    assert len(conds) >= 2


# --------------------------------------------------------------------------
# _remove_self_assignments — to_remove branch (actual removals)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_remove_self_assignments_with_removals(frontend):
    """
    _remove_self_assignments should detect and remove self-assignment
    no-ops (x = x) from inside vertical loops and return the count.
    """

    fcode = """
  SUBROUTINE test_selfassign(nlon, nz)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL :: x(nlon)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        x(jl) = x(jl)
        x(jl) = 1.0
      END DO
    END DO
  END SUBROUTINE test_selfassign
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    jk_loops = [l for l in FindNodes(Loop).visit(routine.body)
                if l.variable.name.lower() == 'jk']
    assert len(jk_loops) == 1

    n_removed = _remove_self_assignments(routine, jk_loops[0])
    assert n_removed == 1

    # After removal, no self-assignments should remain
    assigns = FindNodes(Assignment).visit(routine.body)
    for a in assigns:
        lhs_str = fgen(a.lhs).strip().lower()
        rhs_str = fgen(a.rhs).strip().lower()
        assert lhs_str != rhs_str, (
            f'Self-assignment should have been removed: {fgen(a)}'
        )

    # The real assignment x(jl) = 1.0 should survive
    assert any(fgen(a.rhs).strip() == '1.0' for a in assigns)
