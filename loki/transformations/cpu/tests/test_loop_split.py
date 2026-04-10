# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`SplitLoopForVectorisation` (T7).
"""

import pytest

from loki import Dimension, Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    FindNodes, FindInlineCalls, Loop, Assignment, CallStatement,
    Conditional, Comment, Pragma, Intrinsic,
)
from loki.backend import fgen

from loki.transformations.cpu.loop_split import SplitLoopForVectorisation
from loki.transformations.cpu.simd_pragmas import InsertSIMDPragmaDirectives
from loki.transformations.cpu.merge_conditionals import ConditionalFPGuardToMerge


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


# -----------------------------------------------------------------
# Basic splitting tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_basic(frontend, horizontal):
    """
    A horizontal loop with assignments, a CALL, and more assignments
    is split into three loops.
    """
    fcode = """
subroutine test_split_basic(kidia, kfdia, klon, za, zb, zc)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon), zb(klon), zc(klon)
  integer :: jl

  do jl = kidia, kfdia
    za(jl) = za(jl) + 1.0
    zb(jl) = zb(jl) * 2.0
    call some_sub(za(jl))
    zc(jl) = za(jl) + zb(jl)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # Should be 3 loops: assignments, call, assignments
    assert len(loops) == 3

    # All loops iterate over the same bounds
    for loop in loops:
        assert loop.variable.name.lower() == 'jl'

    # First loop: 2 assignments, no calls
    assigns_1 = FindNodes(Assignment).visit(loops[0].body)
    calls_1 = FindNodes(CallStatement).visit(loops[0].body)
    assert len(assigns_1) == 2
    assert len(calls_1) == 0

    # Second loop: 1 call, no assignments
    assigns_2 = FindNodes(Assignment).visit(loops[1].body)
    calls_2 = FindNodes(CallStatement).visit(loops[1].body)
    assert len(assigns_2) == 0
    assert len(calls_2) == 1

    # Third loop: 1 assignment, no calls
    assigns_3 = FindNodes(Assignment).visit(loops[2].body)
    calls_3 = FindNodes(CallStatement).visit(loops[2].body)
    assert len(assigns_3) == 1
    assert len(calls_3) == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_no_split_all_vectorisable(frontend, horizontal):
    """
    A horizontal loop with only assignments should not be split.
    """
    fcode = """
subroutine test_no_split_vec(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon), zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    za(jl) = za(jl) + 1.0
    zb(jl) = zb(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_no_split_all_non_vectorisable(frontend, horizontal):
    """
    A horizontal loop with only calls should not be split.
    """
    fcode = """
subroutine test_no_split_nonvec(kidia, kfdia, klon, za)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    call sub_a(za(jl))
    call sub_b(za(jl))
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1


# -----------------------------------------------------------------
# Scalar promotion tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_with_scalar_promotion(frontend, horizontal):
    """
    A scalar assigned before a call and used after the call must be
    promoted to an array with the horizontal dimension.
    """
    fcode = """
subroutine test_promote(kidia, kfdia, klon, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl
  real :: tmp

  do jl = kidia, kfdia
    tmp = za(jl) * 2.0
    call some_sub(za(jl))
    zresult(jl) = tmp + 1.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3

    # 'tmp' should have been promoted to tmp(klon)
    tmp_var = routine.variable_map['tmp']
    assert tmp_var.shape is not None
    assert str(tmp_var.shape[0]).lower() == 'klon'

    # Verify the generated code references tmp(jl) not just tmp
    code = fgen(routine)
    assert 'tmp(jl)' in code.lower()


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_multiple_scalars_promoted(frontend, horizontal):
    """
    Multiple scalars crossing the split boundary are all promoted.
    """
    fcode = """
subroutine test_multi_promote(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl
  real :: tmp1, tmp2

  do jl = kidia, kfdia
    tmp1 = za(jl) * 2.0
    tmp2 = zb(jl) + 1.0
    call some_sub(za(jl))
    zresult(jl) = tmp1 + tmp2
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3

    # Both scalars promoted
    assert routine.variable_map['tmp1'].shape is not None
    assert str(routine.variable_map['tmp1'].shape[0]).lower() == 'klon'
    assert routine.variable_map['tmp2'].shape is not None
    assert str(routine.variable_map['tmp2'].shape[0]).lower() == 'klon'


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_no_promotion_needed(frontend, horizontal):
    """
    When no scalars cross the split boundary, no promotion occurs.
    """
    fcode = """
subroutine test_no_promote(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon), zb(klon)
  integer :: jl
  real :: tmp

  do jl = kidia, kfdia
    za(jl) = za(jl) + 1.0
    call some_sub(za(jl))
    tmp = zb(jl) * 2.0
    zb(jl) = tmp
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3

    # 'tmp' is defined and used in the same section (the last one),
    # so no promotion needed — it should still be a scalar.
    tmp_var = routine.variable_map['tmp']
    assert isinstance(tmp_var, sym.Scalar)


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_array_not_promoted(frontend, horizontal):
    """
    An array that already has the horizontal dimension should not be
    promoted again.
    """
    fcode = """
subroutine test_array_no_promote(kidia, kfdia, klon, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl
  real :: zbuf(klon)

  do jl = kidia, kfdia
    zbuf(jl) = za(jl) * 3.0
    call some_sub(za(jl))
    zresult(jl) = zbuf(jl) + 1.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3

    # zbuf should still have shape (klon,) -- NOT (klon, klon)
    zbuf_var = routine.variable_map['zbuf']
    assert len(zbuf_var.shape) == 1
    assert str(zbuf_var.shape[0]).lower() == 'klon'


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_nested_loop_index_not_promoted(frontend, horizontal):
    """
    A nested loop index (e.g. JO from ``DO JO=1,NCLV``) inside a
    horizontal loop must NOT be promoted to an array, even if it
    appears to cross a split boundary.

    This is the pattern that caused the cloudsc.F90 bug where
    ``JO`` was incorrectly promoted to ``JO(KLON)``, producing
    nonsensical ``DO JO(JL)=1,NCLV`` in generated code.
    """
    fcode = """
subroutine test_nested_idx(kidia, kfdia, klon, nclv, za, zresult, iorder)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, nclv
  real, intent(inout) :: za(klon, nclv)
  real, intent(out) :: zresult(klon)
  integer, intent(in) :: iorder(klon, nclv)
  integer :: jl, jo, jm
  real :: ztmp

  do jl = kidia, kfdia
    ! Section 1: nested loop using JO
    do jo = 1, nclv
      za(jl, jo) = za(jl, jo) * 2.0
    end do
    ! Section 2: CALL forces a split
    call some_sub(za(jl, 1))
    ! Section 3: another nested loop using JO again
    ztmp = 0.0
    do jm = 1, nclv
      jo = iorder(jl, jm)
      ztmp = ztmp + za(jl, jo)
    end do
    zresult(jl) = ztmp
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # The horizontal loop is split into 3 sections (pre-call, call, post-call)
    assert len(loops) >= 3

    # JO must remain a plain scalar, NOT promoted to JO(KLON)
    jo_var = routine.variable_map.get('jo')
    assert jo_var is not None, "JO variable should still exist"
    if hasattr(jo_var, 'shape') and jo_var.shape:
        assert False, f"JO was incorrectly promoted to an array: shape={jo_var.shape}"

    # JM must also remain a plain scalar
    jm_var = routine.variable_map.get('jm')
    assert jm_var is not None, "JM variable should still exist"
    if hasattr(jm_var, 'shape') and jm_var.shape:
        assert False, f"JM was incorrectly promoted to an array: shape={jm_var.shape}"

    # ztmp SHOULD be promoted (it's a real scalar crossing the boundary)
    ztmp_var = routine.variable_map.get('ztmp')
    assert ztmp_var is not None
    assert ztmp_var.shape is not None, "ztmp should be promoted to an array"
    assert str(ztmp_var.shape[0]).lower() == 'klon'

    # The generated code must NOT contain 'do jo(' or 'do jm(' patterns
    # (which would indicate array promotion of loop indices)
    code = fgen(routine).lower()
    assert 'do jo(' not in code, "Generated code has nonsensical DO JO( pattern"
    assert 'do jm(' not in code, "Generated code has nonsensical DO JM( pattern"
    # Also check the declaration: JO should NOT be declared as an array
    assert 'jo(klon)' not in code, "JO was promoted to JO(KLON) in generated code"
    assert 'jm(klon)' not in code, "JM was promoted to JM(KLON) in generated code"


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_index_reused_as_scalar_not_promoted(frontend, horizontal):
    """
    Reproducer for the cloudsc.F90 bug: ``JO`` is used as a plain
    scalar (``JO = IORDER(JL,JM)``) in one horizontal loop and as a
    nested loop index (``DO JO=1,NCLV``) in a different horizontal
    loop.  Global promotion to ``JO(KLON)`` would produce nonsensical
    ``DO JO(JL)=1,NCLV`` in the second loop.

    The fix: never promote a variable that is used as a loop index in
    ANY horizontal loop, even if it is a legitimate scalar-crossing-
    split-boundary candidate in another loop.
    """
    fcode = """
subroutine test_idx_reuse(kidia, kfdia, klon, nclv, za, zresult, iorder)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, nclv
  real, intent(inout) :: za(klon, nclv)
  real, intent(out) :: zresult(klon)
  integer, intent(in) :: iorder(klon, nclv)
  integer :: jl, jo, jn
  real :: zzratio

  ! Loop 1: JO is a scalar assigned then used after a non-vec section
  do jl = kidia, kfdia
    jo = iorder(jl, 1)
    zzratio = za(jl, jo)
    !DIR$ IVDEP
    do jn = 1, nclv
      za(jl, jn) = za(jl, jn) * zzratio
    end do
  end do

  ! Loop 2: JO is a proper loop index
  do jl = kidia, kfdia
    zresult(jl) = 0.0
    do jo = 1, nclv
      zresult(jl) = zresult(jl) + za(jl, jo)
    end do
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # JO must remain a plain scalar, NOT promoted to JO(KLON)
    jo_var = routine.variable_map.get('jo')
    assert jo_var is not None, "JO variable should still exist"
    if hasattr(jo_var, 'shape') and jo_var.shape:
        assert False, f"JO was incorrectly promoted to an array: shape={jo_var.shape}"

    code = fgen(routine).lower()
    assert 'do jo(' not in code, "Generated code has nonsensical DO JO( pattern"
    assert 'jo(klon)' not in code, "JO was promoted to JO(KLON) in generated code"


# -----------------------------------------------------------------
# Conditional handling tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_conditional_stays_together(frontend, horizontal):
    """
    A Conditional node (vectorisable) stays in the vectorisable
    section and is not split internally.
    """
    fcode = """
subroutine test_cond_together(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (za(jl) > 0.0) then
      zresult(jl) = za(jl) + zb(jl)
    else
      zresult(jl) = zb(jl)
    end if
    call some_sub(za(jl))
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2

    # First loop contains the conditional, second contains the call
    conds = FindNodes(Conditional).visit(loops[0].body)
    assert len(conds) == 1

    calls = FindNodes(CallStatement).visit(loops[1].body)
    assert len(calls) == 1


# -----------------------------------------------------------------
# Non-vectorisable node type tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI cannot handle intrinsic WRITE without full context')]
))
def test_split_loop_intrinsic_non_vectorisable(frontend, horizontal):
    """
    An Intrinsic statement (WRITE) causes a split.
    """
    fcode = """
subroutine test_intrinsic_split(kidia, kfdia, klon, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) * 2.0
    write(*,*) 'debug: ', za(jl)
    zresult(jl) = zresult(jl) + 1.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_nested_loop_non_vectorisable(frontend, horizontal):
    """
    A nested loop inside the horizontal loop triggers splitting.
    """
    fcode = """
subroutine test_nested_split(kidia, kfdia, klon, klev, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, klev
  real, intent(in) :: za(klon, klev)
  real, intent(out) :: zresult(klon)
  integer :: jl, jk
  real :: zsum

  do jl = kidia, kfdia
    zresult(jl) = 0.0
    zsum = 0.0
    do jk = 1, klev
      zsum = zsum + za(jl, jk)
    end do
    zresult(jl) = zsum
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # The outer horizontal loop is split; the inner jk loop is in the
    # non-vectorisable section.
    # Expect: [vec: zresult=0, zsum=0] [non: DO jk ... END DO] [vec: zresult=zsum]
    # The horizontal loops are top-level, plus the jk loop inside one of them.
    horizontal_loops = [l for l in loops if l.variable.name.lower() == 'jl']
    assert len(horizontal_loops) == 3

    # The nested jk loop should be inside one of the horizontal loops
    jk_loops = [l for l in loops if l.variable.name.lower() == 'jk']
    assert len(jk_loops) == 1


# -----------------------------------------------------------------
# Comment handling tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_comments_attached(frontend, horizontal):
    """
    Comments between sections are attached to the appropriate loop.
    Split boundary markers are inserted.
    """
    fcode = """
subroutine test_comments(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon), zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    ! compute za
    za(jl) = za(jl) + 1.0
    call some_sub(za(jl))
    ! compute zb
    zb(jl) = zb(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3

    # Check that split boundary comment markers exist
    code = fgen(routine)
    assert 'Loki loop-split' in code


# -----------------------------------------------------------------
# Driver/sequential skip tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_driver_noop(frontend, horizontal):
    """
    Driver-role routines are skipped.
    """
    fcode = """
subroutine test_driver(kidia, kfdia, klon, za)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    za(jl) = za(jl) + 1.0
    call some_sub(za(jl))
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='driver')

    # Should remain 1 loop (no splitting)
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1


# -----------------------------------------------------------------
# Pipeline integration test
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_pipeline_integration(frontend, horizontal):
    """
    Full pipeline test: after T1 (MERGE) + T7 (split) + T6 (SIMD),
    the vectorisable loops should have SIMD pragmas while the
    non-vectorisable ones should not.
    """
    fcode = """
subroutine test_pipeline_split(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + zb(jl)
    call some_sub(za(jl))
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Apply T7 (split) then T6 (SIMD pragmas)
    t7 = SplitLoopForVectorisation(horizontal=horizontal)
    t7.apply(routine, role='kernel')

    t6 = InsertSIMDPragmaDirectives(horizontal=horizontal)
    t6.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3

    # The vectorisable loops (first and third) should have SIMD pragmas
    code = fgen(routine)
    assert 'OMP SIMD' in code.upper()

    # Count SIMD pragmas -- should be 2 (one for each vectorisable loop)
    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_multiple_horizontal_loops(frontend, horizontal):
    """
    Multiple horizontal loops in the same routine are each handled
    independently.
    """
    fcode = """
subroutine test_multi_loops(kidia, kfdia, klon, za, zb, zc)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon), zb(klon), zc(klon)
  integer :: jl

  do jl = kidia, kfdia
    za(jl) = za(jl) + 1.0
    call sub_a(za(jl))
  end do

  do jl = kidia, kfdia
    zb(jl) = zb(jl) * 2.0
    call sub_b(zb(jl))
    zc(jl) = zb(jl) + za(jl)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # First loop splits into 2, second loop splits into 3 => 5 total
    assert len(loops) == 5


# -----------------------------------------------------------------
# fp_strict mode tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_fp_strict_exp_splits(frontend, horizontal):
    """
    With ``fp_strict=True``, an assignment containing ``EXP`` is
    classified as non-vectorisable and split out from plain
    arithmetic assignments.
    """
    fcode = """
subroutine test_fp_strict_exp(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + zb(jl)
    zresult(jl) = exp(zresult(jl))
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(+), non(exp), vec(*2) => 3 loops
    assert len(loops) == 3

    # First loop: plain addition (vectorisable)
    assigns_0 = FindNodes(Assignment).visit(loops[0].body)
    assert len(assigns_0) == 1
    assert 'exp' not in fgen(assigns_0[0]).lower()

    # Second loop: the exp call (non-vectorisable)
    assigns_1 = FindNodes(Assignment).visit(loops[1].body)
    assert len(assigns_1) == 1
    assert 'exp' in fgen(assigns_1[0]).lower()

    # Third loop: multiplication (vectorisable)
    assigns_2 = FindNodes(Assignment).visit(loops[2].body)
    assert len(assigns_2) == 1
    assert 'exp' not in fgen(assigns_2[0]).lower()


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_fp_strict_disabled_no_split(frontend, horizontal):
    """
    With ``fp_strict=False`` (default), assignments containing
    ``EXP`` are treated as vectorisable and no extra splitting
    occurs.
    """
    fcode = """
subroutine test_fp_strict_disabled(kidia, kfdia, klon, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + 1.0
    zresult(jl) = exp(zresult(jl))
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Default: fp_strict=False
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # No split -- all assignments are vectorisable by default
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_fp_strict_custom_set(frontend, horizontal):
    """
    A custom ``fp_strict_intrinsics`` set restricts which intrinsics
    trigger a split.  Only ``SIN`` is in the custom set, so ``EXP``
    does not cause a split but ``SIN`` does.
    """
    fcode = """
subroutine test_fp_strict_custom(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + zb(jl)
    zresult(jl) = exp(zresult(jl))
    zresult(jl) = sin(zresult(jl))
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(
        horizontal=horizontal,
        fp_strict=True,
        fp_strict_intrinsics={'SIN'},
    )
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # The add and exp are both vec (EXP not in custom set).
    # The sin is non-vec. The multiply is vec.
    # => vec(add, exp), non(sin), vec(mul) => 3 loops
    assert len(loops) == 3

    # Second loop should be the sin call
    assigns_1 = FindNodes(Assignment).visit(loops[1].body)
    assert len(assigns_1) == 1
    assert 'sin' in fgen(assigns_1[0]).lower()

    # First loop should have the add and exp (both vec in this config)
    assigns_0 = FindNodes(Assignment).visit(loops[0].body)
    assert len(assigns_0) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_fp_strict_multiple_intrinsics(frontend, horizontal):
    """
    Multiple unsafe intrinsics (``EXP`` and ``LOG``) in the same
    loop body are grouped together when adjacent.
    """
    fcode = """
subroutine test_fp_strict_multi(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + zb(jl)
    zresult(jl) = exp(zresult(jl))
    zresult(jl) = log(zresult(jl))
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(add), non(exp, log), vec(mul) => 3 loops
    assert len(loops) == 3

    # Middle loop has exp and log (adjacent non-vec merged)
    assigns_mid = FindNodes(Assignment).visit(loops[1].body)
    assert len(assigns_mid) == 2
    mid_code = fgen(loops[1]).lower()
    assert 'exp' in mid_code
    assert 'log' in mid_code


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_fp_strict_with_scalar_promotion(frontend, horizontal):
    """
    When ``fp_strict`` splits a loop, scalar variables crossing
    the split boundary are promoted to arrays.
    """
    fcode = """
subroutine test_fp_strict_promote(kidia, kfdia, klon, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl
  real :: tmp

  do jl = kidia, kfdia
    tmp = za(jl) + 1.0
    zresult(jl) = exp(tmp)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(tmp = ...), non(exp(tmp)) => 2 loops
    assert len(loops) == 2

    # 'tmp' is defined in the first section and used in the second,
    # so it should be promoted to an array with dimension (klon).
    tmp_var = routine.variable_map['tmp']
    assert isinstance(tmp_var, sym.Array)
    assert str(tmp_var.shape[0]).lower() == 'klon'


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_fp_strict_conditional_with_intrinsic(frontend, horizontal):
    """
    A :any:`Conditional` whose body contains an ``EXP`` call is
    classified as non-vectorisable when ``fp_strict=True``.
    """
    fcode = """
subroutine test_fp_strict_cond(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + zb(jl)
    if (za(jl) > 0.0) then
      zresult(jl) = exp(za(jl))
    end if
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(add), non(if with exp), vec(mul) => 3 loops
    assert len(loops) == 3

    # The middle loop contains the conditional with exp
    conds = FindNodes(Conditional).visit(loops[1].body)
    assert len(conds) == 1
    assert 'exp' in fgen(conds[0]).lower()


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_fp_strict_all_unsafe_no_split(frontend, horizontal):
    """
    When all statements in a horizontal loop are non-vectorisable
    under ``fp_strict``, no split occurs (the loop body is
    homogeneous).
    """
    fcode = """
subroutine test_fp_strict_all_unsafe(kidia, kfdia, klon, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = exp(za(jl))
    zresult(jl) = log(zresult(jl))
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    # All non-vec => no split
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_fp_strict_sqrt_splits(frontend, horizontal):
    """
    ``SQRT`` is in the default ``fp_strict_intrinsics`` set and
    causes a split when ``fp_strict=True``.
    """
    fcode = """
subroutine test_fp_strict_sqrt(kidia, kfdia, klon, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) * 2.0
    zresult(jl) = sqrt(zresult(jl))
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(mul), non(sqrt) => 2 loops
    assert len(loops) == 2

    # Second loop has sqrt
    code_1 = fgen(loops[1]).lower()
    assert 'sqrt' in code_1


# -----------------------------------------------------------------
# Indirect array indexing tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_indirect_indexing_rhs(frontend, horizontal):
    """
    An assignment that reads via indirect indexing on the horizontal
    dimension (gather pattern: ``za(KIDX(jl))``) is classified as
    non-vectorisable and split out.
    """
    fcode = """
subroutine test_indirect_rhs(kidia, kfdia, klon, za, zresult, kidx)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer, intent(in) :: kidx(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + 1.0
    zresult(jl) = zresult(jl) + za(kidx(jl))
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(+1), non(gather), vec(*2) => 3 loops
    assert len(loops) == 3

    # First loop: plain addition, no indirect indexing
    code_0 = fgen(loops[0]).lower()
    assert 'kidx' not in code_0

    # Second loop: the indirect-indexed access
    code_1 = fgen(loops[1]).lower()
    assert 'kidx' in code_1

    # Third loop: plain multiplication
    code_2 = fgen(loops[2]).lower()
    assert 'kidx' not in code_2


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_indirect_indexing_lhs(frontend, horizontal):
    """
    An assignment that writes via indirect indexing on the horizontal
    dimension (scatter pattern: ``za(KIDX(jl)) = ...``) is classified
    as non-vectorisable and split out.
    """
    fcode = """
subroutine test_indirect_lhs(kidia, kfdia, klon, za, zb, kidx)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon)
  real, intent(in) :: zb(klon)
  integer, intent(in) :: kidx(klon)
  integer :: jl

  do jl = kidia, kfdia
    za(jl) = za(jl) + 1.0
    za(kidx(jl)) = zb(jl)
    za(jl) = za(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(+1), non(scatter), vec(*2) => 3 loops
    assert len(loops) == 3

    # The middle loop should contain the scatter assignment
    code_1 = fgen(loops[1]).lower()
    assert 'kidx' in code_1

    # The first and last loops should not contain kidx
    assert 'kidx' not in fgen(loops[0]).lower()
    assert 'kidx' not in fgen(loops[2]).lower()


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_indirect_indexing_both_sides(frontend, horizontal):
    """
    An assignment with indirect indexing on both LHS and RHS is
    classified as non-vectorisable.
    """
    fcode = """
subroutine test_indirect_both(kidia, kfdia, klon, za, zb, kidx)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon)
  real, intent(in) :: zb(klon)
  integer, intent(in) :: kidx(klon)
  integer :: jl

  do jl = kidia, kfdia
    za(jl) = za(jl) + 1.0
    za(kidx(jl)) = zb(kidx(jl))
    za(jl) = za(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(+1), non(gather+scatter), vec(*2) => 3 loops
    assert len(loops) == 3

    # The middle loop should contain the indirect-indexed assignment
    code_1 = fgen(loops[1]).lower()
    assert 'kidx' in code_1


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_indirect_indexing_no_false_positive(frontend, horizontal):
    """
    A loop with only direct array indexing (``za(jl)``) should NOT be
    split — no false positive from the indirect-indexing check.
    """
    fcode = """
subroutine test_indirect_no_fp(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: za(klon), zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    za(jl) = za(jl) + 1.0
    zb(jl) = za(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # All assignments are directly indexed — no split
    assert len(loops) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_indirect_indexing_non_horizontal_dim(frontend, horizontal):
    """
    Indirect indexing on a non-horizontal dimension (dim 1 in a 2-D
    array whose first dimension is horizontal) should NOT trigger a
    split.  Only the horizontal dimension matters.
    """
    fcode = """
subroutine test_indirect_non_horiz(kidia, kfdia, klon, klev, za, kidx)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, klev
  real, intent(inout) :: za(klon, klev)
  integer, intent(in) :: kidx(klon)
  integer :: jl

  do jl = kidia, kfdia
    za(jl, kidx(jl)) = za(jl, kidx(jl)) + 1.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # Indirect indexing is on dimension 1 (klev), not dimension 0 (klon).
    # The horizontal dimension (klon, position 0) is indexed directly
    # by jl, so this should remain vectorisable — no split.
    assert len(loops) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_indirect_indexing_with_promotion(frontend, horizontal):
    """
    A scalar variable defined before an indirect-indexing statement
    and used after it must be promoted to an array.
    """
    fcode = """
subroutine test_indirect_promote(kidia, kfdia, klon, za, zresult, kidx)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer, intent(in) :: kidx(klon)
  integer :: jl
  real :: tmp

  do jl = kidia, kfdia
    tmp = za(jl) * 2.0
    zresult(jl) = za(kidx(jl))
    zresult(jl) = zresult(jl) + tmp
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(tmp=...), non(gather), vec(+tmp) => 3 loops
    assert len(loops) == 3

    # 'tmp' is defined in section 0 and used in section 2,
    # so it must be promoted to tmp(klon).
    tmp_var = routine.variable_map['tmp']
    assert isinstance(tmp_var, sym.Array)
    assert str(tmp_var.shape[0]).lower() == 'klon'

    # Verify the generated code references tmp(jl)
    code = fgen(routine).lower()
    assert 'tmp(jl)' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_indirect_indexing_conditional(frontend, horizontal):
    """
    A :any:`Conditional` whose body contains an indirect-indexed
    array access is classified as non-vectorisable.
    """
    fcode = """
subroutine test_indirect_cond(kidia, kfdia, klon, za, zresult, kidx)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer, intent(in) :: kidx(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + 1.0
    if (za(jl) > 0.0) then
      zresult(jl) = za(kidx(jl))
    end if
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SplitLoopForVectorisation(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    # vec(+1), non(if with gather), vec(*2) => 3 loops
    assert len(loops) == 3

    # The middle loop contains the conditional with indirect indexing
    conds = FindNodes(Conditional).visit(loops[1].body)
    assert len(conds) == 1
    assert 'kidx' in fgen(conds[0]).lower()


@pytest.mark.parametrize('frontend', available_frontends())
def test_split_loop_indirect_indexing_pipeline(frontend, horizontal):
    """
    Pipeline test: after T7 (split) + T6 (SIMD), the original loop
    is split into three.  T6 annotates all three with SIMD pragmas
    because each loop body contains only assignments (T6 checks
    node types, not subscript patterns).  The split still has value:
    it isolates the indirect-indexed statement so that a future T6
    enhancement or compiler heuristic can treat it differently.
    """
    fcode = """
subroutine test_indirect_pipeline(kidia, kfdia, klon, za, zresult, kidx)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(out) :: zresult(klon)
  integer, intent(in) :: kidx(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = za(jl) + 1.0
    zresult(jl) = zresult(jl) + za(kidx(jl))
    zresult(jl) = zresult(jl) * 2.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Apply T7 (split) then T6 (SIMD pragmas)
    t7 = SplitLoopForVectorisation(horizontal=horizontal)
    t7.apply(routine, role='kernel')

    t6 = InsertSIMDPragmaDirectives(horizontal=horizontal)
    t6.apply(routine, role='kernel')

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3

    # T6 annotates all loops whose body consists of simple node types
    # (Assignment, Conditional, Comment, Pragma).  Since all three
    # split loops satisfy this criterion, all three receive SIMD pragmas.
    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 3

    # Verify that the indirect-indexed statement is isolated in its
    # own loop (the middle one).
    code_1 = fgen(loops[1]).lower()
    assert 'kidx' in code_1
    assert 'kidx' not in fgen(loops[0]).lower()
    assert 'kidx' not in fgen(loops[2]).lower()
