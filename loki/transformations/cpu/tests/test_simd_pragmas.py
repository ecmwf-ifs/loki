# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`InsertSIMDPragmaDirectives` (T6).
"""

import pytest

from loki import Dimension, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, Loop, Pragma, Comment, Assignment, Conditional
from loki.backend import fgen

from loki.transformations.cpu.simd_pragmas import InsertSIMDPragmaDirectives


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


# -----------------------------------------------------------------
# Basic insertion tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_simple_loop_omp_simd(frontend, horizontal):
    """
    A simple horizontal loop with only assignments gets ``!$OMP SIMD``.
    """
    fcode = """
subroutine test_simple(kidia, kfdia, klon, za, zb, zc)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(inout) :: zc(klon)
  integer :: jl

  do jl = kidia, kfdia
    zc(jl) = za(jl) + zb(jl)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    code = fgen(routine)
    # Should contain OMP SIMD pragma before the loop
    pragmas = FindNodes(Pragma).visit(routine.body)
    assert len(pragmas) == 1
    assert pragmas[0].keyword.upper() == 'OMP'
    assert 'SIMD' in pragmas[0].content.upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_simple_loop_dir_simd(frontend, horizontal):
    """
    With ``directive='DIR$ SIMD'``, a Comment node ``!DIR$ SIMD`` is
    inserted before the horizontal loop.
    """
    fcode = """
subroutine test_dir_simd(kidia, kfdia, klon, za, zb)
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
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal,
                                        directive='DIR$ SIMD')
    trafo.apply(routine, role='kernel')

    code = fgen(routine)
    # DIR$ directives are rendered as Comment nodes
    assert '!DIR$ SIMD' in code

    comments = FindNodes(Comment).visit(routine.body)
    simd_comments = [c for c in comments if c.text and 'DIR$ SIMD' in c.text]
    assert len(simd_comments) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_ivdep_plus_omp_simd(frontend, horizontal):
    """
    With ``insert_ivdep=True``, both ``!DIR$ IVDEP`` and the primary
    directive are inserted.
    """
    fcode = """
subroutine test_ivdep(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    zb(jl) = za(jl) + 1.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal,
                                        insert_ivdep=True)
    trafo.apply(routine, role='kernel')

    code = fgen(routine)
    assert '!DIR$ IVDEP' in code
    # Should also have OMP SIMD
    pragmas = FindNodes(Pragma).visit(routine.body)
    omp_pragmas = [p for p in pragmas if 'SIMD' in (p.content or '').upper()]
    assert len(omp_pragmas) == 1


# -----------------------------------------------------------------
# Non-simple loop bodies (should NOT get pragmas)
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_with_conditional_no_pragma(frontend, horizontal):
    """
    A horizontal loop containing a conditional IS annotated because
    masked branches are safe under SIMD execution.
    """
    fcode = """
subroutine test_cond(kidia, kfdia, klon, za, zb, llmask)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  logical, intent(in) :: llmask(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      zb(jl) = za(jl) * 2.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_with_call_no_pragma(frontend, horizontal):
    """
    A horizontal loop containing a subroutine call should NOT be annotated.
    """
    fcode = """
subroutine test_call(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    call some_sub(za(jl), zb(jl))
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 0


# -----------------------------------------------------------------
# PRIVATE clause tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_private_clause_scalar_in_conditional(frontend, horizontal):
    """
    A scalar variable assigned inside a conditional within a horizontal
    loop gets a ``PRIVATE(...)`` clause on the ``!$OMP SIMD`` directive.

    This is the pattern that causes Intel optrpt #15316 (false vector
    dependence on a loop-private scalar).
    """
    fcode = """
subroutine test_private_cond(kidia, kfdia, klon, za, zb, llmask)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  logical, intent(in) :: llmask(klon)
  integer :: jl
  real :: zgridlen

  do jl = kidia, kfdia
    zgridlen = 0.0
    if (llmask(jl)) then
      zgridlen = za(jl) * 2.0
    end if
    zb(jl) = zb(jl) + zgridlen
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 1
    assert 'PRIVATE' in simd_pragmas[0].content.upper()
    assert 'ZGRIDLEN' in simd_pragmas[0].content.upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_private_clause_multiple_scalars(frontend, horizontal):
    """
    Multiple scalar variables assigned inside a horizontal loop all
    appear in the ``PRIVATE(...)`` clause, sorted alphabetically.
    """
    fcode = """
subroutine test_private_multi(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl
  real :: ztmp1, ztmp2

  do jl = kidia, kfdia
    ztmp2 = za(jl) * 3.0
    ztmp1 = za(jl) * 2.0
    zb(jl) = ztmp1 + ztmp2
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 1
    content = simd_pragmas[0].content.upper()
    assert 'PRIVATE' in content
    assert 'ZTMP1' in content
    assert 'ZTMP2' in content
    # Verify alphabetical ordering: ZTMP1 before ZTMP2
    assert content.index('ZTMP1') < content.index('ZTMP2')


@pytest.mark.parametrize('frontend', available_frontends())
def test_no_private_clause_array_only(frontend, horizontal):
    """
    When all assignments target array elements (no scalar LHS), no
    ``PRIVATE(...)`` clause is added.
    """
    fcode = """
subroutine test_no_private(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    zb(jl) = za(jl) + 1.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 1
    assert 'PRIVATE' not in simd_pragmas[0].content.upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_private_clause_dir_simd_no_private(frontend, horizontal):
    """
    When using ``DIR$ SIMD`` directive, no ``PRIVATE(...)`` clause is
    added (DIR$ has no PRIVATE syntax).
    """
    fcode = """
subroutine test_dir_no_private(kidia, kfdia, klon, za, zb, llmask)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  logical, intent(in) :: llmask(klon)
  integer :: jl
  real :: ztmp

  do jl = kidia, kfdia
    ztmp = 0.0
    if (llmask(jl)) then
      ztmp = za(jl) * 2.0
    end if
    zb(jl) = zb(jl) + ztmp
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal,
                                        directive='DIR$ SIMD')
    trafo.apply(routine, role='kernel')

    code = fgen(routine)
    assert '!DIR$ SIMD' in code
    # DIR$ directives are rendered as Comment nodes — verify no PRIVATE clause
    comments = FindNodes(Comment).visit(routine.body)
    simd_comments = [c for c in comments if c.text and 'DIR$ SIMD' in c.text]
    assert len(simd_comments) == 1
    assert 'PRIVATE' not in simd_comments[0].text.upper()


# -----------------------------------------------------------------
# VECTOR ALWAYS tests
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_vector_always(frontend, horizontal):
    """
    With ``insert_vector_always=True``, a ``!DIR$ VECTOR ALWAYS``
    comment is inserted before the horizontal loop.
    """
    fcode = """
subroutine test_vecalways(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    zb(jl) = za(jl) + 1.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal,
                                        insert_vector_always=True)
    trafo.apply(routine, role='kernel')

    code = fgen(routine)
    assert '!DIR$ VECTOR ALWAYS' in code
    # Should also have OMP SIMD
    pragmas = FindNodes(Pragma).visit(routine.body)
    omp_pragmas = [p for p in pragmas if 'SIMD' in (p.content or '').upper()]
    assert len(omp_pragmas) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_ivdep_vector_always_omp_simd_private(frontend, horizontal):
    """
    All directives combined: ``!DIR$ IVDEP``, ``!DIR$ VECTOR ALWAYS``,
    and ``!$OMP SIMD PRIVATE(...)`` are inserted in order.
    """
    fcode = """
subroutine test_all_directives(kidia, kfdia, klon, za, zb, llmask)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  logical, intent(in) :: llmask(klon)
  integer :: jl
  real :: ztmp

  do jl = kidia, kfdia
    ztmp = 0.0
    if (llmask(jl)) ztmp = za(jl)
    zb(jl) = zb(jl) + ztmp
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(
        horizontal=horizontal,
        insert_ivdep=True,
        insert_vector_always=True
    )
    trafo.apply(routine, role='kernel')

    code = fgen(routine)
    # All three directives should be present
    assert '!DIR$ IVDEP' in code
    assert '!DIR$ VECTOR ALWAYS' in code

    pragmas = FindNodes(Pragma).visit(routine.body)
    omp_pragmas = [p for p in pragmas if 'SIMD' in (p.content or '').upper()]
    assert len(omp_pragmas) == 1
    assert 'PRIVATE' in omp_pragmas[0].content.upper()
    assert 'ZTMP' in omp_pragmas[0].content.upper()

    # Verify ordering: IVDEP before VECTOR ALWAYS before OMP SIMD
    ivdep_pos = code.index('!DIR$ IVDEP')
    vecalways_pos = code.index('!DIR$ VECTOR ALWAYS')
    omp_pos = code.index('OMP SIMD')
    assert ivdep_pos < vecalways_pos < omp_pos


@pytest.mark.parametrize('frontend', available_frontends())
def test_vector_always_collapse(frontend, horizontal):
    """
    ``!DIR$ VECTOR ALWAYS`` is also inserted for collapsible nests.
    """
    fcode = """
subroutine test_vecalways_collapse(kidia, kfdia, klon, klev, za)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, klev
  real, intent(inout) :: za(klon, klev)
  integer :: jl, jk

  do jk = 1, klev
    do jl = kidia, kfdia
      za(jl, jk) = za(jl, jk) * 2.0
    end do
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(
        horizontal=horizontal,
        collapse_outer=True,
        insert_vector_always=True
    )
    trafo.apply(routine, role='kernel')

    code = fgen(routine)
    assert '!DIR$ VECTOR ALWAYS' in code

    pragmas = FindNodes(Pragma).visit(routine.body)
    collapse_pragmas = [p for p in pragmas
                        if p.content and 'COLLAPSE' in p.content.upper()]
    assert len(collapse_pragmas) == 1


# -----------------------------------------------------------------
# Non-horizontal loops (should NOT get pragmas)
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_non_horizontal_loop_no_pragma(frontend, horizontal):
    """
    A loop that is NOT over the horizontal dimension should not be annotated.
    """
    fcode = """
subroutine test_non_horiz(kidia, kfdia, klon, klev, za)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, klev
  real, intent(inout) :: za(klon, klev)
  integer :: jl, jk

  do jk = 1, klev
    za(1, jk) = 0.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 0


# -----------------------------------------------------------------
# Duplicate pragma avoidance
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_no_duplicate_pragma(frontend, horizontal):
    """
    If a SIMD pragma already exists before the loop, no second one
    should be inserted.
    """
    fcode = """
subroutine test_dup(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  !$OMP SIMD
  do jl = kidia, kfdia
    zb(jl) = za(jl) + 1.0
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 1  # Only the original, no duplicate


# -----------------------------------------------------------------
# Multiple loops
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_multiple_loops(frontend, horizontal):
    """
    Multiple horizontal loops: only simple ones get annotated.
    """
    fcode = """
subroutine test_multi(kidia, kfdia, klon, za, zb, zc, llmask)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon), zc(klon)
  logical, intent(in) :: llmask(klon)
  integer :: jl

  ! Loop 1: simple -- should get pragma
  do jl = kidia, kfdia
    zb(jl) = za(jl) * 2.0
  end do

  ! Loop 2: has conditional -- still gets pragma (conditionals are SIMD-safe)
  do jl = kidia, kfdia
    if (llmask(jl)) then
      zc(jl) = za(jl)
    end if
  end do

  ! Loop 3: simple -- should get pragma
  do jl = kidia, kfdia
    zc(jl) = zb(jl) + za(jl)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 3


# -----------------------------------------------------------------
# Collapse mode
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_collapsible_nest(frontend, horizontal):
    """
    An outer vertical loop with exactly one inner horizontal loop gets
    ``!$OMP SIMD COLLAPSE(2)`` when ``collapse_outer=True``.
    """
    fcode = """
subroutine test_collapse(kidia, kfdia, klon, klev, za)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, klev
  real, intent(inout) :: za(klon, klev)
  integer :: jl, jk

  do jk = 1, klev
    do jl = kidia, kfdia
      za(jl, jk) = za(jl, jk) * 2.0
    end do
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal,
                                        collapse_outer=True)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    collapse_pragmas = [p for p in pragmas
                        if p.content and 'COLLAPSE' in p.content.upper()]
    assert len(collapse_pragmas) == 1
    assert 'SIMD' in collapse_pragmas[0].content.upper()
    assert 'COLLAPSE(2)' in collapse_pragmas[0].content.upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_non_collapsible_nest_extra_stmts(frontend, horizontal):
    """
    An outer loop with statements besides the inner loop is NOT
    collapsible, but the inner horizontal loop still gets a regular
    SIMD pragma if its body is simple.
    """
    fcode = """
subroutine test_not_collapse(kidia, kfdia, klon, klev, za, ztmp)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, klev
  real, intent(inout) :: za(klon, klev)
  real, intent(inout) :: ztmp(klon)
  integer :: jl, jk

  do jk = 1, klev
    ztmp(1) = 0.0
    do jl = kidia, kfdia
      za(jl, jk) = za(jl, jk) + ztmp(jl)
    end do
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal,
                                        collapse_outer=True)
    trafo.apply(routine, role='kernel')

    pragmas = FindNodes(Pragma).visit(routine.body)
    # No COLLAPSE pragma on the outer loop
    collapse_pragmas = [p for p in pragmas
                        if p.content and 'COLLAPSE' in p.content.upper()]
    assert len(collapse_pragmas) == 0

    # But the inner loop should still get a regular SIMD pragma
    simd_pragmas = [p for p in pragmas
                    if p.content and 'SIMD' in p.content.upper()
                    and 'COLLAPSE' not in p.content.upper()]
    assert len(simd_pragmas) == 1


# -----------------------------------------------------------------
# Driver role skip
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_driver_role_skipped(frontend, horizontal):
    """
    When role='driver', no transformation is applied.
    """
    fcode = """
subroutine test_driver(kidia, kfdia, klon, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zb(klon)
  integer :: jl

  do jl = kidia, kfdia
    zb(jl) = za(jl)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InsertSIMDPragmaDirectives(horizontal=horizontal)
    trafo.apply(routine, role='driver')

    pragmas = FindNodes(Pragma).visit(routine.body)
    simd_pragmas = [p for p in pragmas if p.content and 'SIMD' in p.content.upper()]
    assert len(simd_pragmas) == 0
