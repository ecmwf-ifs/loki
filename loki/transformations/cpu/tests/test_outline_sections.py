# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`ExtractOutlinePhysicsSection` (T4).
"""

import pytest

from loki import Dimension, Subroutine, Sourcefile
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, CallStatement, Assignment, Comment
from loki.backend import fgen

from loki.transformations.cpu.outline_sections import (
    ExtractOutlinePhysicsSection,
)


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


# -----------------------------------------------------------------
# Pragma mode — basic test
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_pragma_mode_basic(frontend, horizontal):
    """
    A routine with ``!$loki outline`` pragma regions is split into
    subroutine calls via pragma mode.
    """
    fcode = """
subroutine test_pragma_outline(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1

!$loki outline in(a) out(b)
  b = a + 1
!$loki end outline

  c = a + b
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Before transformation: 3 assignments, no calls
    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    trafo.apply(routine, role='kernel')

    # After transformation: the outlined region is replaced by a call
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1

    # The remaining assignments should be a=1 and c=a+b (2 left)
    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_pragma_mode_multiple_regions(frontend, horizontal):
    """
    Multiple ``!$loki outline`` regions are each replaced by a call.
    """
    fcode = """
subroutine test_multi_outline(a, b, c, d)
  integer, intent(out) :: a, b, c, d

  a = 1

!$loki outline in(a) out(b)
  b = a + 1
!$loki end outline

!$loki outline in(a,b) out(c)
  c = a + b
!$loki end outline

  d = c * 2
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(FindNodes(Assignment).visit(routine.body)) == 4

    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    trafo.apply(routine, role='kernel')

    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 2

    # a=1 and d=c*2 remain
    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 2


# -----------------------------------------------------------------
# Driver role — skipped
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_driver_role_skipped(frontend, horizontal):
    """
    When role='driver', no outlining is applied.
    """
    fcode = """
subroutine test_driver_outline(a, b)
  integer, intent(out) :: a, b

  a = 1

!$loki outline in(a) out(b)
  b = a + 1
!$loki end outline
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    trafo.apply(routine, role='driver')

    # No calls should be created — outline regions untouched
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 0


# -----------------------------------------------------------------
# No pragma regions — routine untouched
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_pragma_mode_no_regions(frontend, horizontal):
    """
    A routine with no ``!$loki outline`` pragmas is left untouched.
    """
    fcode = """
subroutine test_no_outline(a, b)
  integer, intent(out) :: a, b

  a = 1
  b = a + 1
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    trafo.apply(routine, role='kernel')

    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 0

    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 2


# -----------------------------------------------------------------
# Heuristic mode — section header detection
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_heuristic_mode_dashes(frontend, horizontal):
    """
    Heuristic mode detects sections delimited by lines of dashes.
    With min_section_lines=1 (lowered for testing), the section
    between the two dash-comment headers is outlined.
    """
    # Build a source with two comment-delimited sections.
    # Each section has assignments between the headers.
    fcode = """
subroutine test_heuristic(a, b, c, d)
  integer, intent(inout) :: a, b, c, d

! ---------- Section 1 ----------
  a = 1
  b = a + 1
! ---------- Section 2 ----------
  c = b + 1
  d = c + 1
! ---------- End ----------
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assigns_before = len(FindNodes(Assignment).visit(routine.body))
    assert assigns_before == 4

    trafo = ExtractOutlinePhysicsSection(
        horizontal=horizontal, mode='heuristic', min_section_lines=1
    )
    trafo.apply(routine, role='kernel')

    # The heuristic mode should have detected section boundaries
    # and created call statements for the outlined regions
    calls = FindNodes(CallStatement).visit(routine.body)
    # With 3 headers, there are 2 sections (between consecutive headers)
    assert len(calls) >= 1

    code = fgen(routine).upper()
    assert 'CALL' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_heuristic_mode_min_section_lines(frontend, horizontal):
    """
    Sections shorter than ``min_section_lines`` are NOT outlined.
    """
    fcode = """
subroutine test_heuristic_min(a, b, c)
  integer, intent(inout) :: a, b, c

! ---------- Section 1 ----------
  a = 1
  b = a + 1
! ---------- Section 2 ----------
  c = b + 1
! ---------- End ----------
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # With a high min_section_lines, no sections should be outlined
    trafo = ExtractOutlinePhysicsSection(
        horizontal=horizontal, mode='heuristic', min_section_lines=100
    )
    trafo.apply(routine, role='kernel')

    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_heuristic_mode_too_few_headers(frontend, horizontal):
    """
    With fewer than two section headers, nothing happens.
    """
    fcode = """
subroutine test_one_header(a, b)
  integer, intent(inout) :: a, b

! ---------- Single header ----------
  a = 1
  b = a + 1
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ExtractOutlinePhysicsSection(
        horizontal=horizontal, mode='heuristic', min_section_lines=1
    )
    trafo.apply(routine, role='kernel')

    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 0


# -----------------------------------------------------------------
# Section name extraction
# -----------------------------------------------------------------

def test_extract_section_name():
    """
    Test the _extract_section_name helper for various comment patterns.
    """
    trafo = ExtractOutlinePhysicsSection.__new__(ExtractOutlinePhysicsSection)

    # "SECTION N: title" pattern
    assert trafo._extract_section_name('! SECTION 1: condensation', 0) == 'condensation'
    assert trafo._extract_section_name('! SECTION 3', 0) == 'SECTION_3'

    # "N. Title" pattern
    assert trafo._extract_section_name('! 2. Precipitation', 0) == 'Precipitation'

    # Fallback
    assert trafo._extract_section_name('! ----------', 5) == 'SECTION_5'


# -----------------------------------------------------------------
# Sequential routine — skipped
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_sequential_routine_skipped(frontend, horizontal):
    """
    A routine marked with ``!$loki routine seq`` should be skipped.
    """
    fcode = """
subroutine test_seq_outline(a, b)
  integer, intent(out) :: a, b
!$loki routine seq

  a = 1

!$loki outline in(a) out(b)
  b = a + 1
!$loki end outline
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    trafo.apply(routine, role='kernel')

    # Should be skipped — no calls created
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 0


# -----------------------------------------------------------------
# Sourcefile-level placement — new routines at file scope
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_placement_pragma(frontend, horizontal):
    """
    When applied to a Sourcefile, newly created subroutines are
    appended at file scope (critical for free-standing routines
    where routine.parent is None).
    """
    fcode = """
subroutine test_file_outline(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1

!$loki outline in(a) out(b)
  b = a + 1
!$loki end outline

  c = a + b
end subroutine
""".strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['test_file_outline']
    assert routine is not None

    # Before: 1 routine in the file
    assert len(source.subroutines) == 1

    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    trafo.apply(source, role='kernel')

    # After: the original routine has a CALL replacing the outlined region
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1

    # The new outlined subroutine is placed at file scope
    assert len(source.subroutines) == 2

    # The new routine is callable (its name matches the call)
    call_name = calls[0].name.name.lower()
    new_routine_names = [r.name.lower() for r in source.subroutines]
    assert call_name in new_routine_names


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_placement_multiple(frontend, horizontal):
    """
    Multiple outline regions produce multiple new subroutines at
    file scope.
    """
    fcode = """
subroutine test_multi_file(a, b, c, d)
  integer, intent(out) :: a, b, c, d

  a = 1

!$loki outline in(a) out(b)
  b = a + 1
!$loki end outline

!$loki outline in(a,b) out(c)
  c = a + b
!$loki end outline

  d = c * 2
end subroutine
""".strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    assert len(source.subroutines) == 1

    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    trafo.apply(source, role='kernel')

    # 2 new routines + the original = 3
    assert len(source.subroutines) == 3

    # The original routine has 2 calls
    routine = source['test_multi_file']
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_no_regions_unchanged(frontend, horizontal):
    """
    When there are no pragma regions, the Sourcefile is unchanged.
    """
    fcode = """
subroutine test_nochange(a, b)
  integer, intent(out) :: a, b
  a = 1
  b = a + 1
end subroutine
""".strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    assert len(source.subroutines) == 1

    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    trafo.apply(source, role='kernel')

    # Still just 1 routine
    assert len(source.subroutines) == 1


# -----------------------------------------------------------------
# Class attributes
# -----------------------------------------------------------------

def test_class_attributes():
    """
    Verify that the transformation declares the correct class
    attributes for scheduler integration.
    """
    assert ExtractOutlinePhysicsSection.traverse_file_graph is True
    assert ExtractOutlinePhysicsSection.creates_items is True


# -----------------------------------------------------------------
# ASSOCIATE block resolution before outlining
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_associate_resolved_before_outline(frontend, horizontal):
    """
    When a routine contains ASSOCIATE blocks, they must be resolved
    before outlining.  Without resolution, derived-type component
    accesses aliased through ASSOCIATE cause ``order_variables_by_type``
    to crash because the derived-type name is not among the imported
    symbols.

    This test creates a routine with:
    - A derived-type argument (TYPE(MY_TYPE))
    - An ASSOCIATE block mapping ``val => obj%field``
    - A ``!$loki outline`` region *inside* the ASSOCIATE that uses
      the alias ``val``

    The transformation must resolve the ASSOCIATE (turning ``val``
    back into ``obj%field``) so that outlining can correctly build
    the argument list with ``obj`` as an INOUT derived-type argument.
    """
    fcode = """
module my_type_mod
  implicit none
  type :: my_type
    integer :: field
  end type
end module

subroutine test_assoc_outline(obj, klon, a)
  use my_type_mod, only: my_type
  implicit none
  type(my_type), intent(inout) :: obj
  integer, intent(in) :: klon
  integer, intent(out) :: a(klon)
  integer :: jl

  associate(val => obj%field)

!$loki outline
    do jl = 1, klon
      a(jl) = val + jl
    end do
!$loki end outline

  end associate
end subroutine
""".strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['test_assoc_outline']
    assert routine is not None

    # Before: 1 routine, no calls, has ASSOCIATE
    assert len(source.subroutines) == 1
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    # This must not crash with ValueError in order_variables_by_type
    trafo.apply(source, role='kernel')

    # After: the ASSOCIATE is resolved and the outlined region
    # is replaced by a CALL
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1

    # The new subroutine is placed at file scope
    assert len(source.subroutines) == 2

    # The generated code should reference obj%field, not 'val'
    new_routine = [r for r in source.subroutines if r.name != 'test_assoc_outline'][0]
    new_code = fgen(new_routine).lower()
    assert 'obj' in new_code


@pytest.mark.parametrize('frontend', available_frontends())
def test_associate_multiple_derived_types(frontend, horizontal):
    """
    Outlining with multiple derived-type arguments aliased through
    ASSOCIATE.  This mirrors the CLOUDSC pattern where many
    ``YD*%component`` accesses are aliased.
    """
    fcode = """
module types_mod
  implicit none
  type :: t_phys
    real :: gravity
    real :: rd
  end type
  type :: t_cloud
    integer :: ntop
    real :: rlmin
  end type
end module

subroutine test_multi_assoc(ydphys, ydcloud, klon, result)
  use types_mod, only: t_phys, t_cloud
  implicit none
  type(t_phys), intent(in) :: ydphys
  type(t_cloud), intent(in) :: ydcloud
  integer, intent(in) :: klon
  real, intent(out) :: result(klon)
  integer :: jl

  associate(rg => ydphys%gravity, ntop => ydcloud%ntop, &
            rlmin => ydcloud%rlmin)

!$loki outline
    do jl = 1, klon
      result(jl) = rg * rlmin + real(ntop)
    end do
!$loki end outline

  end associate
end subroutine
""".strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['test_multi_assoc']

    trafo = ExtractOutlinePhysicsSection(horizontal=horizontal, mode='pragma')
    # Must not crash
    trafo.apply(source, role='kernel')

    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert len(source.subroutines) == 2
