"""
Test identity of source-to-source translation.

The tests in here do rarely verify correct representation internally,
they mostly check whether at the end comes out what went in at the beginning.

"""
from pathlib import Path
import pytest
from conftest import clean_test, available_frontends
from loki import (
  Sourcefile, Subroutine, OMNI, fgen, FindNodes, as_tuple, ir
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI stores no source.string')]))
def test_raw_source_loop(here, frontend):
    """Verify that the raw_source property is correctly used to annotate
    AST nodes with source strings for loops."""
    fcode = """
subroutine routine_raw_source_loop (ia, ib, ic)
integer, intent(in) :: ia, ib, ic

outer: do ia=1,10
  ib = ia
  do 6 while (ib .lt. 20)
    ic = ib
    if (ic .gt. 10) then
      print *, ic
    else
      print *, ib
    end if
6 end do
end do outer
end subroutine routine_raw_source_loop
    """.strip()
    filename = here / ('routine_raw_source_loop_%s.f90' % frontend)
    Sourcefile.to_file(fcode, filename)

    source = Sourcefile.from_file(filename, frontend=frontend)
    routine = source['routine_raw_source_loop']
    assert source.source.string == fcode
    assert routine.source.string == fcode

    fcode = fcode.splitlines()
    assert source.source.lines == (1, len(fcode))
    assert routine.source.lines == (1, len(fcode))

    # Check the intrinsics
    intrinsic_lines = (9, 11)
    for node in FindNodes(ir.Intrinsic).visit(routine.body):
        # Verify that source string is subset of the relevant line in the original source
        assert node.source is not None
        assert node.source.lines in ((l, l) for l in intrinsic_lines)
        assert node.source.string in fcode[node.source.lines[0]-1]

    # Check the do loops
    do_construct_name_found = 0  # Note: this is the construct name 'outer'
    loop_label_found = 0  # Note: this is the do label '6'
    do_lines = ((4, 14), (6, 13))
    for node in FindNodes((ir.Loop, ir.WhileLoop)).visit(routine.ir):
        # Verify that source string is subset of the relevant line in the original source
        assert node.source is not None
        assert node.source.lines in do_lines
        assert node.source.string in ('\n'.join(fcode[start-1:end]) for start, end in do_lines)
        # Make sure the labels and names are correctly identified and contained
        if node.name:
            do_construct_name_found += 1
            assert node.name == 'outer'
        if node.loop_label:
            loop_label_found += 1
            assert node.loop_label == '6'
    assert do_construct_name_found == 1
    assert loop_label_found == 1

    # Assert output of body matches original string (except for case)
    ref = '\n'.join(fcode[3:-1]).replace('.lt.', '<').replace('.gt.', '>')
    assert fgen(routine.body).strip().lower() == ref

    clean_test(filename)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI stores no source.string')]))
def test_raw_source_conditional(here, frontend):
    """Verify that the raw_source property is correctly used to annotate
    AST nodes with source strings for conditionals."""
    fcode = """
subroutine routine_raw_source_cond (ia, ib, ic)
integer, intent(in) :: ia, ib, ic

check: if (ib > 0) then
  print *, ia
else if (ib == 0) then check
  print *, ib
else check
  print *, ic
end if check
if (ic == 1) print *, ic
end subroutine routine_raw_source_cond
    """.strip()
    filename = here / ('routine_raw_source_cond_%s.f90' % frontend)
    Sourcefile.to_file(fcode, filename)

    source = Sourcefile.from_file(filename, frontend=frontend)
    routine = source['routine_raw_source_cond']
    assert source.source.string == fcode
    assert routine.source.string == fcode

    fcode = fcode.splitlines()
    assert source.source.lines == (1, len(fcode))
    assert routine.source.lines == (1, len(fcode))

    # Check the intrinsics
    intrinsic_lines = (5, 7, 9, 11)
    for node in FindNodes(ir.Intrinsic).visit(routine.body):
        # Verify that source string is subset of the relevant line in the original source
        assert node.source is not None
        assert node.source.lines in ((l, l) for l in intrinsic_lines)
        assert node.source.string in fcode[node.source.lines[0]-1]

    # Check the conditionals
    cond_name_found = 0
    cond_lines = ((4, 10), (6, 10), (11, 11))
    for node in FindNodes(ir.Conditional).visit(routine.ir):
        assert node.source is not None
        assert node.source.lines in cond_lines
        # Make sure that conditionals have source information
        assert node.condition.source.lines[0] == node.condition.source.lines[0]
        assert node.condition.source.lines[0] == node.source.lines[0]
        # Verify that source string is subset of the relevant lines in the original source
        assert node.source.string in ('\n'.join(fcode[start-1:end]) for start, end in cond_lines)
        if node.name:
            cond_name_found += 1
            assert node.name == 'check'
    assert cond_name_found == 1

    # Assert output of body matches original string (except for case)
    assert fgen(routine.body).strip().lower() == '\n'.join(fcode[3:-1])

    clean_test(filename)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI stores no source.string')]))
def test_raw_source_multicond(here, frontend):
    """Verify that the raw_source property is correctly used to annotate
    AST nodes with source strings for multi conditionals."""
    fcode = """
subroutine routine_raw_source_multicond (ia, ib, ic)
integer, intent(in) :: ia, ib, ic

multicond: select case (ic)
case (10) multicond
  print *, ic
case (ia) multicond
  print *, ia
case default multicond
  print *, ib
end select multicond
end subroutine routine_raw_source_multicond
    """.strip()
    filename = here / ('routine_raw_source_multicond_%s.f90' % frontend)
    Sourcefile.to_file(fcode, filename)

    source = Sourcefile.from_file(filename, frontend=frontend)
    routine = source['routine_raw_source_multicond']
    assert source.source.string == fcode
    assert routine.source.string == fcode

    fcode = fcode.splitlines()
    assert source.source.lines == (1, len(fcode))
    assert routine.source.lines == (1, len(fcode))

    # Check the intrinsics
    intrinsic_lines = (6, 8, 10)
    for node in FindNodes(ir.Intrinsic).visit(routine.body):
        # Verify that source string is subset of the relevant line in the original source
        assert node.source is not None
        assert node.source.lines in ((l, l) for l in intrinsic_lines)
        assert node.source.string in fcode[node.source.lines[0]-1]

    # Check the conditional
    cond_name_found = 0
    cond_lines = ((4, 11),)
    conditions = {4: (5, 7)}
    for node in FindNodes(ir.MultiConditional).visit(routine.ir):
        assert node.source is not None
        assert node.source.lines in cond_lines
        # Make sure that cases have source information
        for value in node.values:
            assert all(val.source.lines[0] == val.source.lines[1] and
                       val.source.lines[0] in conditions[node.source.lines[0]]
                       for val in as_tuple(value))
        # Verify that source string is subset of the relevant lines in the original source
        assert node.source.string in ('\n'.join(fcode[start-1:end]) for start, end in cond_lines)
        if node.name:
            cond_name_found += 1
            assert node.name == 'multicond'
    assert cond_name_found == 1

    # Assert output of body matches original string (except for case)
    assert fgen(routine.body).strip().lower() == '\n'.join(fcode[3:-1])

    clean_test(filename)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'This is outright impossible')]))
def test_subroutine_conservative(frontend):
    """
    Test that conservative output of fgen reproduces the original source string for
    a simple subroutine.
    This has a few limitations, in particular with respect to the signature of the
    subroutine.
    """
    fcode = """
SUBROUTINE CONSERVATIVE (X, Y, SCALAR, VECTOR, MATRIX)
  IMPLICIT NONE
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)
  INTEGER, INTENT(IN) :: X, Y
  REAL(KIND=JPRB), INTENT(IN) :: SCALAR
  REAL(KIND=JPRB), INTENT(INOUT) :: VECTOR(X)
  REAL(KIND=JPRB), DIMENSION(X, Y), INTENT(OUT) :: MATRIX
  INTEGER :: I, SOME_VERY_LONG_INTEGERS, TO_SEE_IF_LINE_CONTUATION, IS_NOT_ENFORCED_IN_THIS_CASE
  ! Some comment that is very very long and exceeds the line width but should not be wrapped
  DO I=1, X
    VECTOR(I) = VECTOR(I) + SCALAR
!$LOKI SOME NONSENSE PRAGMA WITH VERY LONG TEXT THAT EXCEEDS THE LINE WIDTH LIMIT OF THE OUTPUT
    MATRIX(I, :) = I * VECTOR(I)
  ENDDO
END SUBROUTINE CONSERVATIVE
    """.strip()

    # Parse and re-generate the code
    routine = Subroutine.from_source(fcode, frontend=frontend)
    source = fgen(routine, linewidth=90, conservative=True)
    assert source == fcode


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'This is outright impossible')]))
def test_subroutine_simple_fgen(frontend):
    """
    Test that non-conservative output produces the original source string for
    a simple subroutine.
    This has a few limitations, in particular for formatting of expressions.
    """
    fcode = """
SUBROUTINE SIMPLE_FGEN (X, Y, SCALAR, VECTOR, MATRIX)
  IMPLICIT NONE
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)
  INTEGER, INTENT(IN) :: X, Y
  REAL(KIND=JPRB), INTENT(IN) :: SCALAR  ! A very long inline comment that should not be wrapped but simply appended
  REAL(KIND=JPRB), INTENT(INOUT) :: VECTOR(X)
  REAL(KIND=JPRB), DIMENSION(X, Y), INTENT(OUT) :: MATRIX
  INTEGER :: I, SOME_VERY_LONG_INTEGERS, TO_SEE_IF_LINE_CONTINUATION,  &
  & WORKS_AS_EXPECTED_IN_ITEM_LISTS
  ! Some comment that is very very long and exceeds the line width but should not be wrapped
  DO I=1,X
    VECTOR(I) = VECTOR(I) + SCALAR
!$LOKI SOME NONSENSE PRAGMA WITH VERY LONG TEXT THAT EXCEEDS THE LINE WIDTH LIMIT OF THE OUTPUT
    MATRIX(I, :) = I*VECTOR(I)
    IF (SOME_VERY_LONG_INTEGERS > X) THEN
      ! Some comment to have more than one line
      ! in the body of the condtional
      IF (TO_SEE_IF_LINE_CONTINUATION > Y) THEN
        PRINT *, 'Intrinsic statement'
      END IF
    END IF
  END DO
END SUBROUTINE SIMPLE_FGEN
    """.strip()

    # Parse and write the code
    routine = Subroutine.from_source(fcode, frontend=frontend)
    source = fgen(routine, linewidth=90)
    assert source == fcode


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does it for you BUT WITHOUT DELETING THE KEYWORD!!!')])
)
def test_multiline_pragma(frontend):
    """
    Test that multi-line pragmas are combined correctly.
    """
    fcode = """
subroutine multiline_pragma
  implicit none
  integer :: dummy
!$foo some very long pragma &
!$foo with a line break
  dummy = 1
!$bar some pragma         &
!$bar with more than      &
!$bar one line break
!$bar followed by    &
!$bar    another multiline pragma &
!$bar    with same keyword
  dummy = dummy + 1
!$foobar and yet &
!$foobar another multiline pragma
end subroutine multiline_pragma
    """.strip()

    # Parse the code
    routine = Subroutine.from_source(fcode, frontend=frontend)
    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    pragma_content = {
        'foo': ['some very long pragma with a line break'],
        'bar': ['some pragma with more than one line break',
                'followed by another multiline pragma with same keyword'],
        'foobar': ['and yet another multiline pragma']
    }

    assert len(pragmas) == 4
    assert all(pragma.content in pragma_content[pragma.keyword] for pragma in pragmas)
