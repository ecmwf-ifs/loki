"""
Test identity of source-to-source translation.

The tests in here do rarely verify correct representation internally,
they mostly check whether at the end comes out what went in at the beginning.

"""
from pathlib import Path
import pytest
from loki import SourceFile, Subroutine, OFP, OMNI, FP, fgen, FindNodes
import loki.ir as ir


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='OMNI stores no source.string')),
    FP])
def test_raw_source(here, frontend):
    """Verify that the raw_source property is correctly used to annotate
    AST nodes with source strings."""
    fcode = """
subroutine routine_raw_source (ia, ib, ic)
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

check: if (ib > 0) then
  print *, ia
end if check

multicond: select case (ic)
  case (10)
    print *, ic
  case (ia)
    print *, ia
  case default
    print *, ib
end select multicond

if (ic == 1) print *, ic
end subroutine routine_raw_source
    """.strip()
    filename = here / ('routine_raw_source_%s.f90' % frontend)
    SourceFile.to_file(fcode, filename)

    source = SourceFile.from_file(filename, frontend=frontend)
    routine = source['routine_raw_source']

    fcode = fcode.splitlines()

    # Check the intrinsics
    intrinsic_lines = (9, 11, 17, 22, 24, 26, 29)
    for node in FindNodes(ir.Intrinsic).visit(routine.body):
        # Verify that source string is subset of the relevant line in the original source
        assert node.source is not None
        assert node.source.lines in ((l, l) for l in intrinsic_lines)
        assert node.source.string in fcode[node.source.lines[0]-1]

    # Check the do loops
    do_construct_label_found = False  # Note: this is the construct name 'outer'
    loop_label_found = False  # Note: this is the do label '6'
    do_lines = ((4, 14), (6, 13))
    for node in FindNodes((ir.Loop, ir.WhileLoop)).visit(routine.ir):
        # Verify that source string is subset of the relevant line in the original source
        assert node.source is not None
        assert node.source.lines in do_lines
        assert node.source.string in ('\n'.join(fcode[start-1:end]) for start, end in do_lines)
        # Make sure the label is correctly identified and contained
        if node.label:
            do_construct_label_found = ~do_construct_label_found  # This way to ensure it is found only once
            assert node.label == 'outer:'
        if node.loop_label:
            loop_label_found = ~loop_label_found
            assert node.loop_label == '6'
    assert do_construct_label_found
    assert loop_label_found

    # Check the conditionals
    cond_label_found = 0
    cond_lines = ((8, 12), (16, 18), (20, 27), (29, 29))
    conditions = {8: (8,), 16: (16,), 20: (21, 23), 29: (29,)}
    for node in FindNodes((ir.Conditional, ir.MultiConditional)).visit(routine.ir):
        assert node.source is not None
        assert node.source.lines in cond_lines
        # Make sure that conditionals/cases have source information
        if isinstance(node, ir.Conditional):
            assert all(cond.source.lines[0] == cond.source.lines[1] and
                       cond.source.lines[0] in conditions[node.source.lines[0]]
                       for cond in node.conditions)
        elif isinstance(node, ir.MultiConditional):
            assert all(val.source.lines[0] == val.source.lines[1] and
                       val.source.lines[0] in conditions[node.source.lines[0]]
                       for val in node.values)
        # Verify that source string is subset of the relevant lines in the original source
        assert node.source.string in ('\n'.join(fcode[start-1:end]) for start, end in cond_lines)
        if node.label:
            cond_label_found += 1
            assert node.label in ('check:', 'multicond:')

    assert cond_label_found == 2


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='This is outright impossible')),
    FP
])
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


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='This is outright impossible')),
    FP
])
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
