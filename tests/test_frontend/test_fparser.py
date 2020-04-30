from pathlib import Path
import pytest

from loki import SourceFile, FP, FindNodes
import loki.ir as ir


@pytest.fixture(scope='module', name='testpath')
def fixture_testpath():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='frontend')
def fixture_frontend():
    return FP


def test_raw_source(testpath, frontend):
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
end subroutine routine_raw_source
    """.strip()
    filename = testpath / ('routine_raw_source_%s.f90' % frontend)
    SourceFile.to_file(fcode, filename)

    source = SourceFile.from_file(filename, frontend=frontend)
    routine = source['routine_raw_source']

    fcode = fcode.splitlines(keepends=True)

    # Check the intrinsics
    intrinsic_lines = (9, 11, 17, 22, 24, 26)
    for node in FindNodes(ir.Intrinsic).visit(routine.ir):
        assert node._source is not None
        assert node._source.lines in ((l, l) for l in intrinsic_lines)
        assert node._source.string in (fcode[l-1] for l in intrinsic_lines)

    # Check the do loops
    loop_label_found = False  # Note: this is the construct name 'outer'
    labeled_do_found = False  # Note: this is the do label '6'
    do_lines = ((4, 14), (6, 13))
    for node in FindNodes((ir.Loop, ir.WhileLoop)).visit(routine.ir):
        assert node._source is not None
        assert node._source.lines in do_lines
        assert node._source.string in (''.join(fcode[start-1:end]) for start, end in do_lines)
        if node._source.label:
            loop_label_found = ~loop_label_found  # This way to ensure it is found only once
            assert node._source.label == 'outer'
        if node.label:
            labeled_do_found = ~labeled_do_found
            assert node.label == '6'
    assert loop_label_found
    assert labeled_do_found

    # Check the conditionals
    cond_label_found = 0
    cond_lines = ((8, 12), (16, 18), (20, 27))
    for node in FindNodes((ir.Conditional, ir.MultiConditional)).visit(routine.ir):
        assert node._source is not None
        assert node._source.lines in cond_lines
        assert node._source.string in (''.join(fcode[start-1:end]) for start, end in cond_lines)
        if node._source.label:
            cond_label_found += 1
            assert node._source.label in ('check', 'multicond')
    assert cond_label_found == 2
