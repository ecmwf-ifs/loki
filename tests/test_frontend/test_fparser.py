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
end subroutine routine_raw_source
    """.strip()
    filename = testpath / ('routine_raw_source_%s.f90' % frontend)
    SourceFile.to_file(fcode, filename)

    source = SourceFile.from_file(filename, frontend=frontend)
    routine = source['routine_raw_source']

    fcode = fcode.splitlines(keepends=True)

    # Check the intrinsics
    for node in FindNodes(ir.Intrinsic).visit(routine.ir):
        assert node._source is not None
        assert node._source.lines in ((9, 9), (11, 11), (17, 17))
        assert node._source.string in (fcode[8], fcode[10], fcode[16])

    # Check the do loops
    loop_label_found = False  # Note: this is the construct name 'outer'
    labeled_do_found = False  # Note: this is the do label '6'
    for node in FindNodes(ir.Loop).visit(routine.ir):
        assert node._source is not None
        assert node._source.lines in ((4, 14), (6, 13))
        assert node._source.string in (''.join(fcode[3:14]), ''.join(fcode[5:13]))
        if node._source.label:
            loop_label_found = ~loop_label_found  # This way to ensure it is found only once
            assert node._source.label == 'outer'
        if node.label:
            labeled_do_found = ~labeled_do_found
            assert node.label == '6'
    assert loop_label_found
    assert labeled_do_found

    # Check the conditional
    cond_label_found = False
    for node in FindNodes(ir.Conditional).visit(routine.ir):
        assert node._source is not None
        assert node._source.lines in ((8, 12), (16, 18))
        assert node._source.string in (''.join(fcode[7:12]), ''.join(fcode[15:18]))
        if node._source.label:
            cond_label_found = ~cond_label_found  # This way to make sure it is found only once
            assert node._source.label == 'check'
    assert cond_label_found
