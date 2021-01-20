import pytest

from loki import FP, OFP, OMNI, Subroutine, FindNodes, Assignment
from loki.analyse import defined_symbols_attached


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_analyse_defined_symbols_attached(frontend):
    fcode = """
subroutine analyse_defined_symbols_attached(v1, v2, v3)
  integer, intent(in) :: v1
  integer, intent(inout) :: v2
  integer, intent(out) :: v3
  integer :: i, j, n=10, tmp, a

  do i=1,n
    do j=1,n
      tmp = j + 1
    end do
    a = v2 + tmp
  end do

  v3 = v1 + v2
  v2 = a
end subroutine analyse_defined_symbols_attached
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assignments = FindNodes(Assignment).visit(routine.body)
    assert len(assignments) == 4

    with pytest.raises(RuntimeError):
        for assignment in assignments:
            _ = assignment.defined_symbols

    ref_defined_symbols = {
        'tmp': {'i', 'j', 'n', 'v1', 'v2'},
        'a': {'i', 'tmp', 'n', 'v1', 'v2'},
        'v3': {'tmp', 'a', 'n', 'v1', 'v2'},
        'v2': {'tmp', 'a', 'n', 'v1', 'v2', 'v3'}
    }

    with defined_symbols_attached(routine):
        for assignment in assignments:
            assert assignment.defined_symbols == ref_defined_symbols[str(assignment.lhs).lower()]

    with pytest.raises(RuntimeError):
        for assignment in assignments:
            _ = assignment.defined_symbols
