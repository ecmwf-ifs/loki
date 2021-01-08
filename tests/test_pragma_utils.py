import pytest

from loki import Module, Subroutine, pprint, FindNodes, flatten
from loki.frontend import OFP, OMNI, FP
from loki.ir import Pragma, Loop, Declaration, PragmaRegion
from loki.pragma_utils import (
    is_loki_pragma, get_pragma_parameters, attach_pragmas, detach_pragmas,
    pragmas_attached, pragma_regions_attached
)


@pytest.mark.parametrize('keyword, content, starts_with, ref', [
    ('foo', None, None, False),
    ('foo', 'bar', None, False),
    ('foo', 'loki', None, False),
    ('foo', 'loki', 'loki', False),
    ('loki', None, None, True),
    ('loki', None, 'foo', False),
    ('loki', 'dataflow', None, True),
    ('loki', 'dataflow', 'dataflow', True),
    ('loki', 'dataflow', 'foobar', False),
    ('loki', 'fusion group(1)', None, True),
    ('loki', 'fusion group(1)', 'fusion', True),
    ('loki', 'fusion group(1)', 'group', False),
])
def test_is_loki_pragma(keyword, content, starts_with, ref):
    """
    Test correct identification of Loki pragmas.
    """
    pragma = Pragma(keyword, content)
    pragma_list = (pragma,)
    if starts_with is not None:
        assert is_loki_pragma(pragma, starts_with=starts_with) == ref
        assert is_loki_pragma(pragma_list, starts_with=starts_with) == ref
    else:
        assert is_loki_pragma(pragma) == ref
        assert is_loki_pragma(pragma_list) == ref


@pytest.mark.parametrize('content, starts_with, ref', [
    (None, None, {}),
    ('', None, {}),
    ('', 'foo', {}),
    ('dataflow', None, {'dataflow': None}),
    ('dataflow', 'dataflow', {}),
    ('dataflow group(1)', None, {'dataflow': None, 'group': '1'}),
    ('dataflow group(1)', 'dataflow', {'group': '1'}),
    ('dataflow group(1)', 'foo', {}),
    ('dataflow group(1) group(2)', 'dataflow', {'group': '2'}),
    ('foo bar(^£!$%*[]:@+-_=~#/?.,<>;) baz foobar(abc_123")', 'foo',
     {'bar':'^£!$%*[]:@+-_=~#/?.,<>;', 'baz': None, 'foobar': 'abc_123"'}),
])
def test_get_pragma_parameters(content, starts_with, ref):
    """
    Test correct extraction of Loki pragma parameters.
    """
    pragma = Pragma('loki', content)
    pragma_list = (pragma,)
    if starts_with is None:
        assert get_pragma_parameters(pragma) == ref
        assert get_pragma_parameters(pragma_list) == ref
    else:
        assert get_pragma_parameters(pragma, starts_with=starts_with) == ref
        assert get_pragma_parameters(pragma_list, starts_with=starts_with) == ref


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_tools_pragma_inlining(frontend):
    """
    A short test that verifies pragmas that are the first statement
    in a routine's body are correctly identified and inlined.
    """
    fcode = """
subroutine test_tools_pragma_inlining (in, out, n)
  implicit none
  real, intent(in) :: in(:)
  real, intent(out) :: out(:)
  integer, intent(in) :: n
  integer :: i
  !$loki some pragma
  do i=1,n
    out(i) = in(i)
  end do
end subroutine test_tools_pragma_inlining
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Check that pragmas are not inlined
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].pragma is None

    # Now inline pragmas and see if everything matches
    routine.body = attach_pragmas(routine.body, Loop)
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].pragma is not None
    assert isinstance(loops[0].pragma, tuple) and len(loops[0].pragma) == 1
    assert loops[0].pragma[0].keyword == 'loki' and loops[0].pragma[0].content == 'some pragma'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_tools_pragma_inlining_multiple(frontend):
    """
    A short test that verifies that multiple pragmas are inlined
    and kept in the right order.
    """
    fcode = """
subroutine test_tools_pragma_inlining_multiple (in, out, n)
  implicit none
  real, intent(in) :: in(:)
  real, intent(out) :: out(:)
  integer, intent(in) :: n
  integer :: i
  !$blub other pragma
  !$loki some pragma(5)
  !$loki more
  do i=1,n
    out(i) = in(i)
  end do
end subroutine test_tools_pragma_inlining_multiple
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Check that pragmas are not inlined
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].pragma is None

    # Now inline pragmas and see if everything matches
    routine.body = attach_pragmas(routine.body, Loop)
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].pragma is not None
    assert isinstance(loops[0].pragma, tuple) and len(loops[0].pragma) == 3
    assert [p.keyword for p in loops[0].pragma] == ['blub', 'loki', 'loki']
    assert loops[0].pragma[0].content == 'other pragma'
    assert loops[0].pragma[1].content == 'some pragma(5)'
    assert loops[0].pragma[2].content == 'more'

    # A few checks for the pragma utility functions
    assert is_loki_pragma(loops[0].pragma)
    assert is_loki_pragma(loops[0].pragma, starts_with='some')
    assert is_loki_pragma(loops[0].pragma, starts_with='more')
    assert not is_loki_pragma(loops[0].pragma, starts_with='other')
    assert get_pragma_parameters(loops[0].pragma) == {'some': None, 'pragma': '5', 'more': None}
    assert get_pragma_parameters(loops[0].pragma, starts_with='some') == {'pragma': '5'}
    assert get_pragma_parameters(loops[0].pragma, only_loki_pragmas=False) == \
            {'some': None, 'pragma': '5', 'more': None, 'other': None}


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_tools_pragma_detach(frontend):
    """
    A short test that verifies that multiple pragmas are inlined
    and kept in the right order.
    """
    fcode = """
subroutine test_tools_pragma_detach (in, out, n)
  implicit none
  real, intent(in) :: in(:)
  real, intent(out) :: out(:)
  integer, intent(in) :: n
  integer :: i, j

!$blub other pragma
!$loki some pragma(5)
!$loki more
  do i=1,n
    out(i) = in(i)

!$loki inner pragma
    do j=1,n
      out(i) = out(i) + 1.0
    end do

  end do
end subroutine test_tools_pragma_detach
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Originally, pragmas shouldn't be inlined
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert all(loop.pragma is None for loop in loops)
    pragmas = FindNodes(Pragma).visit(routine.body)
    assert len(pragmas) == 4

    # Inline pragmas
    ir = attach_pragmas(routine.body, Loop)
    orig_loops = FindNodes(Loop).visit(ir)
    assert len(orig_loops) == 2
    assert all(loop.pragma is not None for loop in orig_loops)
    assert not FindNodes(Pragma).visit(ir)

    # Serialize pragmas
    ir = detach_pragmas(ir, Loop)

    loops = FindNodes(Loop).visit(ir)
    assert len(loops) == 2
    assert all(loop.pragma is None for loop in loops)
    pragmas = FindNodes(Pragma).visit(ir)
    assert len(pragmas) == 4

    # Inline pragmas again
    ir = attach_pragmas(ir, Loop)

    assert pprint(ir) == pprint(routine.body)
    loops = FindNodes(Loop).visit(ir)
    assert len(loops) == 2
    assert all(loop.pragma is not None for loop in loops)
    assert not FindNodes(Pragma).visit(ir)

    for loop, orig_loop in zip(loops, orig_loops):
        pragma = [p.keyword + ' ' + p.content for p in loop.pragma]
        orig_pragma = [p.keyword + ' ' + p.content for p in orig_loop.pragma]
        assert '\n'.join(pragma) == '\n'.join(orig_pragma)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_tools_pragmas_attached_loop(frontend):
    """
    A short test that verifies that the context manager to attach
    pragmas works as expected.
    """
    fcode = """
subroutine test_tools_pragmas_attached_loop(in, out, n)
  implicit none
  real, intent(in) :: in(:)
  real, intent(out) :: out(:)
  integer, intent(in) :: n
  integer :: i, j

!$blub other pragma
!$loki some pragma(5)
!$loki more
  do i=1,n
    out(i) = in(i)

!$loki inner pragma
    do j=1,n
      out(i) = out(i) + 1.0
    end do

  end do
end subroutine test_tools_pragmas_attached_loop
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert all(loop.pragma is None for loop in loops)
    assert len(FindNodes(Pragma).visit(routine.body)) == 4

    with pragmas_attached(routine, Loop):
        # Verify that pragmas are attached
        attached_loops = FindNodes(Loop).visit(routine.body)
        assert len(attached_loops) == 2
        assert all(loop.pragma is not None for loop in attached_loops)
        assert not FindNodes(Pragma).visit(routine.body)

        # Make sure that existing references to nodes still work
        # (and have been changed, too)
        assert all(loop.pragma is not None for loop in loops)

    # Check that the original state is restored
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert all(loop.pragma is None for loop in loops)
    assert len(FindNodes(Pragma).visit(routine.body)) == 4

    # Make sure that reference from inside the context still work
    # (and have their pragmas detached)
    assert all(loop.pragma is None for loop in attached_loops)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_tools_pragmas_attached_example(frontend):
    """
    A short test that verifies that the example from the docstring works.
    """
    fcode = """
subroutine test_tools_pragmas_attached_example (in, out, n)
  implicit none
  real, intent(in) :: in(:)
  real, intent(out) :: out(:)
  integer, intent(in) :: n
  integer :: i

  do i=1,n
    out(i) = 0.0
  end do

!$loki foobar
  do i=1,n
    out(i) = in(i)
  end do
end subroutine test_tools_pragmas_attached_example
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    loop_of_interest = None
    with pragmas_attached(routine, Loop):
        for loop in FindNodes(Loop).visit(routine.body):
            if is_loki_pragma(loop.pragma, starts_with='foobar'):
                loop_of_interest = loop
                break

    assert loop_of_interest is not None
    assert loop_of_interest.pragma is None


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_tools_pragmas_attached_post(frontend):
    """
    Verify the inlining of pragma_post.
    """
    fcode = """
subroutine test_tools_pragmas_attached_post(a, jtend, iend, jend)
  ! Code snippet example adapted from CLAW manual
  integer, intent(out) :: a(jend, iend, jtend)
  integer, intent(in) :: jtend, iend, jend
  integer :: jt, i, j

!$acc parallel loop gang vector collapse(2)
  DO jt=1,jtend
    DO i=1,iend
      DO j=1,jend
        a(j, i, jt) = j + i + jt
      END DO
    END DO
  END DO
!$acc end parallel
end subroutine test_tools_pragmas_attached_post
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(FindNodes(Pragma).visit(routine.body)) == 2
    loop = FindNodes(Loop).visit(routine.body)
    assert len(loop) == 3
    loop = loop[0]
    assert loop.pragma is None and loop.pragma_post is None

    with pragmas_attached(routine, Loop, attach_pragma_post=False):
        assert isinstance(loop.pragma, tuple) and len(loop.pragma) == 1
        assert loop.pragma[0].keyword.lower() == 'acc'
        assert loop.pragma_post is None
        assert len(FindNodes(Pragma).visit(routine.body)) == 1

    assert loop.pragma is None and loop.pragma_post is None

    # default behaviour: attach_pragma_post=True
    with pragmas_attached(routine, Loop):
        assert isinstance(loop.pragma, tuple) and len(loop.pragma) == 1
        assert loop.pragma[0].keyword.lower() == 'acc'
        assert isinstance(loop.pragma_post, tuple) and len(loop.pragma_post) == 1
        assert loop.pragma_post[0].keyword.lower() == 'acc'
        assert not FindNodes(Pragma).visit(routine.body)

    assert loop.pragma is None and loop.pragma_post is None
    assert len(FindNodes(Pragma).visit(routine.body)) == 2


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_tools_pragmas_attached_module(frontend):
    """
    Verify pragmas_attached works for Module objects.
    """
    fcode = """
module test_tools_pragmas_attached_module
  integer, allocatable :: a(:)
!$loki dimension(10, 20)
  integer, allocatable :: b(:,:)
end module test_tools_pragmas_attached_module
    """
    module = Module.from_source(fcode, frontend=frontend)

    assert len(FindNodes(Pragma).visit(module.spec)) == 1
    decl = FindNodes(Declaration).visit(module.spec)[1]
    assert len(decl.variables) == 1 and decl.variables[0].name.lower() == 'b'
    assert decl.pragma is None

    with pragmas_attached(module, Declaration):
        assert not FindNodes(Pragma).visit(module.spec)
        assert isinstance(decl.pragma, tuple) and is_loki_pragma(decl.pragma, starts_with='dimension')

    assert decl.pragma is None
    assert len(FindNodes(Pragma).visit(module.spec)) == 1


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_tools_pragma_regions_attached(frontend):
    """
    Verify ``pragma_regions_attached`` creates and removes `PragmaRegion` objects.
    """
    fcode = """
subroutine test_tools_pragmas_attached_region (in, out, n)
  implicit none
  real, intent(in) :: in(:)
  real, intent(out) :: out(:)
  integer, intent(in) :: n
  integer :: i

  out(0) = -1.0

!$loki whatever

  out(0) = -2.0

  !$loki do_something
  do i=1,n
    out(i) = 0.0
  end do
!$loki end whatever

  do i=1,n
    out(i) = 1.0
  end do

!$foo bar
  do i=1,n
    out(i) = in(i)
  end do
!$end foo bar
end subroutine test_tools_pragmas_attached_region
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3
    assert all(loop.pragma is None for loop in loops)
    assert len(FindNodes(Pragma).visit(routine.body)) == 5

    with pragma_regions_attached(routine):
        assert len(FindNodes(Pragma).visit(routine.body)) == 1
        assert len(FindNodes(PragmaRegion).visit(routine.body)) == 2
        # Find loops inside regions
        regions = FindNodes(PragmaRegion).visit(routine.body)
        region_loops = flatten(FindNodes(Loop).visit(r) for r in regions)
        assert len(region_loops) == 2
        assert all(l in loops for l in region_loops)

    # Verify that loops from context are still valid
    assert all(l in loops for l in region_loops)

    # Ensure that everything is back to where it was
    loops_after = FindNodes(Loop).visit(routine.body)
    assert len(loops_after) == 3
    assert loops_after == loops
    assert all(loop.pragma is None for loop in loops_after)
    assert len(FindNodes(Pragma).visit(routine.body)) == 5
