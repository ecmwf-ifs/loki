"""
Unit tests for utility functions and classes in loki.tools.
"""

import pytest
from loki import Subroutine, pprint, FindNodes
from loki.frontend import OFP, OMNI, FP
from loki.ir import Pragma, Loop
from loki.tools import (
    JoinableStringList, truncate_string, is_loki_pragma, get_pragma_parameters,
    inline_pragmas, detach_pragmas, pragmas_attached
)


@pytest.mark.parametrize('items, sep, width, cont, ref', [
    ([''], ' ', 90, '\n', ''),
    ([], ' ', 90, '\n', ''),
    (('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'), '', 90, '\n', 'Hello world!'),
    (('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'), '', 7, '\n', 'Hello \nworld!'),
    (('Hello', 'world!'), ' ', 90, '\n', 'Hello world!'),
    (('Hello', 'world!'), ' ', 7, '\n', 'Hello \nworld!'),
    (('Hello', 'world!'), ' ', 5, '\n', 'Hello \nworld!'),
    ((JoinableStringList(['H', 'e', 'l', 'l', 'o'], '', 5, '\n'), 'world!'), ' ', 5, '\n',
     'Hell\no \nworld!'),
    (('Hello', JoinableStringList(['w', 'o', 'r', 'l', 'd', '!'], '', 8, '\n', separable=False)),
     ' ', 8, '\n', 'Hello \nworld!'),
    (('Hello', JoinableStringList(['w', 'o', 'r', 'l', 'd', '!'], '', 8, '\n', separable=True)),
     ' ', 8, '\n', 'Hello w\norld!'),
])
def test_joinable_string_list(items, sep, width, cont, ref):
    """
    Test JoinableStringList for some common scenarios.
    """
    obj = JoinableStringList(items, sep, width, cont)
    assert str(obj) == ref


def test_joinable_string_list_long():
    """
    Test JoinableStringList with some long edge cases.
    """
    attributes = ['REAL(KIND=JPRB)', 'INTENT(IN)']
    attributes = JoinableStringList(attributes, ', ', 132, ' &\n   & ')
    variables = ['PDHTLS(KPROMA, YDMODEL%YRML_PHY_G%YRDPHY%NTILES, '
                 'YDMODEL%YRML_DIAG%YRMDDH%NDHVTLS + YDMODEL%YRML_DIAG%YRMDDH%NDHFTLS)']
    variables = JoinableStringList(variables, ', ', 132, ' &\n   & ')
    items = ['  ', attributes, ' :: ', variables]
    obj = JoinableStringList(items, '', 132, ' &\n  & ')
    ref = ('  REAL(KIND=JPRB), INTENT(IN) ::  &\n'
           '  & PDHTLS(KPROMA, YDMODEL%YRML_PHY_G%YRDPHY%NTILES, '
           'YDMODEL%YRML_DIAG%YRMDDH%NDHVTLS + YDMODEL%YRML_DIAG%YRMDDH%NDHFTLS)')
    assert str(obj) == ref

    name = 'io.output'
    args = ['"tensor_out"', 'tensor_out',
            'new DFEVectorType<DFEVector<DFEVar>>(new DFEVectorType<DFEVar>(dfeFloat(11, 53), m), n)']
    args_list = JoinableStringList(args, sep=', ', width=90, cont='\n      ', separable=True)
    items = ['    ', name, '(', args_list, ');']
    items_list = JoinableStringList(items, sep='', width=90, cont='\n      ', separable=True)
    line = str(items_list)
    ref = ('    io.output("tensor_out", tensor_out, \n'
           '      new DFEVectorType<DFEVector<DFEVar>>(new DFEVectorType<DFEVar>(dfeFloat(11, 53), \n'
           '      m), n));')
    assert line == ref


@pytest.mark.parametrize('string, length, continuation, ref', [
    ('short string', 16, '...', 'short string'),
    ('short string', 12, '...', 'short string'),
    ('short string', 11, '...', 'short st...'),
])
def test_truncate_string(string, length, continuation, ref):
    """
    Test string truncation for different string lengths.
    """
    assert truncate_string(string, length, continuation) == ref


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

    orig_loops = FindNodes(Loop).visit(routine.body)
    assert len(orig_loops) == 2
    assert all(loop.pragma is not None for loop in orig_loops)
    assert not FindNodes(Pragma).visit(routine.body)

    # Serialize pragmas
    ir = detach_pragmas(routine.body)

    loops = FindNodes(Loop).visit(ir)
    assert len(loops) == 2
    assert all(loop.pragma is None for loop in loops)
    pragmas = FindNodes(Pragma).visit(ir)
    assert len(pragmas) == 4

    # Inline pragmas again
    ir = inline_pragmas(ir)

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
    routine.spec = detach_pragmas(routine.spec)
    routine.body = detach_pragmas(routine.body)

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
    routine.spec = detach_pragmas(routine.spec)
    routine.body = detach_pragmas(routine.body)

    loop_of_interest = None
    with pragmas_attached(routine, Loop):
        for loop in FindNodes(Loop).visit(routine.body):
            if is_loki_pragma(loop.pragma, starts_with='foobar'):
                loop_of_interest = loop
                break

    assert loop_of_interest is not None
    assert loop_of_interest.pragma is None
