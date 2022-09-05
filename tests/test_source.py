from pathlib import Path
import re

import pytest

from loki import read_file, Source, source_to_lines, join_source_list


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def test_source(here):
    """Test :any:`Source` constructor"""
    filepath = here/'sources/sourcefile.f90'
    fcode = read_file(filepath)
    lines = (1, fcode.count('\n') + 1)

    source = Source([1, 1], None)
    assert source.string is None
    assert source.lines == [1, 1]
    assert source.file is None

    source = Source([3, None], fcode)
    assert source.string == fcode
    assert source.lines == [3, None]
    assert source.file is None

    source = Source(lines, fcode, filepath)
    assert source.string == fcode
    assert source.lines == lines
    assert source.file == filepath


def test_source_find(here):
    """Test the `find` utility of :any:`Source`"""
    filepath = here/'sources/sourcefile.f90'
    fcode = read_file(filepath)
    lines = (1, fcode.count('\n') + 1)

    routine_b_match = re.search(r'(subroutine routine_b.*?end subroutine routine_b)', fcode, re.DOTALL)
    assert routine_b_match
    routine_b_fcode = routine_b_match.group(0)

    cstart, cend = Source((1, 1), None).find(routine_b_fcode)
    assert cstart is None
    assert cend is None

    cstart, cend = Source(lines, fcode).find(routine_b_fcode)
    assert (cstart, cend) == routine_b_match.span()

    cstart, cend = Source(lines, fcode).find(routine_b_fcode.upper())
    assert (cstart, cend) == routine_b_match.span()

    cstart, cend = Source(lines, fcode).find(routine_b_fcode.upper(), ignore_case=False)
    assert (cstart, cend) == (None, None)

    bstart = routine_b_match.span()[0] + len('subroutine ')
    bend = bstart + len('routine_b')
    cstart, cend = Source(lines, fcode).find('   routine_b')
    assert (cstart, cend) == (bstart, bend)

    cstart, cend = Source(lines, fcode).find('   routine_b', ignore_space=False)
    assert (cstart, cend) == (None, None)

    cstart, cend = Source(lines, fcode).find(' routine_b', ignore_space=False)
    assert (cstart, cend) == (bstart - 1, bend)  # start offset by 1 because leading whitespace is taken into account


def test_source_clone_with_string(here):
    """Test the `clone_with_string` utility of :any:`Source`"""
    filepath = here/'sources/sourcefile.f90'
    fcode = read_file(filepath)
    lines = (1, fcode.count('\n') + 1)

    routine_b_match = re.search(r'(subroutine routine_b.*?end subroutine routine_b)', fcode, re.DOTALL)
    assert routine_b_match
    routine_b_fcode = routine_b_match.group(0)

    routine_b_start = fcode[:routine_b_match.span()[0]].count('\n') + 1
    routine_b_end = routine_b_start + routine_b_fcode.count('\n')
    routine_b_lines = (routine_b_start, routine_b_end)

    source = Source([3, None], None).clone_with_string(routine_b_fcode)
    assert source.string == routine_b_fcode
    assert source.lines == [3, None]
    assert source.file is None

    source = Source([3, None], fcode).clone_with_string(routine_b_fcode)
    assert source.string == routine_b_fcode
    assert source.lines == (20, 28)
    assert source.file is None

    source = Source(lines, fcode, filepath).clone_with_string(routine_b_fcode)
    assert source.string == routine_b_fcode
    assert source.lines == routine_b_lines
    assert source.file == filepath

    source = Source((1, routine_b_fcode.count('\n')+1), routine_b_fcode).clone_with_string(routine_b_fcode)
    assert source.string == routine_b_fcode
    assert source.lines == (1, routine_b_fcode.count('\n')+1)
    assert source.file is None

    source = Source(lines, fcode, filepath).clone_with_string(routine_b_fcode.upper(), ignore_case=True)
    assert source.string == routine_b_fcode
    assert source.lines == routine_b_lines
    assert source.file == filepath


def test_source_clone_with_span(here):
    """Test the `clone_with_span` utility of :any:`Source`"""
    filepath = here/'sources/sourcefile.f90'
    fcode = read_file(filepath)
    lines = (1, fcode.count('\n') + 1)

    routine_b_match = re.search(r'(subroutine routine_b.*?end subroutine routine_b)', fcode, re.DOTALL)
    assert routine_b_match
    routine_b_fcode = routine_b_match.group(0)

    routine_b_start = fcode[:routine_b_match.span()[0]].count('\n') + 1
    routine_b_end = routine_b_start + routine_b_fcode.count('\n')
    routine_b_lines = (routine_b_start, routine_b_end)

    source = Source(lines, fcode, filepath).clone_with_span(routine_b_match.span())
    assert source.string == routine_b_fcode
    assert source.lines == routine_b_lines
    assert source.file == filepath

    source = Source(lines, fcode.upper(), filepath).clone_with_span(routine_b_match.span())
    assert source.string == routine_b_fcode.upper()
    assert source.lines == routine_b_lines
    assert source.file == filepath


def test_source_clone_lines(here):
    """Test the `clone_lines` utility of :any:`Source`"""
    filepath = here/'sources/sourcefile.f90'
    fcode = read_file(filepath)
    lines = (1, fcode.count('\n') + 1)
    source = Source(lines, fcode, filepath)

    source_lines = source.clone_lines()
    str_lines = fcode.splitlines()
    assert len(source_lines) == len(str_lines)

    for idx, (source_line, str_line) in enumerate(zip(source_lines, str_lines)):
        assert source_line.string == str_line
        assert source_line.lines[0] == idx+1
        assert source_line.lines[1] == idx+1
        assert source_line.file == filepath

    routine_b_match = re.search(r'(subroutine routine_b.*?end subroutine routine_b)', fcode, re.DOTALL)
    routine_b_source = source.clone_with_span(routine_b_match.span())

    source_lines = source.clone_lines(routine_b_match.span())
    routine_b_str_lines = str_lines[routine_b_source.lines[0]-1:routine_b_source.lines[1]]
    assert len(source_lines) == len(routine_b_str_lines)

    for idx, (source_line, str_line) in enumerate(zip(source_lines, routine_b_str_lines)):
        assert source_line.string == str_line
        assert source_line.lines[0] == idx+routine_b_source.lines[0]
        assert source_line.lines[1] == idx+routine_b_source.lines[0]
        assert source_line.file == filepath


def test_source_to_lines():
    """Test the `source_to_lines` utility"""
    fcode = """
module some_module
    implicit none

    integer :: var1, &
        & var2, &
        & var3, &
& var4, &
            &var5,&
            &var6, &
            & var7

    ! This is a &
    ! & comment
contains
    subroutine my_routine
      integer j
      j = var1 &
        &+1
    end subroutine my_routine
end module some_module
    """.strip()

    lines = (1, fcode.count('\n') + 1)
    source = Source(lines, fcode)

    source_lines = source_to_lines(source)

    # All line numbers present?
    assert set(range(lines[0], lines[1] + 1)) == {
        n for s in source_lines for n in range(s.lines[0], s.lines[1] + 1)
    }
    # Line numbers don't overlap?
    assert all(
        l1.lines[0] <= l1.lines[1] and l1.lines[1] + 1 == l2.lines[0] and l2.lines[0] <= l2.lines[1]
        for l1, l2 in zip(source_lines[:-1], source_lines[1:])
    )

    # The known line continuations:
    assert source_lines[3].lines == (4, 10)
    assert source_lines[10].lines == (17, 18)
    assert '    integer ::' in source_lines[3].string
    assert ',  var7' in source_lines[3].string
    assert '      j = var1 +1' in source_lines[10].string


@pytest.mark.parametrize('source_list,expected',(
    (
        [], None
    ),  (
        [Source([1, 2], 'subroutine my_routine\nimplicit none'), Source([3, None], 'end subroutine my_routine')],
        Source([1, 3], 'subroutine my_routine\nimplicit none\nend subroutine my_routine')
    ), (
        [
            Source([1, None], 'subroutine my_routine'),
            Source([2, None], '  use iso_fortran_env, only: real64'),
            Source([3, None], '  implicit none'),
            Source([4, None], '  real(kind=real64) ::'),
            Source([4, 5], ' var_1, &\n   & var_2'),
            Source([6, 7], '  var_1 = 1._real64\n  var_2 = 2._real64'),
            Source([8, None], 'end subroutine my_routine'),
        ],
        Source([1, 8], '''
subroutine my_routine
  use iso_fortran_env, only: real64
  implicit none
  real(kind=real64) :: var_1, &
   & var_2
  var_1 = 1._real64
  var_2 = 2._real64
end subroutine my_routine
        '''.strip())
    ), (
        [
            Source([5, 5], 'integer ::'),
            Source([5, None], ' var1,'),
            Source([5, 5], ' var2')
        ],
        Source([5, 5], 'integer :: var1, var2')
    ), (
        [Source([1, 1], 'print *,* "hello world!"')], Source([1, 1], 'print *,* "hello world!"')
    ), (
        [Source([13, 19], '! line with less line breaks than reported'), Source([20, None], '! here')],
        Source([13, 20], '! line with less line breaks than reported\n! here')
    ), (
        [Source([7, None], '! Some line'), Source([12, None], '! Some other line')],
        Source([7, 12], '! Some line\n\n\n\n\n! Some other line')
    ), (
        [Source([3, 4], '! Some line\n! With line break'), Source([6, None], '! Other line\n! And new line')],
        Source([3, 7], '! Some line\n! With line break\n\n! Other line\n! And new line')
    )
))
def test_join_source_list(source_list, expected):
    """
    Test the `join_source_list` utility
    """
    result = join_source_list(source_list)
    if expected is None:
        assert result is None
    else:
        assert isinstance(result, Source)
        assert result.lines == expected.lines
        assert result.string == expected.string
        assert result.file == expected.file
