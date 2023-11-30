# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import re

import pytest

from loki import read_file, Source, source_to_lines, join_source_list, FortranReader


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
 var4, &
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


def test_fortran_reader(here):
    """Test :any:`FortranReader` constructor"""
    filepath = here/'sources/Fortran-extract-interface-source.f90'
    fcode = read_file(filepath)
    lines = (1, fcode.count('\n') + 1)

    reader = FortranReader(fcode)

    # Check for line continuation in sanitized string
    _re_line_cont = re.compile(r'&([ \t]*)\n([ \t]*)(?:&|(?!\!)(?=\S))', re.MULTILINE)
    assert _re_line_cont.search(fcode) is not None
    assert _re_line_cont.search(reader.sanitized_string) is None
    assert 'end subroutine sub_simple_3' in reader.sanitized_string

    # Sanity check for line numbers
    assert reader.sanitized_lines[0].span[0] >= lines[0]
    assert reader.sanitized_lines[1].span[1] <= lines[1]
    assert len(reader.source_lines) == lines[1] - lines[0]

    # Check for comments at the top that are removed
    source = reader.source_from_head()
    assert source.lines == (1, 4)
    assert all(line.strip().startswith('!') or not line.strip() for line in source.string.splitlines())

    assert reader.source_from_tail() is None

    # Test extracting substrings
    start = reader.sanitized_string.find('module foo')
    end = reader.sanitized_string.find('end module foo') + len('end module foo')
    assert 0 < start < end

    # without padding
    new_reader = reader.reader_from_sanitized_span((start, end))
    assert new_reader.sanitized_lines[0].span[0] == 51
    assert new_reader.sanitized_lines[-1].span[1] == 64

    source = new_reader.to_source()
    assert source.lines == (51, 64)
    assert source.string.startswith('module foo')
    assert source.string.endswith('end module foo')

    assert new_reader.source_from_tail() is None

    # with padding
    new_reader = reader.reader_from_sanitized_span((start, end), include_padding=True)
    assert new_reader.sanitized_lines[0].span[0] == 51
    assert new_reader.sanitized_lines[-1].span[1] == 64

    source = new_reader.to_source()
    assert source.lines == (51, 64)
    assert source.string.startswith('module foo')
    assert source.string.endswith('end module foo')

    source = new_reader.to_source(include_padding=True)
    assert source.lines == (49, 66)
    assert source.string.startswith('\n!')
    assert source.string.splitlines()[-1].startswith('!')

    source = new_reader.source_from_tail()
    assert source.lines == (65, 66)
    assert all(line.strip().startswith('!') or not line.strip() for line in source.string.splitlines())

    source = reader.source_from_sanitized_span((start, end))
    assert source.lines == (51, 64)
    assert source.string.startswith('module foo')
    assert source.string.endswith('end module foo')

    source = reader.source_from_sanitized_span((start, end), include_padding=True)
    assert source.lines == (49, 66)
    assert source.string.startswith('\n!')
    assert source.string.splitlines()[-1].startswith('!')

    # Test nested reader
    start = new_reader.sanitized_string.find('subroutine foo_sub')
    end = new_reader.sanitized_string.find('end subroutine foo_sub') + len('end subroutine foo_sub')
    assert 0 < start < end < len(new_reader.sanitized_string)

    nested_reader = new_reader.reader_from_sanitized_span((start, end))
    assert nested_reader.sanitized_lines[0].span[0] == 55
    assert nested_reader.sanitized_lines[-1].span[1] == 59

    source = nested_reader.to_source()
    assert source.lines == (55, 59)
    assert source.string.startswith('subroutine foo_sub')

    source = new_reader.source_from_sanitized_span((start, end))
    assert source.lines == (55, 59)
    assert source.string.startswith('subroutine foo_sub')
    assert source.string.splitlines()[-1].startswith('end subroutine foo_sub')

    # Test extracting substring at the start
    start = reader.sanitized_string.find('logical function func_simple')
    end = reader.sanitized_string.find('end function func_simple') + len('end function func_simple')
    assert start == 0 and end > start

    new_reader = reader.reader_from_sanitized_span((start, end))
    assert new_reader.sanitized_lines[0].span[0] == 5
    assert new_reader.sanitized_lines[-1].span[1] == 7

    source = reader.source_from_sanitized_span((start, end), include_padding=True)
    assert source.lines == (1, 9)
    assert source.string.startswith('!')
    assert source.string.splitlines()[-1].startswith('!')

    source = reader.source_from_sanitized_span((start, end))
    assert source.lines == (5, 7)
    assert source.string.startswith('logical function func_simple')
    assert source.string.endswith('end function func_simple')

    # Test extracting substring at the end
    start = reader.sanitized_string.find('subroutine sub_with_end')
    end = reader.sanitized_string.find('end subroutine sub_with_end') + len('end subroutine sub_with_end')
    assert 0 < start < end == len(reader.sanitized_string)

    new_reader = reader.reader_from_sanitized_span((start, end))
    assert new_reader.sanitized_lines[0].span[0] == 181
    assert new_reader.sanitized_lines[-1].span[1] == 184

    source = reader.source_from_sanitized_span((start, end))
    assert source.lines == (181, 184)
    assert source.string.startswith('subroutine sub_with_end')
    assert source.string.splitlines()[-1].startswith('end subroutine')

    # Test extracting open-ended substring
    end = None

    new_reader = reader.reader_from_sanitized_span((start, end))
    assert new_reader.sanitized_lines[0].span[0] == 181
    assert new_reader.sanitized_lines[-1].span[1] == 184

    source = reader.source_from_sanitized_span((start, end))
    assert source.lines == (181, 184)
    assert source.string.startswith('subroutine sub_with_end')
    assert source.string.splitlines()[-1].startswith('end subroutine')


def test_fortran_reader_iterate(here):
    """Test :any:`FortranReader` iteration"""
    filepath = here/'sources/Fortran-extract-interface-source.f90'
    fcode = read_file(filepath)

    reader = FortranReader(fcode)
    sanitized_code = reader.sanitized_string

    assert reader.current_line is None

    # Test that iterating reproduces the sanitized code
    assert sanitized_code == '\n'.join(item.line for item in reader)

    # Test that we can request the current line string within the iteration range
    iterated_code = ''
    for _ in reader:
        iterated_code += reader.current_line.line + '\n'
    iterated_code = iterated_code[:-1]
    assert sanitized_code == iterated_code

    assert reader.current_line is None

    # Test that we can generate source objects while iterating, that contain
    # the original formatting (this excludes lines missing due to sanitzation)

    def sanitize_empty_lines_and_comments(string):
        sanitized_string = ''
        for line in string.splitlines():
            if not line.lstrip() or line.lstrip().startswith('!'):
                continue
            sanitized_string += line + '\n'
        return sanitized_string

    iterated_code = ''
    for _ in reader:
        iterated_code += reader.source_from_current_line().string + '\n'
    assert sanitize_empty_lines_and_comments(fcode) == iterated_code


@pytest.mark.parametrize('fcode', ['', '\n'])
def test_fortran_reader_empty(fcode):
    """Test :any:`FortranReader` for empty strings"""
    reader = FortranReader(fcode)
    assert isinstance(reader, FortranReader)
    assert not reader.source_lines
    assert not reader.sanitized_lines
    source = reader.to_source()
    assert isinstance(source, Source)
    assert source.lines == (1, 1)
    assert source.string == ''
