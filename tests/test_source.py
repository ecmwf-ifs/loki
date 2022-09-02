from pathlib import Path
import re

import pytest

from loki import read_file, Source


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def test_source(here):
    """Test :any:`Source` constructor"""
    filepath = here/'sources/sourcefile.f90'
    fcode = read_file(filepath)
    lines = (1, fcode.count('\n') + 1)

    source = Source(None, None)
    assert source.string is None
    assert source.lines is None
    assert source.file is None

    source = Source(None, fcode)
    assert source.string == fcode
    assert source.lines is None
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

    cstart, cend = Source(None, None).find(routine_b_fcode)
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

    source = Source(None, None).clone_with_string(routine_b_fcode)
    assert source.string == routine_b_fcode
    assert source.lines is None
    assert source.file is None

    source = Source(None, fcode).clone_with_string(routine_b_fcode)
    assert source.string == routine_b_fcode
    assert source.lines is None
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
