# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Implementation of :any:`Source` and adjacent utilities
"""
from bisect import bisect_left
from itertools import accumulate
import re
from codetiming import Timer

try:
    from fparser.common.readfortran import FortranStringReader
except ImportError:
    FortranStringReader = None

from loki.logging import debug, warning

__all__ = [
    'Source', 'FortranReader', 'extract_source', 'extract_source_from_range', 'source_to_lines',
    'join_source_list'
]


class Source:
    """
    Store information about the original source for an IR node.

    Parameters
    ----------
    line : tuple
        Start and (optional) end line number in original source file
    string : str (optional)
        The original raw source string
    file : str (optional)
        The file name
    """

    def __init__(self, lines, string=None, file=None):
        assert lines and len(lines) == 2 and (lines[1] is None or lines[1] >= lines[0])
        self.lines = lines
        self.string = string
        self.file = file

    def __repr__(self):
        line_end = f'-{self.lines[1]}' if self.lines[1] else ''
        return f'Source<line {self.lines[0]}{line_end}>'

    def __eq__(self, o):
        if isinstance(o, Source):
            return self.__dict__ == o.__dict__
        return super().__eq__(o)

    def __hash__(self):
        return hash((self.lines, self.string, self.file))

    def find(self, string, ignore_case=True, ignore_space=True):
        """
        Find the given string in the source and return start and end index or None if not found.
        """
        if not self.string:
            return None, None
        if ignore_case:
            string = string.lower()
            self_string = self.string.lower()
        else:
            self_string = self.string
        if string in self_string:
            # string is contained as is
            idx = self_string.find(string)
            return idx, idx + len(string)
        if ignore_space:
            # Split the key and try to find individual parts
            strings = string.strip().split()
            if strings[0] in self_string:
                if all(substr in self_string for substr in strings):
                    return (self_string.find(strings[0]),
                            self_string.find(strings[-1]) + len(strings[-1]))
        return None, None

    def clone_with_string(self, string, ignore_case=True, ignore_space=True):
        """
        Clone the source object and extract the given string from the original source string
        or use the provided string.
        """
        cstart, cend = self.find(string, ignore_case=ignore_case, ignore_space=ignore_space)
        if None not in (cstart, cend):
            string = self.string[cstart:cend]
            lstart = self.lines[0] + self.string[:cstart].count('\n')
            lend = lstart + string.count('\n')
            lines = (lstart, lend)
        else:
            lines = self.lines
        return Source(lines=lines, string=string, file=self.file)

    def clone_with_span(self, span):
        """
        Clone the source object and extract the given line span from the original source
        string (relative to the string length).
        """
        string = self.string[span[0]:span[1]]
        lstart = self.lines[0] + self.string[:span[0]].count('\n')
        lend = lstart + string.count('\n')
        return Source(lines=(lstart, lend), string=string, file=self.file)

    def clone_lines(self, span=None):
        """
        Create source object clones for each line.
        """
        if span is not None:
            return self.clone_with_span(span).clone_lines()
        return [
            Source(lines=(self.lines[0]+idx,)*2, string=line, file=self.file)
            for idx, line in enumerate(self.string.splitlines())
        ]


class FortranReader:
    """
    Reader for Fortran source strings that provides a sanitized version of the source code

    It performs the following sanitizer steps:

    - Remove all comments and preprocessor directives
    - Remove empty lines
    - Remove all whitespace at the beginning and end of lines
    - Resolve all line continuations

    This enables easier pattern matching in the source code. The original source code
    can be recovered (with some restrictions) for each position in the sanitized source string.

    Parameters
    ----------
    raw_source : str
        The Fortran source code

    Attributes
    ----------
    source_lines : list
        The lines of the original source code
    sanitized_lines : list of :class:`fparser.common.Line`
        Lines in the sanitized source code
    sanitized_string : str
        The sanitized source code
    sanitized_spans : list of int
        Start index of each line in the sanitized string
    """

    def __init__(self, raw_source):
        self.line_offset = 0
        raw_source = raw_source.strip()
        self.source_lines = raw_source.splitlines()
        self._sanitize_raw_source(raw_source)

    @Timer(logger=debug, text=lambda s: f'[Loki::Frontend] Executed _sanitize_raw_source in {s:.2f}s')
    def _sanitize_raw_source(self, raw_source):
        """
        Helper routine to create a sanitized Fortran source string
        with comments removed and whitespace stripped from line beginning and end
        """
        if FortranStringReader is None:
            raise RuntimeError('FortranReader needs fparser2')
        reader = FortranStringReader(raw_source)
        self.sanitized_lines = tuple(item for item in reader)
        self.sanitized_spans = (0,) + tuple(accumulate(len(item.line)+1 for item in self.sanitized_lines))
        self.sanitized_string = '\n'.join(item.line for item in self.sanitized_lines)

    def get_line_index(self, line_number):
        """
        Yield the index in :attr:`source_lines` for the given :data:`line_number`
        """
        return line_number - self.line_offset - 1

    def get_line_indices_from_span(self, span, include_padding=False):
        """
        Yield the line indices in :attr:`source_lines` and :attr:`sanitized_lines` for
        the given :data:`span` in the :attr:`sanitized_string`

        Parameters
        ----------
        span : tuple
            Start and end in the :attr:`sanitized_string`. The end can optionally be `None`,
            which includes everything up to the end
        include_padding : bool (optional)
            Includes lines from the original source that are missing in the sanitized string
            (i.e. comments etc.) and that are located immediately before/after the specified
            span.

        Returns
        -------
        sanitized_start, sanitized_end, source_start, source_end
            Start and end indices corresponding to :attr:`sanitized_lines` and
            :attr:`source_lines`, respectively. Indices for `start` are inclusive and for
            `end` exclusive (i.e. ``[start, end)``).
        """
        # First, find the corresponding line indices in the sanitized string
        sanitized_start = bisect_left(self.sanitized_spans, span[0])
        if span[1] is None:
            sanitized_end = len(self.sanitized_lines)
        else:
            sanitized_end = bisect_left(self.sanitized_spans, span[1], lo=sanitized_start)
            sanitized_end = min(len(self.sanitized_lines), sanitized_end)

        # Next, find the corresponding line indices in the original string
        if include_padding:
            if sanitized_start == 0:
                # Span starts at the beginning of the sanitized string: include everything
                # before as well
                source_start = 0
            elif sanitized_start >= len(self.sanitized_lines):
                # Span starts after the sanitized string: include only lines after it
                source_start = self.get_line_index(self.sanitized_lines[-1].span[1] + 1)
            elif self.sanitized_lines[sanitized_start].span[0] - self.sanitized_lines[sanitized_start-1].span[1] > 1:
                # There are lines in the original string that are missing in the sanitized string
                # between the previous and the start line
                source_start = self.get_line_index(self.sanitized_lines[sanitized_start-1].span[1] + 1)
            else:
                source_start = self.get_line_index(self.sanitized_lines[sanitized_start].span[0])

            if sanitized_end == len(self.sanitized_lines):
                # Span reaches until the end of the sanitized_string: include everything
                # after it as well
                source_end = len(self.source_lines)
            else:
                # Include everything until (but not including) the line corresponding to the
                # first line after the span in the sanitized string
                source_end = self.get_line_index(self.sanitized_lines[sanitized_end].span[0])
        elif sanitized_start >= len(self.sanitized_lines):
            # Span starts after the sanitized string: Point to the first line after it
            source_start = self.get_line_index(self.sanitized_lines[-1].span[1] + 1)
            source_end = source_start
        else:
            source_start = self.get_line_index(self.sanitized_lines[sanitized_start].span[0])
            source_end = self.get_line_index(self.sanitized_lines[sanitized_end-1].span[1] + 1)

        return sanitized_start, sanitized_end, source_start, source_end

    def to_source(self, include_padding=False):
        """
        Create a :any:`Source` object with the content of the reader
        """
        if not self.source_lines:
            string = ''
            lines = (self.line_offset + 1, self.line_offset + 1)
        elif include_padding:
            string = '\n'.join(self.source_lines)
            lines = (self.line_offset + 1, self.line_offset + len(self.source_lines))
        else:
            lines = (self.sanitized_lines[0].span[0], self.sanitized_lines[-1].span[1])
            index = (lines[0] - self.line_offset - 1, lines[1] - self.line_offset)
            string = '\n'.join(self.source_lines[index[0]:index[1]])
        return Source(lines=lines, string=string)

    def source_from_head(self):
        """
        Create a :any:`Source` object that contains raw source lines present in the
        original source string before the sanitized source string

        This means typically comments or preprocessor directives. Returns `None` if there
        is nothing.
        """
        if not self.source_lines:
            return None

        if not self.sanitized_lines:
            string = '\n'.join(self.source_lines)
            lines = (self.line_offset + 1, self.line_offset + len(self.source_lines))
            return Source(lines=lines, string=string)

        line_diff = self.sanitized_lines[0].span[0] - self.line_offset
        if line_diff == 1:
            return None
        assert line_diff > 0

        string = '\n'.join(self.source_lines[:line_diff - 1])
        lines = (self.line_offset + 1, self.sanitized_lines[0].span[0] - 1)
        return Source(lines=lines, string=string)

    def source_from_tail(self):
        """
        Create a :any:`Source` object that contains raw source lines present in the
        original source string after the sanitized source string

        This means typically comments or preprocessor directives. Returns `None` if there
        is nothing.
        """
        if not self.sanitized_lines:
            return None

        line_diff = len(self.source_lines) + self.line_offset - self.sanitized_lines[-1].span[1]
        if line_diff == 0:
            return None
        assert line_diff > 0

        start = self.sanitized_lines[-1].span[1] + 1
        string = '\n'.join(self.source_lines[self.get_line_index(start):])
        lines = (start, start + line_diff - 1)
        return Source(lines=lines, string=string)

    def source_from_sanitized_span(self, span, include_padding=False):
        """
        Create a :any:`Source` object containing the original source string corresponding
        to the given span in the sanitized string
        """
        *_, source_start, source_end = self.get_line_indices_from_span(span, include_padding)
        string = '\n'.join(self.source_lines[source_start:source_end])
        if not string:
            return None
        lines = (self.line_offset + source_start + 1, self.line_offset + source_end)
        return Source(lines=lines, string=string)

    def reader_from_sanitized_span(self, span, include_padding=False):
        """
        Create a new :any:`FortranReader` object covering only the source code section corresponding
        to the given span in the sanitized string
        """
        sanit_start, sanit_end, source_start, source_end = self.get_line_indices_from_span(span, include_padding)
        if sanit_start >= len(self.sanitized_lines):
            return None

        new_reader = FortranReader.__new__(FortranReader)
        new_reader.line_offset = self.line_offset + source_start
        new_reader.source_lines = self.source_lines[source_start:source_end]
        new_reader.sanitized_lines = self.sanitized_lines[sanit_start:sanit_end]
        span_offset = self.sanitized_spans[sanit_start]
        new_reader.sanitized_spans = tuple(span - span_offset for span in self.sanitized_spans[sanit_start:sanit_end+1])

        if sanit_end + 1 < len(self.sanitized_spans):
            sanitized_span = [self.sanitized_spans[sanit_start], self.sanitized_spans[sanit_end + 1]]
        else:
            sanitized_span = [self.sanitized_spans[sanit_start], None]
        new_reader.sanitized_string = self.sanitized_string[sanitized_span[0]:sanitized_span[1]]

        return new_reader

    def __iter__(self):
        """Initialize iteration over lines in the sanitized string"""
        self._current_index = 0
        return self

    def __next__(self):
        self._current_index += 1
        if self._current_index > len(self.sanitized_lines):
            raise StopIteration
        return self.current_line

    @property
    def current_line(self):
        """
        Return the current line of the iterator or `None` if outside of iteration range
        """
        _current_index = getattr(self, '_current_index', 0)
        if _current_index <= 0 or _current_index > len(self.sanitized_lines):
            return None
        return self.sanitized_lines[_current_index - 1]

    def source_from_current_line(self):
        """
        Return a :class:`Source` object for the current line
        """
        line = self.current_line
        start = self.get_line_index(line.span[0])
        end = self.get_line_index(line.span[1])
        return Source(lines=line.span, string='\n'.join(self.source_lines[start:end+1]))


def extract_source(ast, text, label=None, full_lines=False):
    """
    Extract the marked string from source text.
    """
    attrib = getattr(ast, 'attrib', ast)
    lstart = int(attrib['line_begin'])
    lend = int(attrib['line_end'])
    cstart = int(attrib['col_begin'])
    cend = int(attrib['col_end'])
    return extract_source_from_range((lstart, lend), (cstart, cend), text, label=label, full_lines=full_lines)


def extract_source_from_range(lines, columns, text, label=None, full_lines=False):
    """
    Extract the marked string from source text.
    """
    text = text.splitlines(keepends=True)
    lstart, lend = lines
    cstart, cend = columns

    if full_lines:
        return Source(string=''.join(text[lstart-1:lend]).strip('\n'), lines=lines)

    lines = text[lstart-1:lend]

    # Scan for line continuations and honour inline
    # comments in between continued lines
    def continued(line):
        if '!' in line:
            line = line.split('!')[0]
        return line.strip().endswith('&')

    def is_comment(line):
        return line.strip().startswith('!')

    # We only honour line continuation if we're not parsing a comment
    if not is_comment(lines[-1]):
        while continued(lines[-1]) or is_comment(lines[-1]):
            lend += 1
            # TODO: Strip the leading empty space before the '&'
            lines.append(text[lend-1])

    # If line continuation is used, move column index to the relevant parts
    while cstart >= len(lines[0]):
        if not is_comment(lines[0]):
            cstart -= len(lines[0])
            cend -= len(lines[0])
        lines = lines[1:]
        lstart += 1

    # Move column index by length of the label if given
    if label is not None:
        cstart += len(label)
        cend += len(label)

    # Avoid stripping indentation
    if lines[0][:cstart].strip() == '':
        cstart = 0

    # TODO: The column indexes are still not right, so source strings
    # for sub-expressions are likely wrong!
    if lstart == lend:
        lines[0] = lines[0][cstart:cend]
    else:
        lines[0] = lines[0][cstart:]
        lines[-1] = lines[-1][:cend]

    return Source(string=''.join(lines).strip('\n'), lines=(lstart, lend))


def _merge_source_match_source(pre, match, post):
    """
    Merge a triple of :class:`Source`, :class:`re.Match`, :class:`Source` objects
    into a single :class:`Source` object spanning multiple lines

    Helper routine for :any:`source_to_lines`.
    """
    assert isinstance(pre, Source)
    assert isinstance(match, re.Match)
    assert isinstance(post, Source)
    lines = (pre.lines[0], post.lines[1])
    return Source(lines, pre.string + post.string, pre.file)


def _create_lines_and_merge(source_lines, source, span, lineno=None):
    """
    Create line-wise :class:`Source` objects for the substring in :data:`source`
    given by :data:`span`

    If the existing list of source lines ends with (:class:`Source`, :class:`re.Match`),
    they are joined with the first line in the new substring.

    Helper routine for :any:`source_to_lines`.
    """
    if lineno is None:
        new_lines = source.clone_lines(span)
    else:
        new_lines = Source((lineno, None), source.string[span[0]:span[1]], source.file).clone_lines()

    if len(source_lines) >= 2 and isinstance(source_lines[-1], re.Match):
        source_lines = (
            source_lines[:-2]
            + [_merge_source_match_source(source_lines[-2], source_lines[-1], new_lines[0])]
            + new_lines[1:]
        )
    else:
        source_lines += new_lines
    return source_lines


_re_line_cont = re.compile(r'&([ \t]*)\n([ \t]*)(?:&|(?!\!)(?=\S))')
"""Pattern to match Fortran line continuation."""


def source_to_lines(source):
    """
    Create line-wise :class:`Source` objects, resolving Fortran line-continuation.
    """
    source_lines = []
    ptr = 0
    lineno = source.lines[0]
    for match in _re_line_cont.finditer(source.string):
        source_lines = _create_lines_and_merge(source_lines, source, (ptr, match.span()[0]), lineno=lineno)
        lineno = source_lines[-1].lines[1] + 1
        source_lines += [match]
        ptr = match.span()[1]
    if ptr < len(source.string):
        source_lines = _create_lines_and_merge(source_lines, source, (ptr, len(source.string)), lineno=lineno)
    return source_lines


def join_source_list(source_list):
    """
    Combine a list of :class:`Source` objects into a single object containing
    the joined source string.

    This will annotate the joined source object with the maximum range of line
    numbers provided in :data:`source_list` objects and insert empty lines for
    any missing line numbers inbetween the provided source objects.
    """
    if not source_list:
        return None
    string = source_list[0].string
    lines = [source_list[0].lines[0], source_list[0].lines[1] or source_list[0].lines[0]]
    for source in source_list[1:]:
        newlines = source.lines[0] - lines[1]
        if newlines < 0:
            warning('join_source_list: overlapping line range')
            newlines = 0
        string += '\n' * newlines + source.string
        lines[1] = source.lines[1] if source.lines[1] else lines[1] + newlines + source.string.count('\n')
    return Source(lines, string, source_list[0].file)
