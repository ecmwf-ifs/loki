"""
Implementation of :any:`Source` and adjacent utilities
"""

import re
from loki.logging import warning

__all = [
    'Source', 'extract_source', 'extract_source_from_range', 'source_to_lines',
    'join_source_list'
]


class Source:
    """
    Store information about the original source for an IR node.

    :param tuple line: tuple with start and (optional) end line number
                       in original source file.
    :param str string: the original source string.
    :pram str file: the file name.
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


def _create_lines_and_merge(source_lines, source, span):
    """
    Create line-wise :class:`Source` objects for the substring in :data:`source`
    given by :data:`span`

    If the existing list of source lines ends with (:class:`Source`, :class:`re.Match`),
    they are joined with the first line in the new substring.

    Helper routine for :any:`source_to_lines`.
    """
    new_lines = source.clone_lines(span)
    if len(source_lines) >= 2 and isinstance(source_lines[-1], re.Match):
        source_lines = (
            source_lines[:-2]
            + [_merge_source_match_source(source_lines[-2], source_lines[-1], new_lines[0])]
            + new_lines[1:]
        )
    else:
        source_lines += new_lines
    return source_lines


_re_line_cont = re.compile(r'&([ \t]*)\n([ \t]*)&')
"""Pattern to match Fortran line continuation."""


def source_to_lines(source):
    """
    Create line-wise :class:`Source` objects, resolving Fortran line-continuation.
    """
    source_lines = []
    ptr = 0
    for match in _re_line_cont.finditer(source.string):
        source_lines = _create_lines_and_merge(source_lines, source, (ptr, match.span()[0]))
        source_lines += [match]
        ptr = match.span()[1]
    if ptr < len(source.string):
        source_lines = _create_lines_and_merge(source_lines, source, (ptr, len(source.string)))
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
