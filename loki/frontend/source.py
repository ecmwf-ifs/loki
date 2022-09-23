

__all = ['Source', 'extract_source', 'extract_source_from_range']


class Source:
    """
    Store information about the original source for an IR node.

    :param tuple line: tuple with start and (optional) end line number
                       in original source file.
    :param str string: the original source string.
    :pram str file: the file name.
    """

    def __init__(self, lines, string=None, file=None):
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
        if cstart is not None and cend is not None:
            string = self.string[cstart:cend]
        return Source(self.lines, string, self.file)


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
