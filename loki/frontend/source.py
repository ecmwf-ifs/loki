

__all = ['Source', 'extract_source']


class Source:

    def __init__(self, lines, string=None, file=None, label=None):
        self.lines = lines
        self.string = string
        self.file = file
        self.label = label

    def __repr__(self):
        label = ', label %d' % self.label if self.label else ''
        line_end = '-%s' % self.lines[1] if self.lines[1] else ''
        return 'Source<line %s%s%s>' % (self.lines[0], line_end, label)


def extract_source(ast, text, full_lines=False):
    """
    Extract the marked string from source text.
    """
    attrib = ast.attrib if hasattr(ast, 'attrib') else ast
    lstart = int(attrib['line_begin'])
    lend = int(attrib['line_end'])
    cstart = int(attrib['col_begin'])
    cend = int(attrib['col_end'])

    text = text.splitlines(keepends=True)

    if full_lines:
        return Source(string=''.join(text[lstart-1:lend]),
                      lines=(lstart, lend))

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

    # TODO: The column indexes are still not right, so source strings
    # for sub-expressions are likely wrong!
    if lstart == lend:
        lines[0] = lines[0][cstart:cend]
    else:
        lines[0] = lines[0][cstart:]
        lines[-1] = lines[-1][:cend]

    return Source(string=''.join(lines), lines=(lstart, lend))
