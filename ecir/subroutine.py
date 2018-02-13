class Section(object):
    """
    Class to handle and manipulate a source code section.

    :param name: Name of the source section.
    :param source: String with the contained source code.
    """

    def __init__(self, name, source):
        self.name = name

        self._source = source

    @property
    def source(self):
        """
        The raw source code contained in this section.
        """
        return self._source

    @property
    def lines(self):
        """
        Sanitizes source content into long lines with continuous statements.

        Note: This does not change the content of the file
        """
        return self._source.splitlines(keepends=True)

    @property
    def longlines(self):
        from ecir.helpers import assemble_continued_statement_from_iterator
        srciter = iter(self.lines)
        return [assemble_continued_statement_from_iterator(line, srciter)[0] for line in srciter]

    def replace(self, mapping):
        """
        Performs a line-by-line string-replacement from a given mapping

        Note: The replacement is performed on each raw line. Might
        need to improve this later to unpick linebreaks in the search
        keys.
        """
        rawlines = self.lines
        for k, v in mapping.items():
            rawlines = [line.replace(k, v) for line in rawlines]
        self._source = ''.join(rawlines)


class Subroutine(Section):

    def __init__(self, name, ast, source):
        self.name = name
        self._ast = ast
        self._source = source

        # Separate body and declaration sections
        body = self._ast.find('body')
        b_start = int(body.attrib['line_begin'])
        b_end = int(body.attrib['line_end'])

        spec = self._ast.find('body/specification')
        s_start = int(spec.attrib['line_begin'])
        s_end = int(spec.attrib['line_end'])
        
        # A few small shortcuts:
        # We assume every routine starts with declarations, which might also
        # include a comment block. This will be refined soonish...
        self._pre = Section(name='pre', source=''.join(self.lines[:b_start]))
        self._post = Section(name='post', source=''.join(self.lines[b_end:]))
        self.declarations = Section(name='declarations', 
                                    source=''.join(self.lines[b_start:s_end]))
        self.body = Section(name='body', source=''.join(self.lines[s_end:b_end]))

    @property
    def source(self):
        """
        The raw source code contained in this section.
        """
        content = [self._pre, self.declarations, self.body, self._post]
        return ''.join(s.source for s in content)        
