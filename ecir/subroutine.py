from ecir.loop import IRGenerator  #generate


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

    def __init__(self, name, ast, source, raw_source):
        self.name = name
        self._ast = ast
        self._source = source
        # The original source string in the file
        self._raw_source = raw_source

        # Separate body and declaration sections
        body_ast = self._ast.find('body')
        bstart = int(body_ast.attrib['line_begin'])
        bend = int(body_ast.attrib['line_end'])

        spec_ast = self._ast.find('body/specification')
        sstart = int(spec_ast.attrib['line_begin'])
        send = int(spec_ast.attrib['line_end'])
        
        # A few small shortcuts:
        # We assume every routine starts with declarations, which might also
        # include a comment block. This will be refined soonish...
        self._pre = Section(name='pre', source=''.join(self.lines[:bstart]))
        self._post = Section(name='post', source=''.join(self.lines[bend:]))
        self.declarations = Section(name='declarations', 
                                    source=''.join(self.lines[bstart:send]))
        self.body = Section(name='body', source=''.join(self.lines[send:bend]))

        # Create a separate IR for the statements and loops in the body
        if self._ast.find('body/associate'):
            routine_body = self._ast.find('body/associate/body')
        else:
            routine_body = self._ast.find('body')

        self._ir = IRGenerator(self._raw_source).visit(routine_body)

    @property
    def source(self):
        """
        The raw source code contained in this section.
        """
        content = [self._pre, self.declarations, self.body, self._post]
        return ''.join(s.source for s in content)        
