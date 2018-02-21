import re
from collections import OrderedDict

from ecir.generator import generate
from ecir.ir import Declaration, Allocation, Import
from ecir.visitors import FindNodes
from ecir.tools import flatten
from ecir.helpers import assemble_continued_statement_from_list

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
        # The original source string in the file, split into lines
        self._raw_source = raw_source

        # Separate body and declaration sections
        # Note: The declaration includes the SUBROUTINE key and dummy
        # variable list, so no _pre section is required.
        body_ast = self._ast.find('body')
        bstart = int(body_ast.attrib['line_begin']) - 1
        bend = int(body_ast.attrib['line_end'])
        spec_ast = self._ast.find('body/specification')
        sstart = int(spec_ast.attrib['line_begin']) - 1
        send = int(spec_ast.attrib['line_end'])
        self._post = Section(name='post', source=''.join(self.lines[bend:]))
        self.declarations = Section(name='declarations', 
                                    source=''.join(self.lines[:send]))
        self.body = Section(name='body', source=''.join(self.lines[send:bend]))

        # Create a IRs for declarations section and the loop body
        self._spec = generate(spec_ast, self._raw_source)
        if self._ast.find('body/associate'):
            routine_body = self._ast.find('body/associate/body')
        else:
            routine_body = self._ast.find('body')
        self._ir = generate(routine_body, self._raw_source)

        # Create a map of all internally used variables
        decls = FindNodes(Declaration).visit(self._spec)
        allocs = FindNodes(Allocation).visit(self._ir)
        variables = flatten([d.variables for d in decls])
        self._variables = OrderedDict([(v.name, v) for v in variables])

        # Try to infer variable dimensions for ALLOCATABLEs
        for v in self.variables:
            if v.allocatable:
                alloc = [a for a in allocs if a.variable.name == v.name]
                if len(alloc) > 0:
                    v.dimensions = alloc[0].variable.dimensions

    @property
    def source(self):
        """
        The raw source code contained in this section.
        """
        content = [self.declarations, self.body, self._post]
        return ''.join(s.source for s in content)        

    @property
    def arguments(self):
        """
        List of argument names as defined in the subroutine signature.
        """
        argnames = [arg.attrib['name'] for arg in self._ast.findall('header/arguments/argument')]
        return [self._variables[name] for name in argnames]

    @property
    def variables(self):
        """
        List of all declared variables
        """
        return list(self._variables.values())

    @property
    def imports(self):
        """
        List of all module imports via USE statements
        """
        return [i for i in self._spec if isinstance(i, Import)]
