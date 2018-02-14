import re
from collections import OrderedDict

from ecir.loop import IRGenerator
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

class Variable(object):
    """
    Object representing a variable of arbitrary dimension.
    """

    def __init__(self, name, type, kind, allocatable, ast, source):
        self.name = name
        self.type = type
        self.kind = kind
        self.allocatable = allocatable

        # If the variable has dimensions, record them
        self.dimensions = None
        if self.allocatable:
            # Allocatable dimensions are defined further down in the source
            for line in source:
                if 'ALLOCATE' in line and self.name in line:
                    # We have the allocation line, pick out the dimensions
                    re_dims = re.compile('ALLOCATE\(%s\((?P<dims>.*)\)\)' % self.name)
                    match = re_dims.search(line)
                    if match is None:
                        print("Failed to derive dimensions for allocatable variable %s" % self.name)
                        print("Allocation line: %s" % line)
                        raise ValueError("Could not derive variable dimensions for %s" % self.name)
                    self.dimensions = tuple(match.groupdict()['dims'].split(','))
                    break

        elif ast.find('dimensions'):
            # Extract dimensions for mulit-dimensional arrays
            # Note: Since complex expressions cannot be re-created
            # identically (matching each character) from an AST, we
            # now simply pull out the strings from the source.
            lstart = int(ast.attrib['line_begin'])
            _, line = assemble_continued_statement_from_list(lstart-1, source, return_orig=False)
            re_dims = re.compile('%s\((?P<dims>.*?)(?:\)\s*,|\)\s*\n|\)\s*\!)' % self.name, re.DOTALL)
            match = re_dims.search(line)
            if match is None:
                print("Failed to derive dimensions for variable %s" % self.name)
                print("Declaration line: %s" % line)
                raise ValueError("Could not derive variable dimensions for %s" % self.name)
            self.dimensions = tuple(match.groupdict()['dims'].split(','))

    def __repr__(self):
        return "Variable::%s(type=%s, kind=%s, dims=%s)" % (
            self.name, self.type, self.kind, str(self.dimensions))


class Subroutine(Section):

    def __init__(self, name, ast, source, raw_source):
        self.name = name
        self._ast = ast
        self._source = source
        # The original source string in the file, split into lines
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

        # Record variable definitions as a name->variable dict
        self._variables = OrderedDict()
        spec = self._ast.find('body/specification')
        decls = [d for d in spec.findall('declaration')
                 if 'type' in d.attrib and d.attrib['type'] == 'variable']
        for d in decls:
            # Get type information from the declaration node
            vtype = d.find('type').attrib['name']
            has_kind = d.find('type').attrib['hasKind'] in ['true', 'True']
            kind = d.find('type/kind/name').attrib['id'] if has_kind else None
            allocatable = d.find('attribute-allocatable') is not None
            # Create internal :class:`Variable` objects that store definitions
            var_asts = [v for v in d.findall('variables/variable') if 'name' in v.attrib]
            for v in var_asts:
                vname = v.attrib['name']
                self._variables[vname] = Variable(name=vname, type=vtype, kind=kind,
                                                  allocatable=allocatable,
                                                  ast=v, source=self._raw_source)

    @property
    def source(self):
        """
        The raw source code contained in this section.
        """
        content = [self._pre, self.declarations, self.body, self._post]
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
