import re
from collections import OrderedDict, Mapping

from ecir.generator import generate, extract_source
from ecir.ir import Declaration, Allocation, Import, Statement, TypeDef, Call, Conditional
from ecir.expression import ExpressionVisitor
from ecir.types import DerivedType, DataType
from ecir.visitors import FindNodes
from ecir.tools import flatten
from ecir.helpers import assemble_continued_statement_from_list


__all__ = ['Section', 'Subroutine', 'Module']

class InsertLiteralKinds(ExpressionVisitor):
    """
    Re-insert explicit _KIND casts for literals dropped during pre-processing.

    :param pp_info: List of `(literal, kind)` tuples to be inserted
    """

    def __init__(self, pp_info):
        super(InsertLiteralKinds, self).__init__()

        self.pp_info = dict(pp_info)

    def visit_Literal(self, o):
        if o._line in self.pp_info:
            literals = dict(self.pp_info[o._line])
            if o.value in literals:
                o.value = '%s_%s' % (o.value, literals[o.value])


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

    def replace(self, repl, new=None):
        """
        Performs a line-by-line string-replacement from a given mapping

        Note: The replacement is performed on each raw line. Might
        need to improve this later to unpick linebreaks in the search
        keys.
        """
        if isinstance(repl, Mapping):
            for old, new in repl.items():
                self._source = self._source.replace(old, new)
        else:
            self._source = self._source.replace(repl, new)


class Module(Section):
    """
    Class to handle and manipulate source modules.

    :param name: Name of the module
    :param ast: OFP parser node for this module
    :param raw_source: Raw source string, broken into lines(!), as it
                       appeared in the parsed source file.
    """

    def __init__(self, ast, raw_source, name=None):
        self.name = name or ast.attrib['name']
        self._ast = ast
        self._raw_source = raw_source

        # The actual lines in the source for this subroutine
        self._source = extract_source(self._ast.attrib, raw_source).string

        # Process module-level type specifications
        spec_ast = self._ast.find('body/specification')
        self._spec = generate(spec_ast, self._raw_source)

        # Process 'dimension' pragmas to override deferred dimensions
        self._typedefs = FindNodes(TypeDef).visit(self._spec)
        for typedef in self._typedefs:
            pragmas = {p._source.lines[0]: p for p in typedef.pragmas}
            for v in typedef.variables:
                if v._source.lines[0]-1 in pragmas:
                    pragma = pragmas[v._line-1]
                    if pragma.keyword == 'dimension':
                        # Found dimension override for variable
                        dims = pragma._source.string.split('dimension(')[-1]
                        dims = dims.split(')')[0].split(',')
                        dims = [d.strip() for d in dims]
                        # Override dimensions (hacky: not transformer-safe!)
                        v.dimensions = dims

    @property
    def typedefs(self):
        """
        Map of names and :class:`DerivedType`s defined in this module.
        """
        return {td.name.upper(): td for td in self._typedefs}


class Subroutine(Section):
    """
    Class to handle and manipulate a single subroutine.

    :param name: Name of the subroutine
    :param ast: OFP parser node for this subroutine
    :param raw_source: Raw source string, broken into lines(!), as it
                       appeared in the parsed source file.
    :param typedefs: Optional list of external definitions for derived
                     types that allows more detaild type information.
    """

    def __init__(self, ast, raw_source, name=None, typedefs=None, pp_info=None):
        self.name = name or ast.attrib['name']
        self._ast = ast
        self._raw_source = raw_source

        # The actual lines in the source for this subroutine
        # TODO: Turn Section._source into a real `Source` object
        self._source = extract_source(self._ast.attrib, raw_source).string

        # Separate body and declaration sections
        # Note: The declaration includes the SUBROUTINE key and dummy
        # variable list, so no _pre section is required.
        body_ast = self._ast.find('body')
        bstart = int(body_ast.attrib['line_begin']) - 1
        bend = int(body_ast.attrib['line_end'])
        spec_ast = self._ast.find('body/specification')
        sstart = int(spec_ast.attrib['line_begin']) - 1
        send = int(spec_ast.attrib['line_end'])
        self.header = Section(name='header', source=''.join(self.lines[:sstart]))
        self.declarations = Section(name='declarations', source=''.join(self.lines[sstart:send]))
        self.body = Section(name='body', source=''.join(self.lines[send:bend]))
        self._post = Section(name='post', source=''.join(self.lines[bend:]))

        # Create a IRs for declarations section and the loop body
        self._ir = generate(self._ast.find('body'), self._raw_source)

        # Create a map of all internally used variables
        decls = FindNodes(Declaration).visit(self.ir)
        allocs = FindNodes(Allocation).visit(self.ir)
        variables = flatten([d.variables for d in decls])
        self._variables = OrderedDict([(v.name, v) for v in variables])

        # Try to infer variable dimensions for ALLOCATABLEs
        for v in self.variables:
            if v.type.allocatable:
                alloc = [a for a in allocs if a.variable.name == v.name]
                if len(alloc) > 0:
                    v.dimensions = alloc[0].variable.dimensions

        # Attach derived-type information to variables from given typedefs
        for v in self.variables:
            if typedefs is not None and v.type.name in typedefs:
                typedef = typedefs[v.type.name]
                derived_type = DerivedType(name=typedef.name, variables=typedef.variables,
                                           intent=v.type.intent, allocatable=v.type.allocatable,
                                           pointer=v.type.pointer, optional=v.type.optional)
                v._type = derived_type

        # Re-insert literal _KIND type casts from pre-processing info
        # Note, that this is needed to get accurate data _KIND
        # attributes for literal values, as these have been stripped
        # in a preprocessing step to avoid OFP bugs.
        if pp_info is not None:
            insert_kind = InsertLiteralKinds(pp_info)

            for decl in FindNodes(Declaration).visit(self.ir):
                for v in decl.variables:
                    if v.initial is not None:
                        insert_kind.visit(v.initial)

            for stmt in FindNodes(Statement).visit(self.ir):
                insert_kind.visit(stmt)

            for cnd in FindNodes(Conditional).visit(self.ir):
                for c in cnd.conditions:
                    insert_kind.visit(c)

    @property
    def source(self):
        """
        The raw source code contained in this section.
        """
        content = [self.header, self.declarations, self.body, self._post]
        return ''.join(s.source for s in content)        

    @property
    def ir(self):
        """
        Intermediate representation (AST) of the body in this subroutine
        """
        return self._ir

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
        return FindNodes(Import).visit(self.ir)
