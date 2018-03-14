import re
from collections import OrderedDict, Mapping

from ecir.generator import generate
from ecir.ir import Declaration, Allocation, Import, Statement, TypeDef, Call, Conditional
from ecir.expression import ExpressionVisitor
from ecir.types import DerivedType, DataType
from ecir.visitors import FindNodes
from ecir.tools import flatten, extract_lines
from ecir.helpers import assemble_continued_statement_from_list


__all__ = ['Section', 'Subroutine', 'Module']


class InjectFortranType(ExpressionVisitor):
    """
    Inject the full Fortran type from declarations into variable occurences.
    """

    def __init__(self, variable_list):
        super(InjectFortranType, self).__init__()

        self.variable_list = variable_list

    def visit_Variable(self, o):
        if o.type is None:
            if o.name in self.variable_list:
                # TODO: Shouldn't really set things this way...
                o._type = self.variable_list[o.name].type
            elif '%' in o.name:
                # We are dealing with a derived-type member
                base_name = o.name.split('%')[0]
                var_name = o.name.split('%')[1]
                if base_name in self.variable_list:
                    derived = self.variable_list[base_name]
                    if isinstance(derived.type, DerivedType):
                        # TODO: Shouldn't really set things this way...
                        o._type = derived.type.variables[var_name].type


class InferDataType(ExpressionVisitor):
    """
    Top-down visitor to insert data type information into individual
    sub-expression leaves.
    """

    def __init__(self, dtype):
        super(InferDataType, self).__init__()

        self.dtype = dtype

    def visit_Literal(self, o):
        if o.type is None:
            if '.' in o.value:  # skip INTs
                # Shouldn't really set things this way...
                o._type = self.dtype

    def visit_InlineCall(self, o):
        for c in o.children:
            self.visit(c)


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
        self._source = extract_lines(self._ast.attrib, raw_source)

        # Process module-level type specifications
        spec_ast = self._ast.find('body/specification')
        self._spec = generate(spec_ast, self._raw_source)

        # Process 'dimension' pragmas to override deferred dimensions
        self._typedefs = FindNodes(TypeDef).visit(self._spec)
        for typedef in self._typedefs:
            pragmas = {p._line: p for p in typedef.pragmas}
            for v in typedef.variables:
                if v._line-1 in pragmas:
                    pragma = pragmas[v._line-1]
                    if pragma.keyword == 'dimension':
                        # Found dimension override for variable
                        dims = pragma._source.split('dimension(')[-1]
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

    def __init__(self, ast, raw_source, name=None, typedefs=None):
        self.name = name or ast.attrib['name']
        self._ast = ast
        self._raw_source = raw_source

        # The actual lines in the source for this subroutine
        self._source = extract_lines(self._ast.attrib, raw_source)

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

        # Set the basic data type on all expression components
        # Note, that this is needed to get accurate data type for
        # literal values, as these have been stripped in a
        # preprocessing step to avoid OFP bugs.
        for stmt in FindNodes(Statement).visit(self.ir):
            # Inject declaration type information into expression variables
            InjectFortranType(self._variables).visit(stmt)

            # Infer data type of expression components from target variable
            if stmt.target.type is not None:
                InferDataType(dtype=stmt.target.type.dtype).visit(stmt.expr)

        for cnd in FindNodes(Conditional).visit(self.ir):
            for c in cnd.conditions:
                InjectFortranType(self._variables).visit(c)
                InferDataType(dtype=DataType.JPRB).visit(c)

        # Infer data types for initial parameter values
        for v in self.variables:
            if v.type.parameter:
                InferDataType(dtype=v.type.dtype).visit(v.initial)

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
