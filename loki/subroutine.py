from collections import Mapping

from loki.generator import generate, extract_source
from loki.ir import Declaration, Allocation, Import, TypeDef
from loki.expression import Variable
from loki.types import BaseType, DerivedType
from loki.visitors import FindNodes
from loki.tools import flatten
from loki.preprocessing import blacklist


__all__ = ['Section', 'Subroutine', 'Module']


class InterfaceBlock(object):

    def __init__(self, name, arguments, imports, declarations):
        self.name = name
        self.arguments = arguments
        self.imports = imports
        self.declarations = declarations


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

    def __init__(self, name=None, spec=None, routines=None, ast=None,
                 raw_source=None):
        self.name = name or ast.attrib['name']
        self.spec = spec
        self.routines = routines

        self._ast = ast
        self._raw_source = raw_source

    @classmethod
    def from_source(cls, ast, raw_source, name=None):
        # Process module-level type specifications
        name = name or ast.attrib['name']

        # Parse type definitions into IR and store
        spec_ast = ast.find('body/specification')
        spec = generate(spec_ast, raw_source)

        # TODO: Add routine parsing

        # Process pragmas to override deferred dimensions
        cls._process_pragmas(spec)

        return cls(name=name, spec=spec, ast=ast, raw_source=raw_source)

    @classmethod
    def _process_pragmas(self, spec):
        """
        Process any '!$ecir dimension' pragmas to override deferred dimensions
        """
        for typedef in FindNodes(TypeDef).visit(spec):
            pragmas = {p._source.lines[0]: p for p in typedef.pragmas}
            for v in typedef.variables:
                if v._source.lines[0]-1 in pragmas:
                    pragma = pragmas[v._source.lines[0]-1]
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
        types = FindNodes(TypeDef).visit(self.spec)
        return {td.name.upper(): td for td in types}


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

        # Store the names of variables in the subroutine signature
        arg_ast = self._ast.findall('header/arguments/argument')
        self._argnames = [arg.attrib['name'] for arg in arg_ast]

        # Attach derived-type information to variables from given typedefs
        for v in self.variables:
            if typedefs is not None and v.type.name in typedefs:
                typedef = typedefs[v.type.name]
                derived_type = DerivedType(name=typedef.name, variables=typedef.variables,
                                           intent=v.type.intent, allocatable=v.type.allocatable,
                                           pointer=v.type.pointer, optional=v.type.optional)
                v._type = derived_type

        # Apply postprocessing rules to re-insert information lost during preprocessing
        for name, rule in blacklist.items():
            self._ir = rule.postprocess(self._ir, pp_info[name])

        # And finally we parse "member" subroutines
        self.members = None
        if self._ast.find('members'):
            self.members = [Subroutine(ast=s, raw_source=self._raw_source,
                                       typedefs=typedefs, pp_info=pp_info)
                            for s in self._ast.findall('members/subroutine')]

    def _infer_variable_dimensions(self):
        """
        Try to infer variable dimensions for ALLOCATABLEs
        """
        allocs = FindNodes(Allocation).visit(self.ir)
        for v in self.variables:
            if v.type.allocatable:
                alloc = [a for a in allocs if a.variable.name == v.name]
                if len(alloc) > 0:
                    v.dimensions = alloc[0].variable.dimensions

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
    def argnames(self):
        return self._argnames

    @property
    def arguments(self):
        """
        List of argument names as defined in the subroutine signature.
        """
        vmap = self.variable_map
        return [vmap[name] for name in self.argnames]

    @property
    def variables(self):
        """
        List of all declared variables
        """
        decls = FindNodes(Declaration).visit(self.ir)
        return flatten([d.variables for d in decls])

    @property
    def variable_map(self):
        """
        Map of variable names to `Variable` objects
        """
        return {v.name: v for v in self.variables}

    @property
    def imports(self):
        """
        List of all module imports via USE statements
        """
        return FindNodes(Import).visit(self.ir)

    @property
    def interface(self):
        arguments = self.arguments
        declarations = tuple(d for d in FindNodes(Declaration).visit(self.ir)
                             if any(v in arguments for v in d.variables))

        # Collect unknown symbols that we might need to import
        undefined = set()
        anames = [a.name for a in arguments]
        for a in arguments:
            # Add potentially unkown TYPE and KIND symbols to 'undefined'
            if a.type.name.upper() not in BaseType._base_types:
                undefined.add(a.type.name)
            if a.type.kind and not a.type.kind.isdigit():
                undefined.add(a.type.kind)
            # Add (pure) variable dimensions that might be defined elsewhere
            undefined.update([str(d) for d in a.dimensions
                              if isinstance(d, Variable) and d not in anames])

        # Create a sub-list of imports based on undefined symbols
        imports = []
        for use in self.imports:
            symbols = tuple(s for s in use.symbols if s in undefined)
            if len(symbols) > 0:
                imports += [Import(module=use.module, symbols=symbols)]

        return InterfaceBlock(name=self.name, imports=imports,
                              arguments=arguments, declarations=declarations)
