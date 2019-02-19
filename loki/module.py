from loki.frontend.omni import parse_omni_ast
from loki.frontend.ofp import parse_ofp_ast
from loki.ir import TypeDef, Section
from loki.expression import Literal, Variable
from loki.visitors import FindNodes
from loki.subroutine import Subroutine
from loki.tools import as_tuple


__all__ = ['Module']


class Module(object):
    """
    Class to handle and manipulate source modules.

    :param name: Name of the module
    :param ast: OFP parser node for this module
    :param raw_source: Raw source string, broken into lines(!), as it
                       appeared in the parsed source file.
    """

    def __init__(self, name=None, spec=None, routines=None,
                 ast=None, raw_source=None):
        self.name = name or ast.attrib['name']
        self.spec = spec
        self.routines = routines

        self._ast = ast
        self._raw_source = raw_source

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None):
        # Process module-level type specifications
        name = name or ast.attrib['name']

        # Parse type definitions into IR and store
        spec_ast = ast.find('body/specification')
        spec = parse_ofp_ast(spec_ast, raw_source=raw_source)

        # TODO: Add routine parsing
        routine_asts = ast.findall('members/subroutine')
        routines = tuple(Subroutine.from_ofp(ast, raw_source)
                         for ast in routine_asts)

        # Process pragmas to override deferred dimensions
        cls._process_pragmas(spec)

        return cls(name=name, spec=spec, routines=routines,
                   ast=ast, raw_source=raw_source)

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, name=None, symbol_map=None):
        name = name or ast.attrib['name']

        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}

        # Generate spec, filter out external declarations and insert `implicit none`
        spec = parse_omni_ast(ast.find('declarations'), type_map=type_map,
                              symbol_map=symbol_map, raw_source=raw_source)
        spec = Section(body=spec)

        # TODO: Parse member functions properly
        contains = ast.find('FcontainsStatement')
        routines = None
        if contains is not None:
            routines = [Subroutine.from_omni(ast=s, typetable=typetable,
                                             symbol_map=symbol_map,
                                             raw_source=raw_source)
                        for s in contains]

        return cls(name=name, spec=spec, routines=routines, ast=ast)

    @classmethod
    def _process_pragmas(self, spec):
        """
        Process any '!$loki dimension' pragmas to override deferred dimensions
        """
        for typedef in FindNodes(TypeDef).visit(spec):
            pragmas = {p._source.lines[0]: p for p in typedef.pragmas}
            for decl in typedef.declarations:
                # Map pragmas by declaration line, not var line
                if decl._source.lines[0]-1 in pragmas:
                    pragma = pragmas[decl._source.lines[0]-1]
                    for v in decl.variables:
                        if pragma.keyword == 'loki' and pragma.content.startswith('dimension'):
                            # Found dimension override for variable
                            dims = pragma._source.string.split('dimension(')[-1]
                            dims = dims.split(')')[0].split(',')
                            dims = [d.strip() for d in dims]
                            # Override dimensions (hacky: not transformer-safe!)
                            v._shape = as_tuple(Literal(value=d) if d.isnumeric() else Variable(name=d)
                                                for d in dims)

    @property
    def typedefs(self):
        """
        Map of names and :class:`DerivedType`s defined in this module.
        """
        types = FindNodes(TypeDef).visit(self.spec)
        return {td.name.lower(): td for td in types}

    @property
    def subroutines(self):
        """
        List of :class:`Subroutine` objects that are members of this :class:`Module`.
        """
        return self.routines

    def __getitem__(self, name):
        subroutine_map = {s.name.lower(): s for s in self.subroutines}
        if name.lower() in subroutine_map:
            return subroutine_map[name.lower()]

        return None
