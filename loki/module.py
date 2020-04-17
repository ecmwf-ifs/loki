import weakref

from fparser.two import Fortran2003
from fparser.two.utils import get_child

from loki.frontend.omni import parse_omni_ast
from loki.frontend.ofp import parse_ofp_ast
from loki.frontend.fparser import parse_fparser_ast
from loki.ir import TypeDef, Section
from loki.expression import Literal, Variable
from loki.visitors import FindNodes
from loki.subroutine import Subroutine
from loki.tools import as_tuple
from loki.types import TypeTable, DataType, SymbolType


__all__ = ['Module']


class Module:
    """
    Class to handle and manipulate source modules.

    :param name: Name of the module
    :param ast: OFP parser node for this module
    :param raw_source: Raw source string, broken into lines(!), as it
                       appeared in the parsed source file.
    :param symbols: Instance of class:``TypeTable`` used to cache type information
                    for all symbols defined within this module's scope.
    :param types: Instance of class:``TypeTable`` used to cache type information
                  for all (derived) types defined within this module's scope.
    :param parent: Optional enclosing scope, to which a weakref can be held for symbol lookup.
    """

    def __init__(self, name=None, spec=None, routines=None, ast=None,
                 raw_source=None, symbols=None, types=None, parent=None):
        self.name = name or ast.attrib['name']
        self.spec = spec
        self.routines = routines
        self._parent = weakref.ref(parent) if parent is not None else None

        self.symbols = symbols
        if self.symbols is None:
            parent = self.parent.symbols if self.parent is not None else None
            self.symbols = TypeTable(parent)

        self.types = types
        if self.types is None:
            parent = self.parent.types if self.parent is not None else None
            self.types = TypeTable(parent)

        self._ast = ast
        self._raw_source = raw_source

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None, parent=None):
        # Process module-level type specifications
        name = name or ast.attrib['name']
        obj = cls(name=name, ast=ast, raw_source=raw_source, parent=parent)

        # Parse type definitions into IR and store
        spec_ast = ast.find('body/specification')
        spec = parse_ofp_ast(spec_ast, raw_source=raw_source, scope=obj)

        # TODO: Add routine parsing
        routines = tuple(Subroutine.from_ofp(ast, raw_source, parent=obj)
                         for ast in ast.findall('members/subroutine'))
        routines += tuple(Subroutine.from_ofp(ast, raw_source, parent=obj)
                          for ast in ast.findall('members/function'))

        # Process pragmas to override deferred dimensions
        cls._process_pragmas(spec)

        obj.__init__(name=name, spec=spec, routines=routines, ast=ast, raw_source=raw_source,
                     parent=parent, symbols=obj.symbols, types=obj.types)
        return obj

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, name=None, symbol_map=None, parent=None):
        name = name or ast.attrib['name']

        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}

        obj = cls(name=name, ast=ast, raw_source=raw_source, parent=parent)

        # Generate spec, filter out external declarations and insert `implicit none`
        spec = parse_omni_ast(ast.find('declarations'), type_map=type_map,
                              symbol_map=symbol_map, raw_source=raw_source, scope=obj)
        spec = Section(body=spec)

        # TODO: Parse member functions properly
        contains = ast.find('FcontainsStatement')
        routines = None
        if contains is not None:
            routines = [Subroutine.from_omni(ast=s, typetable=typetable,
                                             symbol_map=symbol_map,
                                             raw_source=raw_source, parent=obj)
                        for s in contains]

        obj.__init__(name=name, spec=spec, routines=routines, ast=ast, raw_source=raw_source,
                     parent=parent, symbols=obj.symbols, types=obj.types)
        return obj

    @classmethod
    def from_fparser(cls, ast, name=None, parent=None):
        name = name or ast.content[0].items[1].tostr()
        obj = cls(name, ast=ast, parent=parent)

        spec_ast = get_child(ast, Fortran2003.Specification_Part)
        spec = []
        if spec_ast is not None:
            spec = parse_fparser_ast(spec_ast, scope=obj)
            spec = Section(body=spec)

        routines_ast = get_child(ast, Fortran2003.Module_Subprogram_Part)
        routines = None
        routine_types = (Fortran2003.Subroutine_Subprogram, Fortran2003.Function_Subprogram)
        if routines_ast is not None:
            routines = [Subroutine.from_fparser(ast=s, parent=obj)
                        for s in routines_ast.content if isinstance(s, routine_types)]

        # Process pragmas to override deferred dimensions
        cls._process_pragmas(spec)

        obj.__init__(name=name, spec=spec, routines=routines, ast=ast,
                     symbols=obj.symbols, types=obj.types, parent=parent)
        return obj

    @staticmethod
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
                            dims = pragma.content.split('dimension(')[-1]
                            dims = dims.split(')')[0].split(',')
                            dims = [d.strip() for d in dims]
                            shape = []
                            for d in dims:
                                if d.isnumeric():
                                    shape += [Literal(value=int(d), type=DataType.INTEGER)]
                                else:
                                    _type = SymbolType(DataType.INTEGER)
                                    shape += [Variable(name=d, scope=typedef.symbols, type=_type)]
                            v.type = v.type.clone(shape=as_tuple(shape))

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

    @property
    def parent(self):
        """
        Access the enclosing parent.
        """
        return self._parent() if self._parent is not None else None

    def __getitem__(self, name):
        subroutine_map = {s.name.lower(): s for s in self.subroutines}
        if name.lower() in subroutine_map:
            return subroutine_map[name.lower()]

        return None
