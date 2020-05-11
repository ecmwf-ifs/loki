import weakref

from fparser.two import Fortran2003
from fparser.two.utils import get_child

from loki.frontend import Frontend
from loki.frontend.omni import parse_omni_ast, parse_omni_source
from loki.frontend.ofp import parse_ofp_ast, parse_ofp_source
from loki.frontend.fparser import parse_fparser_ast, parse_fparser_source
from loki.ir import TypeDef, Section
from loki.visitors import FindNodes
from loki.subroutine import Subroutine
from loki.types import TypeTable


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
    def from_source(cls, source, xmods=None, typedefs=None, frontend=Frontend.FP):
        """
        Create `Module` objectfrom raw source string using given frontend.

        :param source: Fortran source string
        :param xmods: Locations of "xmods" module directory for OMNI frontend
        :param frontend: Choice of frontend to use for parsing source (default FP)
        """

        if frontend == Frontend.OMNI:
            ast = parse_omni_source(source, xmods=xmods)
            typetable = ast.find('typeTable')
            f_ast = ast.find('globalDeclarations/FmoduleDefinition')
            return cls.from_omni(ast=f_ast, raw_source=source,
                                 typedefs=typedefs, typetable=typetable)

        if frontend == Frontend.OFP:
            ast = parse_ofp_source(source)
            m_ast = ast.find('file/module')
            return cls.from_ofp(ast=m_ast, typedefs=typedefs, raw_source=source)

        if frontend == Frontend.FP:
            ast = parse_fparser_source(source)
            m_ast = get_child(ast, Fortran2003.Module)
            return cls.from_fparser(ast=m_ast, typedefs=typedefs, raw_source=source)

        raise NotImplementedError('Unknown frontend: %s' % frontend)

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None, typedefs=None, parent=None):
        # Process module-level type specifications
        name = name or ast.attrib['name']
        obj = cls(name=name, ast=ast, raw_source=raw_source, parent=parent)

        # Parse type definitions into IR and store
        spec_ast = ast.find('body/specification')
        spec = parse_ofp_ast(spec_ast, raw_source=raw_source, typedefs=typedefs, scope=obj)

        # Parse member subroutines and functions
        routines = tuple(Subroutine.from_ofp(ast, raw_source, typedefs=typedefs, parent=obj)
                         for ast in ast.findall('members/subroutine'))
        routines += tuple(Subroutine.from_ofp(ast, raw_source, typedefs=typedefs, parent=obj)
                          for ast in ast.findall('members/function'))

        obj.__init__(name=name, spec=spec, routines=routines, ast=ast, raw_source=raw_source,
                     parent=parent, symbols=obj.symbols, types=obj.types)
        return obj

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, name=None, typedefs=None,
                  symbol_map=None, parent=None):
        name = name or ast.attrib['name']

        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}

        obj = cls(name=name, ast=ast, raw_source=raw_source, parent=parent)

        # Generate spec, filter out external declarations and insert `implicit none`
        spec = parse_omni_ast(ast.find('declarations'), type_map=type_map, symbol_map=symbol_map,
                              typedefs=typedefs, raw_source=raw_source, scope=obj)
        spec = Section(body=spec)

        # TODO: Parse member functions properly
        contains = ast.find('FcontainsStatement')
        routines = None
        if contains is not None:
            routines = [Subroutine.from_omni(ast=s, typetable=typetable, typedefs=typedefs,
                                             symbol_map=symbol_map,
                                             raw_source=raw_source, parent=obj)
                        for s in contains]

        obj.__init__(name=name, spec=spec, routines=routines, ast=ast, raw_source=raw_source,
                     parent=parent, symbols=obj.symbols, types=obj.types)
        return obj

    @classmethod
    def from_fparser(cls, ast, raw_source, name=None, typedefs=None, parent=None):
        name = name or ast.content[0].items[1].tostr()
        obj = cls(name, ast=ast, raw_source=raw_source, parent=parent)

        spec_ast = get_child(ast, Fortran2003.Specification_Part)
        spec = []
        if spec_ast is not None:
            spec = parse_fparser_ast(spec_ast, typedefs=typedefs, scope=obj,
                                     raw_source=raw_source)
            spec = Section(body=spec)

        routines_ast = get_child(ast, Fortran2003.Module_Subprogram_Part)
        routines = None
        routine_types = (Fortran2003.Subroutine_Subprogram, Fortran2003.Function_Subprogram)
        if routines_ast is not None:
            routines = [Subroutine.from_fparser(ast=s, typedefs=typedefs, parent=obj,
                                                raw_source=raw_source)
                        for s in routines_ast.content if isinstance(s, routine_types)]

        obj.__init__(name=name, spec=spec, routines=routines, ast=ast,
                     symbols=obj.symbols, types=obj.types, parent=parent)
        return obj

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
