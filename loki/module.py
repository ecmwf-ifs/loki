import weakref

from fparser.two import Fortran2003
from fparser.two.utils import get_child

from loki.frontend import Frontend, Source, extract_source
from loki.frontend.omni import parse_omni_ast, parse_omni_source
from loki.frontend.ofp import parse_ofp_ast, parse_ofp_source
from loki.frontend.fparser import parse_fparser_ast, parse_fparser_source, extract_fparser_source
from loki.backend.fgen import fgen
from loki.ir import TypeDef, Section
from loki.visitors import FindNodes
from loki.subroutine import Subroutine
from loki.types import Scope, ProcedureType
from loki.tools import as_tuple


__all__ = ['Module']


class Module:
    """
    Class to handle and manipulate source modules.

    :param str name: Name of the module.
    :param Section spec: the spec section of the module.
    :param tuple routines: the routines contained in the module.
    :param ast: Frontend node for this module.
    :param Source source: Source object representing the raw source string information from
            the read file.
    :param scope: Instance of class:``Scope`` that holds :class:``TypeTable`` objects to
                  cache type information for all symbols defined within this module's scope.
    """

    def __init__(self, name=None, spec=None, routines=None, ast=None, source=None, scope=None):
        self.name = name or ast.attrib['name']
        self.spec = spec
        self.routines = routines

        # Ensure we always have a local scope, and register ourselves with it
        self._scope = Scope() if scope is None else scope
        self.scope.defined_by = self

        self._ast = ast
        self._source = source

    @classmethod
    def from_source(cls, source, xmods=None, definitions=None, frontend=Frontend.FP):
        """
        Create `Module` object from raw source string using given frontend.

        :param str source: Fortran source string
        :param xmods: Locations of "xmods" module directory for OMNI frontend
        :param frontend: Choice of frontend to use for parsing source (default FP)
        """
        if frontend == Frontend.OMNI:
            ast = parse_omni_source(source, xmods=xmods)
            typetable = ast.find('typeTable')
            f_ast = ast.find('globalDeclarations/FmoduleDefinition')
            return cls.from_omni(ast=f_ast, raw_source=source, definitions=definitions, typetable=typetable)

        if frontend == Frontend.OFP:
            ast = parse_ofp_source(source)
            m_ast = ast.find('file/module')
            return cls.from_ofp(ast=m_ast, definitions=definitions, raw_source=source)

        if frontend == Frontend.FP:
            ast = parse_fparser_source(source)
            m_ast = get_child(ast, Fortran2003.Module)
            return cls.from_fparser(ast=m_ast, definitions=definitions, raw_source=source)

        raise NotImplementedError('Unknown frontend: %s' % frontend)

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None, definitions=None, pp_info=None):
        source = extract_source(ast, raw_source, full_lines=True)

        # Process module-level type specifications
        name = name or ast.attrib['name']
        scope = Scope()

        # Parse type definitions into IR and store
        spec_ast = ast.find('body/specification')
        spec = parse_ofp_ast(spec_ast, raw_source=raw_source, definitions=definitions,
                             scope=scope, pp_info=pp_info)

        # Parse member subroutines and functions
        routines = None
        if ast.find('members'):
            # We need to pre-populate the ProcedureType type table to
            # correctly class inline function calls within the module
            routine_asts = [s for s in ast.find('members') if s.tag in ('subroutine', 'function')]
            for ast in routine_asts:
                fname = ast.attrib['name']
                scope.types[fname] = ProcedureType(fname, is_function=ast.tag == 'function')

            routines = [Subroutine.from_ofp(ast=routine, raw_source=raw_source, definitions=definitions,
                                            parent_scope=scope, pp_info=pp_info)
                        for routine in routine_asts if routine.tag in ('subroutine', 'function')]
            routines = as_tuple(routines)

        return cls(name=name, spec=spec, routines=routines, ast=ast, source=source, scope=scope)

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, name=None, definitions=None, symbol_map=None):
        name = name or ast.attrib['name']
        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}
        source = Source((ast.attrib['lineno'], ast.attrib['lineno']))
        scope = Scope()

        # Generate spec, filter out external declarations and insert `implicit none`
        spec = parse_omni_ast(ast.find('declarations'), type_map=type_map, symbol_map=symbol_map,
                              definitions=definitions, raw_source=raw_source, scope=scope)
        spec = Section(body=spec)

        # Parse member functions
        routines = [Subroutine.from_omni(ast=s, typetable=typetable, symbol_map=symbol_map,
                                         definitions=definitions, raw_source=raw_source,
                                         parent_scope=scope)
                    for s in ast.findall('FcontainsStatement/FfunctionDefinition')]

        return cls(name=name, spec=spec, routines=routines, ast=ast, source=source, scope=scope)

    @classmethod
    def from_fparser(cls, ast, raw_source, name=None, definitions=None, pp_info=None):
        name = name or ast.content[0].items[1].tostr()
        source = extract_fparser_source(ast, raw_source)
        scope = Scope()

        spec_ast = get_child(ast, Fortran2003.Specification_Part)
        spec = []
        if spec_ast is not None:
            spec = parse_fparser_ast(spec_ast, definitions=definitions, scope=scope,
                                     pp_info=pp_info, raw_source=raw_source)
            spec = Section(body=spec)

        routines_ast = get_child(ast, Fortran2003.Module_Subprogram_Part)
        routines = None
        if routines_ast is not None:
            # We need to pre-populate the ProcedureType type table to
            # correctly class inline function calls within the module
            routine_types = (Fortran2003.Subroutine_Subprogram, Fortran2003.Function_Subprogram)
            routine_asts = [s for s in routines_ast.content if isinstance(s, routine_types)]
            for s in routine_asts:
                is_function = isinstance(s, Fortran2003.Function_Subprogram)
                if is_function:
                    routine_stmt = get_child(s, Fortran2003.Function_Stmt)
                    fname = routine_stmt.items[1].tostr()
                else:
                    routine_stmt = get_child(s, Fortran2003.Subroutine_Stmt)
                    fname = routine_stmt.get_name().string
                scope.types[fname] = ProcedureType(fname, is_function=is_function)

            # Now create the actual Subroutine objects
            routines = [Subroutine.from_fparser(ast=s, definitions=definitions, parent_scope=scope,
                                                pp_info=pp_info, raw_source=raw_source)
                        for s in routines_ast.content if isinstance(s, routine_types)]
            routines = as_tuple(routines)

        return cls(name=name, spec=spec, routines=routines, ast=ast, source=source, scope=scope)

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
        return as_tuple(self.routines)

    @property
    def scope(self):
        return self._scope

    @property
    def symbols(self):
        return self.scope.symbols

    @property
    def types(self):
        return self.scope.types

    @property
    def source(self):
        return self._source

    def to_fortran(self, conservative=False):
        return fgen(self, conservative=conservative)

    def __contains__(self, name):
        subroutine_map = {s.name.lower(): s for s in self.subroutines}
        return name in subroutine_map

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError('Subroutine lookup requires a string!')

        subroutine_map = {s.name.lower(): s for s in self.subroutines}
        if name.lower() in subroutine_map:
            return subroutine_map[name.lower()]

        return None

    def apply(self, op, **kwargs):
        """
        Apply a given transformation to the source file object.

        Note that the dispatch routine `op.apply(source)` will ensure
        that all entities of this `Sourcefile` are correctly traversed.
        """
        # TODO: Should type-check for an `Operation` object here
        op.apply(self, **kwargs)

    def __repr__(self):
        """
        String representation.
        """
        return 'Module:: {}'.format(self.name)
