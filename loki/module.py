"""
Contains the declaration of :any:`Module` to represent Fortran modules.
"""
from loki.frontend import Frontend, get_fparser_node
from loki.frontend.omni import parse_omni_ast, parse_omni_source
from loki.frontend.ofp import parse_ofp_ast, parse_ofp_source
from loki.frontend.fparser import parse_fparser_ast, parse_fparser_source
from loki.backend.fgen import fgen
from loki.ir import (
    ProcedureDeclaration, TypeDef, Section, VariableDeclaration, Import, Enumeration,
    Interface
)
from loki.visitors import FindNodes, Transformer
from loki.subroutine import Subroutine
from loki.types import ProcedureType
from loki.scope import Scope
from loki.tools import as_tuple, flatten, CaseInsensitiveDict
from loki.pragma_utils import pragmas_attached, process_dimension_pragmas


__all__ = ['Module']


class Module(Scope):
    """
    Class to handle and manipulate source modules.

    Parameters
    ----------
    name : str
        Name of the module.
    spec : :any:`Section`, optional
        The spec section of the module.
    contains : tuple of :any:`Subroutine`, optional
        The module-subprogram part following a ``CONTAINS`` statement declaring
        member procedures.
    ast : optional
        The node for this module from the parse tree produced by the frontend.
    source : :any:`Source`, optional
        Object representing the raw source string information from the read file.
    rescope_symbols : bool, optional
        Ensure that the type information for all :any:`TypedSymbol` in the
        module's IR exist in the module's scope. Defaults to `False`.
    parent : :any:`Scope`, optional
        The enclosing parent scope of the module. Declarations from the parent
        scope remain valid within the module's scope (unless shadowed by local
        declarations).
    symbol_attrs : :any:`SymbolTable`, optional
        Use the provided :any:`SymbolTable` object instead of creating a new
    """

    def __init__(self, name=None, spec=None, contains=None, ast=None, source=None,
                 rescope_symbols=False, parent=None, symbol_attrs=None):
        # First, store all local properties
        self.name = name or ast.attrib['name']
        assert isinstance(spec, Section) or spec is None
        self.spec = spec
        assert isinstance(contains, Section) or contains is None
        self.contains = contains

        self._ast = ast
        self._source = source

        with pragmas_attached(self, VariableDeclaration):
            self.spec = process_dimension_pragmas(self.spec)

        # Then call the parent constructor to take care of symbol table and rescoping
        super().__init__(parent=parent, symbol_attrs=symbol_attrs, rescope_symbols=rescope_symbols)

    @classmethod
    def from_source(cls, source, definitions=None, xmods=None, frontend=Frontend.FP):
        """
        Create `Module` object from raw source string using given frontend.

        Parameters
        ----------
        source : str
            Fortran source string
        definitions : list, optional
            List of external :any:`Module` to provide derived-type and procedure declarations
        xmods : list, optional
            List of locations with "xmods" module files. Only relevant for :any:`OMNI` frontend
        frontend : :any:`Frontend`, optional
            Choice of frontend to use for parsing source (default :any:`Frontend.FP`)
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
            m_ast = get_fparser_node(ast, 'Module')
            return cls.from_fparser(ast=m_ast, definitions=definitions, raw_source=source)

        raise NotImplementedError(f'Unknown frontend: {frontend}')

    @classmethod
    def from_ofp(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        """
        Create :any:`Module` from :any:`OFP` parse tree

        Parameters
        ----------
        ast :
            The OFP parse tree node corresponding to the module
        raw_source : str
            Fortran source string
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        pp_info :
            Preprocessing info as obtained by :any:`sanitize_input`
        parent : :any:`Scope`, optional
            The enclosing parent scope of the module.
        """
        return parse_ofp_ast(ast=ast, pp_info=pp_info, raw_source=raw_source, definitions=definitions, scope=parent)

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, definitions=None, symbol_map=None, parent=None):
        """
        Create :any:`Module` from :any:`OMNI` parse tree

        Parameters
        ----------
        ast :
            The OMNI parse tree node corresponding to the module
        raw_source : str
            Fortran source string
        typetable :
            The ``typeTable`` AST node from the OMNI parse tree, containing the mapping from
            type hash identifiers to type definitions
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        symbol_map : dict, optional
            The mapping from symbol hash identifiers to symbol attributes
        parent : :any:`Scope`, optional
            The enclosing parent scope of the module
        """
        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {}
        symbol_map.update({s.attrib['type']: s for s in ast.find('symbols')})

        return parse_omni_ast(
            ast=ast, definitions=definitions, raw_source=raw_source,
            type_map=type_map, symbol_map=symbol_map, scope=parent
        )


    @classmethod
    def from_fparser(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        """
        Create :any:`Module` from :any:`FP` parse tree

        Parameters
        ----------
        ast :
            The FParser parse tree node corresponding to the module
        raw_source : str
            Fortran source string
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        pp_info :
            Preprocessing info as obtained by :any:`sanitize_input`
        parent : :any:`Scope`, optional
            The enclosing parent scope of the module.
        """
        return parse_fparser_ast(ast, pp_info=pp_info, definitions=definitions, raw_source=raw_source, scope=parent)

    @property
    def typedefs(self):
        """
        Map of names and :any:`DerivedType` defined in this module.
        """
        types = FindNodes(TypeDef).visit(self.spec)
        return CaseInsensitiveDict((td.name, td) for td in types)

    @property
    def variables(self):
        """
        Return the variables declared in this module
        """
        return as_tuple(
            flatten(
                decl.symbols for decl in FindNodes((VariableDeclaration, ProcedureDeclaration)).visit(self.spec)
            )
        )

    @variables.setter
    def variables(self, variables):
        """
        Set the variables property and ensure that the internal declarations match.
        """
        # First map variables to existing declarations
        declarations = FindNodes((VariableDeclaration, ProcedureDeclaration)).visit(self.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.symbols)

        for v in as_tuple(variables):
            if v not in decl_map:
                # By default, append new variables to the end of the spec
                if isinstance(v.type.dtype, ProcedureType):
                    new_decl = ProcedureDeclaration(symbols=[v])
                else:
                    new_decl = VariableDeclaration(symbols=[v])
                self.spec.append(new_decl)

        # Run through existing declarations and check that all variables still exist
        dmap = {}
        for decl in FindNodes((VariableDeclaration, ProcedureDeclaration)).visit(self.spec):
            new_vars = as_tuple(v for v in decl.symbols if v in variables)
            if len(new_vars) > 0:
                decl._update(symbols=new_vars)
            else:
                dmap[decl] = None  # Mark for removal

        # Remove all redundant declarations
        self.spec = Transformer(dmap).visit(self.spec)

    @property
    def variable_map(self):
        """
        Map of variable names to :any:`Variable` objects
        """
        return CaseInsensitiveDict((v.name, v) for v in self.variables)

    @property
    def imported_symbols(self):
        """
        Return the symbols imported in this module
        """
        return as_tuple(flatten(imprt.symbols for imprt in FindNodes(Import).visit(self.spec or ())))

    @property
    def imported_symbol_map(self):
        """
        Map of imported symbol names to objects
        """
        return CaseInsensitiveDict((s.name, s) for s in self.imported_symbols)

    @property
    def interfaces(self):
        """
        Return the list of interfaces declared in this module
        """
        return as_tuple(FindNodes(Interface).visit(self.spec))

    @property
    def interface_symbols(self):
        """
        Return the list of symbols declared via interfaces in this module
        """
        return as_tuple(flatten(intf.symbols for intf in self.interfaces))

    @property
    def interface_map(self):
        """
        Map of declared interface names to interfaces
        """
        return CaseInsensitiveDict(
            (s.name, intf) for intf in self.interfaces for s in intf.symbols
        )

    @property
    def interface_symbol_map(self):
        """
        Map of declared interface names to symbols
        """
        return CaseInsensitiveDict(
            (s.name, s) for s in self.interface_symbols
        )

    @property
    def enum_symbols(self):
        """
        List of symbols defined via an enum
        """
        return as_tuple(flatten(enum.symbols for enum in FindNodes(Enumeration).visit(self.spec or ())))

    @property
    def symbols(self):
        """
        Return list of all symbols declared or imported in this module scope
        """
        return (
            self.variables + self.imported_symbols + self.interface_symbols + self.enum_symbols +
            tuple(routine.procedure_symbol for routine in self.subroutines)
        )

    @property
    def symbol_map(self):
        """
        Map of symbol names to symbols
        """
        return CaseInsensitiveDict(
            (s.name, s) for s in self.symbols
        )

    @property
    def routines(self):
        """
        List of :class:`Subroutine` objects that are members of this :class:`Module`.
        """
        if self.contains is None:
            return ()
        return as_tuple([
            routine for routine in self.contains.body if isinstance(routine, Subroutine)
        ])

    subroutines = routines

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

    def __iter__(self):
        raise TypeError('Modules alone cannot be traversed! Try traversing "Module.subroutines".')

    def __bool__(self):
        """
        Ensure existing objects register as True in boolean checks, despite
        raising exceptions in `__iter__`.
        """
        return True

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
        return f'Module:: {self.name}'

    def clone(self, **kwargs):
        """
        Create a deep copy of the module with the option to override individual
        parameters.

        .. note:
            This does not clone contained routines.

        Parameters
        ----------
        **kwargs :
            Any parameters from the constructor of :any:`Module`.

        Returns
        -------
        :any:`Module`
            The cloned module object.
        """
        if self.name and 'name' not in kwargs:
            kwargs['name'] = self.name
        if self._ast and 'ast' not in kwargs:
            kwargs['ast'] = self._ast
        if self.source and 'source' not in kwargs:
            kwargs['source'] = self.source

        if 'rescope_symbols' not in kwargs:
            kwargs['rescope_symbols'] = True

        kwargs['spec'] = Transformer({}).visit(self.spec)

        return super().clone(**kwargs)
