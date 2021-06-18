"""
Contains the declaration of :any:`Module` to represent Fortran modules.
"""
from fparser.two import Fortran2003
from fparser.two.utils import get_child

from loki.frontend import Frontend, Source, extract_source
from loki.frontend.omni import parse_omni_ast, parse_omni_source
from loki.frontend.ofp import parse_ofp_ast, parse_ofp_source
from loki.frontend.fparser import parse_fparser_ast, parse_fparser_source, extract_fparser_source
from loki.backend.fgen import fgen
from loki.ir import TypeDef, Section, Declaration, Import
from loki.expression import FindTypedSymbols, SubstituteExpressions
from loki.logging import debug
from loki.visitors import FindNodes, Transformer
from loki.subroutine import Subroutine
from loki.types import Scope, ProcedureType, SymbolAttributes
from loki.tools import as_tuple, flatten, CaseInsensitiveDict
from loki.pragma_utils import pragmas_attached, process_dimension_pragmas


__all__ = ['Module']


class Module:
    """
    Class to handle and manipulate source modules.

    Parameters
    ----------
    name : str
        Name of the module.
    spec : :any:`Section`, optional
        The spec section of the module.
    routines : tuple of :any:`Subroutine`, optional
        The routines contained in the module.
    ast : optional
        The node for this module from the parse tree produced by the frontend.
    source : :any:`Source`, optional
        Object representing the raw source string information from the read file.
    scope : :any:`Scope`, optional
        Prepopulated type and symbol information object to be used in this module.
    rescope_variables : bool, optional
        Ensure that the type information for all :any:`TypedSymbol` in the
        module's IR exist in the module's scope. Defaults to `False`.
    """

    def __init__(self, name=None, spec=None, routines=None, ast=None, source=None, scope=None, rescope_variables=False):
        self.name = name or ast.attrib['name']
        self.spec = spec
        self.routines = routines

        # Ensure we always have a local scope, and register ourselves with it
        self._scope = Scope() if scope is None else scope
        self.scope.defined_by = self
        if rescope_variables:
            self.rescope_variables()

        self._ast = ast
        self._source = source

        with pragmas_attached(self, Declaration):
            self.spec = process_dimension_pragmas(self.spec)

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
            for routine_ast in routine_asts:
                fname = routine_ast.attrib['name']
                scope.types[fname] = SymbolAttributes(ProcedureType(fname, is_function=routine_ast.tag == 'function'))

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
                scope.types[fname] = SymbolAttributes(ProcedureType(fname, is_function=is_function))

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
    def variables(self):
        """
        Return the variables declared in this module
        """
        return as_tuple(flatten(decl.variables for decl in FindNodes(Declaration).visit(self.spec)))

    @variables.setter
    def variables(self, variables):
        """
        Set the variables property and ensure that the internal declarations match.
        """
        # First map variables to existing declarations
        declarations = FindNodes(Declaration).visit(self.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.variables)

        for v in as_tuple(variables):
            if v not in decl_map:
                # By default, append new variables to the end of the spec
                new_decl = Declaration(variables=[v])
                self.spec.append(new_decl)

        # Run through existing declarations and check that all variables still exist
        dmap = {}
        for decl in FindNodes(Declaration).visit(self.spec):
            new_vars = as_tuple(v for v in decl.variables if v in variables)
            if len(new_vars) > 0:
                decl._update(variables=new_vars)
            else:
                dmap[decl] = None  # Mark for removal
        # Remove all redundant declarations
        self.spec = Transformer(dmap).visit(self.spec)

    @property
    def variable_map(self):
        """
        Map of variable names to `Variable` objects
        """
        return CaseInsensitiveDict((v.name, v) for v in self.variables)

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
        return 'Module:: {}'.format(self.name)

    def rescope_variables(self):
        """
        Verify that all :any:`TypedSymbol` objects in the IR are in the
        module's scope.
        """
        # The local variable map. These really need to be in *this* scope.
        variable_map = self.variable_map
        imports_map = CaseInsensitiveDict(
            (s.name, s) for imprt in FindNodes(Import).visit(self.spec or ()) for s in imprt.symbols
        )

        # Check for all variables that they are associated with the scope
        rescope_map = {}
        for var in FindTypedSymbols().visit(self.spec):
            if (var.name in variable_map or var.name in imports_map) and var.scope is not self.scope:
                # This takes care of all local variables or imported symbols
                rescope_map[var] = var.clone(scope=self.scope)
            elif var not in rescope_map:
                # Put this in the local scope just to be on the safe side
                debug('Module.rescope_variables: type for %s not found in any scope.', var.name)
                rescope_map[var] = var.clone(scope=self.scope)

        # Now apply the rescoping map
        if rescope_map and self.spec:
            self.spec = SubstituteExpressions(rescope_map).visit(self.spec)

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

        if 'scope' not in kwargs:
            kwargs['scope'] = Scope()
        if 'rescope_variables' not in kwargs:
            kwargs['rescope_variables'] = True

        kwargs['spec'] = Transformer({}).visit(self.spec)

        return type(self)(**kwargs)
