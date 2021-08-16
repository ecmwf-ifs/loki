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
from loki.visitors import FindNodes, Transformer
from loki.subroutine import Subroutine
from loki.types import ProcedureType, SymbolAttributes
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
    routines : tuple of :any:`Subroutine`, optional
        The routines contained in the module.
    ast : optional
        The node for this module from the parse tree produced by the frontend.
    source : :any:`Source`, optional
        Object representing the raw source string information from the read file.
    rescope_variables : bool, optional
        Ensure that the type information for all :any:`TypedSymbol` in the
        module's IR exist in the module's scope. Defaults to `False`.
    """

    def __init__(self, name=None, spec=None, routines=None, ast=None, source=None, rescope_variables=False,
                 parent=None, symbols=None):
        # First, store all local properties
        self.name = name or ast.attrib['name']
        self.spec = spec
        self.routines = routines

        self._ast = ast
        self._source = source

        with pragmas_attached(self, Declaration):
            self.spec = process_dimension_pragmas(self.spec)

        # Then call the parent constructor to take care of symbol table and rescoping
        super().__init__(parent=parent, symbols=symbols, rescope_variables=rescope_variables)

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

        module = cls(name=name, ast=ast, source=source)

        # Parse type definitions into IR and store
        spec_ast = ast.find('body/specification')
        module.spec = parse_ofp_ast(spec_ast, raw_source=raw_source, definitions=definitions,
                                    scope=module, pp_info=pp_info)

        # Parse member subroutines and functions
        routines = None
        if ast.find('members'):
            # We need to pre-populate the ProcedureType symbol table to
            # correctly classify inline function calls within the module
            routine_asts = [s for s in ast.find('members') if s.tag in ('subroutine', 'function')]
            for routine_ast in routine_asts:
                fname = routine_ast.attrib['name']
                is_function = routine_ast.tag == 'function'
                module.symbols[fname] = SymbolAttributes(ProcedureType(fname, is_function=is_function))

            routines = [Subroutine.from_ofp(ast=routine, raw_source=raw_source, definitions=definitions,
                                            parent=module, pp_info=pp_info)
                        for routine in routine_asts if routine.tag in ('subroutine', 'function')]
            module.routines = as_tuple(routines)

        return module

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, name=None, definitions=None, symbol_map=None):
        name = name or ast.attrib['name']
        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}
        source = Source((ast.attrib['lineno'], ast.attrib['lineno']))

        module = cls(name=name, ast=ast, source=source)

        # Generate spec, filter out external declarations and insert `implicit none`
        module.spec = parse_omni_ast(ast.find('declarations'), type_map=type_map, symbol_map=symbol_map,
                                     definitions=definitions, raw_source=raw_source, scope=module)

        # Parse member functions
        routines = [Subroutine.from_omni(ast=s, typetable=typetable, symbol_map=symbol_map,
                                         definitions=definitions, raw_source=raw_source, parent=module)
                    for s in ast.findall('FcontainsStatement/FfunctionDefinition')]
        module.routines = as_tuple(routines)

        return module

    @classmethod
    def from_fparser(cls, ast, raw_source, name=None, definitions=None, pp_info=None):
        name = name or ast.content[0].items[1].tostr()
        source = extract_fparser_source(ast, raw_source)

        module = cls(name=name, ast=ast, source=source)

        spec_ast = get_child(ast, Fortran2003.Specification_Part)
        if spec_ast is not None:
            spec = parse_fparser_ast(spec_ast, definitions=definitions, scope=module,
                                     pp_info=pp_info, raw_source=raw_source)
        else:
            spec = Section(body=())
        module.spec = spec

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
                module.symbols[fname] = SymbolAttributes(ProcedureType(fname, is_function=is_function))

            # Now create the actual Subroutine objects
            routines = [Subroutine.from_fparser(ast=s, definitions=definitions, parent=module,
                                                pp_info=pp_info, raw_source=raw_source)
                        for s in routines_ast.content if isinstance(s, routine_types)]
            module.routines = as_tuple(routines)

        return module

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
    def subroutines(self):
        """
        List of :class:`Subroutine` objects that are members of this :class:`Module`.
        """
        return as_tuple(self.routines)

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

        if 'rescope_variables' not in kwargs:
            kwargs['rescope_variables'] = True

        kwargs['spec'] = Transformer({}).visit(self.spec)

        return super().clone(**kwargs)
