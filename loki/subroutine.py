from loki.frontend import Frontend, extract_source
from loki.frontend.omni import parse_omni_ast, parse_omni_source
from loki.frontend.ofp import parse_ofp_ast, parse_ofp_source
from loki.frontend.fparser import get_fparser_node, parse_fparser_ast, parse_fparser_source
from loki.backend.fgen import fgen
from loki.ir import (
    VariableDeclaration, ProcedureDeclaration, Allocation, Import, Section, CallStatement,
    CallContext, Interface, TypeDef, Enumeration
)
from loki.expression import FindVariables, Array, SubstituteExpressions, Variable
from loki.pragma_utils import is_loki_pragma, pragmas_attached, process_dimension_pragmas
from loki.visitors import FindNodes, Transformer
from loki.tools import as_tuple, flatten, CaseInsensitiveDict
from loki.types import ProcedureType, SymbolAttributes, BasicType
from loki.scope import Scope


__all__ = ['Subroutine']


class Subroutine(Scope):
    """
    Class to handle and manipulate a single subroutine.

    Parameters
    ----------
    name : str
        Name of the subroutine.
    args : iterable of str, optional
        The names of the dummy args.
    docstring : tuple of :any:`Node`, optional
        The subroutine docstring in the original source.
    spec : :any:`Section`, optional
        The spec of the subroutine.
    body : :any:`Section`, optional
        The body of the subroutine.
    contains : :any:`Section`, optional
        The internal-subprogram part following a ``CONTAINS`` statement
        declaring member procedures
    ast : optional
        Frontend node for this subroutine (from parse tree of the frontend).
    prefix : iterable, optional
        Prefix specifications for the procedure
    bind : optional
        Bind information (e.g., for Fortran ``BIND(C)`` annotation).
    is_function : bool, optional
        Flag to indicate this is a function instead of subroutine
        (in the Fortran sense). Defaults to `False`.
    rescope_symbols : bool, optional
        Ensure that the type information for all :any:`TypedSymbol` in the
        subroutine's IR exist in the subroutine's scope or the scope's parents.
        Defaults to `False`.
    source : :any:`Source`
        Source object representing the raw source string information from the
        read file.
    parent : :any:`Scope`, optional
        The enclosing parent scope of the subroutine, typically a :any:`Module`
        or :any:`Subroutine` object. Declarations from the parent scope remain
        valid within the subroutine's scope (unless shadowed by local
        declarations).
    symbol_attrs : :any:`SymbolTable`, optional
        Use the provided :any:`SymbolTable` object instead of creating a new
    """

    def __init__(self, name, args=None, docstring=None, spec=None, body=None, contains=None,
                 ast=None, prefix=None, bind=None, is_function=False, rescope_symbols=False,
                 source=None, parent=None, symbol_attrs=None):
        # First, store all local poperties
        self.name = name
        self._ast = ast
        self._dummies = as_tuple(a.lower() for a in as_tuple(args))  # Order of dummy arguments
        self._source = source

        self.prefix = as_tuple(prefix)
        self.bind = bind
        self.is_function = is_function

        # The primary IR components
        self.docstring = as_tuple(docstring)
        assert isinstance(spec, Section) or spec is None
        self.spec = spec
        assert isinstance(body, Section) or body is None
        self.body = body
        assert isinstance(contains, Section) or contains is None
        self.contains = contains

        # Then call the parent constructor to take care of symbol table and rescoping
        super().__init__(parent=parent, symbol_attrs=symbol_attrs, rescope_symbols=rescope_symbols)

        # Finally, register this procedure in the parent scope
        if self.parent:
            self.parent.symbol_attrs[self.name] = SymbolAttributes(ProcedureType(procedure=self))

    @staticmethod
    def _infer_allocatable_shapes(spec, body):
        """
        Infer variable symbol shapes from allocations of ``allocatable`` arrays.
        """
        alloc_map = {}
        for alloc in FindNodes(Allocation).visit(body):
            for v in alloc.variables:
                if isinstance(v, Array):
                    if alloc.data_source:
                        alloc_map[v.name.lower()] = alloc.data_source.type.shape
                    else:
                        alloc_map[v.name.lower()] = v.dimensions
        vmap = {}
        for v in FindVariables().visit(body):
            if v.name.lower() in alloc_map:
                vtype = v.type.clone(shape=alloc_map[v.name.lower()])
                vmap[v] = v.clone(type=vtype)
        smap = {}
        for v in FindVariables().visit(spec):
            if v.name.lower() in alloc_map:
                vtype = v.type.clone(shape=alloc_map[v.name.lower()])
                smap[v] = v.clone(type=vtype)
        return (SubstituteExpressions(smap, invalidate_source=False).visit(spec),
                SubstituteExpressions(vmap, invalidate_source=False).visit(body))

    @classmethod
    def from_source(cls, source, definitions=None, xmods=None, frontend=Frontend.FP):
        """
        Create ``Subroutine`` entry node from raw source string using given frontend.

        :param source: Fortran source string
        :param typdedefs: Derived-type definitions from external modules
        :param xmods: Locations of "xmods" module directory for OMNI frontend
        :param frontend: Choice of frontend to use for parsing source (default FP)
        """
        # TODO: Enable pre-processing on-the-fly
        if frontend == Frontend.OMNI:
            ast = parse_omni_source(source, xmods=xmods)
            typetable = ast.find('typeTable')
            f_ast = ast.find('globalDeclarations/FfunctionDefinition')
            return cls.from_omni(ast=f_ast, raw_source=source, typetable=typetable, definitions=definitions)

        if frontend == Frontend.OFP:
            ast = parse_ofp_source(source)
            f_ast = [r for r in list(ast.find('file')) if r.tag in ('subroutine', 'function')].pop()
            return cls.from_ofp(ast=f_ast, raw_source=source, definitions=definitions)

        if frontend == Frontend.FP:
            ast = parse_fparser_source(source)
            f_ast = get_fparser_node(ast, ('Subroutine_Subprogram', 'Function_Subprogram'))
            return cls.from_fparser(ast=f_ast, raw_source=source, definitions=definitions)

        raise NotImplementedError(f'Unknown frontend: {frontend}')

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None, definitions=None, pp_info=None, parent=None):
        name = name or ast.attrib['name']
        is_function = ast.tag == 'function'
        source = extract_source(ast, raw_source, full_lines=True)

        # Store the names of variables in the subroutine signature
        if is_function:
            arg_ast = ast.findall('header/names/name')
            args = [arg.attrib['id'].upper() for arg in arg_ast]
        else:
            arg_ast = ast.findall('header/arguments/argument')
            args = [arg.attrib['name'].upper() for arg in arg_ast]

        # Decompose the body into known sections
        ast_body = list(ast.find('body'))
        ast_spec = ast.find('body/specification')
        idx_spec = ast_body.index(ast_spec)
        ast_docs = ast_body[:idx_spec]
        ast_body = ast_body[idx_spec+1:]

        # Instantiate the subroutine
        routine = cls(name=name, args=args, ast=ast, is_function=is_function, source=source, parent=parent)

        # Create IRs for the docstring and the declaration spec
        routine.docstring = parse_ofp_ast(ast_docs, pp_info=pp_info, raw_source=raw_source, scope=routine)
        routine.spec = parse_ofp_ast(ast_spec, definitions=definitions, pp_info=pp_info,
                                     raw_source=raw_source, scope=routine)

        # Parse "member" subroutines and functions recursively
        if ast.find('members'):
            # We need to pre-populate the ProcedureType symbol table to
            # correctly classify inline function calls within the module
            routine_asts = [s for s in ast.find('members') if s.tag in ('subroutine', 'function')]
            for routine_ast in routine_asts:
                fname = routine_ast.attrib['name']
                if routine_ast.tag == 'function':
                    return_type = SymbolAttributes(BasicType.DEFERRED)
                    dtype = ProcedureType(fname, is_function=True, return_type=return_type)
                else:
                    dtype = ProcedureType(fname, is_function=False)
                routine.symbol_attrs[fname] = SymbolAttributes(dtype)
            members = [Subroutine.from_ofp(ast=member, raw_source=raw_source, definitions=definitions, parent=routine)
                       for member in routine_asts]
            routine._members = as_tuple(members)

        # Generate the subroutine body with all shape and type info
        body = parse_ofp_ast(ast_body, pp_info=pp_info, raw_source=raw_source, scope=routine)
        routine.body = Section(body=body)

        # Now make sure all symbols have their scope attached
        routine.rescope_symbols()

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        routine.spec, routine.body = cls._infer_allocatable_shapes(routine.spec, routine.body)

        # Update array shapes with Loki dimension pragmas
        with pragmas_attached(routine, VariableDeclaration):
            routine.spec = process_dimension_pragmas(routine.spec)

        return routine

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, definitions=None, symbol_map=None, parent=None):
        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {}
        symbol_map.update({s.attrib['type']: s for s in ast.find('symbols')})

        return parse_omni_ast(
            ast=ast, definitions=definitions, raw_source=raw_source,
            type_map=type_map, symbol_map=symbol_map, scope=parent
        )

    @classmethod
    def from_fparser(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        return parse_fparser_ast(ast, pp_info=pp_info, definitions=definitions, raw_source=raw_source, scope=parent)

    @property
    def procedure_symbol(self):
        """
        Return the procedure symbol for this subroutine
        """
        return Variable(name=self.name, type=SymbolAttributes(self.procedure_type), scope=self.parent)

    @property
    def procedure_type(self):
        """
        Return the :any:`ProcedureType` of this subroutine
        """
        return ProcedureType(procedure=self)

    @property
    def return_type(self):
        """
        Return the return_type of this subroutine
        """
        if not self.is_function:
            return None
        return self.symbol_attrs.get(self.name)

    @property
    def variables(self):
        """
        Return the variables (including arguments) declared in this subroutine
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

        Note that arguments also count as variables and therefore any
        removal from this list will also remove arguments from the subroutine signature.
        """
        # First map variables to existing declarations
        declarations = FindNodes((VariableDeclaration, ProcedureDeclaration)).visit(self.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.symbols)

        for v in as_tuple(variables):
            if v not in decl_map:
                # By default, append new symbols to the end of the spec
                if isinstance(v.type.dtype, ProcedureType):
                    new_decl = ProcedureDeclaration(symbols=[v])
                else:
                    new_decl = VariableDeclaration(symbols=[v])
                self.spec.append(new_decl)

        # Run through existing declarations and check that all variables still exist
        dmap = {}
        typedef_decls = set(decl for typedef in FindNodes(TypeDef).visit(self.spec)
                            for decl in typedef.declarations)
        for decl in FindNodes((VariableDeclaration, ProcedureDeclaration)).visit(self.spec):
            if decl in typedef_decls:
            # Slightly hacky: We need to exclude declarations inside TypeDef explicitly
                continue
            new_vars = as_tuple(v for v in decl.symbols if v in variables)
            if len(new_vars) > 0:
                decl._update(symbols=new_vars)
            else:
                dmap[decl] = None  # Mark for removal

        # Remove all redundant declarations
        self.spec = Transformer(dmap).visit(self.spec)

        # Filter the dummy list in case we removed an argument
        varnames = [str(v.name).lower() for v in variables]
        self._dummies = as_tuple(arg for arg in self._dummies if str(arg).lower() in varnames)

    @property
    def variable_map(self):
        """
        Map of variable names to :any:`Variable` objects
        """
        return CaseInsensitiveDict((v.name, v) for v in self.variables)

    @property
    def imported_symbols(self):
        """
        Return the symbols imported in this procedure
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
        Return the list of interfaces declared in this procedure
        """
        return as_tuple(FindNodes(Interface).visit(self.spec))

    @property
    def interface_symbols(self):
        """
        Return the list of symbols declared via interfaces in this subroutine
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
        Return list of all symbols declared or imported in this subroutine scope
        """
        return (
            self.variables + self.imported_symbols + self.interface_symbols + self.enum_symbols +
            tuple(routine.procedure_symbol for routine in self.members)
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
    def arguments(self):
        """
        Return arguments in order of the defined signature (dummy list).
        """
        return as_tuple(self.symbol_map[arg] for arg in self._dummies)

    @arguments.setter
    def arguments(self, arguments):
        """
        Set the arguments property and ensure that internal declarations and signature match.

        Note that removing arguments from this property does not actually remove declarations.
        """
        # FIXME: This will fail if one of the argument is declared via an interface!

        # First map variables to existing declarations
        declarations = FindNodes((VariableDeclaration, ProcedureDeclaration)).visit(self.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.symbols)

        arguments = as_tuple(arguments)
        for arg in arguments:
            if arg not in decl_map:
                # By default, append new variables to the end of the spec
                assert arg.type.intent is not None
                if isinstance(arg.type, ProcedureType):
                    new_decl = ProcedureDeclaration(symbols=[arg])
                else:
                    new_decl = VariableDeclaration(symbols=[arg])
                self.spec.append(new_decl)

        # Set new dummy list according to input
        self._dummies = as_tuple(arg.name.lower() for arg in arguments)

    @property
    def argnames(self):
        """
        Return names of arguments in order of the defined signature (dummy list)
        """
        return [a.name for a in self.arguments]

    @property
    def ir(self):
        """
        Intermediate representation (AST) of the body in this subroutine
        """
        return (self.docstring, self.spec, self.body)

    @property
    def source(self):
        """
        Return the :any:`Source` object associated with this subroutine
        """
        return self._source

    def to_fortran(self, conservative=False):
        """
        Convert subroutine to Fortran source code
        """
        return fgen(self, conservative=conservative)

    @property
    def members(self):
        """
        Tuple of member functions defined in this subroutine
        """
        if self.contains is None:
            return ()
        return as_tuple([
            member for member in self.contains.body if isinstance(member, Subroutine)
        ])

    @property
    def interface(self):
        """
        Interface object that defines the `Subroutine` signature in header files.
        """

        # Remove all local variable declarations from interface routine spec
        # and duplicate all argument symbols within a new subroutine scope
        arg_names = [arg.name for arg in self.arguments]
        routine = Subroutine(name=self.name, args=arg_names, spec=None, body=None)
        decl_map = {}
        for decl in FindNodes((VariableDeclaration, ProcedureDeclaration)).visit(self.spec):
            if any(v in arg_names for v in decl.symbols):
                assert all(v in arg_names and v.type.intent is not None for v in decl.symbols), \
                    "Declarations must have intents and dummy and local arguments cannot be mixed."
                # Replicate declaration with re-scoped variables
                variables = as_tuple(v.clone(scope=routine) for v in decl.symbols)
                decl_map[decl] = decl.clone(symbols=variables)
            else:
                decl_map[decl] = None  # Remove local variable declarations
        routine.spec = Transformer(decl_map).visit(self.spec)
        return Interface(body=(routine,))

    def enrich_calls(self, routines):
        """
        Attach target :class:`Subroutine` object to :class:`CallStatement`
        objects in the IR tree.

        :param call_targets: :class:`Subroutine` objects for corresponding
                             :class:`CallStatement` nodes in the IR tree.
        :param active: Additional flag indicating whether this :call:`CallStatement`
                       represents an active/inactive edge in the
                       interprocedural callgraph.
        """
        routine_map = {r.name.upper(): r for r in as_tuple(routines)}

        with pragmas_attached(self, CallStatement, attach_pragma_post=False):
            for call in FindNodes(CallStatement).visit(self.body):
                name = str(call.name).upper()
                if name in routine_map:
                    # Calls marked as 'reference' are inactive and thus skipped
                    active = not is_loki_pragma(call.pragma, starts_with='reference')
                    context = CallContext(routine=routine_map[name], active=active)
                    call._update(context=context)

        # TODO: Could extend this to module and header imports to
        # facilitate user-directed inlining.

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
        return f'{"Function" if self.is_function else "Subroutine"}:: {self.name}'

    def clone(self, **kwargs):
        """
        Create a copy of the subroutine with the option to override individual
        parameters.

        Parameters
        ----------
        **kwargs :
            Any parameters from the constructor of :any:`Subroutine`.

        Returns
        -------
        :any:`Subroutine`
            The cloned subroutine object.
        """
        if self.name and 'name' not in kwargs:
            kwargs['name'] = self.name
        if self.argnames and 'args' not in kwargs:
            kwargs['args'] = self.argnames
        if self.docstring and 'docstring' not in kwargs:
            kwargs['docstring'] = self.docstring
        if self.spec and 'spec' not in kwargs:
            kwargs['spec'] = self.spec
        if self.body and 'body' not in kwargs:
            kwargs['body'] = self.body
        if self._ast and 'ast' not in kwargs:
            kwargs['ast'] = self._ast
        if self.prefix and 'prefix' not in kwargs:
            kwargs['prefix'] = self.prefix
        if self.bind and 'bind' not in kwargs:
            kwargs['bind'] = self.bind
        if self.is_function and 'is_function' not in kwargs:
            kwargs['is_function'] = self.is_function
        if self.source and 'source' not in kwargs:
            kwargs['source'] = self.source
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent

        if 'rescope_symbols' not in kwargs:
            kwargs['rescope_symbols'] = True

        if 'docstring' in kwargs:
            kwargs['docstring'] = Transformer({}).visit(kwargs['docstring'])
        if 'spec' in kwargs:
            kwargs['spec'] = Transformer({}).visit(kwargs['spec'])
        if 'body' in kwargs:
            kwargs['body'] = Transformer({}).visit(kwargs['body'])

        # Clone the routine and continue with any members
        routine = super().clone(**kwargs)

        if self.contains and 'contains' not in kwargs:
            contains = [
                node.clone(parent=routine, rescope_symbols=kwargs['rescope_symbols']) if isinstance(node, Subroutine)
                else node.clone() for node in self.contains.body
            ]
            routine.contains = self.contains.clone(body=as_tuple(contains))

        return routine
