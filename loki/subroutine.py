# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import ir
from loki.expression import FindVariables, SubstituteExpressions, symbols as sym
from loki.frontend import (
    parse_omni_ast, parse_ofp_ast, parse_fparser_ast, get_fparser_node,
    parse_regex_source
)
from loki.pragma_utils import is_loki_pragma, pragmas_attached
from loki.program_unit import ProgramUnit
from loki.visitors import FindNodes, Transformer
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import BasicType, ProcedureType, SymbolAttributes


__all__ = ['Subroutine']


class Subroutine(ProgramUnit):
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
    prefix : iterable, optional
        Prefix specifications for the procedure
    bind : optional
        Bind information (e.g., for Fortran ``BIND(C)`` annotation).
    result_name : str, optional
        The name of the result variable for functions.
    is_function : bool, optional
        Flag to indicate this is a function instead of subroutine
        (in the Fortran sense). Defaults to `False`.
    ast : optional
        Frontend node for this subroutine (from parse tree of the frontend).
    source : :any:`Source`
        Source object representing the raw source string information from the
        read file.
    parent : :any:`Scope`, optional
        The enclosing parent scope of the subroutine, typically a :any:`Module`
        or :any:`Subroutine` object. Declarations from the parent scope remain
        valid within the subroutine's scope (unless shadowed by local
        declarations).
    rescope_symbols : bool, optional
        Ensure that the type information for all :any:`TypedSymbol` in the
        subroutine's IR exist in the subroutine's scope or the scope's parents.
        Defaults to `False`.
    symbol_attrs : :any:`SymbolTable`, optional
        Use the provided :any:`SymbolTable` object instead of creating a new
    incomplete : bool, optional
        Mark the object as incomplete, i.e. only partially parsed. This is
        typically the case when it was instantiated using the :any:`Frontend.REGEX`
        frontend and a full parse using one of the other frontends is pending.
    """

    def __init__(self, name, args=None, docstring=None, spec=None, body=None, contains=None,
                 prefix=None, bind=None, result_name=None, is_function=False, ast=None, source=None, parent=None,
                 rescope_symbols=False, symbol_attrs=None, incomplete=False):
        # First, store additional Subroutine-specific properties
        self._dummies = as_tuple(a.lower() for a in as_tuple(args))  # Order of dummy arguments
        self.prefix = as_tuple(prefix)
        self.bind = bind
        self.result_name = result_name
        self.is_function = is_function

        # Additional IR components
        if body is not None and not isinstance(body, ir.Section):
            body = ir.Section(body=body)
        self.body = body

        # Then call the parent constructor to store common properties
        super().__init__(
            name=name, docstring=docstring, spec=spec, contains=contains,
            ast=ast, source=source, parent=parent, rescope_symbols=rescope_symbols,
            symbol_attrs=symbol_attrs, incomplete=incomplete
        )

    def __getstate__(self):
        _ignore = ('_ast', '_parent')
        return dict((k, v) for k, v in self.__dict__.items() if k not in _ignore)

    def __setstate__(self, s):
        self.__dict__.update(s)

        self._ast = None
        self._parent = None

        # Re-register all encapulated member procedures
        for member in self.members:
            self.symbol_attrs[member.name] = SymbolAttributes(ProcedureType(procedure=member))

        # Ensure that we are attaching all symbols to the newly create ``self``.
        self.rescope_symbols()

    @staticmethod
    def _infer_allocatable_shapes(spec, body):
        """
        Infer variable symbol shapes from allocations of ``allocatable`` arrays.
        """
        alloc_map = {}
        for alloc in FindNodes(ir.Allocation).visit(body):
            for v in alloc.variables:
                if isinstance(v, sym.Array):
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
    def from_omni(cls, ast, raw_source, definitions=None, parent=None, type_map=None):
        """
        Create :any:`Subroutine` from :any:`OMNI` parse tree

        Parameters
        ----------
        ast :
            The OMNI parse tree
        raw_source : str
            Fortran source string
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        parent : :any:`Scope`, optional
            The enclosing parent scope of the subroutine, typically a :any:`Module`.
        type_map : dict, optional
            A mapping from type hash identifiers to type definitions, as provided in
            OMNI's ``typeTable`` parse tree node
        """
        type_map = type_map or {}
        if ast.tag != 'FfunctionDefinition':
            ast = ast.find('globalDeclarations/FfunctionDefinition')
        return parse_omni_ast(
            ast=ast, definitions=definitions, raw_source=raw_source,
            type_map=type_map, scope=parent
        )

    @classmethod
    def from_ofp(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        """
        Create :any:`Subroutine` from :any:`OFP` parse tree

        Parameters
        ----------
        ast :
            The OFP parse tree
        raw_source : str
            Fortran source string
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        pp_info :
            Preprocessing info as obtained by :any:`sanitize_input`
        parent : :any:`Scope`, optional
            The enclosing parent scope of the subroutine, typically a :any:`Module`.
        """
        if ast.tag not in ('subroutine', 'function'):
            ast = [r for r in as_tuple(ast.find('file')) if r.tag in ('subroutine', 'function')].pop()
        return parse_ofp_ast(
            ast=ast, pp_info=pp_info, raw_source=raw_source,
            definitions=definitions, scope=parent
        )

    @classmethod
    def from_fparser(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        """
        Create :any:`Subroutine` from :any:`FP` parse tree

        Parameters
        ----------
        ast :
            The FParser parse tree
        raw_source : str
            Fortran source string
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        pp_info :
            Preprocessing info as obtained by :any:`sanitize_input`
        parent : :any:`Scope`, optional
            The enclosing parent scope of the subroutine, typically a :any:`Module`.
        """
        if ast.__class__.__name__ not in ('Subroutine_Subprogram', 'Function_Subprogram'):
            ast = get_fparser_node(ast, ('Subroutine_Subprogram', 'Function_Subprogram'))
        # Note that our Fparser interface returns a tuple with the
        # Subroutine object always last but potentially containing
        # comments before the Subroutine object
        return parse_fparser_ast(
            ast, pp_info=pp_info, definitions=definitions,
            raw_source=raw_source, scope=parent
        )[-1]

    @classmethod
    def from_regex(cls, raw_source, parser_classes=None, parent=None):
        """
        Create :any:`Subroutine` from source regex'ing

        Parameters
        ----------
        raw_source : str
            Fortran source string
        parent : :any:`Scope`, optional
            The enclosing parent scope of the subroutine, typically a :any:`Module`.
        """
        ir_ = parse_regex_source(raw_source, parser_classes=parser_classes, scope=parent)
        return [node for node in ir_.body if isinstance(node, cls)][0]

    def register_in_parent_scope(self):
        """
        Insert the type information for this object in the parent's symbol table

        If :attr:`parent` is `None`, this does nothing.
        """
        if self.parent:
            self.parent.symbol_attrs[self.name] = SymbolAttributes(self.procedure_type)

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
        # Collect all properties bespoke to Subroutine
        if self.argnames and 'args' not in kwargs:
            kwargs['args'] = self.argnames
        if self.body and 'body' not in kwargs:
            kwargs['body'] = self.body
        if self.prefix and 'prefix' not in kwargs:
            kwargs['prefix'] = self.prefix
        if self.bind and 'bind' not in kwargs:
            kwargs['bind'] = self.bind
        if self.result_name and 'result_name' not in kwargs:
            kwargs['result_name'] = self.result_name
        if self.is_function and 'is_function' not in kwargs:
            kwargs['is_function'] = self.is_function

        # Rebuild body (other IR components are taken care of in super class)
        if 'body' in kwargs:
            kwargs['body'] = Transformer({}).visit(kwargs['body'])

        # Escalate to parent class
        return super().clone(**kwargs)

    @property
    def _canonical(self):
        """
        Base definition for comparing :any:`Subroutine` objects.
        """
        return (
            self.name, self.is_function, self._dummies, self.prefix,
            self.bind, self.docstring, self.spec, self.body,
            self.contains, self.symbol_attrs
        )

    def __eq__(self, other):
        if isinstance(other, Subroutine):
            return self._canonical == other._canonical
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._canonical)

    @property
    def procedure_symbol(self):
        """
        Return the procedure symbol for this subroutine
        """
        return sym.Variable(name=self.name, type=SymbolAttributes(self.procedure_type), scope=self.parent)

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
        if self.result_name is not None:
            return self.symbol_attrs.get(self.result_name)
        return self.symbol_attrs.get(self.name)

    variables = ProgramUnit.variables

    @variables.setter
    def variables(self, variables):
        """
        Set the variables property and ensure that the internal declarations match.

        Note that arguments also count as variables and therefore any
        removal from this list will also remove arguments from the subroutine signature.
        """
        # Use the parent's property setter
        super(self.__class__, self.__class__).variables.__set__(self, variables)

        # Filter the dummy list in case we removed an argument
        varnames = [str(v.name).lower() for v in variables]
        self._dummies = as_tuple(arg for arg in self._dummies if str(arg).lower() in varnames)

    @property
    def arguments(self):
        """
        Return arguments in order of the defined signature (dummy list).
        """

        #Load symbol_map
        #Note that if the map is not loaded, Python will recreate it for every arguement,
        #resulting in a large overhead.
        symbol_map = self.symbol_map
        return as_tuple(symbol_map.get(arg, sym.Variable(name=arg)) for arg in self._dummies)

    @arguments.setter
    def arguments(self, arguments):
        """
        Set the arguments property and ensure that internal declarations and signature match.

        Note that removing arguments from this property does not actually remove declarations.
        """
        # FIXME: This will fail if one of the argument is declared via an interface!

        # First map variables to existing declarations
        declarations = FindNodes((ir.VariableDeclaration, ir.ProcedureDeclaration)).visit(self.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.symbols)

        arguments = as_tuple(arguments)
        for arg in arguments:
            if arg not in decl_map:
                # By default, append new variables to the end of the spec
                assert arg.type.intent is not None
                if isinstance(arg.type, ProcedureType):
                    new_decl = ir.ProcedureDeclaration(symbols=(arg, ))
                else:
                    new_decl = ir.VariableDeclaration(symbols=(arg, ))
                self.spec.append(new_decl)

        # Set new dummy list according to input
        self._dummies = as_tuple(arg.name.lower() for arg in arguments)

    @property
    def argnames(self):
        """
        Return names of arguments in order of the defined signature (dummy list)
        """
        return [a.name for a in self.arguments]

    members = ProgramUnit.subroutines

    @property
    def ir(self):
        """
        All components of the intermediate representation in this subroutine
        """
        return (self.docstring, self.spec, self.body, self.contains)

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
        for decl in FindNodes((ir.VariableDeclaration, ir.ProcedureDeclaration)).visit(self.spec):
            if any(v in arg_names for v in decl.symbols):
                assert all(v in arg_names and v.type.intent is not None for v in decl.symbols), \
                    "Declarations must have intents and dummy and local arguments cannot be mixed."
                # Replicate declaration with re-scoped variables
                variables = as_tuple(v.clone(scope=routine) for v in decl.symbols)
                decl_map[decl] = decl.clone(symbols=variables)
            else:
                decl_map[decl] = None  # Remove local variable declarations
        routine.spec = Transformer(decl_map).visit(self.spec)
        return ir.Interface(body=(routine,))

    def enrich_calls(self, routines):
        """
        Update :any:`SymbolAttributes` for the ``name`` property of
        :any:`CallStatement` nodes to provide links to the :any:`Subroutine`
        nodes given in :data:`routines`.

        Parameters
        ----------
        routines : (list of) :any:`Subroutine`
            Possible targets of :any:`CallStatement` calls
        """
        routine_map = CaseInsensitiveDict((r.name, r) for r in as_tuple(routines))

        with pragmas_attached(self, ir.CallStatement, attach_pragma_post=False):
            for call in FindNodes(ir.CallStatement).visit(self.body):
                name = str(call.name)
                # Calls marked as 'reference' are inactive and thus skipped
                not_active = is_loki_pragma(call.pragma, starts_with='reference')

                # Update symbol table if necessary and present in routine_map
                routine = routine_map.get(name)
                if isinstance(routine, sym.ProcedureSymbol):
                    # Type-bound procedure: shortcut to bound procedure if not generic
                    if routine.type.bind_names and len(routine.type.bind_names) == 1:
                        routine = routine.type.bind_names[0].type.dtype.procedure
                    else:
                        routine = None
                if routine is not None:
                    name_type = call.name.type
                    update_symbol = (
                        call.name.scope is None or                # No scope attached
                        name_type.dtype is BasicType.DEFERRED or  # No ProcedureType attached
                        name_type.dtype.procedure is not routine  # ProcedureType not linked to routine
                    )
                    if update_symbol:
                        # Need to update the call's symbol to establish link to routine
                        name_type = name_type.clone(dtype=routine.procedure_type)
                        call._update(name=call.name.clone(scope=self, type=name_type), not_active=not_active)

                # In any case, update the not_active attribute
                if call.not_active is not not_active:
                    # Need to update only the active status of the call
                    call._update(not_active=not_active)

        # TODO: Could extend this to module and header imports to
        # facilitate user-directed inlining.

    def __repr__(self):
        """
        String representation.
        """
        return f'{"Function" if self.is_function else "Subroutine"}:: {self.name}'
