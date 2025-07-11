# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki.frontend import (
    parse_omni_ast, parse_fparser_ast, get_fparser_node,
    parse_regex_source
)
from loki.ir import (
    nodes as ir, FindNodes, Transformer, ExpressionTransformer,
    pragmas_attached
)
from loki.logging import debug
from loki.program_unit import ProgramUnit
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
    parser_classes : :any:`RegexParserClass`, optional
        Provide the list of parser classes used during incomplete regex parsing
    """

    is_function = False

    def __init__(self, *args, parent=None, symbol_attrs=None, **kwargs):
        super().__init__(parent=parent)

        if symbol_attrs:
            self.symbol_attrs.update(symbol_attrs)

        self.__initialize__(*args, **kwargs)

    def __initialize__(
            self, name, docstring=None, spec=None, contains=None,
            ast=None, source=None, rescope_symbols=False, incomplete=False,
            parser_classes=None, body=None, args=None, prefix=None, bind=None,
    ):
        # First, store additional Subroutine-specific properties
        self._dummies = as_tuple(a.lower() for a in as_tuple(args))  # Order of dummy arguments
        self.prefix = as_tuple(prefix)
        self.bind = bind

        # Additional IR components
        if body is not None and not isinstance(body, ir.Section):
            body = ir.Section(body=body)
        self.body = body

        super().__initialize__(
            name=name, docstring=docstring, spec=spec, contains=contains,
            ast=ast, source=source, rescope_symbols=rescope_symbols,
            incomplete=incomplete, parser_classes=parser_classes
        )

    def __getstate__(self):
        _ignore = ('_ast', '_parent')
        return dict((k, v) for k, v in self.__dict__.items() if k not in _ignore)

    def __setstate__(self, s):
        self.__dict__.update(s)

        self._ast = None

        # Re-register all encapulated member procedures
        for member in self.members:
            self.symbol_attrs[member.name] = SymbolAttributes(ProcedureType(procedure=member))

        # Ensure that we are attaching all symbols to the newly create ``self``.
        self.rescope_symbols()

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

        # Rebuild body (other IR components are taken care of in super class)
        if 'body' in kwargs:
            kwargs['body'] = Transformer({}, rebuild_scopes=True).visit(kwargs['body'])

        # Escalate to parent class
        return super().clone(**kwargs)

    @property
    def _canonical(self):
        """
        Base definition for comparing :any:`Subroutine` objects.
        """
        return (
            self.name, self._dummies, self.prefix, self.bind,
            self.docstring, self.spec, self.body, self.contains,
            self.symbol_attrs
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

    variables = ProgramUnit.variables

    @variables.setter
    def variables(self, variables):
        """
        Set the variables property and ensure that the internal declarations match.

        Note that arguments also count as variables and therefore any
        removal from this list will also remove arguments from the subroutine signature.
        """
        # Use the parent's property setter
        ProgramUnit.variables.__set__(self, variables)

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
            if any(v.name in arg_names for v in decl.symbols):
                assert all(v.name in arg_names and v.type.intent is not None for v in decl.symbols), \
                    "Declarations must have intents and dummy and local arguments cannot be mixed."
                # Replicate declaration with re-scoped variables
                variables = as_tuple(v.clone(scope=routine) for v in decl.symbols)
                decl_map[decl] = decl.clone(symbols=variables)
            else:
                decl_map[decl] = None  # Remove local variable declarations
        routine.spec = Transformer(decl_map).visit(self.spec)
        return ir.Interface(body=(routine,))

    def enrich(self, definitions, recurse=False):
        """
        Apply :any:`ProgramUnit.enrich` and expand enrichment to calls declared
        via interfaces

        Parameters
        ----------
        definitions : list of :any:`ProgramUnit`
            A list of all available definitions
        recurse : bool, optional
            Enrich contained scopes
        """
        # First, enrich imported symbols
        super().enrich(definitions, recurse=recurse)

        # Secondly, take care of procedures that are declared via interface block includes
        # and therefore are not discovered via module imports
        definitions_map = CaseInsensitiveDict((r.name, r) for r in as_tuple(definitions))
        with pragmas_attached(self, ir.CallStatement, attach_pragma_post=False):
            for call in FindNodes(ir.CallStatement).visit(self.body):

                # Clone symbol to ensure Deferred symbols are
                # recognised ProcedureSymbols
                symbol = call.name.clone()
                routine = definitions_map.get(symbol.name)

                if not routine and symbol.parent:
                    # Type-bound procedure: try to obtain procedure from typedef
                    if (dtype := symbol.parent.type.dtype) is not BasicType.DEFERRED:
                        if (typedef := dtype.typedef) is not BasicType.DEFERRED:
                            if proc_symbol := typedef.variable_map.get(symbol.name_parts[-1]):
                                if (dtype := proc_symbol.type.dtype) is not BasicType.DEFERRED:
                                    if dtype.procedure is not BasicType.DEFERRED:
                                        routine = dtype.procedure

                is_not_enriched = (
                    symbol.scope is None or                         # No scope attached
                    symbol.type.dtype is BasicType.DEFERRED or      # Wrong datatype
                    symbol.type.dtype.procedure is not routine      # ProcedureType not linked
                )

                # Always update the call symbol to ensure it is up-to-date
                call._update(name=symbol)

                # Skip already enriched symbols and routines without definitions
                if not (routine and is_not_enriched):
                    debug('Cannot enrich call to %s', symbol)
                    continue

                # Remove existing symbol from symbol table if defined in interface block
                for node in [node for intf in self.interfaces for node in intf.body]:
                    if getattr(node, 'name', None) == symbol:
                        if node.parent == self:
                            node.parent = None

                # Need to update the call's symbol to establish link to routine
                symbol = symbol.clone(scope=self, type=symbol.type.clone(dtype=routine.procedure_type))
                call._update(name=symbol)

        # Rebuild local symbols to ensure correct symbol types
        self.body = ExpressionTransformer(inplace=True).visit(self.body)

    def __repr__(self):
        """ String representation """
        return f'Subroutine:: {self.name}'
