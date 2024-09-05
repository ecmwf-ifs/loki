# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import abstractmethod

from loki.expression import Variable
from loki.frontend import (
    Frontend, parse_omni_source, parse_ofp_source, parse_fparser_source,
    RegexParserClass, preprocess_cpp, sanitize_input
)
from loki.ir import nodes as ir, FindNodes, Transformer
from loki.logging import debug
from loki.scope import Scope
from loki.tools import CaseInsensitiveDict, as_tuple, flatten
from loki.types import BasicType, DerivedType, ProcedureType


__all__ = ['ProgramUnit']


class ProgramUnit(Scope):
    """
    Common base class for :any:`Module` and :any:`Subroutine`

    Parameters
    ----------
    name : str
        Name of the program unit.
    docstring : tuple of :any:`Node`, optional
        The docstring in the original source.
    spec : :any:`Section`, optional
        The spec of the program unit.
    contains : :any:`Section`, optional
        The internal-subprogram part following a ``CONTAINS`` statement
        declaring module or member procedures
    ast : optional
        Parse tree node from the frontend for this program unit
    source : :any:`Source`
        Source object representing the raw source string information from the
        original file.
    parent : :any:`Scope`, optional
        The enclosing parent scope of the program unit. Declarations from
        the parent scope remain valid within the program unit's scope
        (unless shadowed by local declarations).
    rescope_symbols : bool, optional
        Ensure that the type information for all :any:`TypedSymbol` in the
        IR exist in this program unit's scope or the scope's parents.
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

    def __initialize__(self, name, docstring=None, spec=None, contains=None,
                       ast=None, source=None, rescope_symbols=False, incomplete=False,
                       parser_classes=None):
        # Common properties
        assert name and isinstance(name, str)
        self.name = name
        self._ast = ast
        self._source = source
        self._incomplete = incomplete
        self._parser_classes = parser_classes

        # Bring arguments into shape
        if spec is not None and not isinstance(spec, ir.Section):
            spec = ir.Section(body=as_tuple(spec))
        if contains is not None:
            if not isinstance(contains, ir.Section):
                contains = ir.Section(body=as_tuple(contains))
            for node in contains.body:
                if isinstance(node, ir.Intrinsic) and 'contains' in node.text.lower():
                    break
                if isinstance(node, ProgramUnit):
                    contains.prepend(ir.Intrinsic(text='CONTAINS'))
                    break

        # Primary IR components
        self.docstring = as_tuple(docstring)
        self.spec = spec
        self.contains = contains

        # Finally, register this object in the parent scope
        self.register_in_parent_scope()

        if rescope_symbols:
            self.rescope_symbols()

    @classmethod
    def from_source(cls, source, definitions=None, preprocess=False,
                    includes = None, defines=None, xmods=None, omni_includes=None,
                    frontend=Frontend.FP, parser_classes=None, parent=None):
        """
        Instantiate an object derived from :any:`ProgramUnit` from raw source string

        This calls the frontend-specific factory method implemented in the derived class,
        such as :any:`Module` or :any:`Subroutine`

        Parameters
        ----------
        source : str
            Fortran source string
        definitions : list of :any:`Module`, optional
            :any:`Module` object(s) that may supply external type or procedure
            definitions.
        preprocess : bool, optional
            Flag to trigger CPP preprocessing (by default `False`).

            .. attention::
                Please note that, when using the OMNI frontend, C-preprocessing
                will always be applied, so :data:`includes` and :data:`defines`
                may have to be defined even when disabling :data:`preprocess`.

        includes : list of str, optional
            Include paths to pass to the C-preprocessor.
        defines : list of str, optional
            Symbol definitions to pass to the C-preprocessor.
        xmods : str, optional
            Path to directory to find and store ``.xmod`` files when using the
            OMNI frontend.
        omni_includes: list of str, optional
            Additional include paths to pass to the preprocessor run as part of
            the OMNI frontend parse. If set, this **replaces** (!)
            :data:`includes`, otherwise :data:`omni_includes` defaults to the
            value of :data:`includes`.
        frontend : :any:`Frontend`, optional
            Frontend to use for producing the AST (default :any:`FP`).
        parent : :any:`Scope`, optional
            The parent scope this module or subroutine is nested into
        """
        if isinstance(frontend, str):
            frontend = Frontend[frontend.upper()]

        if preprocess:
            # Trigger CPP-preprocessing explicitly, as includes and
            # defines can also be used by our OMNI frontend
            if frontend == Frontend.OMNI and omni_includes:
                includes = omni_includes
            source = preprocess_cpp(source=source, includes=includes, defines=defines)

        # Preprocess using internal frontend-specific PP rules
        # to sanitize input and work around known frontend problems.
        if frontend != Frontend.OMNI:
            source, pp_info = sanitize_input(source=source, frontend=frontend)

        if frontend == Frontend.REGEX:
            return cls.from_regex(raw_source=source, parser_classes=parser_classes, parent=parent)

        if frontend == Frontend.OMNI:
            ast = parse_omni_source(source, xmods=xmods)
            type_map = {t.attrib['type']: t for t in ast.find('typeTable')}
            return cls.from_omni(ast=ast, raw_source=source, definitions=definitions,
                                 type_map=type_map, parent=parent)

        if frontend == Frontend.OFP:
            ast = parse_ofp_source(source)
            return cls.from_ofp(ast=ast, raw_source=source, definitions=definitions,
                                pp_info=pp_info, parent=parent) # pylint: disable=possibly-used-before-assignment

        if frontend == Frontend.FP:
            ast = parse_fparser_source(source)
            return cls.from_fparser(ast=ast, raw_source=source, definitions=definitions,
                                    pp_info=pp_info, parent=parent)

        raise NotImplementedError(f'Unknown frontend: {frontend}')

    @classmethod
    @abstractmethod
    def from_omni(cls, ast, raw_source, definitions=None, parent=None, type_map=None):
        """
        Create the :any:`ProgramUnit` object from an :any:`OMNI` parse tree.

        This method must be implemented by the derived class.

        Parameters
        ----------
        ast :
            The OMNI parse tree
        raw_source : str
            Fortran source string
        definitions : list, optional
            List of external :any:`Module` to provide derived-type and procedure declarations
        parent : :any:`Scope`, optional
            The enclosing parent scope of the module
        typetable : dict, optional
            A mapping from type hash identifiers to type definitions, as provided in
            OMNI's ``typeTable`` parse tree node
        """

    @classmethod
    @abstractmethod
    def from_ofp(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        """
        Create the :any:`ProgramUnit` object from an :any:`OFP` parse tree.

        This method must be implemented by the derived class.

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
            The enclosing parent scope of the module.
        """

    @classmethod
    @abstractmethod
    def from_fparser(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        """
        Create the :any:`ProgramUnit` object from an :any:`FP` parse tree.

        This method must be implemented by the derived class.

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
            The enclosing parent scope of the module.
        """

    @classmethod
    @abstractmethod
    def from_regex(cls, raw_source, parser_classes=None, parent=None):
        """
        Create the :any:`ProgramUnit` object from source regex'ing.

        This method must be implemented by the derived class.

        Parameters
        ----------
        raw_source : str
            Fortran source string
        parent : :any:`Scope`, optional
            The enclosing parent scope of the module.
        """

    @abstractmethod
    def register_in_parent_scope(self):
        """
        Insert the type information for this object in the parent's symbol table

        If :attr:`parent` is `None`, this does nothing.

        This method must be implemented by the derived class.
        """

    def make_complete(self, **frontend_args):
        """
        Trigger a re-parse of the object if incomplete to produce a full Loki IR

        If the object is marked to be incomplete, i.e. when using the `lazy` constructor
        option, this triggers a new parsing of all :any:`ProgramUnit` objects and any
        :any:`RawSource` nodes in the :attr:`ir`.

        Existing :any:`Module` and :any:`Subroutine` objects continue to exist and references
        to them stay valid, as they will only be updated instead of replaced.
        """
        if not self._incomplete:
            return
        frontend = frontend_args.pop('frontend', Frontend.FP)
        if isinstance(frontend, str):
            frontend = Frontend[frontend.upper()]
        definitions = frontend_args.get('definitions')
        xmods = frontend_args.get('xmods')
        parser_classes = frontend_args.get('parser_classes', RegexParserClass.AllClasses)
        if frontend == Frontend.REGEX and self._parser_classes:
            if self._parser_classes == parser_classes:
                return
            parser_classes = parser_classes | self._parser_classes

        # If this object does not have a parent, we create a temporary parent scope
        # and make sure the node exists in the parent scope. This way, the existing
        # object is re-used while converting the parse tree to Loki-IR.
        has_parent = self.parent is not None
        if not has_parent:
            parent_scope = Scope(parent=None)
            self._reset_parent(parent_scope)
        if self.name not in self.parent.symbol_attrs:
            self.register_in_parent_scope()

        ir_ = self.from_source(
            self.source.string, frontend=frontend, definitions=definitions, xmods=xmods,
            parser_classes=parser_classes, parent=self.parent
        )
        assert ir_ is self

        if not has_parent:
            self._reset_parent(None)

    def enrich(self, definitions, recurse=False):
        """
        Enrich the current scope with inter-procedural annotations

        This updates the :any:`SymbolAttributes` in the scope's :any:`SymbolTable`
        with :data:`definitions` for all imported symbols.

        Note that :any:`Subroutine.enrich` expands this to interface-declared calls.

        Parameters
        ----------
        definitions : list of :any:`ProgramUnit`
            A list of all available definitions
        recurse : bool, optional
            Enrich contained scopes
        """
        definitions_map = CaseInsensitiveDict((r.name, r) for r in as_tuple(definitions))

        for imprt in self.imports:
            if not (module := definitions_map.get(imprt.module)):
                # Skip modules that are not available in the definitions list
                continue

            # Build a list of symbols that are imported
            if imprt.symbols:
                # Import only symbols listed in the only list
                symbols = imprt.symbols
            else:
                # Import all symbols
                rename_list = CaseInsensitiveDict((k, v) for k, v in as_tuple(imprt.rename_list))
                symbols = [
                    Variable(name=rename_list.get(symbol.name, symbol.name), scope=self)
                    for symbol in module.symbols
                ]

            updated_symbol_attrs = {}
            for symbol in symbols:
                # Take care of renaming upon import
                local_name = symbol.name
                remote_name = symbol.type.use_name or local_name
                remote_node = module[remote_name]

                if hasattr(remote_node, 'procedure_type'):
                    # This is a subroutine/function defined in the remote module
                    updated_symbol_attrs[local_name] = symbol.type.clone(
                        dtype=remote_node.procedure_type, imported=True, module=module
                    )
                elif hasattr(remote_node, 'dtype'):
                    # This is a derived type defined in the remote module
                    updated_symbol_attrs[local_name] = symbol.type.clone(
                        dtype=remote_node.dtype, imported=True, module=module
                    )
                    # Update dtype for local variables using this type
                    variables_with_this_type = {
                        name: type_.clone(dtype=remote_node.dtype)
                        for name, type_ in self.symbol_attrs.items()
                        if getattr(type_.dtype, 'name') == remote_node.dtype.name
                    }
                    updated_symbol_attrs.update(variables_with_this_type)
                elif hasattr(remote_node, 'type'):
                    # This is a global variable or interface import
                    updated_symbol_attrs[local_name] = remote_node.type.clone(
                        imported=True, module=module, use_name=symbol.type.use_name
                    )
                else:
                    debug('Cannot enrich import of %s from module %s', local_name, module.name)
            self.symbol_attrs.update(updated_symbol_attrs)

            if imprt.symbols:
                # Rebuild the symbols in the import's symbol list to obtain the correct
                # expression nodes
                imprt._update(symbols=tuple(symbol.clone() for symbol in imprt.symbols))

        # Update any symbol table entries that have been inherited from the parent
        if self.parent:
            updated_symbol_attrs = {}
            for name, attrs in self.symbol_attrs.items():
                if name not in self.parent.symbol_attrs:
                    continue

                if attrs.imported and not attrs.module:
                    updated_symbol_attrs[name] = self.parent.symbol_attrs[name]
                elif isinstance(attrs.dtype, ProcedureType) and attrs.dtype.procedure is BasicType.DEFERRED:
                    updated_symbol_attrs[name] = self.parent.symbol_attrs[name]
                elif isinstance(attrs.dtype, DerivedType) and attrs.dtype.typedef is BasicType.DEFERRED:
                    updated_symbol_attrs[name] = attrs.clone(dtype=self.parent.symbol_attrs[name].dtype)
            self.symbol_attrs.update(updated_symbol_attrs)

        if recurse:
            for routine in self.subroutines:
                routine.enrich(definitions, recurse=True)

    def clone(self, **kwargs):
        """
        Create a deep copy of the object with the option to override individual
        parameters

        Parameters
        ----------
        **kwargs :
            Any parameters from the constructor of the class.

        Returns
        -------
        Object of type ``self.__class__``
            The cloned object.
        """
        # Collect all properties that have not been overriden
        if self.name is not None and 'name' not in kwargs:
            kwargs['name'] = self.name
        if self.docstring and 'docstring' not in kwargs:
            kwargs['docstring'] = self.docstring
        if self.spec and 'spec' not in kwargs:
            kwargs['spec'] = self.spec
        if self.contains and 'contains' not in kwargs:
            contains_needs_clone = True
            kwargs['contains'] = self.contains
        else:
            contains_needs_clone = False
        if self._ast is not None and 'ast' not in kwargs:
            kwargs['ast'] = self._ast
        if self._source is not None and 'source' not in kwargs:
            kwargs['source'] = self._source
        kwargs.setdefault('incomplete', self._incomplete)

        # Rebuild IRs
        rebuild = Transformer({}, rebuild_scopes=True)
        if 'docstring' in kwargs:
            kwargs['docstring'] = rebuild.visit(kwargs['docstring'])
        if 'spec' in kwargs:
            kwargs['spec'] = rebuild.visit(kwargs['spec'])
        if 'contains' in kwargs:
            kwargs['contains'] = rebuild.visit(kwargs['contains'])

        # Rescope symbols if not explicitly disabled
        kwargs.setdefault('rescope_symbols', True)

        # Escalate to Scope's clone function
        obj = super().clone(**kwargs)

        # Update contained routines with new parent scope
        # TODO: Convert ProgramUnit to an IR node(-like) object and make this
        #       work via `Transformer`
        if obj.contains:
            if contains_needs_clone:
                contains = [
                    node.clone(parent=obj, rescope_symbols=kwargs['rescope_symbols'])
                    if isinstance(node, ProgramUnit) else node
                    for node in obj.contains.body
                ]
                obj.contains = obj.contains.clone(body=as_tuple(contains))
            else:
                for node in obj.contains.body:
                    if isinstance(node, ProgramUnit):
                        node._reset_parent(obj)
                        node.register_in_parent_scope()

            # Rescope to ensure that symbol references are up to date
            obj.rescope_symbols()

        obj.register_in_parent_scope()

        return obj

    @property
    def typedefs(self):
        """
        Return the :any:`TypeDef` defined in the :attr:`spec` of this unit
        """
        return as_tuple(FindNodes(ir.TypeDef).visit(self.spec))

    @property
    def typedef_map(self):
        """
        Map of names and :any:`TypeDef` defined in the :attr:`spec` of this unit
        """
        return CaseInsensitiveDict((td.name, td) for td in self.typedefs)

    @property
    def declarations(self):
        """
        Return the declarations from the :attr:`spec` of this unit
        """
        return as_tuple(FindNodes((ir.VariableDeclaration, ir.ProcedureDeclaration)).visit(self.spec))

    @property
    def variables(self):
        """
        Return the variables declared in the :attr:`spec` of this unit
        """
        return as_tuple(flatten(decl.symbols for decl in self.declarations))

    @variables.setter
    def variables(self, variables):
        """
        Set the variables property and ensure that the internal declarations match.
        """
        # First map variables to existing declarations
        decl_map = dict((v, decl) for decl in self.declarations for v in decl.symbols)

        for v in as_tuple(variables):
            if v not in decl_map:
                # By default, append new variables to the end of the spec
                if isinstance(v.type.dtype, ProcedureType):
                    new_decl = ir.ProcedureDeclaration(symbols=(v, ))
                else:
                    new_decl = ir.VariableDeclaration(symbols=(v, ))
                self.spec.append(new_decl)

        # Run through existing declarations and check that all variables still exist
        dmap = {}
        for decl in self.declarations:
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
    def imports(self):
        """
        Return the list of :any:`Import` in this unit
        """
        return as_tuple(FindNodes(ir.Import).visit(self.spec or ()))

    @property
    def import_map(self):
        """
        Map of imported symbol names to :any:`Import` objects
        """
        return CaseInsensitiveDict((s.name, imprt) for imprt in self.imports for s in imprt.symbols)

    @property
    def imported_symbols(self):
        """
        Return the symbols imported in this unit
        """
        imports = self.imports
        return as_tuple(flatten(
            imprt.symbols or [s[1] for s in imprt.rename_list or []]
            for imprt in imports
        ))

    @property
    def imported_symbol_map(self):
        """
        Map of imported symbol names to objects
        """
        return CaseInsensitiveDict((s.name, s) for s in self.imported_symbols)

    @property
    def all_imports(self):
        """
        Return the list of :any:`Import` in this unit and any parent scopes
        """
        imports = self.imports
        scope = self
        while (scope := scope.parent):
            imports += scope.imports
        return imports

    @property
    def interfaces(self):
        """
        Return the list of :any:`Interface` declared in this unit
        """
        return as_tuple(FindNodes(ir.Interface).visit(self.spec))

    @property
    def interface_symbols(self):
        """
        Return the list of symbols declared via interfaces in this unit
        """
        return as_tuple(flatten(intf.symbols for intf in self.interfaces))

    @property
    def interface_map(self):
        """
        Map of declared interface names to :any:`Interface` nodes
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
        return as_tuple(flatten(enum.symbols for enum in FindNodes(ir.Enumeration).visit(self.spec or ())))

    @property
    def definitions(self):
        """
        The list of IR nodes defined by this program unit.

        Returns an empty tuple by default and can be overwritten by derived nodes.
        """
        return ()

    @property
    def symbols(self):
        """
        Return list of all symbols declared or imported in this module scope
        """

        #Find all nodes that may contain symbols
        nodelist = FindNodes((ir.VariableDeclaration, ir.ProcedureDeclaration,
                    ir.Import, ir.Interface, ir.Enumeration)).visit(self.spec or ())

        #Return all symbols found in nodelist as well as any procedure_symbols
        #in contained subroutines
        return as_tuple(flatten(n.symbols for n in nodelist)) + \
               tuple(routine.procedure_symbol for routine in self.subroutines)

    @property
    def symbol_map(self):
        """
        Map of symbol names to symbols
        """
        return CaseInsensitiveDict(
            (s.name, s) for s in self.symbols
        )

    def get_symbol(self, name):
        """
        Returns the symbol for a given name as defined in its declaration.

        The returned symbol might include dimension symbols if it was
        declared as an array.

        Parameters
        ----------
        name : str
            Base name of the symbol to be retrieved
        """
        return self.get_symbol_scope(name).variable_map.get(name)

    def Variable(self, **kwargs):
        """
        Factory method for :any:`TypedSymbol` or :any:`MetaSymbol` classes.

        This invokes the :any:`Variable` with this node as the scope.

        Parameters
        ----------
        name : str
            The name of the variable.
        type : optional
            The type of that symbol. Defaults to :any:`BasicType.DEFERRED`.
        parent : :any:`Scalar` or :any:`Array`, optional
            The derived type variable this variable belongs to.
        dimensions : :any:`ArraySubscript`, optional
            The array subscript expression.
        """
        kwargs['scope'] = self
        return Variable(**kwargs)

    @property
    def subroutines(self):
        """
        List of :class:`Subroutine` objects that are declared in this unit
        """
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel,cyclic-import
        if self.contains is None:
            return ()
        return as_tuple([
            routine for routine in self.contains.body if isinstance(routine, Subroutine)
        ])

    routines = subroutines

    @property
    def subroutine_map(self):
        """
        Map of subroutine names to :any:`Subroutine` objects in :attr:`subroutines`
        """
        return CaseInsensitiveDict(
            (s.name, s) for s in self.subroutines
        )

    @property
    def spec_parts(self):
        """
        Return the :attr:`spec` subdivided into the parts the Fortran standard
        describes and requires to appear in a specific order

        The parts are:

        * import statements (such as module imports via ``USE``)
        * implicit-part (such as ``IMPLICIT NONE``)
        * declaration constructs (such as access statements, variable declarations etc.)

        This can be useful when adding or looking for statements that have to appear
        in a certain position.

        Note that comments at the interface between parts may be allocated to the
        previous or next part.

        Returns
        -------
        tuple of tuple of :class:`ir.Node`
            The parts of the spec, with empty parts represented by empty tuples.
        """
        if not self.spec:
            return ((),(),())

        intrinsic_nodes = FindNodes(ir.Intrinsic).visit(self.spec)
        implicit_nodes = [node for node in intrinsic_nodes if node.text.lstrip().lower().startswith('implicit')]

        if implicit_nodes:
            # Use 'IMPLICIT' statements as divider
            implicit_start_index = self.spec.body.index(implicit_nodes[0])
            if len(implicit_nodes) == 1:
                implicit_end_index = implicit_start_index
            else:
                implicit_end_index = self.spec.body.index(implicit_nodes[-1])

            return (
                self.spec.body[:implicit_start_index],
                self.spec.body[implicit_start_index:implicit_end_index+1],
                self.spec.body[implicit_end_index+1:]
            )

        # No 'IMPLICIT' statements: find the end of imports
        import_nodes = FindNodes(ir.Import).visit(self.spec)

        if not import_nodes:
            return ((), (), self.spec.body)

        import_nodes_end_index = self.spec.body.index(import_nodes[-1])
        return (
            self.spec.body[:import_nodes_end_index+1],
            (),
            self.spec.body[import_nodes_end_index+1:]
        )

    @property
    def ir(self):
        """
        All components of the intermediate representation in this unit
        """
        return (self.docstring, self.spec, self.contains)

    @property
    def source(self):
        """
        The :any:`Source` object for this unit
        """
        return self._source

    def to_fortran(self, conservative=False, cuf=False):
        """
        Convert this unit to Fortran source representation
        """
        if cuf:
            from loki.backend.cufgen import cufgen # pylint: disable=import-outside-toplevel
            return cufgen(self, conservative=conservative)
        from loki.backend.fgen import fgen  # pylint: disable=import-outside-toplevel
        return fgen(self, conservative=conservative)

    def __repr__(self):
        """
        Short string representation
        """
        return f'{self.__class__.__name__}:: {self.name}'

    def __contains__(self, name):
        """
        Check if a symbol, type or subroutine with the given name is declared
        inside this unit
        """
        return name in self.symbols or name in self.typedef_map

    def __getitem__(self, name):
        """
        Get the IR node of the subroutine, typedef, imported symbol or declared
        variable corresponding to the given name
        """
        if not isinstance(name, str):
            raise TypeError('Name lookup requires a string!')

        item = self.subroutine_map.get(name)
        if item is None:
            item = self.typedef_map.get(name)
        if item is None:
            item = self.symbol_map[name]
        return item

    def __iter__(self):
        """
        Make :any:`ProgramUnit`s non-iterable
        """
        raise TypeError('ProgramUnit nodes can not be traversed. Try `ir` or `subroutines` instead.')

    def __bool__(self):
        """
        Ensure existing objects register as True in boolean checks, despite
        raising exceptions in :meth:`__iter__`.
        """
        return True

    def apply(self, op, **kwargs):
        """
        Apply a given transformation to this program unit

        Note that the dispatch routine ``op.apply(source)`` will ensure
        that all entities of this :any:`ProgramUnit` are correctly traversed.
        """
        # TODO: Should type-check for an `Operation` object here
        op.apply(self, **kwargs)

    def resolve_typebound_var(self, name, variable_map=None):
        """
        A small convenience utility to resolve type-bound variables.

        Parameters
        ----------
        name : str
            The full name of the variable to be resolved, e.g., a%b%c%d.
        variable_map : dict
            A map of the variables defined in the current scope.
        """

        if not (_variable_map := variable_map):
            _variable_map = self.variable_map

        name_parts = name.split('%', maxsplit=1)
        var = _variable_map[name_parts[0]]
        if len(name_parts) > 1:
            var = var.get_derived_type_member(name_parts[1])
        return var
