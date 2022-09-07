from abc import abstractmethod

from loki import ir
from loki.frontend import Frontend, parse_omni_source, parse_ofp_source, parse_fparser_source
from loki.scope import Scope
from loki.tools import CaseInsensitiveDict, as_tuple, flatten
from loki.types import ProcedureType
from loki.visitors import FindNodes, Transformer


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
    """

    def __init__(self, name, docstring=None, spec=None, contains=None,
                 ast=None, source=None, parent=None,
                 rescope_symbols=False, symbol_attrs=None):
        # Common properties
        assert name and isinstance(name, str)
        self.name = name
        self._ast = ast
        self._source = source

        # Bring arguments into shape
        if spec is not None and not isinstance(spec, ir.Section):
            spec = ir.Section(body=spec)
        if contains is not None:
            if not isinstance(contains, ir.Section):
                contains = ir.Section(body=contains)
            for node in contains.body:
                if isinstance(node, ir.Intrinsic) and 'contains' in node.text.lower():
                    break
                else:
                    contains.prepend(ir.Intrinsic(text='CONTAINS'))

        # Primary IR components
        self.docstring = as_tuple(docstring)
        self.spec = spec
        self.contains = contains

        # Call the parent constructor to take care of symbol table and rescoping
        super().__init__(parent=parent, symbol_attrs=symbol_attrs, rescope_symbols=rescope_symbols)

    @classmethod
    def from_source(cls, source, definitions=None, xmods=None, frontend=Frontend.FP, parent=None):
        """
        Instantiate an object derived from :any:`ProgramUnit` from raw source string

        This calls the frontend-specific factory method implemented in the derived class,
        such as :any:`Module` or :any:`Subroutine`

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
        parent : :any:`Scope`, optional
            The parent scope this module or subroutine is nested into
        """
        if frontend == Frontend.REGEX:
            return cls.from_regex(raw_source=source, parent=parent)

        if frontend == Frontend.OMNI:
            ast = parse_omni_source(source, xmods=xmods)
            type_map = {t.attrib['type']: t for t in ast.find('typeTable')}
            return cls.from_omni(ast=ast, raw_source=source, definitions=definitions, type_map=type_map, parent=parent)

        if frontend == Frontend.OFP:
            ast = parse_ofp_source(source)
            return cls.from_ofp(ast=ast, raw_source=source, definitions=definitions, parent=parent)

        if frontend == Frontend.FP:
            ast = parse_fparser_source(source)
            return cls.from_fparser(ast=ast, raw_source=source, definitions=definitions, parent=parent)

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
    def from_regex(cls, raw_source, parent=None):
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
            kwargs['contains'] = self.contains
        if self._ast is not None and 'ast' not in kwargs:
            kwargs['ast'] = self._ast
        if self._source is not None and 'source' not in kwargs:
            kwargs['source'] = self._source

        # Rebuild IRs
        if 'docstring' in kwargs:
            kwargs['docstring'] = Transformer({}).visit(kwargs['docstring'])
        if 'spec' in kwargs:
            kwargs['spec'] = Transformer({}).visit(kwargs['spec'])
        if 'contains' in kwargs:
            kwargs['contains'] = Transformer({}).visit(kwargs['contains'])

        # Rescope symbols if not explicitly disabled
        kwargs.setdefault('rescope_symbols', True)

        # Escalate to Scope's clone function
        obj = super().clone(**kwargs)

        # Update contained routines with new parent scope
        # TODO: Convert ProgramUnit to an IR node(-like) object and make this
        #       work via `Transformer`
        if obj.contains:
            contains = [
                node.clone(parent=obj, rescope_symbols=kwargs['rescope_symbols'])
                if isinstance(node, ProgramUnit) else node
                for node in obj.contains.body
            ]
            obj.contains = obj.contains.clone(body=as_tuple(contains))

        return obj

    @property
    def typedefs(self):
        """
        Map of names and :any:`TypeDef` defined in the :attr:`spec` of this unit
        """
        types = FindNodes(ir.TypeDef).visit(self.spec)
        return CaseInsensitiveDict((td.name, td) for td in types)

    @property
    def declarations(self):
        """
        Return the declarations from the :attr:`spec` of this unit
        """
        return FindNodes((ir.VariableDeclaration, ir.ProcedureDeclaration)).visit(self.spec)

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
    def imported_symbols(self):
        """
        Return the symbols imported in this unit
        """
        imports = FindNodes(ir.Import).visit(self.spec or ())
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

    @property
    def subroutines(self):
        """
        List of :class:`Subroutine` objects that are declared in this unit
        """
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel
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

    def to_fortran(self, conservative=False):
        """
        Convert this unit to Fortran source representation
        """
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
        return name in self.symbols or name in self.typedefs

    def __getitem__(self, name):
        """
        Get the IR node of the subroutine, typedef, imported symbol or declared
        variable corresponding to the given name
        """
        if not isinstance(name, str):
            raise TypeError('Name lookup requires a string!')

        item = self.subroutine_map.get(name)
        if item is None:
            item = self.typedefs.get(name)
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
