# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import abstractmethod
try:
    from functools import cached_property
except ImportError:
    try:
        from cached_property import cached_property
    except ImportError:
        def cached_property(func):
            return func

from loki.logging import warning
from loki.tools import as_tuple
from loki.tools.util import CaseInsensitiveDict
from loki.visitors import FindNodes
from loki.ir import CallStatement, TypeDef, ProcedureDeclaration
from loki.subroutine import Subroutine


__all__ = ['Item', 'SubroutineItem', 'ProcedureBindingItem', 'GlobalVarImportItem', 'GenericImportItem']


class Item:
    """
    Base class of a work item that represents a single source routine to be processed

    Each :any:`Item` spawns new work items according to its
    own subroutine calls and can be configured to ignore individual
    sub-trees.

    Depending of the nature of the work item, the implementation of :class:`Item` is
    done in subclasses :class:`SubroutineItem`, :class:`ProcedureBindingItem`,
    :class:`GlobalVarImportItem` and :class:`GenericImportItem`.

    The :attr:`name` of a :class:`Item` refers to the routine or variable name using
    a fully-qualified name in the format ``<scope_name>#<local_name>``. The
    ``<scope_name>`` corresponds to a Fortran module that a subroutine or variable is declared
    in, or can be empty if the subroutine is not enclosed in a module (i.e. exists
    in the global scope). This is to enable use of routines with the same name that
    are declared in different modules.
    The corresponding parts of the name can be accessed via :attr:`scope_name` and
    :attr:`local_name`.

    For type-bound procedures, the :attr:`local_name` should take the format
    ``<type_name>%<binding_name>``. This may also span across multiple derived types, e.g.,
    to allow calls to type bound procedures of a derived type variable that in turn
    is a member of another derived type, e.g., ``<type_name>%<member_name>%<binding_name>``.
    See :class:`ProcedureBindingItem`` for more details.

    Notes
    -----

    Each work item may have its own configuration settings that
    primarily inherit values from the `'default'`, but can be
    specialised explicitly in the config file or dictionary.

    Possible arguments are:

    * ``role``: Role string to pass to the :any:`Transformation` (eg. "kernel")
    * ``mode``: Transformation "mode" to pass to the transformation
    * ``expand``: Flag to generally enable/disable expansion under this item
    * ``strict``: Flag controlling whether to strictly fail if source file cannot be parsed
    * ``replicated``: Flag indicating whether to mark item as "replicated" in call graphs
    * ``disable``: List of subroutines that are completely ignored and are not reported
      as ``children``. Useful to exclude entire call trees or utility routines.
    * ``block``: List of subroutines that should should not be added to the scheduler
      tree. Note, these might still be shown in the graph visulisation.
    * ``ignore``: Individual list of subroutine calls to "ignore" during expansion.
      Calls to these routines may be processed on the caller side but not the called subroutine
      itself. This facilitates processing across build targets, where caller and callee-side
      are transformed in different Loki passes.
    * ``enrich``: List of subroutines that should still be looked up and used to "enrich"
      :any:`CallStatement` nodes in this :any:`Item` for inter-procedural
      transformation passes.

    Parameters
    ----------
    name : str
        Name to identify items in the schedulers graph
    source : :any:`Sourcefile`
        The underlying source file that contains the associated item
    config : dict
        Dict of item-specific config markers
    """

    def __init__(self, name, source, config=None):
        assert '#' in name
        self.name = name
        self.source = source
        self.config = config or {}
        self.trafo_data = {}

    def __repr__(self):
        return f'loki.bulk.Item<{self.name}>'

    def __eq__(self, other):
        """
        :class:`Item` objects are considered equal if they refer to the same
        fully-qualified name, i.e., :attr:`name` is identical

        This allows also comparison against a string.
        """
        if isinstance(other, Item):
            return self.name.lower() == other.name.lower()
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def clear_cached_property(self, property_name):
        """
        Clear the cached value for a cached property
        """
        if property_name in self.__dict__:
            del self.__dict__[property_name]

    @property
    def scope_name(self):
        """
        The name of this item's scope
        """
        return self.name[:self.name.index('#')] or None

    @property
    def local_name(self):
        """
        The item name without the scope
        """
        return self.name[self.name.index('#')+1:]

    @cached_property
    def scope(self):
        """
        :any:`Module` object that is the enclosing scope of this :any:`Item`

        Note that this property is cached, so that updating the name of an associated
        :any:`Module` (eg. via the :any:`DependencyTransformation`) may not
        break the association with this :any:`Item`.

        Returns
        -------
        :any:`Module` or `NoneType`
        """
        name = self.scope_name
        if name is None:
            return None
        return self.source[name]

    @property
    @abstractmethod
    def routine(self):
        """
        Return the :any:`Subroutine` object associated with this :class:`Item`

        Returns
        -------
        :any:`Subroutine` or NoneType
        """

    @property
    @abstractmethod
    def imports(self):
        """
        Return a tuple of all :any:`Import` nodes relevant to this :class:`Item`

        Note that this includes also imports from the parent scope.

        Returns
        -------
        list of :any:`Import`
        """

    @property
    def qualified_imports(self):
        """
        Return the mapping of named imports (i.e. explicitly qualified imports via a use-list
        or rename-list) to their fully qualified name
        """
        return CaseInsensitiveDict(
            (symbol.name, f'{import_.module}#{symbol.type.use_name or symbol.name}')
            for import_ in self.imports
            for symbol in import_.symbols + tuple(s for _, s in as_tuple(import_.rename_list))
        )

    @property
    def unqualified_imports(self):
        """
        Return names of imported modules without explicit ``ONLY`` list
        """
        return tuple(import_.module for import_ in self.imports if not import_.symbols)

    @property
    @abstractmethod
    def members(self):
        """
        The names of member routines contained in this :class:`Item`
        """

    @property
    @abstractmethod
    def calls(self):
        """
        The local name of all routines called from this :class:`Item`
        """

    @property
    @abstractmethod
    def function_interfaces(self):
        """
        All inline functions defined in this :class:`Item` via an explicit interface
        """

    @property
    def path(self):
        """
        The filepath of the associated source file
        """
        return self.source.path

    @property
    def role(self):
        """
        Role in the transformation chain, for example ``'driver'`` or ``'kernel'``
        """
        return self.config.get('role', None)

    @property
    def mode(self):
        """
        Transformation "mode" to pass to the transformation
        """
        return self.config.get('mode', None)

    @property
    def expand(self):
        """
        Flag to trigger expansion of children under this node
        """
        return self.config.get('expand', False)

    @property
    def strict(self):
        """
        Flag controlling whether to strictly fail if source file cannot be parsed
        """
        return self.config.get('strict', True)

    @property
    def replicate(self):
        """
        Flag indicating whether to mark item as "replicated" in call graphs
        """
        return self.config.get('replicate', False)

    @property
    def disable(self):
        """
        List of sources to completely exclude from expansion and the source tree.
        """
        return self.config.get('disable', tuple())

    @property
    def block(self):
        """
        List of sources to block from processing, but add to the
        source tree for visualisation.
        """
        return self.config.get('block', tuple())

    @property
    def ignore(self):
        """
        List of sources to expand but ignore during processing
        """
        return self.config.get('ignore', tuple())

    @property
    def enrich(self):
        """
        List of sources to to use for IPA enrichment
        """
        return self.config.get('enrich', tuple())

    @property
    def enable_imports(self):
        """
        Configurable option to enable the addition of module imports as children.
        """

        return self.config.get('enable_imports', False)

    @property
    def children(self):
        """
        Set of all child routines that this work item has in the call tree

        Note that this is not the set of active children that a traversal
        will apply a transformation over, but rather the set of nodes that
        defines the next level of the internal call tree.

        This returns the local names of children which can be fully qualified
        via :meth:`qualify_names`.

        Returns
        -------
        list of str
        """
        disabled = as_tuple(str(b).lower() for b in self.disable)

        # Base definition of child is a procedure call
        children = self.calls

        if self.enable_imports:
            if isinstance(self, SubroutineItem):
                children += tuple(self.qualified_imports.keys())
            elif isinstance(self, GenericImportItem):
                children += self.procedure_interface_members

        # Filter out local members and disabled sub-branches
        children = [c for c in children if c not in self.members]
        children = [c for c in children if c not in disabled]

        # Remove duplicates
        return as_tuple(dict.fromkeys(children))

    def qualify_names(self, names, available_names=None):
        """
        Fully qualify names with their scope

        This amends every entry in :data:`names` with their scope name in the
        format ``<scope_name>#<local_name>``. Entries that already have a scope
        name are unchanged.

        The scope is derived using qualified imports and takes into account any
        renaming that may happen as part of that.

        For names that cannot be unambiguously attributed to a scope, either because
        they stem from an unqualified import or because they refer to a subroutine
        declared in the global scope, a tuple of fully-qualified candidate names
        is returned. Of these, only one can be a possible match or the symbol would
        be non-uniquely defined. If :data:`available_names` is provided, these
        candidate lists are resolved by picking the correct fully-qualified name out
        of these candidate lists.

        Parameters
        ----------
        names : list of str
            A list of local or fully-qualified names to process
        available_names: list of str, optional
            A list of all available fully-qualified names, to be provided, e.g.,
            by the :any:`Scheduler`

        Returns
        -------
        qualified_names : list of str
            The fully-qualified names in the same order as :data:`names`. For names
            that cannot be resolved unambiguously, a tuple of candidate names is
            returned.
        """
        qualified_names = []

        if all('#' in name for name in as_tuple(names)):
            return as_tuple(names)

        scope = self.scope
        qualified_imports = self.qualified_imports

        for name in as_tuple(names):
            if '#' in name:
                qualified_names += [name]
                continue

            pos = name.find('%')
            if pos != -1:
                # Search for named import of the derived type
                type_name = name[:pos]
                if type_name in qualified_imports:
                    qualified_names += [qualified_imports[type_name] + name[pos:]]
                    continue

                # Search for definition of the type in parent scope
                if scope is not None and type_name in scope.typedefs:
                    qualified_names += [f'{self.scope_name}#{name}']
                    continue
            else:
                # Search for named import of the subroutine
                if name in qualified_imports:
                    qualified_names += [qualified_imports[name]]
                    continue

                # Search for subroutine in parent scope
                scope = self.scope
                if scope is not None and name in scope:
                    qualified_names += [f'{self.scope_name}#{name}']
                    continue

            # This name has to come from an unqualified import or exists in the global
            # scope (i.e. defined directly in a source file). We add all
            # possibilities, of which only one should match (otherwise we would
            # have duplicate symbols in this compilation unit)
            candidates = [f'#{name}'.lower()]
            candidates += [f'{import_}#{name}'.lower() for import_ in self.unqualified_imports]
            qualified_names += [as_tuple(candidates)]

        if available_names is None:
            return as_tuple(qualified_names)

        available_names = set(available_names)
        def map_to_available_name(candidates):
            # Resolve a tuple of candidate names by picking the matching name
            # from available_names
            pos = candidates[0].find('%')
            if pos != -1:
                candidate_types = {c[:c.index('%')] for c in candidates}
                member_name = candidates[0][pos:]
                matched_names = candidate_types & available_names
            else:
                member_name = ''
                matched_names = set(candidates) & available_names

            if not matched_names:
                warning(f'{candidates} not found in available_names')
                return candidates
            if len(matched_names) != 1:
                name = matched_names[0]
                warning(f'Duplicate symbol: {name[name.find("#")+1:]}, can be one of {matched_names}')
                return tuple(name + member_name for name in matched_names)
            return matched_names.pop() + member_name

        return tuple(
            name if isinstance(name, str) else map_to_available_name(name)
            for name in qualified_names
        )

    @property
    def targets(self):
        """
        Set of "active" child routines that are part of the transformation
        traversal.

        This defines all child routines of an item that will be
        traversed when applying a :any:`Transformation` as well, after
        tree pruning rules are applied.

        This returns the local names of children which can be fully qualified
        via :meth:`qualify_names`.

        Returns
        -------
        list of str
        """
        disabled = as_tuple(str(b).lower() for b in self.disable)
        blocked = as_tuple(str(b).lower() for b in self.block)
        ignored = as_tuple(str(b).lower() for b in self.ignore)

        # Base definition of child is a procedure call
        targets = self.calls

        if self.enable_imports:
            if isinstance(self, SubroutineItem):
                targets += tuple(self.qualified_imports.keys())
            elif isinstance(self, GenericImportItem):
                targets += self.procedure_interface_members

        # Filter out blocked and ignored children
        targets = [c for c in targets if c not in disabled]
        targets = [t for t in targets if t not in blocked]
        targets = [t for t in targets if t not in ignored]
        return as_tuple(targets)


class SubroutineItem(Item):
    """
    Implementation of :class:`Item` to represent a Fortran subroutine work item
    """

    def __init__(self, name, source, config=None):
        assert '%' not in name
        super().__init__(name, source, config)

    @cached_property
    def routine(self):
        """
        :any:`Subroutine` object that this :any:`Item` encapsulates for processing

        Note that this property is cached, so that updating the name of an associated
        :any:`Subroutine` with (eg. via the :any:`DependencyTransformation`) may not
        break the association with this :any:`Item`.

        Returns
        -------
        :any:`Subroutine`
        """
        return self.source[self.local_name]

    @cached_property
    def members(self):
        """
        Names of member routines contained in the subroutine corresponding to this item

        Returns
        -------
        tuple of str
        """
        return tuple(member.name.lower() for member in self.routine.members)

    @cached_property
    def imports(self):
        """
        Return a tuple of all :any:`Import` nodes relevant to this :class:`Item`

        This includes imports in the corresponding :any:`Subroutine` as well as the
        enclosing :any:`Module` scope, if applicable.

        Returns
        -------
        list of :any:`Import`
        """
        scope = self.routine
        imports = []
        while scope is not None:
            imports += scope.imports
            scope = scope.parent
        return imports

    @property
    def calls(self):
        """
        The local name of all routines called from this :class:`Item`

        These are identified via :any:`CallStatement` nodes within the associated routine's IR.
        """
        return tuple(
            self._variable_to_type_name(call.name).lower()
            for call in FindNodes(CallStatement).visit(self.routine.ir)
        ) + self.function_interfaces

    def _variable_to_type_name(self, var):
        """
        For type bound procedure calls, map the variable symbol to the type name, otherwise
        return the name unchanged
        """
        pos = var.name.find('%')
        if pos == -1:
            return var.name
        # Find the type name for the outermost derived type parent
        var_name = var.name[:pos]
        type_name = self.routine.symbol_attrs[var_name].dtype.name
        return type_name + var.name[pos:]

    @property
    def function_interfaces(self):
        """
        Inline functions declared in the corresponding :any:`Subroutine`,
        or its parent :any:`Module`, via an explicit interface.
        """

        names = ()
        interfaces = self.routine.interfaces

        # Named interfaces defined in the parent module should not be included to remove
        # the risk of adding a procedure interface defined in the current scope
        if (scope := self.scope) is not None:
            interfaces += as_tuple(i for i in scope.interfaces if not i.spec)

        names = tuple(
            s.name.lower() for intf in interfaces for s in intf.symbols
            if s.type.dtype.is_function
        )

        return names


class ProcedureBindingItem(Item):
    """
    Implementation of :class:`Item` to represent a Fortran procedure binding

    This does not constitute a work item when applying transformations across the
    call tree in the :any:`Scheduler` and is skipped during the processing phase.
    However, it is necessary to provide the dependency link from calls to type bound
    procedures to their implementation in a Fortran routine.
    """

    def __init__(self, name, source, config=None):
        assert '%' in name
        super().__init__(name, source, config)

    @property
    def routine(self):
        """
        Always returns `None` as this is not associated with a :any:`Subroutine`
        """
        return None

    @property
    def members(self):
        """
        Empty tuple as procedure bindings have no member routines

        Returns
        -------
        tuple
        """
        return ()

    @cached_property
    def imports(self):
        """
        Return modules imported in the parent scope
        """
        return self.scope.imports

    @property
    def function_interfaces(self):
        """
        Empty tuple as procedure bindings cannot include interface blocks

        Returns
        -------
        tuple
        """
        return ()

    @property
    def calls(self):
        """
        The local names of the routines that are bound to the derived type under
        the name of the current :class:`Item`

        For procedure bindings that returns the local name of a subroutine.
        For an item representing a call to a type bound procedure in a derived type
        member, this returns the local name of the type bound procedure.
        For a generic binding, this returns all local names of type bound procedures
        that are combined under a generic binding.
        """
        if '%' not in self.name:
            return ()
        module = self.source[self.scope_name]
        name_parts = self.local_name.split('%')
        typedef = module[name_parts[0]]
        symbol = typedef.variable_map[name_parts[1]]
        type_ = symbol.type
        if len(name_parts) > 2:
            # This is a type-bound procedure in a derived type member of a derived type
            return ('%'.join([type_.dtype.name, *name_parts[2:]]),)
        if type_.dtype.is_generic:
            # This is a generic binding, so we need to refer to other type-bound procedures
            # in this type
            return tuple(f'{name_parts[0]}%{name!s}' for name in type_.bind_names)
        if type_.initial is not None:
            # This has a bind name explicitly specified:
            return (type_.initial.name.lower(), )
        if type_.bind_names is not None and len(type_.bind_names) == 1:
            return (type_.bind_names[0].name.lower(),)
        # The name of the procedure is identical to the name of the binding
        return (name_parts[1],)


class GenericImportItem(Item):
    """
    Implementation of :class:`Item` to represent a generic Fortran module import.

    This does not constitute a work item when applying transformations across the
    call tree in the :any:`Scheduler` and is skipped during the processing phase.
    However, it is needed to determine whether the import corresponds to a type
    declaration or a procedure interface.
    """

    def __init__(self, name, source, config=None):
        name_parts = name.split('#')
        assert len(name_parts) > 1 and all(name_parts) #only accept fully-qualified module imports
        super().__init__(name, source, config)
        assert self.scope

    @property
    def routine(self):
        """
        Always returns `None` as this is not associated with a :any:`Subroutine`
        """
        return None

    @property
    def members(self):
        """
        Empty tuple as generic imports have no member routines

        Returns
        -------
        tuple
        """
        return ()

    @property
    def function_interfaces(self):
        """
        Empty tuple as generic import items cannot include interface blocks

        Returns
        -------
        tuple
        """
        return ()

    @cached_property
    def imports(self):
        """
        Return modules imported in the parent scope
        """
        return self.scope.imports

    @property
    def calls(self):
        return ()

    @property
    def procedure_interface_members(self):
        """
        The set of children unique to items of type :class:`GenericImportItem`.
        Comprises exclusively of function calls bound to a procedure interface.
        """

        _children = ()
        intfs = self.scope.interfaces
        if isinstance(self.source[self.local_name], TypeDef):
            pass
        elif self.local_name in [i.spec.name for i in intfs]:
            for i in intfs:
                if i.spec.name == self.local_name:
                    for b in i.body:
                        if isinstance(b, ProcedureDeclaration):
                            for s in b.symbols:
                                _children += as_tuple(s.name)
                        elif isinstance(b, Subroutine):
                            if b.is_function:
                                _children += as_tuple(b.name)

        return _children


class GlobalVarImportItem(Item):
    """
    Implementation of :class:`Item` to represent a global variable import.
    These only appear at the end of dependency trees and do not have children.
    """

    def __init__(self, name, source, config=None):
        name_parts = name.split('#')
        assert len(name_parts) > 1 and all(name_parts) #only accept fully-qualified module imports
        super().__init__(name, source, config)
        assert self.scope

    @property
    def routine(self):
        """
        Always returns `None` as this is not associated with a :any:`Subroutine`
        """
        return None

    @property
    def members(self):
        """
        Empty tuple as variable imports have no member routines

        Returns
        -------
        tuple
        """
        return ()

    @property
    def function_interfaces(self):
        """
        Empty tuple as global variable imports cannot include interface blocks

        Returns
        -------
        tuple
        """
        return ()

    @cached_property
    def imports(self):
        """
        Return modules imported in the parent scope
        """

        return self.scope.imports

    @property
    def calls(self):
        """
        Empty tuple as items of type :class:`GlobalVarImportItem` cannot have any children.
        """
        return ()
