try:
    from functools import cached_property
except ImportError:
    try:
        from cached_property import cached_property
    except ImportError:
        def cached_property(func):
            return func

from loki.tools import as_tuple
from loki.tools.util import CaseInsensitiveDict
from loki.visitors import FindNodes
from loki.ir import CallStatement


__all__ = ['Item']


class Item:
    """
    A work item that represents a single source routine to be
    processed. Each :any:`Item` spawns new work items according to its
    own subroutine calls and can be configured to ignore individual
    sub-tree.

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
    * ``disable``: List of subroutines that are completely ignored and are not reported as
      ``children``. Useful to exclude entire call trees or utility routines.
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
        self.name = name
        self.source = source
        self.config = config or {}

    def __repr__(self):
        return f'loki.bulk.Item<{self.name}>'

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.name.lower() == other.name.lower()
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name)

    @staticmethod
    def name_is_in(name, list_of_names):
        """
        Check if any of the names in :data:`list_of_names` matches :data:`name`

        It is considered a match if either :data:`name` is in :data:`list_of_names`
        or, if :data:`name` contains a ``#`` and the local name (i.e. everything after
        the ``#``) is in :data:`list_of_names`. The comparison is not case sensitive.
        This allows to check if :data:`name` is in a list of (possibly) not fully-qualified name.
        """
        lower_name = name.lower()
        list_of_names = [name.lower() for name in list_of_names]
        if '#' in lower_name:
            local_name = lower_name.split('#')[-1]
            return local_name in list_of_names or lower_name in list_of_names
        return lower_name in list_of_names

    def is_in(self, list_of_names):
        """
        Check if this item matches any of the names in :data:`list_of_names`

        See :meth:`name_is_in` for the methodology.
        """
        return self.name_is_in(self.name, list_of_names)

    @cached_property
    def scope_name(self):
        """
        The name of this item's scope
        """
        return self.name[:self.name.index('#')] or None

    @cached_property
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
        :any:`Module` with (eg. via the :any:`DependencyTransformation`) may not
        break the association with this :any:`Item`.
        """
        name = self.scope_name
        if name is None:
            return None
        return self.source[name]

    @cached_property
    def routine(self):
        """
        :any:`Subroutine` object that this :any:`Item` encapsulates for processing.

        Note that this property is cached, so that updating the name of an associated
        :any:`Subroutine` with (eg. via the :any:`DependencyTransformation`) may not
        break the association with this :any:`Item`.
        """
        if '%' in self.name:
            return None
        return self.source[self.local_name]

    @cached_property
    def imports(self):
        """
        Return modules imported in the current item and parent scopes
        """
        scope = self.routine
        imports = []
        while scope is not None:
            imports += [import_.module for import_ in scope.imports]
            scope = scope.parent
        return as_tuple(imports)

    @cached_property
    def named_imports(self):
        """
        Return the mapping of named imports (i.e. explicitly qualified imports via a use-list
        or rename-list) to their canonical name
        """
        scope = self.routine
        import_map = CaseInsensitiveDict()
        while scope is not None:
            import_map.update(
                (symbol.name, f'{import_.module}#{symbol.type.use_name or symbol.name}')
                for import_ in scope.imports for symbol in import_.symbols
            )
            scope = scope.parent
        return import_map

    @cached_property
    def unqualified_imports(self):
        """
        Return modules imported without explicit ``ONLY`` list
        """
        scope = self.routine
        imports = []
        while scope is not None:
            imports += [import_.module for import_ in scope.imports if not import_.symbols]
            scope = scope.parent
        return as_tuple(imports)

    @cached_property
    def members(self):
        """
        Names of member routines contained in the subroutine corresponding to this item
        """
        if self.routine is None:
            return ()
        return tuple(member.name.lower() for member in self.routine.members)

    @property
    def bind_names(self):
        """
        For items representing type-bound procedures, this returns the names of the
        routines it binds to
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
            typename = type_.dtype.name
            return as_tuple('%'.join([typename] + name_parts[2:]))
        if type_.dtype.is_generic:
            # This is a generic binding, so we need to refer to other type-bound procedures
            # in this type
            return tuple(name_parts[0] + '%' + name for name in type_.bind_names)
        if type_.initial is not None:
            # This has a bind name explicitly specified:
            return (type_.initial.name.lower(), )
        # The name of the procedure is identical to the name of the binding
        return (name_parts[1],)

    @cached_property
    def calls(self):
        if '%' in self.name:
            # This is a type-bound procedure item: we are mapping to the names it binds to
            return self.bind_names

        def _canonical_name(var):
            # For type bound procedure calls, map the variable name to the type name
            pos = var.name.find('%')
            if pos != -1:
                var_name = var.name[:pos]
                type_name = self.routine.variable_map[var_name].type.dtype.name
                type_name = self.named_imports.get(type_name, type_name)
                return type_name + var.name[pos:]
            return self.named_imports.get(var.name, var.name)

        return tuple(
            _canonical_name(call.name).lower()
            for call in FindNodes(CallStatement).visit(self.routine.ir)
        )

    @property
    def path(self):
        """
        The filepath of the associated source file
        """
        return self.source.path

    @property
    def role(self):
        """
        Role in the transformation chain, for example 'driver' or 'kernel'
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
    def children(self):
        """
        Set of all child routines that this work item has in the call tree

        Note that this is not the set of active children that a traversal
        will apply a transformation over, but rather the set of nodes that
        defines the next level of the internal call tree.

        This includes potentially non-existent routines where
        """
        disabled = as_tuple(str(b).lower() for b in self.disable)

        # Base definition of child is a procedure call (for now)
        children = self.calls

        # Filter out local members and disabled sub-branches
        children = [c for c in children if c not in self.members]
        children = [c for c in children if not any(c.endswith(d) for d in disabled)]

        def _canonical_names(name):
            # Inject module names for unqualified child names
            if '#' in name:
                return as_tuple(name)

            pos = name.find('%')
            if pos != -1:
                # Search for named import of the derived type
                type_name = name[:pos]
                if type_name in self.named_imports:
                    return as_tuple(self.named_imports[type_name])

                # Search for definition of the type in parent scope
                scope = self.scope
                if scope is not None and type_name in scope.typedefs:
                    return as_tuple(f'{self.scope_name}#{name}')
            else:
                # Search for named import of the subroutine
                if name in self.named_imports:
                    return as_tuple(self.named_imports[name])

                # Search for subroutine in parent scope
                scope = self.scope
                if scope is not None and name in scope:
                    return as_tuple(f'{self.scope_name}#{name}')

            # This child has to come from an unqualified import or exists in the global
            # scope (i.e. defined directly in a source file). We return all
            # possibilities, of which only one should match (otherwise we would
            # have duplicate symbols in this compilation unit)
            return tuple(f'{import_}#{name}' for import_ in ('',) + self.unqualified_imports)

        children = [_canonical_names(c) for c in children]
        return as_tuple(children)

    @property
    def targets(self):
        """
        Set of "active" child routines that are part of the transformation
        traversal.

        This defines all child routines of an item that will be
        traversed when applying a :any:`Transformation` as well, after
        tree pruning rules are applied.
        """
        disabled = as_tuple(str(b).lower() for b in self.disable)
        blocked = as_tuple(str(b).lower() for b in self.block)
        ignored = as_tuple(str(b).lower() for b in self.ignore)

        # Base definition of child is a procedure call
        targets = as_tuple(str(call.name).lower() for call in FindNodes(CallStatement).visit(self.routine.ir))

        # Filter out blocked and ignored children
        targets = [c for c in targets if c not in disabled]
        targets = [t for t in targets if t not in blocked]
        targets = [t for t in targets if t not in ignored]
        return as_tuple(targets)
