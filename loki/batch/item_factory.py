# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch.configure import SchedulerConfig
from loki.batch.item import (
    get_all_import_map, ExternalItem, FileItem, InterfaceItem, ModuleItem,
    ProcedureBindingItem, ProcedureItem, TypeDefItem
)
from loki.expression import ProcedureSymbol
from loki.ir import nodes as ir
from loki.logging import warning
from loki.module import Module
from loki.subroutine import Subroutine
from loki.sourcefile import Sourcefile
from loki.tools import CaseInsensitiveDict, as_tuple


__all__ = ['ItemFactory']


class ItemFactory:
    """
    Utility class to instantiate instances of :any:`Item`

    It maintains a :attr:`item_cache` for all created items. Most
    important factory method is :meth:`create_from_ir` to create (or
    return from the cache) a :any:`Item` object corresponding to an
    IR node. Other factory methods exist for more bespoke use cases.

    Attributes
    ----------
    item_cache : :any:`CaseInsensitiveDict`
        This maps item names to corresponding :any:`Item` objects
    """

    def __init__(self):
        self.item_cache = CaseInsensitiveDict()

    def __contains__(self, key):
        """
        Check if an item under the given name exists in the :attr:`item_cache`
        """
        return key in self.item_cache

    def create_from_ir(self, node, scope_ir, config=None, ignore=None):
        """
        Helper method to create items for definitions or dependency

        This is a helper method to determine the fully-qualified item names
        and item type for a given IR :any:`Node`, e.g., when creating the items
        for definitions (see :any:`Item.create_definition_items`) or dependencies
        (see :any:`Item.create_dependency_items`).

        This routine's responsibility is to determine the item name, and then call
        :meth:`get_or_create_item` to look-up an existing item or create it.

        Parameters
        ----------
        node : :any:`Node` or :any:`pymbolic.primitives.Expression`
            The Loki IR node for which to create a corresponding :any:`Item`
        scope_ir : :any:`Scope`
            The scope node in which the IR node is declared or used. Note that this
            is not necessarily the same as the scope of the created :any:`Item` but
            serves as the entry point for the lookup mechanism that underpins the
            creation procedure.
        config : any:`SchedulerConfiguration`, optional
            The config object from which a bespoke item configuration will be derived.
        ignore : list of str, optional
            A list of item names that should be ignored, i.e., not be created as an item.
        """
        if isinstance(node, Module):
            item_name = node.name.lower()
            if self._is_ignored(item_name, config, ignore):
                return None
            return as_tuple(self.get_or_create_item(ModuleItem, item_name, item_name, config))

        if isinstance(node, Subroutine):
            scope_name = getattr(node.parent, 'name', '').lower()
            item_name = f'{scope_name}#{node.name}'.lower()
            if self._is_ignored(item_name, config, ignore):
                return None
            return as_tuple(
                self.get_or_create_item(ProcedureItem, item_name, scope_name, config)
            )

        if isinstance(node, ir.TypeDef):
            # A typedef always lives in a Module
            scope_name = node.parent.name.lower()
            item_name = f'{scope_name}#{node.name}'.lower()
            if self._is_ignored(item_name, config, ignore):
                return None
            return as_tuple(self.get_or_create_item(TypeDefItem, item_name, scope_name, config))

        if isinstance(node, ir.Import):
            # Skip intrinsic modules
            if node.nature == 'intrinsic':
                return None

            # Skip CPP includes
            if node.c_import:
                return None

            # If we have a fully-qualified import (which we hopefully have),
            # we create a dependency for every imported symbol, otherwise we
            # depend only on the imported module
            scope_name = node.module.lower()
            if self._is_ignored(scope_name, config, ignore):
                return None
            if scope_name not in self.item_cache:
                # This will instantiate an ExternalItem
                return as_tuple(self.get_or_create_item(ModuleItem, scope_name, scope_name, config))

            scope_item = self.item_cache[scope_name]

            if node.symbols:
                scope_definitions = {
                    item.local_name: item
                    for item in scope_item.create_definition_items(item_factory=self, config=config)
                }
                symbol_names = tuple(str(smbl.type.use_name or smbl).lower() for smbl in node.symbols)
                non_ignored_symbol_names = tuple(
                    smbl for smbl in symbol_names
                    if not self._is_ignored(f'{scope_name}#{smbl}', config, ignore)
                )
                imported_items = tuple(
                    it for smbl in non_ignored_symbol_names
                    if (it := scope_definitions.get(smbl)) is not None
                )

                # Global variable imports are filtered out in the previous statement because they
                # are not represented by an Item. For these, we introduce a dependency on the
                # module instead
                has_globalvar_import = len(imported_items) != len(non_ignored_symbol_names)

                # Filter out ProcedureItems corresponding to a subroutine:
                # dependencies on subroutines are introduced via the call statements, as this avoids
                # depending on imported but not called subroutines
                imported_items = tuple(
                    it for it in imported_items
                    if not isinstance(it, ProcedureItem) or it.ir.is_function
                )

                if has_globalvar_import:
                    return (scope_item,) + imported_items
                if not imported_items:
                    return None
                return imported_items

            return (scope_item,)

        if isinstance(node, ir.CallStatement):
            procedure_symbols = as_tuple(node.name)
        elif isinstance(node, ProcedureSymbol):
            procedure_symbols = as_tuple(node)
        elif isinstance(node, (ir.ProcedureDeclaration, ir.Interface)):
            procedure_symbols = as_tuple(node.symbols)
        else:
            raise ValueError(f'{node} has an unsupported node type {type(node)}')

        return tuple(
            self._get_procedure_binding_item(symbol, scope_ir, config, ignore=ignore) if '%' in symbol.name
            else self._get_procedure_item(symbol, scope_ir, config, ignore=ignore)
            for symbol in procedure_symbols
        )

    def get_or_create_item(self, item_cls, item_name, scope_name, config=None):
        """
        Helper method to instantiate an :any:`Item` of class :data:`item_cls`
        with name :data:`item_name`.

        This helper method checks for the presence of :data:`item_name` in the
        :attr:`item_cache` and returns that instance. If none is found, an instance
        of :data:`item_cls` is created and stored in the item cache.

        The :data:`scope_name` denotes the name of the parent scope, under which a
        parent :any:`Item` has to exist in :data:`self.item_cache` to find the source
        object to use.

        Item names matching one of the entries in the :data:`config` disable list
        are skipped. If `strict` mode is enabled, this raises a :any:`RuntimeError`
        if no matching parent item can be found in the item cache.

        Parameters
        ----------
        item_cls : subclass of :any:`Item`
            The class of the item to create
        item_name : str
            The name of the item to create
        scope_name : str
            The name under which a parent item can be found in the :attr:`item_cache`
            to find the corresponding source
        config : :any:`SchedulerConfig`, optional
            The config object to use to determine disabled items, and to use when
            instantiating the new item

        Returns
        -------
        :any:`Item` or None
            The item object or `None` if disabled or impossible to create
        """
        if item_name in self.item_cache:
            return self.item_cache[item_name]

        item_conf = config.create_item_config(item_name) if config else None
        scope_item = self.item_cache.get(scope_name)
        if scope_item is None or isinstance(scope_item, ExternalItem):
            warning(f'Module {scope_name} not found in self.item_cache. Marking {item_name} as an external dependency')
            item = ExternalItem(item_name, source=None, config=item_conf, origin_cls=item_cls)
        else:
            source = scope_item.source
            item = item_cls(item_name, source=source, config=item_conf)
        self.item_cache[item_name] = item
        return item

    def get_or_create_item_from_item(self, name, item, config=None):
        """
        Helper method to instantiate an :any:`Item` as a clone of a given :data:`item`
        with the given new :data:`name`.

        This helper method checks for the presence of :data:`name` in the
        :attr:`item_cache` and returns that instance. If none is in the cache, it tries
        a lookup via the scope, if applicable. Otherwise, a new item is created as
        a duplicate.

        This duplication is performed by replicating the corresponding :any:`FileItem`
        and any enclosing scope items, applying name changes for scopes as implied by
        :data:`name`.

        Parameters
        ----------
        name : str
            The name of the item to create
        item : :any:`Item`
            The item to duplicate to create the new item
        config : :any:`SchedulerConfig`, optional
            The config object to use when instantiating the new item

        Returns
        -------
        :any:`Item`
            The new item object
        """
        # Sanity checks and early return if an item by that name exists
        if name in self.item_cache:
            return self.item_cache[name]

        if not isinstance(item, ProcedureItem):
            raise NotImplementedError(f'Cloning of Items is not supported for {type(item)}')

        # Derive name components for the new item
        pos = name.find('#')
        local_name = name[pos+1:]
        if pos == -1:
            scope_name = None
            if local_name == item.local_name:
                raise RuntimeError(f'Cloning item {item.name} with the same name in global scope')
            if item.scope_name:
                raise RuntimeError(f'Cloning item {item.name} from local scope to global scope is not supported')
        else:
            scope_name = name[:pos]
            if scope_name and scope_name == item.scope_name:
                raise RuntimeError(f'Cloning item {item.name} as {name} creates name conflict for scope {scope_name}')
            if scope_name and not item.scope_name:
                raise RuntimeError(f'Cloning item {item.name} from global scope to local scope is not supported')

        # We may need to create a new item as a clone of the given item
        # For this, we start with replicating the source and updating the
        if not scope_name or scope_name not in self.item_cache:
            # Clone the source and update names
            new_source = item.source.clone()
            if scope_name:
                scope = new_source[item.scope_name]
                scope.name = scope_name
                ir = scope[item.local_name]
            else:
                ir = new_source[item.local_name]
            ir.name = local_name

            # Create a new FileItem for the new source
            new_source.path = item.path.with_name(f'{scope_name or local_name}{item.path.suffix}')
            file_item = self.get_or_create_file_item_from_source(new_source, config=config)

            # Get the definition items for the FileItem and return the new item
            definition_items = {
                it.name: it for it in file_item.create_definition_items(item_factory=self, config=config)
            }
            self.item_cache.update(definition_items)

            if name in definition_items:
                return definition_items[name]

        # Check for existing scope item
        if scope_name and scope_name in self.item_cache:
            scope = self.item_cache[scope_name].ir
            if local_name not in scope:
                raise RuntimeError(f'Cloning item {item.name} as {name} failed, {local_name} not found in existing scope {scope_name}')
            return self.create_from_ir(scope[local_name], scope, config=config)

        raise RuntimeError(f'Failed to clone item {item.name} as {name}')

    def get_or_create_file_item_from_path(self, path, config, frontend_args=None):
        """
        Utility method to create a :any:`FileItem` for a given path

        This is used to instantiate items for the first time during the scheduler's
        discovery phase. It will use a cached item if it exists, or parse the source
        file using the given :data:`frontend_args`.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path of the source file
        config : :any:`SchedulerConfig`
            The config object from which the item configuration will be derived
        frontend_args : dict, optional
            Frontend arguments that are given to :any:`Sourcefile.from_file` when
            parsing the file
        """
        item_name = str(path).lower()
        if file_item := self.item_cache.get(item_name):
            return file_item

        if not frontend_args:
            frontend_args = {}
        if config:
            frontend_args = config.create_frontend_args(path, frontend_args)

        source = Sourcefile.from_file(path, **frontend_args)
        item_conf = config.create_item_config(item_name) if config else None
        file_item = FileItem(item_name, source=source, config=item_conf)
        self.item_cache[item_name] = file_item
        return file_item

    def get_or_create_file_item_from_source(self, source, config):
        """
        Utility method to create a :any:`FileItem` corresponding to a given source object

        This can be used to create a :any:`FileItem` for an already parsed :any:`Sourcefile`,
        or when looking up the file item corresponding to a :any:`Item` by providing the
        item's ``source`` object.

        Lookup is not performed via the ``path`` property in :data:`source` but by
        searching for an existing :any:`FileItem` in the cache that has the same source
        object. This allows creating clones of source files during transformations without
        having to ensure their path property is always updated. Only if no item is found
        in the cache, a new one is created.

        Parameters
        ----------
        source : :any:`Sourcefile`
            The existing sourcefile object for which to create the file item
        config : :any:`SchedulerConfig`
            The config object from which the item configuration will be derived
        """
        # Check for file item with the same source object
        for item in self.item_cache.values():
            if isinstance(item, FileItem) and item.source is source:
                return item

        if not source.path:
            raise RuntimeError('Cannot create FileItem from source: Sourcefile has no path')

        # Create a new file item
        item_name = str(source.path).lower()
        item_conf = config.create_item_config(item_name) if config else None
        file_item = FileItem(item_name, source=source, config=item_conf)
        self.item_cache[item_name] = file_item
        return file_item

    def _get_procedure_binding_item(self, proc_symbol, scope_ir, config, ignore=None):
        """
        Utility method to create a :any:`ProcedureBindingItem` for a given
        :any:`ProcedureSymbol`

        Parameters
        ----------
        proc_symbol : :any:`ProcedureSymbol`
            The procedure symbol of the type binding
        scope_ir : :any:`Scope`
            The scope node in which the procedure binding is declared or used. Note that this
            is not necessarily the same as the scope of the created :any:`Item` but
            serves as the entry point for the lookup mechanism that underpins the
            creation procedure.
        config : :any:`SchedulerConfig`
            The config object from which the item configuration will be derived
        ignore : list of str, optional
            A list of item names that should be ignored, i.e., not be created as an item.
        """
        is_strict = not config or config.default.get('strict', True)

        # This is a typebound procedure call: we are only resolving
        # to the type member by mapping the local name to the type name,
        # and creating a ProcedureBindingItem. For that we need to find out
        # the type of the derived type symbol.
        # NB: For nested derived types, we create multiple such ProcedureBindingItems,
        #     resolving one type at a time, e.g.
        #     my_var%member%procedure -> my_type%member%procedure -> member_type%procedure -> procedure
        type_name = proc_symbol.parents[0].type.dtype.name
        scope_name = None

        # Imported in current or parent scopes?
        if imprt := get_all_import_map(scope_ir).get(type_name):
            scope_name = imprt.module
            type_name = self._get_imported_symbol_name(imprt, type_name)

        # Otherwise: must be declared in parent module scope
        if not scope_name:
            scope = scope_ir
            while scope:
                if isinstance(scope, Module):
                    if type_name in scope.typedef_map:
                        scope_name = scope.name
                    break
                scope = scope.parent

        # Unknown: Likely imported via `USE` without `ONLY` list
        if not scope_name:
            # We create definition items for TypeDefs in all modules for which
            # we have unqualified imports, to find the type definition that
            # may have been imported via one of the unqualified imports
            unqualified_import_modules = [
                imprt.module for imprt in scope_ir.all_imports if not imprt.symbols
            ]
            candidates = self.get_or_create_module_definitions_from_candidates(
                type_name, config, module_names=unqualified_import_modules, only=TypeDefItem
            )
            if not candidates:
                msg = f'Unable to find the module declaring {type_name}.'
                if is_strict:
                    raise RuntimeError(msg)
                warning(msg)
                return None
            if len(candidates) > 1:
                msg = f'Multiple definitions for {type_name}: '
                msg += ','.join(item.name for item in candidates)
                if is_strict:
                    raise RuntimeError(msg)
                warning(msg)
            scope_name = candidates[0].scope_name

        item_name = f'{scope_name}#{type_name}%{"%".join(proc_symbol.name_parts[1:])}'.lower()
        if self._is_ignored(item_name, config, ignore):
            return None
        return self.get_or_create_item(ProcedureBindingItem, item_name, scope_name, config)

    def _get_procedure_item(self, proc_symbol, scope_ir, config, ignore=None):
        """
        Utility method to create a :any:`ProcedureItem`, :any:`ProcedureBindingItem`,
        or :any:`InterfaceItem` for a given :any:`ProcedureSymbol`

        Parameters
        ----------
        proc_symbol : :any:`ProcedureSymbol`
            The procedure symbol for which the corresponding item is created
        scope_ir : :any:`Scope`
            The scope node in which the procedure symbol is declared or used. Note that this
            is not necessarily the same as the scope of the created :any:`Item` but
            serves as the entry point for the lookup mechanism that underpins the
            creation procedure.
        config : :any:`SchedulerConfig`
            The config object from which the item configuration will be derived
        ignore : list of str, optional
            A list of item names that should be ignored, i.e., not be created as an item.
        """
        proc_name = proc_symbol.name

        if proc_name in scope_ir:
            if isinstance(scope_ir, ir.TypeDef):
                # This is a procedure binding item
                scope_name = scope_ir.parent.name.lower()
                item_name = f'{scope_name}#{scope_ir.name}%{proc_name}'.lower()
                if self._is_ignored(item_name, config, ignore):
                    return None
                return self.get_or_create_item(ProcedureBindingItem, item_name, scope_name, config)

            if (
                isinstance(scope_ir, Subroutine) and
                any(r.name.lower() == proc_name for r in scope_ir.subroutines)
            ):
                # This is a call to an internal member procedure
                # TODO: Make it configurable whether to include these in the callgraph
                return None

        # Recursively search for the enclosing module
        current_module = None
        scope = scope_ir
        while scope:
            if isinstance(scope, Module):
                current_module = scope
                break
            scope = scope.parent

        if current_module and any(proc_name.lower() == r.name.lower() for r in current_module.subroutines):
            # This is a call to a procedure in the same module
            scope_name = current_module.name
            item_name = f'{scope_name}#{proc_name}'.lower()
            if self._is_ignored(item_name, config, ignore):
                return None
            return self.get_or_create_item(ProcedureItem, item_name, scope_name, config)

        if current_module and proc_name in current_module.interface_symbols:
            # This procedure is declared in an interface in the current module
            scope_name = scope_ir.name
            item_name = f'{scope_name}#{proc_name}'.lower()
            if self._is_ignored(item_name, config, ignore):
                return None
            return self.get_or_create_item(InterfaceItem, item_name, scope_name, config)

        if imprt := get_all_import_map(scope_ir).get(proc_name):
            # This is a call to a module procedure which has been imported via
            # a fully qualified import in the current or parent scope
            scope_name = imprt.module
            proc_name = self._get_imported_symbol_name(imprt, proc_name)
            item_name = f'{scope_name}#{proc_name}'.lower()
            if self._is_ignored(item_name, config, ignore):
                return None
            return self.get_or_create_item(ProcedureItem, item_name, scope_name, config)

        # This may come from an unqualified import
        unqualified_imports = [imprt for imprt in scope_ir.all_imports if not imprt.symbols]
        if unqualified_imports:
            # We try to find the ProcedureItem in the unqualified module imports
            module_names = [imprt.module for imprt in unqualified_imports]
            candidates = self.get_or_create_module_definitions_from_candidates(
                proc_name, config, module_names=module_names, only=ProcedureItem
            )
            if candidates:
                if len(candidates) > 1:
                    candidate_modules = [it.scope_name for it in candidates]
                    raise RuntimeError(
                        f'Procedure {item_name} defined in multiple imported modules: {", ".join(candidate_modules)}'
                    )
                return candidates[0]

        # This is a call to a subroutine declared via header-included interface
        item_name = f'#{proc_name}'.lower()
        if self._is_ignored(item_name, config, ignore):
            return None
        if config and config.is_disabled(item_name):
            return None
        if item_name not in self.item_cache:
            if not config or config.default.get('strict', True):
                raise RuntimeError(f'Procedure {item_name} not found in self.item_cache.')
            warning(f'Procedure {item_name} not found in self.item_cache.')
            return None
        return self.item_cache[item_name]

    def get_or_create_module_definitions_from_candidates(self, name, config, module_names=None, only=None):
        """
        Utility routine to get definition items matching :data:`name`
        from a given list of module candidates

        This can be used to find a dependency that has been introduced via an unqualified
        import statement, where the local name of the dependency is known and a set of
        candidate modules thanks to the unqualified imports on the use side.

        Parameters
        ----------
        name : str
            Local name of the item(s) in the candidate modules
        config : :any:`SchedulerConfig`
            The config object from which the item configuration will be derived
        module_names : list of str, optional
            List of module candidates in which to create the definition items. If not provided,
            all :any:`ModuleItem` in the cache will be considered.
        only : list of :any:`Item` classes, optional
            Filter the generated items to include only those of the type provided in the list

        Returns
        -------
        tuple of :any:`Item`
            The items matching :data:`name` in the modules given in :any:`module_names`.
            Ideally, only a single item will be found (or there would be a name conflict).
        """
        if not module_names:
            module_names = [item.name for item in self.item_cache.values() if isinstance(item, ModuleItem)]
        items = []
        for module_name in module_names:
            module_item = self.item_cache.get(module_name)
            if module_item:
                definition_items = module_item.create_definition_items(
                    item_factory=self, config=config, only=only
                )
                items += [_it for _it in definition_items if _it.name[_it.name.index('#')+1:] == name.lower()]
        return tuple(items)

    @staticmethod
    def _get_imported_symbol_name(imprt, symbol_name):
        """
        For a :data:`symbol_name` and its corresponding :any:`Import` node :data:`imprt`,
        determine the symbol in the defining module.

        This resolves renaming upon import but, in most cases, will simply return the
        original :data:`symbol_name`.

        Returns
        -------
        :any:`MetaSymbol` or :any:`TypedSymbol` :
            The symbol in the defining scope
        """
        imprt_symbol = imprt.symbols[imprt.symbols.index(symbol_name)]
        if imprt_symbol and imprt_symbol.type.use_name:
            symbol_name = imprt_symbol.type.use_name
        return symbol_name

    @staticmethod
    def _is_ignored(name, config, ignore):
        """
        Utility method to check if a given :data:`name` is ignored

        Parameters
        ----------
        name : str
            The name to check
        config : :any:`SchedulerConfig`, optional
            An optional config object, in which :any:`SchedulerConfig.is_disabled`
            is checked for :data:`name`
        ignore : list of str, optional
            An optional list of names, as typically provided in a config value.
            These are matched via :any:`SchedulerConfig.match_item_keys` with
            pattern matching enabled.

        Returns
        -------
        bool
            ``True`` if matched successfully via :data:`config` or :data:`ignore` list,
            otherwise ``False``
        """
        keys = as_tuple(config.disable if config else ()) + as_tuple(ignore)
        return keys and SchedulerConfig.match_item_keys(
            name, keys, use_pattern_matching=True, match_item_parents=True
        )
