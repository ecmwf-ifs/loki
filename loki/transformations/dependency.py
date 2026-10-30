# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from itertools import chain

from loki.batch.item import ProcedureItem
from loki.batch.transformation import Transformation
from loki.ir import nodes as ir, Transformer, FindNodes, FindInlineCalls
from loki.tools.util import OrderedSet, as_tuple, CaseInsensitiveDict
from loki.subroutine import Subroutine
from loki.logging import warning
from loki.transformations.build_system.dependency import do_rename_subroutine
from loki.transformations.build_system.module_wrap import do_module_wrap_subroutine
from loki.transformations.utilities import sanitise_imports

__all__ = [
    "CreateEntryPointsTransformation", "DuplicateKernel", "RemoveKernel",
    "SeparateModesKernel", "get_mode_rename_suffix"
]


def get_mode_rename_suffix(mode, rename_suffix='_loki_{mode}'):
    """
    Get the suffix used to rename items for a given transformation mode.

    Parameters
    ----------
    mode : str
        Transformation mode. Dashes are replaced by underscores to yield a
        valid Fortran name.
    rename_suffix : str, optional
        Format string for the suffix, which accepts a ``mode`` keyword argument.

    Returns
    -------
    str
        The mode-specific rename suffix, e.g. ``_loki_<mode>``.
    """
    return rename_suffix.format(mode=mode.replace('-', '_'))


class CreateEntryPointsTransformation(Transformation):
    """
    Replace configured external subroutines with runtime dispatch entry points.

    Routines are selected via the item configuration key ``entry-point``, and
    the configuration key ``condition`` provides the dispatch expression, which is
    parsed in the scope of the configured routine.

    For every selected routine ``<name>``, two driver variants are created, one
    in transformation mode :data:`mode` and one in transformation mode
    :data:`baseline_mode`. Following the naming convention of
    :any:`SeparateModesKernel`, the variants are named ``<name>_loki_<mode>`` and
    wrapped in a module ``<name>_loki_<mode><module_suffix>`` in a new source file.

    The original routine is retained as a standalone dispatch layer, which
    preserves the original name and interface. It is assigned the
    :data:`baseline_mode`, and its item configuration disables expansion and blocks
    the driver variants, such that the dispatch layer has no children in the
    dependency graph and does not propagate its mode to the driver variants.

    The transformation sets :any:`Transformation.changes_modes`. When applied
    as part of mode-specific pipelines, the :any:`Scheduler` applies it before
    propagating the transformation modes down the call trees of the driver
    variants, which separates their dependencies via :any:`SeparateModesKernel`.

    Parameters
    ----------
    mode : str
        Transformation mode of the Loki-transformed driver variant.
    baseline_mode : str
        Transformation mode of the baseline driver variant and the dispatch layer.
    module_suffix : str, optional
        Suffix used for the generated driver variant modules.
    """

    creates_items = True
    changes_modes = True

    def __init__(self, mode, baseline_mode, module_suffix='_mod'):
        if get_mode_rename_suffix(mode) == get_mode_rename_suffix(baseline_mode):
            raise ValueError(
                f'Entry-point modes must yield distinct variant names: {mode}, {baseline_mode}'
            )

        self.mode = mode
        self.baseline_mode = baseline_mode
        self.module_suffix = module_suffix

    @staticmethod
    def _is_entry_point(routine, item):
        """Validate the entry-point configuration and report whether it is enabled."""

        if not isinstance(item, ProcedureItem) or not item.entry_point:
            return False

        condition = item.config.get('condition')
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f'Entry point {item.name} requires a non-empty string condition')

        if routine.parent is not None:
            raise ValueError(f'Entry point {item.name} must be an external subroutine, not a module procedure')

        if routine.is_function:
            raise ValueError(f'Entry point {item.name} must be a subroutine, not a function')

        return True

    @staticmethod
    def _create_item_config(config, mode, **kwargs):
        """
        Create a driver or dispatch layer config without
        entry-point selection data.
        """

        new_config = config.copy()
        for key in ('entry-point', 'condition', 'seed_routine'):
            new_config.pop(key, None)
        new_config.update(role='driver', mode=mode, **kwargs)
        return new_config

    @staticmethod
    def _update_scheduler_config(scheduler_config, item_name, config):
        """Register the item configuration in the scheduler config."""

        matched_keys = scheduler_config.match_item_keys(item_name, scheduler_config.routines)
        for key in matched_keys or (item_name,):
            scheduler_config.routines[key] = config.copy()

    def _create_driver(self, routine, item, mode, item_factory, scheduler_config):
        """
        Create a module-wrapped driver variant of :data:`routine` in a new source
        file and register the corresponding items with the :data:`item_factory`.
        """
        driver = routine.clone(parent=None, rescope_symbols=True)
        do_rename_subroutine(driver, get_mode_rename_suffix(mode))

        module_name = f'{driver.name}{self.module_suffix}'
        source = item.source
        path = source.path.with_name(f'{module_name.lower()}{source.path.suffix}')
        driver_source = source.clone(path=path, ir=ir.Section(body=(driver,)))
        do_module_wrap_subroutine(driver_source, driver, self.module_suffix)

        # Register the config first, so that the new items pick it up on creation
        driver_item_name = f'{module_name}#{driver.name}'.lower()
        driver_config = self._create_item_config(item.config, mode=mode)
        self._update_scheduler_config(scheduler_config, driver_item_name, driver_config)

        file_item = item_factory.get_or_create_file_item_from_source(driver_source, scheduler_config)
        file_item.create_definition_items(item_factory=item_factory, config=scheduler_config)
        driver_item = item_factory.get_or_create_item(
            ProcedureItem, driver_item_name, module_name.lower(), config=scheduler_config
        )
        return driver_item

    def _configure_dispatcher(self, item, driver_items, scheduler_config):
        """
        Turn the entry-point item into a standalone dispatch layer driver without
        children in the dependency graph.
        """
        block = OrderedSet(as_tuple(item.block))
        for driver_item in driver_items:
            block.add(driver_item.scope_name)
            block.add(driver_item.name)

        # We explicitly disable and block all children here, making the new dispatch an entirely
        # standalone driver from a loki scheduler perspective. This is purely to allow it to be a part of the
        # scheduler graph so that we can reuse existing loki machinery to write it to file.
        dispatcher_config = self._create_item_config(
            item.config, mode=self.baseline_mode, expand=False, replicate=False, block=tuple(block)
        )
        item.config = dispatcher_config
        self._update_scheduler_config(scheduler_config, item.name, dispatcher_config)

    @staticmethod
    def _remove_callee_dependencies(routine, callees):
        """
        Remove unused imports of the original callees from the dispatch layer,
        such that it does not depend on them anymore.
        """
        node_map = {
            imprt: None for imprt in FindNodes(ir.Import).visit(routine.spec)
            if imprt.c_import and imprt.module.split('.', maxsplit=1)[0].lower() in callees
        }
        if node_map:
            routine.spec = Transformer(node_map).visit(routine.spec)
        sanitise_imports(routine)

    @classmethod
    def _create_dispatch_layer(cls, routine, condition, driver_routine, baseline_routine):
        """Replace a routine body with a two-way dispatch to the driver variants."""

        callees = {str(call.name).lower() for call in FindNodes(ir.CallStatement).visit(routine.body)}
        callees |= {str(call.function).lower() for call in FindInlineCalls().visit(routine.body)}

        try:
            condition = routine.parse_expr(condition, strict=True)
        except Exception as exc:
            raise ValueError(
                f'Failed to parse entry-point condition for {routine.name}: {condition}'
            ) from exc

        driver_symbol = driver_routine.procedure_symbol.rescope(scope=routine)
        baseline_symbol = baseline_routine.procedure_symbol.rescope(scope=routine)

        arguments = tuple(
            argument.rescope(scope=routine).clone(dimensions=None)
            for argument in routine.arguments
        )

        driver_call = ir.CallStatement(name=driver_symbol, arguments=arguments)
        baseline_call = ir.CallStatement(name=baseline_symbol, arguments=arguments)

        routine.body = ir.Section(body=(ir.Conditional(
            condition=condition, body=(driver_call,), else_body=(baseline_call,)
        ),))

        imports = (
            ir.Import(module=driver_routine.parent.name, symbols=(driver_symbol,)),
            ir.Import(module=baseline_routine.parent.name, symbols=(baseline_symbol,)),
        )
        routine.spec.prepend(imports)

        cls._remove_callee_dependencies(routine, callees)

    def _create_entry_point(self, routine, item, item_factory, scheduler_config):
        """Create the driver variants and turn the entry-point item into the dispatch layer."""

        driver_items = tuple(
            self._create_driver(routine, item, mode, item_factory, scheduler_config)
            for mode in (self.mode, self.baseline_mode)
        )
        self._configure_dispatcher(item, driver_items, scheduler_config)
        return driver_items

    def transform_subroutine(self, routine, **kwargs):
        """Create the dispatch layer and driver variants for a selected entry point."""

        item = kwargs.get('item')
        if not self._is_entry_point(routine, item):
            return

        condition = item.config['condition']
        driver_item, baseline_item = self._create_entry_point(
            routine, item, kwargs['item_factory'], kwargs['scheduler_config']
        )
        self._create_dispatch_layer(routine, condition, driver_item.ir, baseline_item.ir)

    def plan_subroutine(self, routine, **kwargs):
        """Create the driver variants and dispatch layer configuration in planning mode."""

        item = kwargs.get('item')
        if not self._is_entry_point(routine, item):
            return

        self._create_entry_point(routine, item, kwargs['item_factory'], kwargs['scheduler_config'])


class DuplicateKernel(Transformation):
    """
    Duplicate subroutines which includes the creation of new :any:`Item`s
    as well as the addition of the corresponding new dependencies.

    Therefore, this transformation creates a new item and also implements
    the relevant routines for dry-run pipeline planning runs.

    Parameters
    ----------
    duplicate_kernels : str|tuple|list, optional
        Kernel name(s) to be duplicated.
    duplicate_suffix : str, optional
        Suffix to be used to append the original kernel name(s).
    duplicate_module_suffix : str, optional
        Suffix to be used to append the original module name(s),
        if defined, otherwise `duplicate_suffix`
    duplicate_subgraph : bool, optional
        Whether or not duplicate the subgraph beneath the kernel(s)
        that are duplicated.
    """

    creates_items = True
    reverse_traversal = True

    def __init__(self, duplicate_kernels=None, duplicate_suffix='duplicated',
                 duplicate_module_suffix=None, duplicate_subgraph=False):
        self.suffix = duplicate_suffix
        self.module_suffix = duplicate_module_suffix or duplicate_suffix
        self.duplicate_kernels = tuple(kernel.lower() for kernel in as_tuple(duplicate_kernels))
        self.duplicate_subgraph = duplicate_subgraph

    def _get_new_item_name(self, item):
        """
        Get new/duplicated item name, more specifically ``local_name``,
        ``scope_name`` and ``new_item_name``.

        Parameters
        ----------
        item : :any:`Item`
            The item used to derive ``local_name``,
            ``scope_name`` and ``new_item_name``.

        Returns
        -------
        scope_name : str
            New item scope name.
        new_item_name : str
            New item name.
        local_name : str
            New item local name.
        """
        # Determine new item name
        scope_name = item.scope_name
        local_name = f'{item.local_name}{self.suffix}'
        if scope_name:
            scope_name = f'{scope_name}{self.module_suffix}'
        # Try to get existing item from cache
        new_item_name = f'{scope_name or ""}#{local_name}'
        return scope_name, local_name, new_item_name

    def _get_or_create_or_rename_item(self, item, item_factory, config):
        """
        Get, create or rename item including the scope item if there is a
        scope.

        Parameters
        ----------
        item : :any:`Item`
            Item to duplicate/to use to derive new item.
        item_factory : :any:`ItemFactory`
            The :any:`ItemFactory` to use when creating the items.
        config : :any:`SchedulerConfig`
            The scheduler config to use when instantiating new items.

        Returns
        -------
        :any:`Item`
            Newly created item.
        """
        scope_name, local_name, new_item_name = self._get_new_item_name(item)
        new_item = item_factory.item_cache.get(new_item_name)

        # Try to get an item for the scope or create that first
        if new_item is None and scope_name:
            scope_item = item_factory.item_cache.get(scope_name)
            if scope_item:
                scope = scope_item.ir
                if local_name not in scope and item.local_name in scope:
                    # Rename the existing item to the new name
                    scope[item.local_name].name = local_name

                if local_name in scope:
                    new_item = item_factory.create_from_ir(
                        scope[local_name], scope, config=config
                    )

        # Create new item
        if new_item is None:
            new_item = item_factory.get_or_create_item_from_item(new_item_name, item, config=config)

        return new_item

    def _modify_sgraph(self, sgraph, item, new_items):
        """
        Add new items to graph.

        Parameters
        ----------
        sgraph : :any:`SGraph`
            Directed graph or rather copy of it to
            be modified.
        item : :any:`Item`
            Node to which add the new items.
        new_items : tuple
            Tuple of :any:`Item` to add to graph.
        """
        sgraph.add_nodes(new_items)
        sgraph.add_edges((item, _item) for _item in new_items)

    def _rename_calls(self, new_item, new_dependencies):
        """
        Rename calls and imports according to the newly created
        duplicated items.

        Parameters
        ----------
        new_item : :any:`Item`
            The newly created item for which calls and imports
            to be renamed.
        new_dependencies : dict
            Dictionary used to get information about how
            to rename calls and imports.
        """
        call_map = {}
        for call in FindNodes(ir.CallStatement).visit(new_item.ir.body):
            call_name = str(call.name).lower()
            new_call_name = f'{call_name}{self.suffix}'.lower()
            if new_call_name in new_dependencies:
                call_new_item = new_dependencies[new_call_name]
                proc_symbol = call_new_item.ir.procedure_symbol.rescope(scope=new_item.ir)
                call_map[call] = call.clone(name=proc_symbol)
        # TODO: imports at module level ...
        imp_map = {}
        for imp in FindNodes(ir.Import).visit(new_item.ir.spec):
            # potentially new symbols
            symbol_map = {symbol: symbol.clone(name=f'{symbol.name}{self.suffix}') for symbol in imp.symbols}
            new_symbols = ()
            orig_symbols = ()
            # distinguish imported symbols that should remain and those which should be altered
            for orig_symbol, new_symbol in symbol_map.items():
                if new_symbol in new_dependencies:
                    new_symbols += (new_symbol,)
                else:
                    orig_symbols += (orig_symbol,)
            new_imports = ()
            if new_symbols:
                new_imports += (imp.clone(module=f'{imp.module.lower()}{self.module_suffix}',
                                         symbols=as_tuple(new_symbols)),)
            if orig_symbols:
                new_imports += (imp.clone(symbols=as_tuple(orig_symbols)),)
            if new_imports:
                imp_map[imp] = new_imports
        if call_map:
            new_item.ir.body = Transformer(call_map).visit(new_item.ir.body)
        if imp_map:
            new_item.ir.spec = Transformer(imp_map).visit(new_item.ir.spec)

    def _create_duplicate_items(self, successors, item_factory, config, item, sub_sgraph,
            rename_calls=False, force_duplicate=False, ignore=None):
        """
        Create new/duplicated items.

        Parameters
        ----------
        successors : tuple
            Tuple of :any:`Item`s representing the successor items for which
            new/duplicated items are created..
        item_factory : :any:`ItemFactory`
            The :any:`ItemFactory` to use when creating the items.
        config : :any:`SchedulerConfig`
            The scheduler config to use when instantiating new items.
        item : :any:`Item`
            Starting point/source item from which the successors
            originate from
        sub_sgraph : :any:`SGraph`
            Sgraph (copy) representing the subgraph of the directed
            overall graph.
        rename_calls : bool, optional
            Rename calls/imports in accordance to the duplicated
            kernels.
        force_duplicate : bool, optional
            Check whether successor is within ``duplicate_kernels``
            or duplicate either way.
        Returns
        -------
        tuple
            Tuple of newly created items.
        """
        ignore = as_tuple(ignore)
        new_items = ()
        for child in successors:
            if child.local_name in self.duplicate_kernels or force_duplicate:
                if child.local_name in ignore:
                    continue
                # get/create/rename item
                new_item = self._get_or_create_or_rename_item(child, item_factory, config)
                new_items += as_tuple(new_item)
                # duplicate subgraph?
                if self.duplicate_subgraph:
                    new_item = as_tuple(new_item)[0]
                    # add new_items to sgraph (copy)
                    self._modify_sgraph(sub_sgraph, item, new_items)
                    # get the successors
                    child_ignore = ignore + as_tuple(child.ignore)
                    child_successors = as_tuple([successor for successor in sub_sgraph.successors(child)
                        if successor.local_name not in child_ignore])
                    if child_successors:
                        # create new/duplicated successors
                        new_dependencies = self._create_duplicate_items(child_successors, item_factory, config,
                                sub_sgraph=sub_sgraph, item=new_item, rename_calls=rename_calls, force_duplicate=True,
                                ignore=ignore)
                        new_dependencies_dic = CaseInsensitiveDict((new_item.local_name, new_item)
                                for new_item in new_dependencies)
                        # add dependencies to new/duplicated successors and remove the "old" ones
                        new_item.plan_data.setdefault('additional_dependencies', ())
                        new_item.plan_data.setdefault('removed_dependencies', ())
                        new_item.plan_data['additional_dependencies'] += as_tuple(new_dependencies)
                        new_item.plan_data['removed_dependencies'] += as_tuple(child_successors)
                        sub_sgraph._add_children(new_item, item_factory, config, dependencies=new_dependencies)
                        # rename calls and imports
                        if rename_calls:
                            self._rename_calls(new_item, new_dependencies_dic)
        return tuple(new_items)

    def transform_subroutine(self, routine, **kwargs):
        # Create new dependency items
        item = kwargs.get('item')
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = sub_sgraph.successors(item) if sub_sgraph is not None else ()
        ignore = tuple(str(t).lower() for t in as_tuple(kwargs.get('ignore', None)))
        new_dependencies = self._create_duplicate_items(
            successors=successors,
            item_factory=kwargs.get('item_factory'),
            config=kwargs.get('scheduler_config'),
            item=kwargs.get('item'),
            sub_sgraph=sub_sgraph,
            rename_calls=True, ignore=ignore
        )
        new_dependencies = CaseInsensitiveDict((new_item.local_name, new_item) for new_item in new_dependencies)

        # Duplicate calls to kernels
        call_map = {}
        new_imports = []
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            call_name = str(call.name).lower()
            if call_name in self.duplicate_kernels:
                # Duplicate the call
                new_call_name = f'{call_name}{self.suffix}'.lower()
                new_item = new_dependencies[new_call_name]
                proc_symbol = new_item.ir.procedure_symbol.rescope(scope=routine)
                call_map[call] = (call, call.clone(name=proc_symbol))

                # Register the module import
                if new_item.scope_name:
                    new_imports += [ir.Import(module=new_item.scope_name, symbols=(proc_symbol,))]

        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)
            if new_imports:
                routine.spec.prepend(as_tuple(new_imports))

    def plan_subroutine(self, routine, **kwargs):
        item = kwargs.get('item')
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = sub_sgraph.successors(item) if sub_sgraph is not None else ()
        ignore = tuple(str(t).lower() for t in as_tuple(kwargs.get('ignore', None)))
        item.plan_data.setdefault('additional_dependencies', ())
        item.plan_data['additional_dependencies'] += self._create_duplicate_items(
            successors=successors,
            item_factory=kwargs.get('item_factory'),
            config=kwargs.get('scheduler_config'),
            item=item,
            sub_sgraph=sub_sgraph,
            ignore=ignore
        )


class RemoveKernel(Transformation):
    """
    Remove subroutines which includes the removal of the relevant :any:`Item`s
    as well as the removal of the corresponding dependencies.

    Therefore, this transformation creates a new item and also implements
    the relevant routines for dry-run pipeline planning runs.

    Parameters
    ----------
    remove_kernels : str|tuple|list, optional
        Kernel name(s) to be removed.
    """

    creates_items = True

    def __init__(self, remove_kernels=None):
        self.remove_kernels = tuple(kernel.lower() for kernel in as_tuple(remove_kernels))

    def transform_subroutine(self, routine, **kwargs):
        call_map = {
            call: None for call in FindNodes(ir.CallStatement).visit(routine.body)
            if str(call.name).lower() in self.remove_kernels
        }
        routine.body = Transformer(call_map).visit(routine.body)

    def plan_subroutine(self, routine, **kwargs):
        item = kwargs.get('item')
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = sub_sgraph.successors(item) if sub_sgraph is not None else ()

        item.plan_data.setdefault('removed_dependencies', ())
        item.plan_data['removed_dependencies'] += tuple(
            child for child in successors if child.local_name in self.remove_kernels
        )

class SeparateModesKernel(Transformation):
    """
    Duplicate subroutines which includes the creation of new :any:`Item`s
    as well as the addition of the corresponding new dependencies.

    Therefore, this transformation creates a new item and also implements
    the relevant routines for dry-run pipeline planning runs.

    Parameters:
    -----------
    rename_suffix : str, optional
        Suffix to be used to append the original kernel name(s) and module name(s).
        Must be a valid format string that accepts a ``mode`` keyword argument,
        which is used to generate the suffix for the new item names.
    """

    creates_items = True
    renames_items = True
    reverse_traversal = True

    def __init__(self, rename_suffix='_loki_{mode}'):
        if '{mode}' not in rename_suffix:
            raise ValueError("rename_suffix must contain '{mode}' placeholder")
        self.rename_suffix = rename_suffix

    def _get_new_item_name(self, item, mode):
        """
        Get new/duplicated item name, more specifically ``local_name``,
        ``scope_name`` and ``new_item_name``.

        Parameters
        ----------
        item : :any:`Item`
            The item used to derive ``local_name``,
            ``scope_name`` and ``new_item_name``.
        mode : str
           Transformation mode.

        Returns
        -------
        local_name : str
            New item fully-qualified name.
        """
        mode_rename = get_mode_rename_suffix(mode, self.rename_suffix)

        # Determine new item name
        scope_name = item.scope_name
        local_name = f'{item.local_name}{mode_rename}'
        if scope_name:
            if scope_name.endswith('_mod'):
                scope_name = f'{scope_name[:-4]}{mode_rename}_mod'
            else:
                scope_name = f'{scope_name}{mode_rename}'

        return f'{scope_name or ""}#{local_name}'

    def _get_or_create_or_rename_item(self, item, mode, item_factory, config):
        """
        Get, create or rename item including the scope item if there is a
        scope.

        Parameters
        ----------
        item : :any:`Item`
            Item to duplicate/to use to derive new item.
        mode : str
            Transformation mode.
        item_factory : :any:`ItemFactory`
            The :any:`ItemFactory` to use when creating the items.
        config : :any:`SchedulerConfig`
            The scheduler config to use when instantiating new items.

        Returns
        -------
        :any:`Item`
            Newly created item.
        """
        new_item_name = self._get_new_item_name(item, mode)
        new_item = item_factory.item_cache.get(new_item_name)

        # Try to get an item for the scope or create that first
        if new_item is None:
            scope_name, local_name = new_item_name.split('#')
            if scope_name and (scope_item := item_factory.item_cache.get(scope_name)):
                scope = scope_item.ir
                if local_name not in scope and item.local_name in scope:
                    # Rename the existing item to the new name
                    scope[item.local_name].name = local_name

                if local_name in scope:
                    new_item = item_factory.create_from_ir(
                        scope[local_name], scope, config=config
                    )

        # Create new item
        if new_item is None:
            new_item = as_tuple(item_factory.get_or_create_item_from_item(new_item_name, item, config=config))[0]

        return new_item

    def _update_plan_dependencies(self, item, successors, item_factory, ignore=None):
        """
        Update plan dependencies for the given item.

        Parameters
        ----------
        item : :any:`Item`
            The item for which to update the plan dependencies.
        successors : tuple
            Tuple of :any:`Item`s representing the successor items for which
            the dependencies of :data:`item` are updated.
        item_factory : :any:`ItemFactory`
            The :any:`ItemFactory` to fetch the new successor items from.
        ignore : tuple, optional
            Tuple of local names to ignore when updating dependencies.

        Returns
        -------
        dict
            A mapping from old successor items to new successor items that
            were added as dependencies.
        """
        ignore = as_tuple(ignore)
        item.plan_data.setdefault('additional_dependencies', ())
        item.plan_data.setdefault('removed_dependencies', ())

        new_dependencies = {}

        for child in successors:
            if child.local_name in ignore:
                continue
            new_child_name = self._get_new_item_name(child, item.mode)
            new_child = item_factory.item_cache.get(new_child_name)

            if new_child:
                item.plan_data['additional_dependencies'] += (new_child,)
                item.plan_data['removed_dependencies'] += (child,)
                new_dependencies[child] = new_child
            else:
                warning((
                    f'[Loki] SeparateModesKernel: Could not find new item {new_child_name} for '
                    f'child {child.name} of item {item.name} in the item cache.'
                ))

        return new_dependencies

    def _create_new_items(self, item, successors, item_factory, config, ignore=None):
        """
        Duplicate :data:`item` for each transformation mode and update the plan
        dependencies accordingly.

        Driver items are not duplicated but their dependencies are still updated to point
        to the new items, if applicable.

        Parameters
        ----------
        item : :any:`Item`
            Starting point/source item from which the successors
            originate from
        successors : tuple
            Tuple of :any:`Item`s representing the successor items for which
            new/duplicated items are created..
        item_factory : :any:`ItemFactory`
            The :any:`ItemFactory` to use when creating the items.
        config : :any:`SchedulerConfig`
            The scheduler config to use when instantiating new items.
        ignore : tuple, optional
            Tuple of local names to ignore when creating new items and updating dependencies.

        Returns
        -------
        tuple
            Tuple of newly created items.
        """
        if item.trafo_data.get('SeparateModes'):
            # Item has already been handled for this stage, so we
            # can skip the item renaming to remain idempotent but
            # still provide the mapping in case they are required for
            # call updates
            additional_dependencies = item.plan_data.get('additional_dependencies', ())
            removed_dependencies = item.plan_data.get('removed_dependencies', ())
            mapping = dict(zip(removed_dependencies, additional_dependencies))
            return {item: mapping}

        # A map containing items that have been modified or created, with corresponding mappings from old
        # to new dependency items
        updated_items = {}

        if item.role == 'kernel':
            modes_to_generate = item.trafo_data.get('inherited_mode', {item.mode})

            for mode in modes_to_generate:
                new_item = self._get_or_create_or_rename_item(item, mode, item_factory, config)
                new_item.config = item.config.copy()
                new_item.config['mode'] = mode


                # Make sure that encapsulating scopes are also renamed
                for scope_item in item_factory.get_scope_and_file_items(new_item):
                    scope_item.config['mode'] = mode

                # Update plan dependencies
                updated_items[new_item] = self._update_plan_dependencies(
                    new_item, successors, item_factory, ignore=ignore
                )

                new_item.trafo_data['SeparateModes'] = True

        else:
            updated_items[item] = self._update_plan_dependencies(
                item, successors, item_factory, ignore=ignore
            )

        item.trafo_data['SeparateModes'] = True

        return updated_items

    def _match_names_to_new_dependencies(self, names, mapping):
        """
        Return the mapping from :data:`names` to new dependency items provided in :data:`mapping`
        """
        from loki.batch.configure import SchedulerConfig  # pylint: disable=import-outside-toplevel
        return CaseInsensitiveDict(
            (name, new_item)
            for old_item, new_item in mapping.items()
            for name in SchedulerConfig.match_item_keys(old_item.name, names)
        )

    def _update_item_config_names(self, item, orig_name, new_name):
        """
        Rename ``ignore`` and ``block`` entries in the item config to reflect new item names.
        """
        from loki.batch.configure import SchedulerConfig  # pylint: disable=import-outside-toplevel

        # Update the ignore and block properties if necessary
        if matched_keys := SchedulerConfig.match_item_keys(orig_name, item.ignore):
            # Add the renamed but ignored items to the block list because we won't be able to
            # find them as dependencies under their new name anymore
            item.config['block'] = as_tuple(item.block) + tuple(
                new_name for name in item.ignore if name in matched_keys
            )
            item.config['ignore'] = tuple(
                new_name if name in matched_keys else name
                for name in item.ignore
            )

    def rename_interfaces(self, item, mapping):
        """
        Update :any:`Interface` names to actively transformed procedures.

        Parameters
        ----------
        item : :any:`Item`
            The item for which to rename interfaces.
        mapping : dict
            A mapping from old dependency item names to new dependency items.
        """
        intf_map = item.ir.interface_map
        intfs_to_rename = self._match_names_to_new_dependencies(list(intf_map), mapping)

        for name, intf in intf_map.items():
            if name in intfs_to_rename:
                for node in intf.body:
                    if isinstance(node, Subroutine):
                        node.name = intfs_to_rename[name].ir.procedure_symbol.name

    def rename_imports(self, item, mapping):
        """
        Update :any:`Import` statements to actively transformed procedures and modules.

        Updates imports of C-style headers and Fortran modules to match the new mode-specific names.

        Parameters
        ----------
        item : :any:`Item`
            The item for which to rename imports.
        mapping : dict
            A mapping from old dependency item names to new dependency items.
        """
        imports = item.ir.imports

        imported_names = [
            import_.module.lower().split('.', maxsplit=1)[0]
            for import_ in imports
            if import_.c_import and not import_.module.lower().endswith('func.h')
        ]
        imported_names += [
            s.name.lower()
            for import_ in imports
            if not import_.c_import and import_.symbols
            for s in import_.symbols
        ]

        imports_to_rename = self._match_names_to_new_dependencies(imported_names, mapping)
        if not imports_to_rename:
            return

        # We go through the IR, as C-imports can be attributed to the body
        new_imports = []
        for import_ in imports:
            if import_.c_import:
                target_symbol = import_.module.lower().split('.', maxsplit=1)[0]
                if target_symbol in imports_to_rename:
                    # Modify the the basename of the C-style header import
                    s = '.'.join(import_.module.split('.')[1:])
                    import_._update(module=f'{imports_to_rename[target_symbol].local_name}.{s}')
            else:
                if import_.symbols:
                    if not any(s.name in imports_to_rename for s in import_.symbols):
                        continue

                    modules_and_symbols = defaultdict(list)
                    for s in import_.symbols:
                        if s.name in imports_to_rename:
                            modules_and_symbols[imports_to_rename[s.name].scope_name].append(
                                s.clone(name=imports_to_rename[s.name].local_name)
                            )
                        else:
                            modules_and_symbols[import_.module].append(s)
                    for module, symbols in modules_and_symbols.items():
                        if module == import_.module or len(modules_and_symbols) == 1:
                            import_._update(module=module,symbols=tuple(symbols))
                        else:
                            new_imports += [import_.clone(module=module, symbols=tuple(symbols))]
                else:
                    warning(f'[Loki] SeparateModesKernel: encountered unqualified blank import for {import_.module}')
        if new_imports:
            item.ir.spec.prepend(as_tuple(new_imports))

    def rename_calls(self, item, mapping):
        """
        Update :any:`CallStatement` and :any:`InlineCall` to actively
        transformed procedures

        Parameters
        ----------
        item : :any:`Item`
            The item for which to rename calls.
        mapping : dict
            A mapping from old dependency item names to new dependency items.
        """
        routine = item.ir
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        inline_calls = FindInlineCalls().visit(routine.body)
        call_names = [str(call.name) for call in chain(calls, inline_calls)]

        calls_to_rename = self._match_names_to_new_dependencies(call_names, mapping)
        if not calls_to_rename:
            return

        for call in chain(calls, inline_calls):
            orig_name = str(call.name)
            if orig_name in calls_to_rename:
                new_item = calls_to_rename[orig_name]
                proc_symbol = new_item.ir.procedure_symbol.rescope(scope=routine)
                if isinstance(call, ir.CallStatement):
                    call._update(name=proc_symbol)
                else:
                    call.function = proc_symbol
                self._update_item_config_names(item, orig_name, proc_symbol.name)

    def transform_subroutine(self, routine, **kwargs):
        """
        Transform the given subroutine by creating new items for the different modes
        and updating the dependencies accordingly.
        """
        item = kwargs['item']
        assert routine is item.ir
        item_factory = kwargs.get('item_factory')
        sub_sgraph = kwargs.get('sub_sgraph')
        successors = sub_sgraph.successors(item) if sub_sgraph is not None else ()
        config = kwargs.get('scheduler_config')
        ignore = tuple(str(t).lower() for t in as_tuple(kwargs.get('ignore')))

        updated_items = self._create_new_items(item, successors, item_factory, config, ignore=ignore)

        for item_, mapping in updated_items.items():
            if mapping:
                self.rename_calls(item_, mapping)
                self.rename_imports(item_, mapping)
                self.rename_interfaces(item_, mapping)

    def plan_subroutine(self, routine, **kwargs):
        """
        Transform the given subroutine by creating new items for the different modes
        and registering the new dependencies in the item's plan_data accordingly.
        """
        item = kwargs['item']
        assert routine is item.ir
        item_factory = kwargs.get('item_factory')
        sub_sgraph = kwargs.get('sub_sgraph')
        successors = sub_sgraph.successors(item) if sub_sgraph is not None else ()
        config = kwargs.get('scheduler_config')
        ignore = tuple(str(t).lower() for t in as_tuple(kwargs.get('ignore')))

        self._create_new_items(item, successors, item_factory, config, ignore=ignore)
