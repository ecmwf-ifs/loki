# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.ir import nodes as ir, Transformer, FindNodes
from loki.tools.util import as_tuple, CaseInsensitiveDict

__all__ = ['DuplicateKernel', 'RemoveKernel']


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
