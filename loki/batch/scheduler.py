# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from os.path import commonpath
from pathlib import Path
from codetiming import Timer

from loki.batch.configure import SchedulerConfig
from loki.batch.sfilter import SFilter
from loki.batch.sgraph import SGraph
from loki.batch.item import (
    FileItem, ModuleItem, ProcedureItem, ProcedureBindingItem,
    InterfaceItem, TypeDefItem, ExternalItem, ItemFactory
)
from loki.frontend import FP, REGEX, RegexParserClass
from loki.tools import as_tuple, CaseInsensitiveDict, flatten
from loki.logging import info, perf, warning, debug, error


__all__ = ['Scheduler']


class Scheduler:
    """
    Work queue manager to discover and capture dependencies for a given
    call tree, and apply transformations for batch processing

    Using a given list of :data:`paths` and :data:`seed_routines` (can be
    inferred from :data:`config`), all scopes and symbols defined in the
    source tree are discovered and a dependency graph is created. The nodes
    of the dependency graph are :any:`Item` objects, each corresponding to
    a specific Loki IR node.

    The dependency graph is re-generated on-the-fly and can be filtered for
    specific dependency classes during traversals (see :any:`SFilter`).
    All items, are stored in the cache of the associated :attr:`item_factory`.

    Under the hood, the Scheduler is initialised in a three-stage procedure:

    1. A `discovery` step, where the minimum top-level definitions (modules
       and procedures) are determined for every source file in the path list.
    2. A `populate` step, which instantiates a first :any:`SGraph` dependency
       graph by chasing dependencies starting from the provided seed nodes.
    3. Optionally, a full parse is triggered for all :any:`Sourcefile` that
       appear in an :any:`Item` in the dependency graph.

    This first two stages rely on an incremental, incomplete parsing of the
    source files that extract only the minimum set of symbols in each file.
    This is driven by the :any:`Item._parser_class` and :any:`Item._depends_class`
    attributes, which declare the minimum :any:`RegexParserClass` classes to
    use with the :any:`REGEX` frontend.

    To discover dependencies, the item's IR is `concretized`. This calls
    ``Scope.make_complete`` with the minimum set of additional parser classes
    (as defined in the ``_parser_class`` attribute) that are required to discover
    the dependencies (e.g., calls, imports).
    When creating the corresponding dependency's item, the defining scope's item
    (e.g., a module containing a derived type declaration) is queried for its
    ``definitions``, which in turn may trigger also a `concretize` step with the
    ``_parser_class`` of all item types that are listed in the scope item's
    ``defines_items`` attribute.

    A :any:`Transformation` can be applied across all nodes in the dependency
    graph using the :meth:`process` method. The class properties in the
    transformation implementation (such as :any:`Transformation.reverse_traversal`,
    :any:`Transformation.traverse_file_graph` or :any:`Transformation.item_filter`)
    determine, what nodes should be processed.

    Attributes
    ----------
    config : :any:`SchedulerConfig`
        The config object describing the Scheduler's behaviour
    full_parse : bool
        Flag to indicate a full parse of scheduler items
    paths : list of :any:`pathlib.Path`
        List of paths where sourcefiles are searched
    seeds : list of str
        Names of seed routines that are the root of dependency graphs
    build_args : dict
        List of frontend arguments that are given to :any:`Sourcefile.from_file`
        when performing a full parse
    item_factory : :any:`ItemFactory`
        Instance of the factory class for :any:`Item` creation and caching

    Parameters
    ----------
    paths : str or list of str
        List of paths to search for automated source file detection.
    config : dict or str, optional
        Configuration dict or path to scheduler configuration file
    seed_routines : list of str, optional
        Names of routines from which to populate the callgraph initially.
        If not provided, these will be inferred from the given config.
    preprocess : bool, optional
        Flag to trigger CPP preprocessing (by default `False`).
    includes : list of str, optional
        Include paths to pass to the C-preprocessor.
    defines : list of str, optional
        Symbol definitions to pass to the C-preprocessor.
    definitions : list of :any:`Module`, optional
        :any:`Module` object(s) that may supply external type or procedure
        definitions.
    xmods : str, optional
        Path to directory to find and store ``.xmod`` files when using
        the OMNI frontend.
    omni_includes: list of str, optional
        Additional include paths to pass to the preprocessor run as part of
        the OMNI frontend parse. If set, this **replaces** (!)
        :data:`includes`, otherwise :data:`omni_includes` defaults to the
        value of :data:`includes`.
    full_parse: bool, optional
        Flag indicating whether a full parse of all sourcefiles is required.
        By default a full parse is executed, use this flag to suppress.
    frontend : :any:`Frontend`, optional
        Frontend to use for full parse of source files (default :any:`FP`).
    """

    # TODO: Should be user-definable!
    source_suffixes = ['.f90', '.F90', '.f', '.F']

    def __init__(self, paths, config=None, seed_routines=None, preprocess=False,
                 includes=None, defines=None, definitions=None, xmods=None,
                 omni_includes=None, full_parse=True, frontend=FP):
        # Derive config from file or dict
        if isinstance(config, SchedulerConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = SchedulerConfig.from_file(config)
        else:
            self.config = SchedulerConfig.from_dict(config)

        self.full_parse = full_parse

        # Build-related arguments to pass to the sources
        self.paths = [Path(p) for p in as_tuple(paths)]
        self.seeds = tuple(
            seed.lower()
            for seed in as_tuple(seed_routines) or self.config.routines.keys()
        )

        # Accumulate all build arguments to pass to `Sourcefile` constructors
        self.build_args = {
            'definitions': definitions,
            'preprocess': preprocess,
            'includes': includes,
            'defines': defines,
            'xmods': xmods,
            'omni_includes': omni_includes,
            'frontend': frontend
        }

        # Internal data structures to store the callgraph
        self.item_factory = ItemFactory()

        self._discover()

        if self.full_parse:
            self._parse_items()

            # Attach interprocedural call-tree information
            self._enrich()

    @Timer(logger=info, text='[Loki::Scheduler] Performed initial source scan in {:.2f}s')
    def _discover(self):
        """
        Scan all source paths and create light-weight :any:`Sourcefile` objects for each file
        """
        frontend_args = {
            'preprocess': self.build_args['preprocess'],
            'includes': self.build_args['includes'],
            'defines': self.build_args['defines'],
            'parser_classes': RegexParserClass.ProgramUnitClass,
            'frontend': REGEX
        }

        # Create a list of initial files to scan with the fast REGEX frontend
        path_list = [path.glob(f'**/*{ext}') for path in self.paths for ext in self.source_suffixes]
        path_list = list(set(flatten(path_list)))  # Filter duplicates and flatten

        # Instantiate FileItem instances for all files in the search path
        for path in path_list:
            self.item_factory.get_or_create_file_item_from_path(path, self.config, frontend_args)

        # Instantiate the basic list of items for files and top-level program units
        #  in each file, i.e., modules and subroutines
        #  Note that we do this separate from the FileItem instantiation above to enable discovery
        #  also for FileItems that have been created as part of a transformation
        file_items = [
            file_item
            for file_item in self.item_factory.item_cache.values()
            if isinstance(file_item, FileItem)
        ]
        for file_item in file_items:
            definition_items = {
                item.name: item
                for item in file_item.create_definition_items(item_factory=self.item_factory, config=self.config)
            }
            self.item_factory.item_cache.update(definition_items)

        # (Re-)build the SGraph after discovery for later traversals
        self._sgraph = SGraph.from_seed(self.seeds, self.item_factory, self.config)

    @property
    def sgraph(self):
        """
        Create and return the :any:`SGraph` constructed from the :attr:`seeds` of the Scheduler.
        """
        return self._sgraph

    @property
    def items(self):
        """
        All :any:`Item` objects contained in the :any:`Scheduler` dependency graph.
        """
        return self.sgraph.items

    @property
    def dependencies(self):
        """
        All individual pairs of :any:`Item` that represent a dependency
        and form an edge in the :any`Scheduler` call graph.
        """
        return self.sgraph.dependencies

    @property
    def definitions(self):
        """
        The list of definitions that the source files in the callgraph provide
        """
        return tuple(
            definition
            for item in self.file_graph
            for definition in item.definitions
        )

    @property
    def file_graph(self):
        """
        Alternative dependency graph based on relations between source files

        Returns
        -------
        :any:`SGraph`
            A dependency graph containing only :any:`FileItem` nodes
        """
        item_filter = None if self.config.enable_imports else ProcedureItem
        return self.sgraph.as_filegraph(
            self.item_factory, self.config, item_filter=item_filter
        )

    def __getitem__(self, name):
        """
        Find and return an item in the Scheduler's dependency graph
        """
        for item in self.items:
            if item == name:
                return item
        return None

    def __iter__(self):
        return self.sgraph._graph.__iter__()

    @Timer(logger=info, text='[Loki::Scheduler] Performed full source parse in {:.2f}s')
    def _parse_items(self):
        """
        Prepare processing by triggering a full parse of the items in
        the execution plan and enriching subroutine calls.
        """
        # Force the parsing of the routines
        default_frontend_args = self.build_args.copy()
        default_frontend_args['definitions'] = as_tuple(default_frontend_args['definitions']) + self.definitions
        for item in SFilter(self.file_graph, reverse=True):
            frontend_args = self.config.create_frontend_args(item.name, default_frontend_args)
            item.source.make_complete(**frontend_args)

        # Re-build the SGraph after parsing to pick up all new connections
        self._sgraph = SGraph.from_seed(self.seeds, self.item_factory, self.config)

    @Timer(logger=info, text='[Loki::Scheduler] Enriched call tree in {:.2f}s')
    def _enrich(self):
        """
        For items that have a specific enrichment list provided as part of their
        config, try to provide this information
        """
        definitions = self.definitions
        for item in SFilter(self.sgraph, item_filter=ProcedureItem):
            # Enrich with the definitions of the scheduler's graph and meta-info from outside the callgraph
            enrich_definitions = definitions
            for name in as_tuple(item.enrich):
                enrich_items = as_tuple(
                    self.sgraph._create_item(name, item_factory=self.item_factory, config=self.config)
                )
                for enrich_item in enrich_items:
                    frontend_args = self.config.create_frontend_args(enrich_item.source.path, self.build_args)
                    enrich_item.source.make_complete(**frontend_args)
                enrich_definitions += tuple(item_.ir for item_ in enrich_items)
            item.ir.enrich(enrich_definitions, recurse=True)

    def rekey_item_cache(self):
        """
        Rebuild the mapping of item names to items in the :attr:`item_factory`'s cache

        This is required when a :any:`Transformation` renames items during processing,
        and is triggered automatically at the end of the :meth:`process` method if
        the transformation has :any:`Transformation.renames_items` specified.

        This update also updates :attr:`config` entries that are affected by the renaming.
        """
        # Find invalid item cache entries
        renamed_keys = {
            key: item.name for key, item in self.item_factory.item_cache.items()
            if item.name != key
        }

        # Find deleted item cache entries
        deleted_keys = set()
        for key, item in self.item_factory.item_cache.items():
            if isinstance(item, FileItem):
                continue
            if isinstance(item, ModuleItem):
                if item.name not in renamed_keys and item.name not in item.source:
                    # The module was in a file (likely with something else) and has been deleted
                    deleted_keys.add(key)
            elif not isinstance(item, ExternalItem):
                if not item.scope_name:
                    # IR node without a scope (i.e., a Procedure without a module)
                    if item.local_name not in item.source:
                        # ...has been deleted
                        deleted_keys.add(key)
                elif item.scope_name in renamed_keys:
                    # The parent module has been renamed...
                    if item.local_name not in item.source[renamed_keys[item.scope_name]]:
                        # ...and the contained item has been deleted from the module
                        deleted_keys.add(key)
                else:
                    if item.scope_name not in item.source:
                        # The parent module has been removed
                        deleted_keys.add(key)

        # Rename item scopes where necessary
        for key, item in self.item_factory.item_cache.items():
            if item.scope_name in renamed_keys and key not in deleted_keys:
                item.name = f'{renamed_keys[item.scope_name]}#{item.local_name}'
                renamed_keys[key] = item.name

        # Search for invalid item cache keys in config entries
        for old_name, new_name in renamed_keys.items():
            if matched_keys := self.config.match_item_keys(old_name, self.config.routines):
                for key in matched_keys:
                    self.config.routines[new_name] = self.config.routines[key].copy()
                    del self.config.routines[key]
            if matched_keys := self.config.match_item_keys(old_name, self.seeds):
                self.seeds = tuple(
                    new_name if seed in matched_keys else seed
                    for seed in self.seeds
                )

        # Find FileItem cache entries for renamed cache entries and rename them.
        # This allows to clone program units _and_ use references to unmodified program units
        # in the original file within the same SGraph. The unmodified program units are
        # re-discovered when running _discover() afterwards.
        if renamed_keys:
            for key, file_item in self.item_factory.item_cache.items():
                if isinstance(file_item, FileItem):
                    if any(file_item.source is self.item_factory.item_cache[key].source for key in renamed_keys):
                        file_item.name = f'duplicate of {file_item.name}'
                        renamed_keys[key] = file_item.name

        # Rebuild item_cache to make keys match entries
        self.item_factory.item_cache = CaseInsensitiveDict(
            (item.name, item) for item in self.item_factory.item_cache.values()
            if item.name not in deleted_keys
        )

    def process(self, transformation):
        """
        Process all :attr:`items` in the scheduler's graph with either
        a :any:`Pipeline` or a single :any:`Transformation`.

        A single :any:`Transformation` pass invokes
        :meth:`process_transformation` individually, while a
        :any:`Pipeline` will apply each contained transformation in
        turn over the full dependency graph of the scheduler.

        Parameters
        ----------
        transformation : :any:`Transformation` or :any:`Pipeline`
            The transformation or transformation pipeline to apply
        """
        from loki.transform import Transformation, Pipeline  # pylint: disable=import-outside-toplevel

        if isinstance(transformation, Transformation):
            self.process_transformation(transformation=transformation)

        elif isinstance(transformation, Pipeline):
            self.process_pipeline(pipeline=transformation)

        else:
            error('[Loki::Scheduler] Batch processing requires Transformation or Pipeline object')
            raise RuntimeError('[Loki] Could not batch process {transformation_or_pipeline}')

    def process_pipeline(self, pipeline):
        """
        Process a given :any:`Pipeline` by applying its assocaited
        transformations in turn.

        Parameters
        ----------
        transformation : :any:`Pipeline`
            The transformation pipeline to apply
        """
        for transformation in pipeline.transformations:
            self.process_transformation(transformation)

    def process_transformation(self, transformation):
        """
        Process all :attr:`items` in the scheduler's graph

        By default, the traversal is performed in topological order, which
        ensures that an item is processed before the items it depends upon
        (e.g., via a procedure call)
        This order can be reversed in the :any:`Transformation` manifest by
        setting :any:`Transformation.reverse_traversal` to ``True``.

        The scheduler applies the transformation to the scope corresponding to
        each item in the scheduler's graph, determined by the :any:`Item.scope_ir`
        property. For example, for a :any:`ProcedureItem`, the transformation is
        applied to the corresponding :any:`Subroutine` object.

        Optionally, the traversal can be performed on a source file level only,
        if the transformation has set :any:`Transformation.traverse_file_graph`
        to ``True``. This uses the :attr:`filegraph` to process the dependency tree.
        If combined with a :any:`Transformation.item_filter`, only source files with
        at least one object corresponding to an item of that type are processed.

        Parameters
        ----------
        transformation : :any:`Transformation`
            The transformation to apply over the dependency tree
        """
        def _get_definition_items(_item, sgraph_items):
            # For backward-compatibility with the DependencyTransform and LinterTransformation
            if not transformation.traverse_file_graph:
                return None

            # Recursively obtain all definition items but exclude any that are not part of the original SGraph
            items = ()
            for item in _item.create_definition_items(item_factory=self.item_factory, config=self.config):
                # Recursion gives us only items that are included in the SGraph, or the parent scopes
                # of items included in the SGraph
                child_items = _get_definition_items(item, sgraph_items)
                # If the current item has relevant children, or is included in the SGraph itself, we
                # include it in the list of items
                if child_items or item in sgraph_items:
                    if transformation.process_ignored_items or not item.is_ignored:
                        items += (item,) + child_items
            return items

        trafo_name = transformation.__class__.__name__
        log = f'[Loki::Scheduler] Applied transformation <{trafo_name}>' + ' in {:.2f}s'
        with Timer(logger=info, text=log):

            # Extract the graph iteration properties from the transformation
            item_filter = as_tuple(transformation.item_filter)
            if transformation.traverse_file_graph:
                sgraph = self.sgraph
                graph = sgraph.as_filegraph(
                    self.item_factory, self.config, item_filter=item_filter,
                    exclude_ignored=not transformation.process_ignored_items
                )
                sgraph_items = sgraph.items
                traversal = SFilter(
                    graph, reverse=transformation.reverse_traversal,
                    include_external=self.config.default.get('strict', True)
                )
            else:
                graph = self.sgraph
                sgraph_items = graph.items
                traversal = SFilter(
                    graph, item_filter=item_filter, reverse=transformation.reverse_traversal,
                    exclude_ignored=not transformation.process_ignored_items,
                    include_external=self.config.default.get('strict', True)
                )

            for _item in traversal:
                if isinstance(_item, ExternalItem):
                    raise RuntimeError(f'Cannot apply {trafo_name} to {_item.name}: Item is marked as external.')

                transformation.apply(
                    _item.scope_ir, role=_item.role, mode=_item.mode,
                    item=_item, targets=_item.targets, items=_get_definition_items(_item, sgraph_items),
                    successors=graph.successors(_item, item_filter=item_filter),
                    depths=graph.depths
                )

        if transformation.renames_items:
            self.rekey_item_cache()

        if transformation.creates_items:
            self._discover()

            self._parse_items()

    def callgraph(self, path, with_file_graph=False, with_legend=False):
        """
        Generate a callgraph visualization and dump to file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to write the callgraph figure to.
        with_filegraph : bool or str or pathlib.Path
            Visualize file dependencies in an additional file. Can be set to `True` or a file path to write to.
        with_legend : bool
            Add a key/legend to the plot. Can be enabled by setting the argument to `True`.
        """
        try:
            import graphviz as gviz  # pylint: disable=import-outside-toplevel
        except ImportError:
            warning('[Loki] Failed to load graphviz, skipping callgraph generation...')
            return

        item_colors = {
            FileItem: '#c0c0c0',       # gray
            ModuleItem: '#2080ff',     # blue
            ProcedureItem: '#60e080',  # green
            TypeDefItem: '#ffc832',    # yellow
            InterfaceItem: '#c0ff40',  # light-green
            ProcedureBindingItem: '#00dcc8', # turquoise
            ExternalItem: '#dc2000'    # red
        }

        cg_path = Path(path)
        callgraph = gviz.Digraph(format='pdf', strict=True, graph_attr=(('rankdir', 'LR'),))

        node_style = {
            'color': 'black',
            'shape': 'box',
            'style': 'filled'
        }

        # Insert all nodes in the schedulers graph
        for item in self.items:
            style = node_style.copy()
            alpha_channel = '33' if item.is_ignored else 'ff'
            style['fillcolor'] = item_colors.get(type(item), '#333333') + alpha_channel
            if item.replicate:
                style['shape'] = 'diamond'
                style['style'] += ',rounded'
            callgraph.node(item.name.upper(), **style)

        # Insert all edges in the schedulers graph
        for parent, child in self.dependencies:
            callgraph.edge(parent.name.upper(), child.name.upper())

        # Insert all nodes we were told to either block or ignore
        for item in self.items:
            blocked_children = set(item.targets_and_blocked_targets) - set(item.targets)
            for child in blocked_children:
                style = node_style.copy()
                style['fillcolor'] = '#ff141499'  # light red
                callgraph.node(child.upper(), **style)
                callgraph.edge(item.name.upper(), child.upper())

        if with_legend:
            for cls, color in item_colors.items():
                style = node_style.copy()
                style['fillcolor'] = color
                callgraph.node(cls.__name__, **style)

        try:
            callgraph.render(cg_path, view=False)
        except gviz.ExecutableNotFound as e:
            warning(f'[Loki] Failed to render callgraph due to graphviz error:\n  {e}')

        if with_file_graph:
            if with_file_graph is True:
                fg_path = cg_path.with_name(f'{cg_path.stem}_file_graph{cg_path.suffix}')
            else:
                fg_path = Path(with_file_graph)
            fg = gviz.Digraph(format='pdf', strict=True, graph_attr=(('rankdir', 'LR'),))
            file_graph = self.file_graph

            basedir = commonpath(item.name for item in file_graph.items)
            name_offset = len(basedir) + 1 if len(basedir) > 0 else 0

            for item in file_graph:
                style = node_style.copy()
                alpha_channel = '33' if item.is_ignored else 'ff'
                style['fillcolor'] = item_colors.get(type(item), '#333333') + alpha_channel
                if item.replicate:
                    style['shape'] = 'diamond'
                    style['style'] += ',rounded'
                fg.node(str(item.name)[name_offset:], **style)

            for parent, child in file_graph.dependencies:
                fg.edge(str(parent.name)[name_offset:], str(child.name)[name_offset:])

            try:
                fg.render(fg_path, view=False)
            except gviz.ExecutableNotFound as e:
                warning(f'[Loki] Failed to render filegraph due to graphviz error:\n  {e}')

    @Timer(logger=perf, text='[Loki::Scheduler] Wrote CMake plan file in {:.2f}s')
    def write_cmake_plan(self, filepath, mode, buildpath, rootpath):
        """
        Generate the "plan file" for CMake

        The plan file is a CMake file defining three lists:

        * ``LOKI_SOURCES_TO_TRANSFORM``: The list of files that are
          processed in the dependency graph
        * ``LOKI_SOURCES_TO_APPEND``: The list of files that are created
          and have to be added to the build target as part of the processing
        * ``LOKI_SOURCES_TO_REMOVE``: The list of files that are no longer
          required (because they have been replaced by transformed files) and
          should be removed from the build target.

        These lists are used by the CMake wrappers to schedule the source
        updates and update the source lists of the CMake target object accordingly.
        """
        info(f'[Loki] Scheduler writing CMake plan: {filepath}')

        rootpath = None if rootpath is None else Path(rootpath).resolve()
        buildpath = None if buildpath is None else Path(buildpath)
        sources_to_append = []
        sources_to_remove = []
        sources_to_transform = []

        # Filter the SGraph to get a pure call-tree
        item_filter = None if self.config.enable_imports else ProcedureItem
        for item in SFilter(self.sgraph, item_filter=item_filter):
            if item.is_ignored:
                continue

            sourcepath = item.path.resolve()
            newsource = sourcepath.with_suffix(f'.{mode.lower()}.F90')
            if buildpath:
                newsource = buildpath/newsource.name

            # Make new CMake paths relative to source again
            if rootpath is not None:
                sourcepath = sourcepath.relative_to(rootpath)

            debug(f'Planning:: {item.name} (role={item.role}, mode={mode})')

            # Inject new object into the final binary libs
            if newsource not in sources_to_append:
                sources_to_transform += [sourcepath]
                if item.replicate:
                    # Add new source file next to the old one
                    sources_to_append += [newsource]
                else:
                    # Replace old source file to avoid ghosting
                    sources_to_append += [newsource]
                    sources_to_remove += [sourcepath]

        with Path(filepath).open('w') as f:
            s_transform = '\n'.join(f'    {s}' for s in sources_to_transform)
            f.write(f'set( LOKI_SOURCES_TO_TRANSFORM \n{s_transform}\n   )\n')

            s_append = '\n'.join(f'    {s}' for s in sources_to_append)
            f.write(f'set( LOKI_SOURCES_TO_APPEND \n{s_append}\n   )\n')

            s_remove = '\n'.join(f'    {s}' for s in sources_to_remove)
            f.write(f'set( LOKI_SOURCES_TO_REMOVE \n{s_remove}\n   )\n')
