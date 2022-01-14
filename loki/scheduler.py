from pathlib import Path
from collections import deque, OrderedDict
import networkx as nx

from loki.build import Obj
from loki.frontend import FP
from loki.ir import CallStatement
from loki.visitors import FindNodes
from loki.sourcefile import Sourcefile
from loki.dimension import Dimension
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.logging import info, warning, error, debug


__all__ = ['Item', 'Scheduler', 'SchedulerConfig']


class SchedulerConfig:
    """
    Configuration object for the transformation :any:`Scheduler` that
    encapsulates default behaviour and item-specific behaviour. Can
    be create either from a raw dictionary or configration file.

    Parameters
    ----------
    default : dict
        Default options for each item
    routines : dict of dicts or list of dicts
        Dicts with routine-specific options.
    dimensions : dict of dicts or list of dicts
        Dicts with options to define :any`Dimension` objects.
    disable : list of str
        Subroutine names that are entirely disabled and will not be
        added to either the callgraph that we traverse, nor the
        visualisation. These are intended for utility routines that
        pop up in many routines but can be ignored in terms of program
        control flow, like ``flush`` or ``abort``.
    """

    def __init__(self, default, routines, disable=None, dimensions=None):
        self.default = default
        if isinstance(routines, dict):
            self.routines = CaseInsensitiveDict(routines)
        else:
            self.routines = CaseInsensitiveDict((r.name, r) for r in as_tuple(routines))
        self.disable = as_tuple(disable)
        self.dimensions = dimensions

    @classmethod
    def from_dict(cls, config):
        default = config['default']
        if 'routine' in config:
            config['routines'] = OrderedDict((r['name'], r) for r in config.get('routine', []))
        else:
            config['routines'] = []
        routines = config['routines']
        disable = default.get('disable', None)

        # Add any dimension definitions contained in the config dict
        dimensions = {}
        if 'dimension' in config:
            dimensions = [Dimension(**d) for d in config['dimension']]
            dimensions = {d.name: d for d in dimensions}

        return cls(default=default, routines=routines, disable=disable, dimensions=dimensions)

    @classmethod
    def from_file(cls, path):
        import toml  # pylint: disable=import-outside-toplevel
        # Load configuration file and process options
        with Path(path).open('r') as f:
            config = toml.load(f)

        return cls.from_dict(config)


class Item:
    """
    A work item that represents a single source routine to be
    processed. Each :any:`Item` spawns new work items according to its
    own subroutine calls and can be configured to ignore individual
    sub-tree.

    Each work item may have its own configuration settings that
    primarily inherit values from the `'default'`, but can be
    specialised explicitly in the config file or dictionary.

    Config markers
    ==============

    * ``role``: Role string to pass to the :any:`Transformation` (eg. "kernel")
    * ``mode``: Transformation "mode" to pass to the transformation
    * ``expand``: Flag to generally enable/disable expansion under this item
    * ``strict``: Flag controlling whether to strictly fail if source file cannot be parsed
    * ``replicated``: Flag indicating whether to mark item as "replicated" in call graphs
    * ``ignore``: Individual list of subroutine calls to "ignore" during expansion.
      The routines will still be added to the schedulers tree, but not
      followed during expansion.
    * ``enrich``: List of subroutines that should still be looked up and used to "enrich"
      :any:`CallStatement` nodes in this :any:`Item` for inter-procedural
      transformation passes.
    * ``block``: List of subroutines that should should not be added to the scheduler
      tree. Note, these might still be shown in the graph visulisation.

    Parameters
    ----------
    name : str
        Name to identify items in the schedulers graph
    path : path
        Filepath to the underlying source file
    config : dict
        Dict of item-specific config markers
    build_args : dict
        Dict of build arguments to pass to ``SourceFile.from_file`` constructors
    """

    def __init__(self, name, path, config=None, build_args=None):
        # Essential item attributes
        self.name = name
        self.path = path
        self.source = None
        self.routine = None

        self.config = config or {}
        self.build_args = build_args or {}

        if path and path.exists():
            try:
                # Read and parse source file and extract subroutine
                self.source = Sourcefile.from_file(path, **self.build_args)
                self.routine = self.source[self.name]

            except Exception as excinfo:  # pylint: disable=broad-except
                warning('Could not parse %s:', path)
                if self.strict:
                    raise excinfo
                error(excinfo)

        else:
            info("Could not find source file %s; skipping...", name)

    def __repr__(self):
        return f'loki.scheduler.Item<{self.name}>'

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.name.lower() == other.name.lower()
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name)

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
        Set of all child routines that this work item has in the call tree.

        Note that this is not the set of active children that a traversal
        will apply a transformation over, but rather the set of nodes that
        defines the next level of the internal call tree.
        """
        members = [m.name.lower() for m in as_tuple(self.routine.members)]
        disabled = as_tuple(str(b).lower() for b in self.disable)

        # Base definition of child is a procedure call (for now)
        children = as_tuple(str(call.name).lower() for call in FindNodes(CallStatement).visit(self.routine.ir))

        # Filter out local members and disabled sub-branches
        children = [c for c in children if c not in members]
        children = [c for c in children if c not in disabled]
        return as_tuple(children)

    @property
    def targets(self):
        """
        Set of "active" child routines that are part of the transformation
        traversal.

        This defines all child routines of an item that will be traversed
        when applying ``Transformation``s as well, after tree pruning rules
        are applied.
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


class Scheduler:
    """
    Work queue manager to enqueue and process individual `Item`
    routines/modules with a given kernel.

    Note: The processing module can create a callgraph and perform
    automated discovery, to enable easy bulk-processing of large
    numbers of source files.

    Parameters
    ----------
    paths : str or list of str
        List of paths to search for automated source file detection.
    config : dict or str, optional
        Configuration dict or path to scheduler configuration file
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
    frontend : :any:`Frontend`, optional
        Frontend to use when parsing source files (default :any:`FP`).
    """

    # TODO: Should be user-definable!
    source_suffixes = ['.f90', '_mod.f90']

    def __init__(self, paths, config=None, preprocess=False, includes=None,
                 defines=None, definitions=None, xmods=None, omni_includes=None,
                 frontend=FP):
        # Derive config from file or dict
        if isinstance(config, SchedulerConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = SchedulerConfig.from_file(config)
        else:
            self.config = SchedulerConfig.from_dict(config)

        # Build-related arguments to pass to the sources
        self.paths = [Path(p) for p in as_tuple(paths)]

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
        self.item_graph = nx.DiGraph()
        self.item_map = {}

        # Scan all source paths and create light-weight `Obj` objects for each file.
        obj_list = []
        for path in self.paths:
            for ext in Obj._ext:
                obj_list += [Obj(source_path=f) for f in path.glob(f'**/*{ext}')]

        # Create a map of all potential target routines for fast lookup later
        self.obj_map = CaseInsensitiveDict((r, obj) for obj in obj_list for r in as_tuple(obj.subroutines))

    @property
    def routines(self):
        return [item.routine for item in self.item_graph.nodes]

    @property
    def items(self):
        """
        All `Items` contained in the `Scheduler`s call graph.
        """
        return as_tuple(self.item_graph.nodes)

    @property
    def dependencies(self):
        """
        All individual pairs of `Item`s that represent a dependency
        and form an edge in the `Scheduler`s call graph.
        """
        return as_tuple(self.item_graph.edges)

    def find_path(self, routine):
        """
        Find path of file containing a given routine from the internal `obj_cache`.

        :param routine: Name of the source routine to locate.
        """
        if routine in self.obj_map:
            return self.obj_map[routine].source_path

        raise FileNotFoundError(f'Source path not found for routine: {routine}')

    def create_item(self, source):
        """
        Create an `Item` by looking up the path and setting all inferred properties.

        If the item cannot be created due to unknown source files, and the default
        configuration does not force ``strict`` behaviour, ``None`` is returned.

        Note that this takes a `SchedulerConfig` object for default options and an
        item-specific dict with override options, as well as given attributes that
        might be forced on this item from its parent.
        """
        if source in self.item_map:
            return self.item_map[source]

        # Use default as base and overrid individual options
        item_conf = self.config.default.copy()
        if source in self.config.routines:
            item_conf.update(self.config.routines[source])

        name = item_conf.pop('name', source)
        try:
            path = self.find_path(source)
        except FileNotFoundError as fnferr:
            warning(f'Scheduler could not create item: {source}')
            if self.config.default['strict']:
                raise fnferr
            return None

        debug(f'[Loki] Scheduler creating item: {name} => {path}')
        return Item(name=name, config=item_conf, path=path, build_args=self.build_args)

    def populate(self, routines):
        """
        Populate the callgraph of this scheduler through automatic expansion of
        subroutine-call induced dependencies from a set of starting routines.

        :param routines: Names of root routines from which to populate the callgraph.
        """
        queue = deque()
        for routine in as_tuple(routines):
            item = self.create_item(routine)
            if item:
                queue.append(item)

                self.item_map[routine] = item
                self.item_graph.add_node(item)

        while len(queue) > 0:
            item = queue.popleft()

            for c in item.children:
                child = self.create_item(c)

                if child is None:
                    continue

                # Skip blocked children as well
                if child in item.block:
                    continue

                # Append child to work queue if expansion is configured
                if item.expand:
                    # Do not propagate to dependencies marked as "ignore"
                    # Note that, unlike blackisted items, "ignore" items
                    # are still marked as targets during bulk-processing,
                    # so that calls to "ignore" routines will be renamed.
                    if child in item.ignore:
                        continue

                    if child not in self.item_map:
                        queue.append(child)
                        self.item_map[child.name] = child
                        self.item_graph.add_node(child)

                    self.item_graph.add_edge(item, child)

        # Enrich subroutine calls for inter-procedural transformations
        for item in self.item_graph:
            item.routine.enrich_calls(routines=self.routines)

            # Enrich item with meta-info from outside of the callgraph
            for routine in item.enrich:
                try:
                    path = self.find_path(routine)
                except FileNotFoundError as err:
                    warning(f'Scheduler could not find file for enrichment:\n{path}')
                    if self.config.default['strict']:
                        raise err
                    continue
                source = Sourcefile.from_file(path, **self.build_args)
                item.routine.enrich_calls(source.all_subroutines)

    def process(self, transformation):
        """
        Process all enqueued source modules and routines with the
        stored kernel. The traversal is performed in topological
        order, which ensures that :class:`CallStatement`s are always processed
        before their target :class:`Subroutine`s.
        """
        for item in nx.topological_sort(self.item_graph):

            # Process work item with appropriate kernel
            transformation.apply(item.source, role=item.role, mode=item.mode,
                                 item=item, targets=item.targets)

    def callgraph(self, path):
        """
        Generate a callgraph visualization and dump to file.

        :param path: Path to write the callgraph figure to.
        """
        import graphviz as gviz  # pylint: disable=import-outside-toplevel

        cg_path = Path(path)
        callgraph = gviz.Digraph(format='pdf', strict=True)

        # Insert all nodes in the schedulers graph
        for item in self.items:
            if item.replicate:
                callgraph.node(item.name.upper(), color='black', shape='diamond',
                                fillcolor='limegreen', style='rounded,filled')
            else:
                callgraph.node(item.name.upper(), color='black', shape='box',
                                fillcolor='limegreen', style='filled')

        # Insert all edges in the schedulers graph
        for parent, child in self.dependencies:
            callgraph.edge(parent.name.upper(), child.name.upper())  # pylint: disable=no-member

        # Insert all nodes we were told to either block or ignore
        for item in self.items:
            for child in item.children:
                if child in item.block:
                    callgraph.node(child.upper(), color='black', shape='box',
                                   fillcolor='orangered', style='filled')
                    callgraph.edge(item.name.upper(), child.upper())

            for child in item.children:
                if child in item.ignore:
                    callgraph.node(child.upper(), color='black', shape='box',
                                   fillcolor='lightblue', style='filled')
                    callgraph.edge(item.name.upper(), child.upper())

        callgraph.render(cg_path, view=False)
