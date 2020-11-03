from pathlib import Path
from collections import deque, OrderedDict
import networkx as nx

from loki.build import Obj
from loki.frontend import FP
from loki.ir import CallStatement
from loki.visitors import FindNodes
from loki.sourcefile import SourceFile
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.logging import info, warning, error


__all__ = ['Item', 'Scheduler']


class SchedulerConfig:
    """
    Configuration object for the transformation `Scheduler` that
    encapsulates default behaviour and item-specific behaviour.
    Cna be create eitehr froma raw dictionary or TOML, configration
    file.

    :param defaults: Dict of default options for each item
    :param routines: List of routine-specific option dicts.
    :param block: List or subroutine names that are blocked from the tree.
                  Note that these will still be shown in the visualisation
                  of the callgraph.
    :param replicate: List or subroutine names that need to be replicated.
                      Note, this only sets a flag for external build systems
                      to align source injection mechanics.
    :param disbale: List or subroutine names that are entirely disabled and
                    will not be added to either the callgraph that we traverse,
                    nor the visualisation. These are intended for utility routines
                    that pop up in many routines but can be ignored in terms of
                    program control flow, like `flush` or `abort`.
    """

    def __init__(self, default, routines, disable=None):
        self.default = default
        if isinstance(routines, dict):
            self.routines = CaseInsensitiveDict(routines)
        else:
            self.routines = CaseInsensitiveDict((r.name, r) for r in as_tuple(routines))
        self.disable = as_tuple(disable)

    @classmethod
    def from_dict(cls, config):
        default = config['default']
        if 'routine' in config:
            config['routines'] = OrderedDict((r['name'], r) for r in config.get('routine', []))
        routines = config['routines']
        disable = default.get('disable', None)
        return cls(default=default, routines=routines, disable=disable)

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
    processed. Each `Item` spawns new work items according to its own
    subroutine calls and can be configured to ignore individual sub-tree.

    Note: Each work item may have its own configuration settings that
    primarily inherit values from the 'default', but can be
    specialised explicitly in the config file or dictionary.

    :param name: Name to identify items in the schedulers graph.
    :param role: Role string to pass to the `Transformation` (eg. "kernel")
    :param expand: Flag to generally enable/disable expansion under this node.
    :param ignore: Individual list of subroutine calls to "ignore" during expansion.
                   The routines will still be added to the schedulers tree, but not
                   followed during expansion.
    :param enrich: List of subroutines that should still be searched for and used to
                   "enrich" `CallStatement` nodes in this `Item` for inter-procedural
                   transformation passes.
    :param block: List of subroutines that should should not be added to the scheduler
                  tree. Note, these might still be shown in the graph visulisation.
    """

    def __init__(self, name, path, role, mode=None, expand=True, strict=True,
                 ignore=None, enrich=None, block=None, replicate=False, xmods=None,
                 includes=None, builddir=None, definitions=None, frontend=FP):
        # Essential item attributes
        self.name = name
        self.path = path
        self.source = None
        self.routine = None

        # Item-specific markers and attributes
        self.role = role
        self.mode = mode
        self.expand = expand
        self.strict = strict
        self.replicate = replicate

        # Lists of subtree markers
        self.ignore = as_tuple(ignore)
        self.enrich = as_tuple(enrich)
        self.block = as_tuple(block)

        if path.exists():
            try:
                # Read and parse source file and extract subroutine
                self.source = SourceFile.from_file(path, preprocess=True,
                                                   xmods=xmods, includes=includes,
                                                   builddir=builddir,
                                                   definitions=definitions, frontend=frontend)
                self.routine = self.source[self.name]

            except Exception as excinfo:  # pylint: disable=broad-except
                warning('Could not parse %s:', path)
                if self.strict:
                    raise excinfo
                error(excinfo)

        else:
            info("Could not find source file %s; skipping...", name)

    def __repr__(self):
        return 'loki.scheduler.Item<{}>'.format(self.name)

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.name.lower() == other.name.lower()
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name)

    @property
    def children(self):
        """
        Set of all child routines that this work item calls.
        """
        members = [m.name.lower() for m in as_tuple(self.routine.members)]
        calls = tuple(call.name.lower() for call in FindNodes(CallStatement).visit(self.routine.ir))
        return tuple(call for call in calls if call not in members)


class Scheduler:
    """
    Work queue manager to enqueue and process individual `Item`
    routines/modules with a given kernel.

    Note: The processing module can create a callgraph and perform
    automated discovery, to enable easy bulk-processing of large
    numbers of source files.

    :param paths: List of locations to search for source files.
    """

    # TODO: Should be user-definable!
    source_suffixes = ['.f90', '_mod.f90']

    def __init__(self, paths, config=None, xmods=None, includes=None,
                 builddir=None, definitions=None, frontend=FP):
        # Derive config from file or dict
        if isinstance(config, SchedulerConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = SchedulerConfig.from_file(config)
        else:
            self.config = SchedulerConfig.from_dict(config)

        # Build-related arguments to pass to the sources
        self.paths = [Path(p) for p in as_tuple(paths)]
        self.xmods = xmods
        self.includes = includes
        self.builddir = builddir
        self.definitions = definitions
        self.frontend = frontend

        # Internal data structures to store the callgraph
        self.item_graph = nx.DiGraph()
        self.item_map = {}

        # Scan all source paths and create light-weight `Obj` objects for each file.
        obj_list = []
        for path in self.paths:
            for ext in Obj._ext:
                obj_list += [Obj(source_path=f) for f in path.glob('**/*%s' % ext)]

        # Create a map of all potential target routines for fast lookup later
        self.obj_map = CaseInsensitiveDict((r, obj) for obj in obj_list for r in obj.subroutines)

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

        raise RuntimeError("Source path not found for routine: %s" % routine)

    def create_item(self, source):
        """
        Create an `Item` by looking up the path and setting all inferred properties.

        Note that this takes a `SchedulerConfig` object for default options and an
        item-specific dict with override options, as well as given attributes that
        might be forced on this item from its paretn (like "is_ignored").
        """
        if source in self.item_map:
            return self.item_map[source]

        # Use default as base and overrid individual options
        item_conf = self.config.default.copy()
        if source in self.config.routines:
            item_conf.update(self.config.routines[source])

        name = item_conf.pop('name', source)
        return Item(name=name, **item_conf, path=self.find_path(source),
                    xmods=self.xmods,
                    includes=self.includes, definitions=self.definitions,
                    builddir=self.builddir, frontend=self.frontend)

    def populate(self, routines):
        """
        Populate the callgraph of this scheduler through automatic expansion of
        subroutine-call induced dependencies from a set of starting routines.

        :param routines: Names of root routines from which to populate the callgraph.
        """
        queue = deque()
        for routine in as_tuple(routines):
            item = self.create_item(routine)
            queue.append(item)

            self.item_map[routine] = item
            self.item_graph.add_node(item)

        while len(queue) > 0:
            item = queue.popleft()

            for c in item.children:
                child = self.create_item(c)

                # Skip "disabled" items immediately
                if child in self.config.disable:
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
                path = self.find_path(routine)
                source = SourceFile.from_file(path, preprocess=True,
                                              xmods=self.xmods,
                                              includes=self.includes,
                                              builddir=self.builddir,
                                              definitions=self.definitions,
                                              frontend=self.frontend)
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
            transformation.apply(item.routine, role=item.role)

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
