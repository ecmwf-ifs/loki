from pathlib import Path
from collections import deque
import networkx as nx
import glob
try:
    import graphviz as gviz
except ImportError:
    gviz = None

from loki import (as_tuple, debug, info, warning, error,
                  FortranSourceFile, FindNodes, Call)


__all__ = ['Task', 'TaskScheduler']


class Task(object):
    """
    A work item that represents a single source routine or module to
    be processed. Each :class:`Task` spawns new work items according
    to its own subroutine calls and the scheduler's blacklist.

    Note: Each work item may have its own configuration settings that
    primarily inherit values from the 'default', but can be
    specialised explicitly in the config file.
    """

    def __init__(self, name, config, path, graph=None, typedefs=None):
        self.name = name
        self.path = path
        self.file = None
        self.routine = None
        self.graph = graph

        # Generate item-specific config settings
        self.config = config['default'].copy()
        if name in config['routines']:
            self.config.update(config['routines'][name])

        if path.exists():
            try:
                # Read and parse source file and extract subroutine
                self.file = FortranSourceFile(path, preprocess=True,
                                              typedefs=typedefs)
                # TODO: Modules should be first-class items too
                self.routine = self.file.subroutines[0]

            except Exception as e:
                if self.graph:
                    self.graph.node(self.name.upper(), color='red', style='filled')

                warning('Could not parse %s:' % path)
                if self.config['strict']:
                    raise e
                else:
                    error(e)

        else:
            if self.graph:
                self.graph.node(self.name.upper(), color='lightsalmon', style='filled')
            info("Could not find source file %s; skipping..." % name)


    @property
    def children(self):
        """
        Set of all child routines that this work item calls.
        """
        return tuple(call.name.lower() for call in FindNodes(Call).visit(self.routine.ir))

    def enrich(self, routines):
        self.routine.enrich_calls(routines=routines)


class TaskScheduler(object):
    """
    Work queue manager to enqueue and process individual :class:`Task`
    routines/modules with a given kernel.

    Note: The processing module can create a callgraph and perform
    automated discovery, to enable easy bulk-processing of large
    numbers of source files.

    :param paths: List of locations to search for source files.
    """

    _deadlist = ['dr_hook', 'abor1']

    def __init__(self, paths, config=None, typedefs=None):
        self.paths = [Path(p) for p in as_tuple(paths)]
        self.config = config
        self.typedefs = typedefs
        # TODO: Remove; should be done per item
        self.blacklist = []

        self.taskgraph = nx.DiGraph()

        self.queue = deque()
        self.processed = []
        self.item_map = {}

        if gviz is not None:
            self.graph = gviz.Digraph(format='pdf', strict=True)
        else:
            self.graph = None

    @property
    def routines(self):
        return [task.routine for task in self.taskgraph.nodes]

    def find_path(self, source):
        """
        Attempts to find a source file from a (no endings) routine
        name across all specified source locations.

        :param source: Name of the source routine to locate.
        """
        for path in self.paths:
            for suffix in ['.F90', '_mod.F90']:
                path_string = '%s/**/%s%s' % (str(path), source, suffix)
                locs = glob.glob(path_string, recursive=True)
                if len(locs) > 0:
                    source_path = Path(locs[0])
                    if source_path.exists():
                        return source_path

        raise RuntimeError("Source path not found: %s" % source)

    def append(self, sources):
        """
        Add names of source routines or modules to find and process.
        """
        for source in as_tuple(sources):
            if source in self.item_map:
                continue
            item = Task(name=source, config=self.config,
                        path=self.find_path(source),
                        graph=self.graph, typedefs=self.typedefs)
            self.queue.append(item)
            self.item_map[source] = item

            self.taskgraph.add_node(item)

    def populate(self):
        """
        Process all enqueued source modules and routines with the
        stored kernel.
        """

        while len(self.queue) > 0:
            item = self.queue.popleft()

            for child in item.children:
                # Skip "deadlisted" items immediately
                if child in self._deadlist:
                    continue

                # Mark blacklisted children in graph, but skip
                if child in item.config['blacklist']:
                    if self.graph:
                        self.graph.node(child.upper(), color='black',
                                        fillcolor='orangered', style='filled')
                        self.graph.edge(item.name.upper(), child.upper())

                    continue

                # Append child to work queue if expansion is configured
                if item.config['expand']:
                    self.append(child)

                    self.taskgraph.add_edge(item, self.item_map[child])

                    # Append newly created edge to graph
                    if self.graph:
                        if child not in [r.name for r in self.processed]:
                            self.graph.node(child.upper(), color='black',
                                            fillcolor='lightblue', style='filled')
                        self.graph.edge(item.name.upper(), child.upper())

        # Enrich subroutine calls for inter-procedural transformations
        for item in self.taskgraph:
            item.enrich(routines=self.routines)

    def process(self, transformation):
        """
        Process all enqueued source modules and routines with the
        stored kernel. The traversal is performed in topological
        order, which ensures that :class:`Call`s are always processed
        before their target :class:`Subroutine`s.
        """
        for task in nx.topological_sort(self.taskgraph):

            # Process work item with appropriate kernel
            transformation.apply(task.routine, task=task, processor=self)

            # Mark item as processed in list and graph
            self.processed.append(task)

            if self.graph:
                if task.name in task.config['whitelist']:
                    self.graph.node(task.name.upper(), color='black', shape='diamond',
                                    fillcolor='limegreen', style='rounded,filled')
                else:
                    self.graph.node(task.name.upper(), color='black',
                                    fillcolor='limegreen', style='filled')
