from pathlib import Path
from collections import deque
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

from loki import (as_tuple, debug, info, warning, error,
                  FortranSourceFile, FindNodes, Call)


class WorkItem(object):
    """
    A single work item that represents a single source routine to be
    processed. Each work item spawns new work items according to its
    own subroutine calls and the scheduler's blacklist.

    Note: Each work item may have its own configuration settings that
    primarily inherit values from the 'default', but can be
    specialised explicitly in the config file.
    """

    def __init__(self, name, config, source_path, graph=None):
        self.name = name
        self.routine = None
        self.source_file = None
        self.graph = graph

        # Generate item-specific config settings
        self.config = config['default'].copy()
        if name in config['routines']:
            self.config.update(config['routines'][name])

        if source_path.exists():
            try:
                # Read and parse source file and extract subroutine
                self.source_file = FortranSourceFile(source_path, preprocess=True)
                self.routine = self.source_file.subroutines[0]

            except Exception as e:
                if self.graph:
                    self.graph.node(self.name.upper(), color='red', style='filled')

                warning('Could not parse %s:' % source)
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


class Scheduler(object):
    """
    Work queue manager to enqueue and process individual source
    routines/modules with a given kernel.

    Note: The processing module can create a callgraph and perform
    automated discovery, to enable easy bulk-processing of large
    numbers of source files.
    """

    _deadlist = ['dr_hook', 'abor1']

    def __init__(self, path, config=None, kernel_map=None):
        self.path = Path(path)
        self.config = config
        self.kernel_map = kernel_map
        # TODO: Remove; should be done per item
        self.blacklist = []

        self.queue = deque()
        self.processed = []
        self.item_map = {}

        if Digraph is not None:
            self.graph = Digraph(format='pdf', strict=True)
        else:
            self.graph = None

    @property
    def routines(self):
        return list(self.processed) + list(self.queue)

    def append(self, sources):
        """
        Add names of source routines or modules to find and process.
        """
        for source in as_tuple(sources):
            if source in self.item_map:
                continue
            source_path = (self.path / source).with_suffix('.F90')
            item = WorkItem(name=source, config=self.config,
                            source_path=source_path, graph=self.graph)
            self.queue.append(item)
            self.item_map[source] = item

    def process(self, discovery=False):
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

                    # Append newly created edge to graph
                    if self.graph:
                        if child not in [r.name for r in self.processed]:
                            self.graph.node(child.upper(), color='black',
                                            fillcolor='lightblue', style='filled')
                        self.graph.edge(item.name.upper(), child.upper())

            # Process worl item with appropriate kernel
            mode = item.config['mode']
            role = item.config['role']
            kernel = self.kernel_map[mode][role]
            if kernel is not None:
                kernel(item.source_file, config=self.config, processor=self)

            # Finally mark item as processed in list and graph
            self.processed.append(item)

            if self.graph:
                if item.name in item.config['whitelist']:
                    self.graph.node(item.name.upper(), color='black', shape='diamond',
                                    fillcolor='limegreen', style='rounded,filled')
                else:
                    self.graph.node(item.name.upper(), color='black',
                                    fillcolor='limegreen', style='filled')
