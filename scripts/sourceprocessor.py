class SourceProcessor(object):
    """
    Work queue manager to enqueue and process individual source
    routines/modules with a given kernel.

    Note: The processing module can create a callgraph and perform
    automated discovery, to enable easy bulk-processing of large
    numbers of source files.
    """

    blacklist = ['DR_HOOK', 'ABOR1']

    def __init__(self, path, config=None, kernel_map=None):
        self.path = Path(path)
        self.config = config
        self.kernel_map = kernel_map

        self.queue = deque()
        self.processed = []

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
        sources = as_tuple(sources)
        self.queue.extend(s for s in sources if s not in self.routines)

    def process(self, discovery=False):
        """
        Process all enqueued source modules and routines with the
        stored kernel.
        """
        while len(self.queue) > 0:
            source = self.queue.popleft()
            source_path = (self.path / source).with_suffix('.F90')

            if source_path.exists():
                try:
                    config = self.config['default'].copy()
                    if source in self.config['routines']:
                        config.update(self.config['routines'][source])

                    # Re-generate target routine and interface block with updated name
                    source_file = FortranSourceFile(source_path, preprocess=True)
                    routine = source_file.subroutines[0]

                    debug("Source: %s, config: %s" % (source, config))

                    if self.graph:
                        if routine.name.lower() in config['whitelist']:
                            self.graph.node(routine.name, color='black', shape='diamond',
                                            fillcolor='limegreen', style='rounded,filled')
                        else:
                            self.graph.node(routine.name, color='black',
                                            fillcolor='limegreen', style='filled')

                    for call in FindNodes(Call).visit(routine.ir):
                        # Yes, DR_HOOK is that(!) special
                        if self.graph and call.name not in ['DR_HOOK', 'ABOR1']:
                            self.graph.edge(routine.name, call.name)
                            if call.name.upper() in self.blacklist:
                                self.graph.node(call.name, color='black',
                                                fillcolor='orangered', style='filled')
                            elif call.name.lower() not in self.processed:
                                self.graph.node(call.name, color='black',
                                                fillcolor='lightblue', style='filled')

                        if call.name.upper() in self.blacklist:
                            continue

                        if config['expand']:
                            self.append(call.name.lower())

                    # Apply the user-defined kernel
                    kernel = self.kernel_map[config['mode']][config['role']]

                    if kernel is not None:
                        kernel(source_file, config=self.config, processor=self)

                    self.processed.append(source)

                except Exception as e:
                    if self.graph:
                        self.graph.node(source.upper(), color='red', style='filled')
                    warning('Could not parse %s:' % source)
                    if config['strict']:
                        raise e
                    else:
                        error(e)

            else:
                if self.graph:
                    self.graph.node(source.upper(), color='lightsalmon', style='filled')
                info("Could not find source file %s; skipping..." % source)
