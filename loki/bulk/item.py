import re
from functools import cached_property
from pathlib import Path

from loki.tools import as_tuple, CaseInsensitiveDict
from loki.visitors import FindNodes
from loki.sourcefile import Sourcefile
from loki.ir import CallStatement


__all__ = ['Item']


_re_call = re.compile(r'^\s*call\s+(?P<routine>[a-zA-Z0-9_% ]+)', re.IGNORECASE | re.MULTILINE)
_re_subroutine = re.compile(r'subroutine\s+(?P<routine>\w+)(?P<body>.*?)end\s+subroutine\s+(?=\1)', re.IGNORECASE | re.DOTALL)


class Item:
    """
    A work item that represents a single source routine to be
    processed. Each :any:`Item` spawns new work items according to its
    own subroutine calls and can be configured to ignore individual
    sub-tree.

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
    * ``ignore``: Individual list of subroutine calls to "ignore" during expansion.
      The routines will still be added to the schedulers tree, but not
      followed during expansion.
    * ``enrich``: List of subroutines that should still be looked up and used to "enrich"
      :any:`CallStatement` nodes in this :any:`Item` for inter-procedural
      transformation passes.
    * ``block``: List of subroutines that should should not be added to the scheduler
      tree. Note, these might still be shown in the graph visulisation.

    """

    def __init__(self, name, path, config=None, source_cache=None, build_args=None):
        self.name = name
        self.path = path
        self.config = config or {}

        # Private constructor arguments for delayed sourcefile creation
        self._source_cache = source_cache
        self._build_args = build_args

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

    @cached_property
    def source_string(self):
        """
        The original source string as read from file. This property is cached.

        This is primarily used for establishing dependency trees via
        regexes during the initial planning stage.
        """
        with self.path.open(encoding='latin1') as f:
            source = f.read()
        return source

    @cached_property
    def _re_subroutines(self):
        """
        A :any:`CaseInsensitiveDict` matching subroutine names to
        their body as determined by fast regex-matching.

        This is intended for fast subroutine detection without triggering full frontend parsers.
        """
        return CaseInsensitiveDict(_re_subroutine.findall(self.source_string))

    @property
    def _re_subroutine_calls(self):
        """
        A :any:`tuple` of strings with subroutine calls in the
        :any:`Item`'s associated subroutine.
        """
        body = self._re_subroutines[self.name]
        return tuple(r.replace(' ', '') for r in _re_call.findall(body))

    @property
    def _re_subroutine_members(self):
        """
        A :any:`tuple` of strings with names of member subroutines
        contained in the :any:`Item`'s associated subroutine.
        """
        body = self._re_subroutines[self.name]
        return tuple(r for r, _ in _re_subroutine.findall(body))

    @cached_property
    def routine(self):
        """
        :any:`Subroutine` object that this :any:`Item` encapsulates for processing.

        Note that this property is cached, so that updating the name of an associated
        :any:`Subroutine` with (eg. via the :any:`DependencyTransformation`) may not
        break the association with this :any:`Item`.
        """
        return self.source[self.name]

    @cached_property
    def source(self):
        """
        :any:`Sourcefile` that this :any:`Item` encapsulates for processing.

        Note that this property is cached, so that we may defer the creation of the
        :any:`Sourcefile` to the processing stage, as it triggers the potentially
        costly full frontend parser.
        """
        if self.path in self._source_cache:
            return self._source_cache[self.path]

        # Parse the sourcefile with build options and store in cache
        source = Sourcefile.from_file(filename=self.path, **self._build_args)
        self._source_cache[self.path]= source

        return source

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
        members = [str(m).lower() for m in self._re_subroutine_members]
        disabled = as_tuple(str(b).lower() for b in self.disable)

        # Base definition of child is a procedure call (for now)
        children = as_tuple(str(call).lower() for call in self._re_subroutine_calls)

        # Filter out local members and disabled sub-branches
        children = [c for c in children if c not in members]
        children = [c for c in children if c not in disabled]
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
