import inspect
from itertools import groupby

from loki.ir import Node, Conditional
from loki.tools import flatten, is_iterable, as_tuple, JoinableStringList

__all__ = [
    'pprint', 'GenericVisitor', 'Visitor', 'Transformer', 'NestedTransformer', 'FindNodes',
    'Stringifier', 'MaskedTransformer', 'SequenceFinder', 'PatternFinder', 'is_parent_of',
    'is_child_of', 'NestedMaskedTransformer', 'FindScopes'
]


class GenericVisitor:

    """
    A generic visitor class, shamelessly copied from:
    https://github.com/opesci/devito/

    To define handlers, subclasses should define :data:`visit_Foo`
    methods for each class :data:`Foo` they want to handle.
    If a specific method for a class :data:`Foo` is not found, the MRO
    of the class is walked in order until a matching method is found.
    The method signature is:
        .. code-block::
           def visit_Foo(self, o, [*args, **kwargs]):
               pass
    The handler is responsible for visiting the children (if any) of
    the node :data:`o`.  :data:`*args` and :data:`**kwargs` may be
    used to pass information up and down the call stack.  You can also
    pass named keyword arguments, e.g.:
        .. code-block::
           def visit_Foo(self, o, parent=None, *args, **kwargs):
               pass
    """

    def __init__(self):
        handlers = {}
        # visit methods are spelt visit_Foo.
        prefix = "visit_"
        # Inspect the methods on this instance to find out which
        # handlers are defined.
        for (name, meth) in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith(prefix):
                continue
            # Check the argument specification
            # Valid options are:
            #    visit_Foo(self, o, [*args, **kwargs])
            argspec = inspect.getfullargspec(meth)
            if len(argspec.args) < 2:
                raise RuntimeError("Visit method signature must be "
                                   "visit_Foo(self, o, [*args, **kwargs])")
            handlers[name[len(prefix):]] = meth
        self._handlers = handlers

    """
    :attr:`default_args`. A dict of default keyword arguments for the visitor.
    These are not used by default in :meth:`visit`, however, a caller may pass
    them explicitly to :meth:`visit` by accessing :attr:`default_args`.
    For example::
        .. code-block::
           v = FooVisitor()
           v.visit(node, **v.default_args)
    """
    default_args = {}

    @classmethod
    def default_retval(cls):
        """
        A method that returns an object to use to populate return values.
        If your visitor combines values in a tree-walk, it may be useful to
        provide a object to combine the results into. :meth:`default_retval`
        may be defined by the visitor to be called to provide an empty object
        of appropriate type.
        """
        return None

    def lookup_method(self, instance):
        """Look up a handler method for a visitee.

        :param instance: The instance to look up a method for.
        """
        cls = instance.__class__
        try:
            # Do we have a method handler defined for this type name
            return self._handlers[cls.__name__]
        except KeyError:
            # No, walk the MRO.
            for klass in cls.mro()[1:]:
                entry = self._handlers.get(klass.__name__)
                if entry:
                    # Save it on this type name for faster lookup next time
                    self._handlers[cls.__name__] = entry
                    return entry
        raise RuntimeError("No handler found for class %s" % cls.__name__)

    def visit(self, o, *args, **kwargs):
        """
        Apply this :class:`Visitor` to an AST.

        :param o: The :class:`Node` to visit.
        :param args: Optional arguments to pass to the visit methods.
        :param kwargs: Optional keyword arguments to pass to the visit methods.
        """
        meth = self.lookup_method(o)
        return meth(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):  # pylint: disable=unused-argument
        return self.default_retval()


class Visitor(GenericVisitor):

    def visit_tuple(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        return self.visit(o.children, **kwargs)

    @staticmethod
    def reuse(o, *args, **kwargs):  # pylint: disable=unused-argument
        """A visit method to reuse a node, ignoring children."""
        return o

    def maybe_rebuild(self, o, *args, **kwargs):
        """A visit method that rebuilds nodes if their children have changed."""
        ops, okwargs = o.operands()
        new_ops = [self.visit(op, *args, **kwargs) for op in ops]
        if all(a is b for a, b in zip(ops, new_ops)):
            return o
        return o._rebuild(*new_ops, **okwargs)

    def always_rebuild(self, o, *args, **kwargs):
        """A visit method that always rebuilds nodes."""
        ops, okwargs = o.operands()
        new_ops = [self.visit(op, *args, **kwargs) for op in ops]
        return o._rebuild(*new_ops, **okwargs)


class Transformer(Visitor):

    """
    Given an Iteration/Expression tree T and a mapper from nodes in T to
    a set of new nodes L, M : N --> L, build a new Iteration/Expression tree T'
    where a node ``n`` in N is replaced with ``M[n]``.

    In the special case in which ``M[n]`` is None, ``n`` is dropped from T'.

    In the special case in which ``M[n]`` is an iterable of nodes,
    all nodes in ``M[n]`` are inserted into the tuple containing ``n``.

    :param dict mapper: the mapping M : N --> L.
    :param bool invalidate_source: if set to True and if ``M[n]`` has `source=None`,
        this triggers invalidating the source property for all nodes enclosing ``n``.
        Note that the source property is not explicitly invalidated for ``M[n]``.
    :param bool inplace: If set, all updated are performed on existing ``Node``
                         objects, keeping the original tree intact.
    """

    def __init__(self, mapper=None, invalidate_source=True, inplace=False):
        super().__init__()
        self.mapper = mapper.copy() if mapper is not None else {}
        self.invalidate_source = invalidate_source
        self.rebuilt = {}
        self.inplace = inplace

    def _rebuild_without_source(self, o, children, **args):
        """
        Rebuild the given node without the source property.
        """
        args_frozen = o.args_frozen
        args_frozen.update(args)
        if 'source' in o.args_frozen:
            args_frozen['source'] = None

        if self.inplace:
            # Updated nodes in place, if requested
            o._update(*children, **args_frozen)
            return o

        # Rebuild updated nodes by default
        return o._rebuild(*children, **args_frozen)

    def _rebuild(self, o, children, **args):
        """
        Rebuild the given node with the provided children.

        If `invalidate_source` is `True`, `source` is set to `None` whenever
        any of the children has `source == None`.
        """
        args_frozen = o.args_frozen
        args_frozen.update(args)
        if self.invalidate_source and 'source' in args_frozen:
            child_has_no_source = [getattr(i, 'source', None) is None for i in flatten(children)]
            if any(child_has_no_source) or len(child_has_no_source) != len(flatten(o.children)):
                return self._rebuild_without_source(o, children, **args_frozen)

        if self.inplace:
            # Updated nodes in place, if requested
            o._update(*children, **args_frozen)
            return o

        # Rebuild updated nodes by default
        return o._rebuild(*children, **args_frozen)

    def visit_object(self, o, **kwargs):
        return o

    def _inject_tuple_mapping(self, o):
        # For one-to-many mappings check iterables for the replacement
        # node and insert the sub-list/tuple into the list/tuple.
        for k, handle in self.mapper.items():
            if k in o and is_iterable(handle):
                i = o.index(k)
                o = o[:i] + tuple(handle) + o[i+1:]
        return o

    def visit_tuple(self, o, **kwargs):
        o = self._inject_tuple_mapping(o)
        visited = tuple(self.visit(i, **kwargs) for i in o)
        return tuple(i for i in visited if i is not None)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        if o in self.mapper:
            handle = self.mapper[o]
            if handle is None:
                # None -> drop /o/
                return None

            # For one-to-many mappings making sure this is not replaced again
            # as it has been inserted by visit_tuple already
            if not is_iterable(handle) or o not in handle:
                return handle._rebuild(**handle.args)

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)

    def visit(self, o, *args, **kwargs):
        obj = super().visit(o, *args, **kwargs)
        if isinstance(o, Node) and obj is not o:
            self.rebuilt[o] = obj
        return obj


class NestedTransformer(Transformer):
    """
    Unlike a :class:`Transformer`, a :class:`NestedTransforer` applies
    replacements in a depth-first fashion.
    """

    def visit_Node(self, o, **kwargs):
        rebuilt = [self.visit(i, **kwargs) for i in o.children]
        handle = self.mapper.get(o, o)
        if handle is None:
            # None -> drop /o/
            return None
        if is_iterable(handle):
            if not o.children:
                raise ValueError
            extended = [tuple(handle) + rebuilt[0]] + rebuilt[1:]
            if self.invalidate_source:
                return self._rebuild_without_source(o, extended)
            return o._rebuild(*extended, **o.args_frozen)
        return self._rebuild(handle, rebuilt)


class MaskedTransformer(Transformer):
    """
    An enriched :class:`Transformer` that allows to selectively activate and
    deactivate including nodes or subtrees in the produced tree.

    This can be used, e.g., to extract everything between two nodes. Or to
    create a copy of the entire tree but without all nodes between two nodes.

    Parameters `start` and `stop` are to be understood in a Pythonic way, i.e.,
    `start` will be included in the result and `stop` excluded.

    :param start: a node or list/set of nodes. When a node from this list is
        encountered, it and all subsequently traversed nodes are included in
        the produced tree until a node from the `stop` list is encountered.
    :param stop: a node or list/set of nodes. When a node from this list is
        encountered, it and all subsequently traversed nodes are excluded from
        the produced tree until a node from the `start` list is encountered.
    :param bool active: if set to `True` nodes are included in the produced tree
        from the beginning.
    :param bool require_all_start: if set to ``True``, nodes will be included
        in the produced tree only after all nodes in ``start`` have been
        encountered.
    :param bool greedy_stop: No more nodes will be added to the produced tree
        as soon as any of the nodes in ``stop`` has been encountered, not even
        if another node in ``start`` matches.

    NB: Enabling ``require_all_start`` and ``greedy_stop`` at the same time is
        useful, e.g., when one requires the minimum number of nodes in-between
        multiple start and end nodes without knowing in which order they appear.
    """

    def __init__(self, start=None, stop=None, active=False,
                 require_all_start=False, greedy_stop=False, **kwargs):
        super().__init__(**kwargs)

        self.start = set(as_tuple(start))
        self.stop = set(as_tuple(stop))
        self.active = active
        self.require_all_start = require_all_start
        self.greedy_stop = greedy_stop

    def visit(self, o, *args, **kwargs):
        # Vertical active status update
        if self.require_all_start:
            if o in self.start:
                # to record encountered nodes we remove them from the set of
                # start nodes and only if it is then empty we set active=True
                self.start.remove(o)
                self.active = self.active or not self.start
            else:
                self.active = self.active and o not in self.stop
        else:
            self.active = (self.active and o not in self.stop) or o in self.start
        if self.greedy_stop and o in self.stop:
            # to make sure that we don't include any following nodes we clear start
            self.start.clear()
            self.active = False
        return super().visit(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):
        if kwargs['parent_active']:
            # this is not an IR node but usually an expression tree or similar
            # we need to retain this only if the "parent" IR node is active
            return o
        return None

    def visit_Node(self, o, **kwargs):
        if o in self.mapper:
            return super().visit_Node(o, **kwargs)

        # pass to children if this node is active
        kwargs['parent_active'] = self.active
        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        if kwargs['parent_active']:
            return self._rebuild(o, rebuilt)
        return tuple(i for i in rebuilt if i is not None) or None


class NestedMaskedTransformer(MaskedTransformer):
    """
    Like :class:``MaskedTransformer`` but retains encapsulating code
    blocks (such as loops, conditionals, etc.).
    """

    # Handler for leaf nodes

    def visit_object(self, o, **kwargs):
        # need to keep them active here because inactive parents may still be
        # retained if other children turn on active status
        return o

    def visit_LeafNode(self, o, **kwargs):
        if o in self.mapper:
            return super().visit_Node(o, **kwargs)
        if not self.active:
            # because any section/scope nodes are treated separately we can
            # simply drop inactive nodes
            return None

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)

    # Handler for block nodes

    def visit_InternalNode(self, o, **kwargs):
        if o in self.mapper:
            return super().visit_Node(o, **kwargs)

        rebuilt = [self.visit(i, **kwargs) for i in o.children]
        body_index = o._traversable.index('body')

        if rebuilt[body_index]:
            rebuilt[body_index] = flatten(rebuilt[body_index])

        # check if body still exists, otherwise delete this node
        if not rebuilt[body_index]:
            return None
        return self._rebuild(o, rebuilt)

    def visit_Conditional(self, o, **kwargs):
        if o in self.mapper:
            return super().visit(o, **kwargs)

        condition = self.visit(o.condition, **kwargs)
        body = flatten(as_tuple(self.visit(o.body, **kwargs)))
        else_body = flatten(as_tuple(self.visit(o.else_body, **kwargs)))

        if not body:
            return else_body

        has_elseif = o.has_elseif and else_body and isinstance(else_body[0], Conditional)
        return self._rebuild(o, tuple((condition,) + (body,) + (else_body,)), has_elseif=has_elseif)

    def visit_MultiConditional(self, o, **kwargs):
        if o in self.mapper:
            return super().visit(o, **kwargs)

        # need to make (value, body) pairs to track vanishing bodies
        expr = self.visit(o.expr, **kwargs)
        branches = [(self.visit(c, **kwargs), self.visit(b, **kwargs))
                    for c, b in zip(o.values, o.bodies)]
        branches = [(c, b) for c, b in branches if flatten(as_tuple(b))]
        else_body = self.visit(o.else_body, **kwargs)

        # retain whatever is in the else body if all other branches are gone
        if not branches:
            return else_body

        # rebuild conditional with remaining branches
        values, bodies = zip(*branches)
        return self._rebuild(o, tuple((expr,) + (values,) + (bodies,) + (else_body,)))


class FindNodes(Visitor):

    @classmethod
    def default_retval(cls):
        return []

    """
    Find :class:`Node` instances.
    :param match: Pattern to look for.
    :param mode: Drive the search. Accepted values are: ::
        * 'type' (default): Collect all instances of type ``match``.
        * 'scope': Return the scope in which the object ``match`` appears.
    :param greedy: Do not recurse for children of a matched node.
    """

    rules = {
        'type': lambda match, o: isinstance(o, match),
        'scope': lambda match, o: match in flatten(o.children)
    }

    def __init__(self, match, mode='type', greedy=False):
        super().__init__()
        self.match = match
        self.rule = self.rules[mode]
        self.greedy = greedy

    def visit_object(self, o, **kwargs):
        ret = kwargs.get('ret')
        return ret or self.default_retval()

    def visit_tuple(self, o, **kwargs):
        ret = kwargs.pop('ret', self.default_retval())
        for i in o:
            ret = self.visit(i, ret=ret, **kwargs)
        return ret or self.default_retval()

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        ret = kwargs.pop('ret', self.default_retval())
        if self.rule(self.match, o):
            ret.append(o)
            if self.greedy:
                return ret
        for i in o.children:
            ret = self.visit(i, ret=ret, **kwargs)
        return ret or self.default_retval()


def is_child_of(node, other):
    """
    Return ``True`` if ``node`` is contained in the IR below ``other``,
    otherwise return ``False``.

    Note that this can be expensive for large subtrees.
    """
    return len(FindNodes(node, mode='scope', greedy=True).visit(other)) > 0


def is_parent_of(node, other):
    """
    Return ``True`` if ``other`` is contained in the IR below ``node``,
    otherwise return ``False``.

    Note that this can be expensive for large subtrees.
    """
    return len(FindNodes(other, mode='scope', greedy=True).visit(node)) > 0


class FindScopes(FindNodes):
    """
    Find all enclosing scopes for object ``match``.
    """
    def __init__(self, match, greedy=True):
        super().__init__(match=match, greedy=greedy)
        self.rule = lambda match, o: match is o

    def visit_Node(self, o, **kwargs):
        ret = kwargs.pop('ret', self.default_retval())
        ancestors = kwargs.pop('ancestors', []) + [o]

        if self.rule(self.match, o):
            ret.append(ancestors)
            if self.greedy:
                return ret

        for i in o.children:
            ret = self.visit(i, ret=ret, ancestors=ancestors, **kwargs)
        return ret or self.default_retval()


class SequenceFinder(Visitor):
    """
    Utility visitor that finds repeated nodes of the same type in
    lists/tuples within a given tree.
    """

    def __init__(self, node_type):
        super().__init__()
        self.node_type = node_type

    @classmethod
    def default_retval(cls):
        return []

    def visit_tuple(self, o, **kwargs):
        groups = []
        for c in o:
            # First recurse...
            subgroups = self.visit(c)
            if subgroups is not None and len(subgroups) > 0:
                groups += subgroups
        for t, group in groupby(o, type):
            # ... then add new groups
            g = tuple(group)
            if t is self.node_type and len(g) > 1:
                groups.append(g)
        return groups

    visit_list = visit_tuple


class PatternFinder(Visitor):
    """
    Utility visitor that finds a pattern of nodes given as tuple/list
    of types within a given tree.
    """

    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    @classmethod
    def default_retval(cls):
        return []

    @staticmethod
    def match_indices(pattern, sequence):
        """ Return indices of matched patterns in sequence. """
        matches = []
        for i, elem in enumerate(sequence):
            if elem == pattern[0]:
                if tuple(sequence[i:i+len(pattern)]) == tuple(pattern):
                    matches.append(i)
        return matches

    def visit_tuple(self, o, **kwargs):
        matches = []
        for c in o:
            # First recurse...
            submatches = self.visit(c)
            if submatches is not None and len(submatches) > 0:
                matches += submatches
        types = list(map(type, o))
        idx = self.match_indices(self.pattern, types)
        for i in idx:
            matches.append(o[i:i+len(self.pattern)])
        return matches

    visit_list = visit_tuple


class Stringifier(Visitor):
    """
    Visitor that converts a given IR tree to string.

    It serves as base class for backends and provides a number of helpful routines that
    ease implementing automatic recursion and line wrapping.

    :param int depth: the current level of indentation.
    :param str indent: the string to be prepended to a line for each level of indentation.
    :param int linewidth: the line width limit.
    :param line_cont: a function handle that takes the current indentation string and yields
                      the string that should be inserted inbetween lines when they need to
                      be wrapped to stay within the line width limit.
    :type line_cont: function expecting 1 str argument
    """

    # pylint: disable=arguments-differ

    def __init__(self, depth=0, indent='  ', linewidth=90, line_cont=lambda indent: '\n' + indent,
                 symgen=str):
        super().__init__()

        self.depth = depth
        self._indent = indent
        self.linewidth = linewidth
        self.line_cont = line_cont
        self._symgen = symgen

    @property
    def symgen(self):
        """
        Formatter for expressions.
        """
        return self._symgen

    @property
    def indent(self):
        """
        Yield indentation according to current depth.

        :rtype: str
        """
        return self._indent * self.depth

    @staticmethod
    def join_lines(*lines):
        """
        Combine multiple lines into a long string, inserting line breaks in between.
        Entries that are `None` are skipped.

        :param list lines: the list of lines to be combined.
        :return: the resulting string or `None` if no lines are given.
        :rtype: str or NoneType
        """
        if not lines:
            return None
        return '\n'.join(line for line in lines if line is not None)

    def join_items(self, items, sep=', ', separable=True):
        """
        Concatenate a list of items by creating a py:class:`JoinableStringList` object.

        The return value can be passed to `format_line` or `format_node` or converted to a string
        by simply calling `str` with it as an argument. Upon expansion, lines will be
        wrapped automatically to stay within the linewidth.

        :param list items: the list of strings to be joined.
        :param str sep: the separator to be inserted between items.
        :param bool separable: an indicator whether cosmetic line breaks between items are
                               permitted.

        :return: a py:class:`JoinableStringList` object for the items.
        :rtype: py:class:`JoinableStringList`
        """
        return JoinableStringList(items, sep=sep, width=self.linewidth,
                                  cont=self.line_cont(self.indent), separable=separable)

    def format_node(self, name, *items):
        """
        Default format for a node.

        Creates a string of the form `<name[, attribute, attribute, ...]>`.
        """
        if items:
            return self.format_line('<', name, ' ', self.join_items(items), '>')
        return self.format_line('<', name, '>')

    def format_line(self, *items, comment=None, no_wrap=False, no_indent=False):
        """
        Format a line by concatenating all items and applying indentation while observing
        the allowed line width limit.

        Note that the provided comment will simply be appended to the line and no line
        width limit will be enforced for that.

        :param list items: the items to be put on that line.
        :param str comment: an optional inline comment to be put at the end of the line.
        :param bool no_wrap: disable line wrapping.
        :param bool no_indent: do not apply indentation.

        :return: the string of the current line, potentially including line breaks if
                 required to observe the line width limit.
        :rtype: str
        """
        if not no_indent:
            items = [self.indent, *items]
        if no_wrap:
            # Simply concatenate items and append the comment
            line = ''.join(str(item) for item in items)
        else:
            # Use join_items to concatenate items
            line = str(self.join_items(items, sep=''))
        if comment:
            return line + comment
        return line

    def visit_all(self, item, *args, **kwargs):
        """
        Convenience function to call `visit` for all given items.
        If only one iterable argument is provided, `visit` is called on all of
        its elements.
        """
        if is_iterable(item) and not args:
            return as_tuple(self.visit(i, **kwargs) for i in item if i is not None)
        return as_tuple(self.visit(i, **kwargs) for i in [item, *args] if i is not None)

    # Handler for outer objects

    def visit_Module(self, o, **kwargs):
        """
        Format as
          <repr(Module)>
            ...spec...
            ...routines...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        spec = self.visit(o.spec, **kwargs)
        routines = self.visit(o.subroutines, **kwargs)
        self.depth -= 1
        return self.join_lines(header, spec, routines)

    def visit_Subroutine(self, o, **kwargs):
        """
        Format as
          <repr(Subroutine)>
            ...docstring...
            ...spec...
            ...body...
            ...members...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        docstring = self.visit(o.docstring, **kwargs)
        spec = self.visit(o.spec, **kwargs)
        body = self.visit(o.body, **kwargs)
        members = self.visit(o.members, **kwargs)
        self.depth -= 1
        return self.join_lines(header, docstring, spec, body, members)

    # Handler for AST base nodes

    def visit_Node(self, o, **kwargs):
        """
        Format as
          <repr(Node)>
        """
        return self.format_node(repr(o))

    def visit_Expression(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Dispatch routine to expression tree stringifier.
        """
        return self.symgen(o)

    def visit_tuple(self, o, **kwargs):
        """
        Recurse for each item in the tuple and return as separate lines.
        """
        lines = (self.visit(item, **kwargs) for item in o)
        return self.join_lines(*lines)

    visit_list = visit_tuple

    # Handler for IR nodes

    def visit_Section(self, o, **kwargs):
        """
        Format as
          <repr(Section)>
            ...body...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body)

    visit_Loop = visit_Section
    visit_WhileLoop = visit_Section
    visit_PragmaRegion = visit_Section

    def visit_Conditional(self, o, **kwargs):
        """
        Format as
          <repr(Conditional)>
            <If [condition]>
              ...
            <Else>
              ...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        conditions = [self.format_node('If', self.visit(o.condition, **kwargs))]
        if o.else_body:
            conditions.append(self.format_node('Else'))
        self.depth += 1
        bodies = self.visit_all(o.body, o.else_body, **kwargs)
        self.depth -= 1
        self.depth -= 1
        body = [item for branch in zip(conditions, bodies) for item in branch]
        return self.join_lines(header, *body)

    def visit_MultiConditional(self, o, **kwargs):
        """
        Format as
          <repr(MultiConditional)>
            <Case [value(s)]>
              ...
            <Case [value(s)]>
              ...
            <Default>
              ...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        values = []
        for expr in o.values:
            value = '({})'.format(', '.join(self.visit_all(expr, **kwargs)))
            values += [self.format_node('Case', value)]
        if o.else_body:
            values += [self.format_node('Default')]
        self.depth += 1
        bodies = self.visit_all(*o.bodies, o.else_body, **kwargs)
        self.depth -= 1
        self.depth -= 1
        body = [item for branch in zip(values, bodies) for item in branch]
        return self.join_lines(header, *body)


def pprint(ir):
    """
    Convert the given IR to string using the py:class:`Stringifier`.
    """
    print(Stringifier().visit(ir))
