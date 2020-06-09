import inspect
from itertools import zip_longest

from loki.ir import Node
from loki.tools import flatten, is_iterable, as_tuple

__all__ = ['pprint', 'GenericVisitor', 'Visitor', 'Transformer', 'NestedTransformer', 'FindNodes',
           'Stringifier']


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
    """

    def __init__(self, mapper=None):
        super(Transformer, self).__init__()
        self.mapper = mapper.copy() if mapper is not None else {}
        self.rebuilt = {}

    def visit_object(self, o, **kwargs):
        return o

    def visit_tuple(self, o, **kwargs):
        # For one-to-many mappings check iterables for the replacement
        # node and insert the sub-list/tuple into the list/tuple.
        for k, handle in self.mapper.items():
            if k in o and is_iterable(handle):
                i = o.index(k)
                o = o[:i] + tuple(handle) + o[i+1:]
        visited = tuple(self.visit(i, **kwargs) for i in o)
        return tuple(i for i in visited if i is not None)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        if o in self.mapper:
            handle = self.mapper[o]
            if handle is None:
                # None -> drop /o/
                return None
            if is_iterable(handle):
                # Original implementation to extend o.children:
                if not o.children:
                    raise ValueError
                extended = (tuple(handle) + o.children[0],) + o.children[1:]
                return o._rebuild(*extended, **o.args_frozen)
            return handle._rebuild(**handle.args)

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return o._rebuild(*rebuilt, **o.args_frozen)

    def visit(self, o, *args, **kwargs):
        obj = super(Transformer, self).visit(o, *args, **kwargs)
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
            return o._rebuild(*extended, **o.args_frozen)
        return handle._rebuild(*rebuilt, **handle.args_frozen)


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
        super(FindNodes, self).__init__()
        self.match = match
        self.rule = self.rules[mode]
        self.greedy = greedy

    def visit_object(self, o, **kwargs):
        ret = kwargs.get('ret')
        return ret or self.default_retval()

    def visit_tuple(self, o, **kwargs):
        ret = kwargs.get('ret')
        for i in o:
            ret = self.visit(i, ret=ret)
        return ret or self.default_retval()

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        ret = kwargs.get('ret')
        if ret is None:
            ret = self.default_retval()
        if self.rule(self.match, o):
            ret.append(o)
            if self.greedy:
                return ret
        for i in o.children:
            ret = self.visit(i, ret=ret)
        return ret or self.default_retval()


class Stringifier(Visitor):

    # pylint: disable=arguments-differ

    class ItemList:
        """
        Helper class that takes a list of items and joins them into a string,
        using custom separators and wrapping long lines.
        """

        def __init__(self, items, sep, width, cont):
            super().__init__()
            self.items = [item for item in items if item is not None]
            self.sep = sep
            self.width = width
            self.cont = cont.splitlines(keepends=True)

        def _add_item_to_line(self, line, item):
            # Let's see if we can fit the current item plus separator
            # onto the line and have enough space left for a line break
            new_line = line + item
            if len(new_line) + len(self.cont[0]) <= self.width:
                return new_line, []

            # Putting the current item plus separator and potential line break
            # onto the current line exceeds the allowed width: we need to break.

            # First, let's see if the item by itself fits on the next line
            new_line = self.cont[1] + item
            if len(new_line) + len(self.cont[0]) <= self.width:
                return new_line, [line + self.cont[0]]

            # The new item does not fit onto a line at all, we have to split it
            # up as good as we can
            # TODO: This is not safe for strings currently and may still exceed
            #       the line limit if the chunks are too big!
            chunks = item.split(' ')
            chunks[:-1] = [chunk + ' ' for chunk in chunks[:-1]]

            # First, add as much as possible to the previous line
            next_chunk = 0
            for idx, chunk in enumerate(chunks):
                new_line = line + chunk
                if len(new_line) + len(self.cont[0]) > self.width:
                    next_chunk = idx
                    break
                line = new_line

            # Now put the rest on new lines
            lines = [line + self.cont[0]]
            line = self.cont[1]
            for chunk in chunks[next_chunk:]:
                new_line = line + chunk
                if len(new_line) + len(self.cont[0]) > self.width:
                    lines += [line + self.cont[0]]
                    line = self.cont[1] + chunk
                else:
                    line = new_line
            return line, lines

        def __str__(self):
            """
            Join the items together using the given separator and wrap lines if
            necessary.
            """
            if not self.items:
                return ''
            if len(self.items) == 1:
                return str(self.items[0])
            lines = []
            # Start with the first item and the separator
            line = str(self.items[0]) + self.sep
            # Add all following items except for the last one
            for item in self.items[1:-1]:
                line, _lines = self._add_item_to_line(line, str(item) + self.sep)
                lines += _lines
            # Add the last item
            line, _lines = self._add_item_to_line(line, str(self.items[-1]))
            lines += _lines
            return ''.join([*lines, line])

    def __init__(self, depth=0, indent='  ', linewidth=90, line_cont=lambda indent: '\n' + indent):
        super().__init__()

        self.depth = depth
        self._indent = indent
        self.linewidth = linewidth
        self.line_cont = line_cont

    @property
    def indent(self):
        """
        Yield indentation according to current depth.
        """
        return self._indent * self.depth

    @staticmethod
    def join_lines(*lines):
        """
        Combine multiple lines.
        """
        if not lines:
            return None
        return '\n'.join(line for line in lines if line is not None)

    def join_items(self, items, sep=', '):
        """
        Create a TokenList object which can be passed to `format_line` or `format_node`
        where it is expanded, observing the allowed linewidth.
        """
        return self.ItemList(items, sep, self.linewidth, self.line_cont(self.indent))

    def format_node(self, name, *items):
        """
        Default format for a node.
        """
        if items:
            return self.format_line('<', name, ' ', self.join_items(items), '>')
        return self.format_line('<', name, '>')

    def format_line(self, *items):
        """
        Format a line and apply indentation.
        """
        return str(self.join_items([self.indent, *items], sep=''))

    def visit_all(self, item, *args, **kwargs):
        """
        Convenience function to call `visit` either for all elements in
        item if item is an iterable, or for multiple items given as arguments.
        """
        if is_iterable(item):
            return as_tuple(self.visit(i, **kwargs) for i in [*item, *args] if i is not None)
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

    @classmethod
    def visit_Expression(cls, o, **kwargs):  # pylint: disable=unused-argument
        """
        Dispatch routine to expression tree stringifier.
        """
        return str(o)

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

    def visit_Conditional(self, o, **kwargs):
        """
        Format as
          <repr(Conditional)>
            <If [condition]>
              ...
            <ElseIf [condition]>
              ...
            <Else>
              ...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        conditions = self.visit_all(o.conditions, **kwargs)
        conditions = [self.format_node(*vals)
                      for vals in zip_longest(['If'], conditions, fillvalue='ElseIf')]
        if o.else_body:
            conditions.append(self.format_node('Else'))
        self.depth += 1
        bodies = self.visit_all(*o.bodies, o.else_body, **kwargs)
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
    print(Stringifier().visit(ir))
