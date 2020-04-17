import inspect
from collections.abc import Iterable

from loki.ir import Node
from loki.tools import flatten

__all__ = ['pprint', 'GenericVisitor', 'Visitor', 'Transformer', 'NestedTransformer', 'FindNodes']


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

    def visit_object(self, o, **kwargs):
        return self.default_retval()


class Visitor(GenericVisitor):

    def visit_tuple(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        return self.visit(o.children, **kwargs)

    @staticmethod
    def reuse(o, *args, **kwargs):
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
            if k in o and isinstance(handle, Iterable):
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
            if isinstance(handle, Iterable):
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
        if isinstance(handle, Iterable):
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
    """

    rules = {
        'type': lambda match, o: isinstance(o, match),
        'scope': lambda match, o: match in flatten(o.children)
    }

    def __init__(self, match, mode='type'):
        super(FindNodes, self).__init__()
        self.match = match
        self.rule = self.rules[mode]

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
        for i in o.children:
            ret = self.visit(i, ret=ret)
        return ret or self.default_retval()


class PrintAST(Visitor):
    # pylint: disable=no-self-use

    _depth = 0

    def __init__(self, verbose=True):
        super(PrintAST, self).__init__()
        self.verbose = verbose

    @classmethod
    def default_retval(cls):
        return "<>"

    @property
    def indent(self):
        return '  ' * self._depth

    def visit_Node(self, o, **kwargs):
        return self.indent + '<%s>' % o.__class__.__name__

    def visit_tuple(self, o, **kwargs):
        return '\n'.join([self.visit(i) for i in o])

    visit_list = visit_tuple

    def visit_Block(self, o, **kwargs):
        self._depth += 2
        body = self.visit(o.body)
        self._depth -= 2
        return self.indent + "<Block>\n%s" % body

    def visit_Loop(self, o, **kwargs):
        self._depth += 2
        body = self.visit(o.children)
        self._depth -= 2
        if self.verbose and o.bounds is not None:
            bounds = ' :: %s' % str(o.bounds)
        else:
            bounds = ''
        return self.indent + "<Loop %s%s>\n%s" % (o.variable, bounds, body)

    def visit_Conditional(self, o, **kwargs):
        self._depth += 2
        bodies = tuple(self.visit(b) for b in o.bodies)
        self._depth -= 2
        out = self.indent + '<If %s>\n%s' % (o.conditions[0], bodies[0])
        for b, c in zip(bodies[1:], o.conditions[1:]):
            out += '\n%s' % self.indent + '<Else-If %s>\n%s' % (c, b)
        if o.else_body is not None:
            self._depth += 2
            else_body = self.visit(o.else_body)
            self._depth -= 2
            out += '\n%s' % self.indent + '<Else>\n%s' % else_body
        return out

    def visit_Statement(self, o, **kwargs):
        expr = (' => ' if o.ptr else ' = ') + str(o.expr) if self.verbose else ''
        if self.verbose and o.comment is not None:
            self._depth += 2
            comment = '\n%s' % self.visit(o.comment)
            self._depth -= 2
        else:
            comment = ''
        return self.indent + '<Stmt %s%s>%s' % (str(o.target), expr, comment)

    def visit_Scope(self, o, **kwargs):
        self._depth += 2
        body = self.visit(o.body)
        self._depth -= 2
        return self.indent + "<Scope>\n%s" % body

    def visit_Declaration(self, o, **kwargs):
        variables = ' :: %s' % ', '.join(v.name for v in o.variables) if self.verbose else ''
        comment = ''
        pragma = ''

        if self.verbose and o.comment is not None:
            self._depth += 2
            comment = '\n%s' % self.visit(o.comment)
            self._depth -= 2
        if self.verbose and o.pragma is not None:
            self._depth += 2
            pragma = '\n%s' % self.visit(o.pragma)
            self._depth -= 2
        return self.indent + '<Declaration%s>%s%s' % (variables, comment, pragma)

    def visit_Allocation(self, o, **kwargs):
        variable = " %s" % o.variable if self.verbose else ''
        return self.indent + '<Alloc%s>' % variable

    def visit_CallStatement(self, o, **kwargs):
        args = '(%s)' % (', '.join(str(a) for a in o.arguments)) if self.verbose else ''
        return self.indent + '<CallStatement %s%s>' % (o.name, args)

    def visit_Comment(self, o, **kwargs):
        body = '::%s::' % o._source.string if self.verbose else ''
        return self.indent + '<Comment%s>' % body

    def visit_CommentBlock(self, o, **kwargs):
        body = ('\n%s' % self.indent).join([b._source.string for b in o.comments])
        return self.indent + '<CommentBlock%s' % (
            ('\n%s' % self.indent) + body + '>' if self.verbose else '>')

    def visit_Pragma(self, o, **kwargs):
        body = ' ::%s::' % o._source.string if self.verbose else ''
        return self.indent + '<Pragma %s%s>' % (o.keyword, body)

    def visit_Variable(self, o, **kwargs):
        dimensions = ('(%s)' % ','.join([str(v) for v in o.dimensions])) if o.dimensions else ''
        _type = self.visit(o.type) if o.type is not None else ''
        return self.indent + '<Var %s%s%s>' % (o.name, dimensions, _type)

    def visit_BaseType(self, o, **kwargs):
        ptr = ', ptr' if o.pointer else ''
        return '<Type %s:%s%s>' % (o.name, o.kind, ptr)

    def visit_DerivedType(self, o, **kwargs):
        variables = ''
        comments = ''
        pragmas = ''
        if self.verbose:
            self._depth += 2
            variables = '\n%s' % self.visit(o.variables)
            self._depth -= 2
        if self.verbose and o.comments is not None:
            self._depth += 2
            comments = '\n%s' % self.visit(o.comments)
            self._depth -= 2
        if self.verbose and o.pragmas is not None:
            self._depth += 2
            pragmas = '\n%s' % self.visit(o.pragmas)
            self._depth -= 2
        return self.indent + '<DerivedType %s>%s%s%s' % (o.name, variables, pragmas, comments)


def pprint(ir, verbose=False):
    print(PrintAST(verbose=verbose).visit(ir))
