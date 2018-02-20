import inspect
from collections import Iterable

from ecir.ir import Node

__all__ = ['pprint', 'Visitor', 'Transformer', 'NestedTransformer']


class Visitor(object):

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
            argspec = inspect.getargspec(meth)
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
        raise RuntimeError("No handler found for class %s", cls.__name__)

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

    def visit_Node(self, o, **kwargs):
        return self.visit(o.children, **kwargs)

    def reuse(self, o, *args, **kwargs):
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

    In the special case in which ``M[n]`` is an iterable of nodes, ``n`` is
    "extended" by pre-pending to its body the nodes in ``M[n]``.
    """

    def __init__(self, mapper={}):
        super(Transformer, self).__init__()
        self.mapper = mapper.copy()
        self.rebuilt = {}

    def visit_object(self, o, **kwargs):
        return o

    def visit_tuple(self, o, **kwargs):
        visited = tuple(self.visit(i, **kwargs) for i in o)
        return tuple(i for i in visited if i is not None)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        if o in self.mapper:
            handle = self.mapper[o]
            if handle is None:
                # None -> drop /o/
                return None
            elif isinstance(handle, Iterable):
                if not o.children:
                    raise VisitorException
                extended = (tuple(handle) + o.children[0],) + o.children[1:]
                return o._rebuild(*extended, **o.args_frozen)
            else:
                ret = handle._rebuild(**handle.args)
                return ret
        else:
            rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
            ret = o._rebuild(*rebuilt, **o.args_frozen)
            return ret

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
        elif isinstance(handle, Iterable):
            if not o.children:
                raise VisitorException
            extended = [tuple(handle) + rebuilt[0]] + rebuilt[1:]
            return o._rebuild(*extended, **o.args_frozen)
        else:
            return handle._rebuild(*rebuilt, **handle.args_frozen)


class PrintAST(Visitor):

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

    def visit_Node(self, o):
        return self.indent + '<%s>' % o.__class__.__name__

    def visit_list(self, o):
        return ('\n').join([self.visit(i) for i in o])

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    def visit_Block(self, o):
        self._depth += 2
        body = self.visit(o.body)
        self._depth -= 2
        return self.indent + "<Block>\n%s" % body

    def visit_Loop(self, o):
        self._depth += 2
        body = self.visit(o.children)
        self._depth -= 2
        if self.verbose and o.bounds is not None:
            bounds = ' :: %s' % (', '.join(str(b) for b in o.bounds))
        else:
            bounds = ''
        return self.indent + "<Loop %s%s>\n%s" % (o.variable, bounds, body)

    def visit_Conditional(self, o):
        self._depth += 2
        bodies = tuple(self.visit(b) for b in o.bodies)
        self._depth -= 2
        out = self.indent + '<Cond %s>\n%s' % (o.conditions[0], bodies[0])
        for b, c in zip(bodies[1:], o.conditions[1:]):
            out += '\n%s' % self.indent + '<Else-If %s>\n%s' % (c, b)
        if o.else_body is not None:
            self._depth += 2
            else_body = self.visit(o.else_body) #[self.visit(b) for b in o.else_body]
            self._depth -= 2
            out += '\n%s' % self.indent + '<Else>\n%s' % else_body
        return out

    def visit_Statement(self, o):
        expr = ' = %s' % o.expr if self.verbose else ''
        return self.indent + '<Stmt %s%s>' % (str(o.target), expr)

    def visit_Comment(self, o):
        body = '::%s::' % o._source if self.verbose else ''
        return self.indent + '<Comment%s>' % body

    def visit_CommentBlock(self, o):
        body = ('\n%s' % self.indent).join([b._source for b in o.comments])
        return self.indent + '<CommentBlock%s' % (('\n%s' % self.indent)+body+'>' if self.verbose else '>')

    def visit_Variable(self, o):
        indices = ('(%s)' % ','.join([str(v) for v in o.indices])) if o.indices else ''
        return 'V<%s%s>' % (o.name, indices)


def pprint(ir, verbose=False):
    print(PrintAST(verbose=verbose).visit(ir))
