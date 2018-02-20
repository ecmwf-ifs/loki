from collections import OrderedDict
import inspect


__all__ = ['Node', 'Loop', 'Statement', 'Conditional', 'Comment',
           'Variable', 'Expression']

class Node(object):

    """
    :attr:`_traversable`. The traversable fields of the Node; that is, fields
    walked over by a :class:`Visitor`. All arguments in __init__ whose name
    appears in this list are treated as traversable fields.
    """
    _traversable = []

    def __new__(cls, *args, **kwargs):
        obj = super(Node, cls).__new__(cls)
        argnames = inspect.getargspec(cls.__init__).args
        obj._args = {k: v for k, v in zip(argnames[1:], args)}
        obj._args.update(kwargs.items())
        obj._args.update({k: None for k in argnames[1:] if k not in obj._args})
        return obj

    def _rebuild(self, *args, **kwargs):
        handle = self._args.copy()  # Original constructor arguments
        argnames = [i for i in self._traversable if i not in kwargs]
        handle.update(OrderedDict([(k, v) for k, v in zip(argnames, args)]))
        handle.update(kwargs)
        return type(self)(**handle)

    @property
    def args(self):
        """Arguments used to construct the Node."""
        return self._args.copy()


    @property
    def args_frozen(self):
        """Arguments used to construct the Node that cannot be traversed."""
        return {k: v for k, v in self.args.items() if k not in self._traversable}

    @property
    def children(self):
        return ()


class Comment(Node):
    """
    Internal representation of a single comment line.
    """

    def __init__(self, source):
        self._source = source


class CommentBlock(Node):
    """
    Internal representation of a block comment.
    """

    def __init__(self, comments, source=None):
        self._source = source

        self.comments = comments


class Loop(Node):
    """
    Internal representation of a loop in source code.

    Importantly, this object will carry around an exact copy of the
    source string that defines it's body.
    """

    _traversable = ['body']

    def __init__(self, variable, source=None, body=None, bounds=None):
        self._source = source
        self.body = body

        self.variable = variable
        self.bounds = bounds

    @property
    def children(self):
        # Note: Needs to be one tuple per `traversable`
        return tuple([self.body])


class Conditional(Node):
    """
    Internal representation of a conditional branching construct.
    """

    _traversable = ['bodies', 'else_body']

    def __init__(self, conditions, bodies, else_body, source=None):
        self._source = source

        self.conditions = conditions
        self.bodies = bodies
        self.else_body = else_body

    @property
    def children(self):
        # Note that we currently ignore the condition itself
        return tuple(tuple([self.bodies]) + tuple([self.else_body]))


class Statement(Node):
    """
    Internal representation of a variable assignment
    """
    def __init__(self, target, expr, source):
        self._source = source

        self.target = target
        self.expr = expr


############################################################
## Utility classes that are not (yet) part of the hierachy
############################################################

class Variable(object):

    def __init__(self, name, indices=None):
        self.name = name
        self.indices = indices

    def __repr__(self):
        idx = '(%s)' % ','.join([str(i) for i in self.indices]) if self.indices else ''
        return '%s%s' % (self.name, idx)


class Expression(object):

    def __init__(self, source):
        self.expr = source

    def __repr__(self):
        return '<Expr::%s>' % (self.expr)
