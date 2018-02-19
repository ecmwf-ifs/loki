from collections import OrderedDict
import inspect


__all__ = ['Node', 'Loop', 'Statement', 'InlineComment']

class Node(object):

    def __new__(cls, *args, **kwargs):
        obj = super(Node, cls).__new__(cls)
        argnames = inspect.getargspec(cls.__init__).args
        obj._args = {k: v for k, v in zip(argnames[1:], args)}
        obj._args.update(kwargs.items())
        obj._args.update({k: None for k in argnames[1:] if k not in obj._args})
        return obj

    def _rebuild(self, *args, **kwargs):
        """Reconstruct self. None of the embedded Sympy expressions are rebuilt."""
        handle = self._args.copy()  # Original constructor arguments
        argnames = [i for i in self._traversable if i not in kwargs]
        handle.update(OrderedDict([(k, v) for k, v in zip(argnames, args)]))
        handle.update(kwargs)
        return type(self)(**handle)

    @property
    def children(self):
        return ()


class InlineComment(Node):
    """
    Internal representation of a single comment line.
    """

    def __init__(self, source):
        self._source = source


class Loop(Node):
    """
    Internal representation of a loop in source code.

    Importantly, this object will carry around an exact copy of the
    source string that defines it's body.
    """

    def __init__(self, variable, source, children=None, bounds=None):
        self._source = source
        self._children = children

        self.variable = variable
        self.bounds = bounds

    @property
    def children(self):
        return self._children


class Statement(Node):
    """
    Internal representation of a variable assignment
    """
    def __init__(self, target, expr, source):
        self._source = source

        self.target = target
        self.expr = expr


class Variable(object):

    def __init__(self, name, indices=None):
        self.name = name
        self.indices = indices

    def __repr__(self):
        idx = '(%s)' % ','.join([str(i) for i in self.indices]) if self.indices else ''
        return '%s%s' % (self.name, idx)
