from abc import ABCMeta, abstractproperty

from loki.visitors import GenericVisitor

__all__ = ['Expression', 'Operation', 'Literal', 'Variable', 'Index',
           'ExpressionVisitor']


class ExpressionVisitor(GenericVisitor):

    def visit_Statement(self, o, **kwargs):
        return tuple([self.visit(o.target, **kwargs), self.visit(o.expr, **kwargs)])

    def visit_Expression(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o.children)


class Expression(object):

    __metaclass__ = ABCMeta

    _traversable = []

    def __init__(self, source=None):
        self._source = source

    @abstractproperty
    def expr(self):
        """
        Symbolic representation - might be used in this raw form
        for code generation.
        """
        pass

    @abstractproperty
    def type(self):
        """
        Data type of (sub-)expressions.

        Note, that this is the pure data type (eg. int32, float64),
        not the full variable declaration type (allocatable, pointer,
        etc.). This is so that we may reason about it recursively.
        """
        pass

    def __repr__(self):
        return self.expr

    @property
    def children(self):
        return ()


class Operation(Expression):

    _traversable = ['operands']

    def __init__(self, ops, operands, parenthesis=False, source=None):
        super(Operation, self).__init__(source=source)
        self.ops = ops
        self.operands = operands
        self.parenthesis = parenthesis

    @property
    def expr(self):
        if len(self.ops) == 1 and len(self.operands) == 1:
            # Special case: a unary operator
            return '%s%s' % (self.ops[0], self.operands[0])

        s = str(self.operands[0])
        s += ''.join(['%s%s' % (o, str(e)) for o, e in zip(self.ops, self.operands[1:])])
        return ('(%s)' % s) if self.parenthesis else s

    @property
    def type(self):
        types = [o.type for o in self.operands]
        assert(all(types == types[0]))
        return types[0]

    @property
    def children(self):
        return self.operands


class Literal(Expression):

    def __init__(self, value, kind=None, type=None, source=None):
        super(Literal, self).__init__(source=source)
        self.value = value
        self.kind = kind
        self._type = type

    @property
    def expr(self):
        return self.value if self.kind is None else '%s_%s' % (self.value, self.kind)

    @property
    def type(self):
        return self._type


class Variable(Expression):

    def __init__(self, name, type=None, dimensions=None, subvar=None,
                 initial=None, source=None):
        super(Variable, self).__init__(source=source)
        self._source = source

        self.name = name
        self._type = type
        self.subvar = subvar
        self.dimensions = dimensions or ()
        self.initial = initial

    @property
    def expr(self):
        idx = '(%s)' % ','.join([str(i) for i in self.dimensions]) if len(self.dimensions) > 0 else ''
        subvar = '' if self.subvar is None else '%%%s' % str(self.subvar)
        return '%s%s%s' % (self.name, idx, subvar)

    @property
    def type(self):
        return self._type

    def __key(self):
        return (self.name, self.type, self.dimensions, self.subvar)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        # Allow direct comparison to string and other Variable objects
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, Variable):
            return self.__key() == other.__key()
        else:
            return super(Variable, self).__eq__(other)

    @property
    def children(self):
        c = self.dimensions
        if self.subvar is not None:
            c += (self.subvar, )
        return c


class InlineCall(Expression):
    """
    Internal representation of an in-line function call
    """
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    @property
    def expr(self):
        return '%s(%s)' % (self.name, ','.join(str(a) for a in self.arguments))

    @property
    def type(self):
        return self._type

    @property
    def children(self):
        return self.arguments


class Index(Expression):

    def __init__(self, name):
        self.name = name

    @property
    def expr(self):
        return '%s' % self.name

    def __key(self):
        return (self.name)

    def __hash__(self):
        return hash(self.__key())

    @property
    def type(self):
        # TODO: Some common form of `INT`, maybe?
        return None

    def __eq__(self, other):
        # Allow direct comparisong to string and other Index objects
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Index):
            return self.name == other.name
        else:
            return super(Index, self).__eq__(other)
