from abc import ABCMeta, abstractproperty
from collections import Iterable

from loki.visitors import GenericVisitor, Visitor
from loki.tools import flatten, as_tuple
from loki.logging import warning
from loki.expression.search import retrieve_variables

__all__ = ['Expression', 'Operation', 'Index', 'Cast',
           'ExpressionVisitor', 'FindVariables']


class ExpressionVisitor(GenericVisitor):

    def visit_Statement(self, o, **kwargs):
        return tuple([self.visit(o.target, **kwargs), self.visit(o.expr, **kwargs)])

    def visit_Expression(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o.children)


class FindVariables(Visitor):
    """
    A dedicated visitor to collect all variables used in an IR tree.

    Note: With `unique=False` all variables instanecs are traversed,
    allowing us to change them in-place. Conversely, `unique=True`
    returns a :class:`set` of unique :class:`Variable` objects that
    can be used to check if a particular variable is used in a given
    context.

    Note: :class:`Variable` objects are not recursed on in themselves.
    That means that individual component variables or dimension indices
    are not traversed or included in the final :class:`set`.
    """

    def __init__(self, unique=True):
        super(FindVariables, self).__init__()
        self.unique = unique

    default_retval = tuple

    def visit_tuple(self, o):
        variables = as_tuple(flatten(self.visit(c) for c in o))
        return set(variables) if self.unique else variables

    visit_list = visit_tuple

    def visit_Statement(self, o, **kwargs):
        # if not hasattr(o.expr, 'is_Symbol'):
        # from IPython import embed; embed()
        variables = as_tuple(retrieve_variables(o.target))
        variables += as_tuple(retrieve_variables(o.expr))
        return set(variables) if self.unique else variables

    def visit_Loop(self, o, **kwargs):
        variables = as_tuple(retrieve_variables(o.variable))
        variables += as_tuple(flatten(retrieve_variables(c) for c in o.bounds
                                      if c is not None))
        variables += as_tuple(flatten(self.visit(c) for c in o.body))
        return set(variables) if self.unique else variables


class Expression(object):
    """
    Base class for aithmetic and logical expressions.

    Note: :class:`Expression` objects are not part of the IR hierarchy,
    because re-building each individual expression tree during
    :class:`Transformer` passes can quickly become much more costly
    than re-building the control flow structures.
    """

    __metaclass__ = ABCMeta

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

    def __init__(self, ops, operands, parenthesis=False, source=None):
        super(Operation, self).__init__(source=source)
        self.ops = as_tuple(ops)
        self.operands = as_tuple(operands)
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

    def __key(self):
        return (self.ops, self.operands, self.parenthesis)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        # Allow direct comparisong to string and other Index objects
        if isinstance(other, str):
            return self.expr.upper() == other.upper()
        elif isinstance(other, Operation):
            return self.__key() == other.__key()
        else:
            return super(Operation, self).__eq__(other)



class Cast(Expression):
    """
    Internal representation of a data cast to a psecific type.
    """

    def __init__(self, expr, type):
        self._expr = expr
        self._type = type

    @property
    def expr(self):
        return '%s' % self._expr

    @property
    def type(self):
        return self._type

    @property
    def children(self):
        return as_tuple(self._expr)


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
            return self.name.upper() == other.upper()
        elif isinstance(other, Index):
            return self.name == other.name
        else:
            return super(Index, self).__eq__(other)
