from abc import ABCMeta, abstractproperty
from collections import Iterable

from loki.visitors import GenericVisitor, Visitor, Transformer
from loki.tools import flatten, as_tuple
from loki.logging import warning
from loki.expression.search import retrieve_variables

__all__ = ['Expression', 'FindVariables', 'SubstituteExpressions', 'ExpressionVisitor']


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
        variables = as_tuple(retrieve_variables(o.target))
        variables += as_tuple(retrieve_variables(o.expr))
        return set(variables) if self.unique else variables

    def visit_Loop(self, o, **kwargs):
        variables = as_tuple(retrieve_variables(o.variable))
        variables += as_tuple(flatten(retrieve_variables(c) for c in o.bounds
                                      if c is not None))
        variables += as_tuple(flatten(self.visit(c) for c in o.body))
        return set(variables) if self.unique else variables


class SubstituteExpressions(Transformer):
    """
    A dedicated visitor to perform expression substitution in all IR nodes.

    :param expr_map: Expression mapping to apply to all expressions in a tree.
    """

    def __init__(self, expr_map):
        super(SubstituteExpressions, self).__init__()

        self.expr_map = expr_map

    def visit_Statement(self, o, **kwargs):
        target = o.target.xreplace(self.expr_map)
        expr = o.expr.xreplace(self.expr_map)
        return o._rebuild(target=target, expr=expr)

    # TODO: Add Loops and Conditionals that have expressions to rebuild.


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
