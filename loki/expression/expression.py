from abc import ABCMeta, abstractproperty
from collections import Iterable
from sympy import Expr, evaluate

from loki.visitors import GenericVisitor, Visitor, Transformer
from loki.tools import flatten, as_tuple
from loki.logging import warning
from loki.expression.search import retrieve_symbols, retrieve_functions, retrieve_variables

__all__ = ['Expression', 'FindSymbols', 'FindFunctions', 'FindVariables',
           'SubstituteExpressions', 'ExpressionFinder', 'ExpressionVisitor']


class ExpressionVisitor(GenericVisitor):

    def visit_Statement(self, o, **kwargs):
        return tuple([self.visit(o.target, **kwargs), self.visit(o.expr, **kwargs)])

    def visit_Expression(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o.children)


class ExpressionFinder(Visitor):
    """
    Base class visitor to collect specific sub-expressions,
    eg. functions or symbls, from all nodes in an IR tree.

    :param retrieve: Custom retrieval function that yields all wanted
                     sub-expressions from an expression.
    :param unique: If ``True`` the visitor will return a set of unique sub-expression
                   instead of a list of possibly repeated instances.

    Note that :class:`FindXXX` classes are provided to find the most
    common sub-expression types, eg. symbols, functions and variables.
    """

    # By default we return nothing
    retrieval_function = lambda x: ()

    def __init__(self, unique=True, retrieve=None):
        super(ExpressionFinder, self).__init__()
        self.unique = unique

        # Use custom retrieval function or the class default
        # TODO: This is pretty hacky, isn't it..?
        if retrieve is not None:
            type(self).retrieval_function = staticmethod(retrieve)

    def retrieve(self, expr):
        """
        Internal retrieval function used on expressions.
        """
        return self.retrieval_function(expr)

    default_retval = tuple

    def visit_tuple(self, o):
        variables = as_tuple(flatten(self.visit(c) for c in o))
        return set(variables) if self.unique else variables

    visit_list = visit_tuple

    def visit_Statement(self, o, **kwargs):
        variables = as_tuple(self.retrieve(o.target))
        variables += as_tuple(self.retrieve(o.expr))
        return set(variables) if self.unique else variables

    def visit_Conditional(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(c) for c in o.conditions))
        variables += as_tuple(flatten(self.visit(c) for c in o.bodies))
        variables += as_tuple(self.visit(o.else_body))
        return set(variables) if self.unique else variables

    def visit_Loop(self, o, **kwargs):
        variables = as_tuple(self.retrieve(o.variable))
        variables += as_tuple(flatten(self.retrieve(c) for c in o.bounds
                                      if c is not None))
        variables += as_tuple(flatten(self.visit(c) for c in o.body))
        return set(variables) if self.unique else variables

    def visit_Call(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(a) for a in o.arguments
                                     if isinstance(a, Expr)))
        variables += as_tuple(flatten(self.retrieve(a) for _, a in o.kwarguments
                                      if isinstance(a, Expr)))
        return set(variables) if self.unique else variables

    def visit_Allocation(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(a) for a in o.variables))
        return set(variables) if self.unique else variables


class FindSymbols(ExpressionFinder):
    """
    A visitor to collect all :class:`sympy.Symbol` symbols in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(retrieve_symbols)


class FindFunctions(ExpressionFinder):
    """
    A visitor to collect all :class:`sympy.Function` symbols in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(retrieve_functions)


class FindVariables(ExpressionFinder):
    """
    A visitor to collect all variables (:class:`loki.Scalar` and
    :class:`loki.Array`) symbols used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(retrieve_variables)


class SubstituteExpressions(Transformer):
    """
    A dedicated visitor to perform expression substitution in all IR nodes.

    :param expr_map: Expression mapping to apply to all expressions in a tree.
    """

    def __init__(self, expr_map):
        super(SubstituteExpressions, self).__init__()

        self.expr_map = expr_map

    def visit_Statement(self, o, **kwargs):
        with evaluate(False):
            target = o.target.xreplace(self.expr_map)
            expr = o.expr.xreplace(self.expr_map)
        return o._rebuild(target=target, expr=expr)

    def visit_Conditional(self, o, **kwargs):
        with evaluate(False):
            conditions = tuple(e.xreplace(self.expr_map) for e in o.conditions)
        bodies = self.visit(o.bodies)
        else_body = self.visit(o.else_body)
        return o._rebuild(conditions=conditions, bodies=bodies, else_body=else_body)

    def visit_Loop(self, o, **kwargs):
        with evaluate(False):
            variable = o.variable.xreplace(self.expr_map)
            bounds = tuple(b if b is None else b.xreplace(self.expr_map) for b in o.bounds)
            body = self.visit(o.body)
        return o._rebuild(variable=variable, bounds=bounds, body=body)

    def visit_Call(self, o, **kwargs):
        with evaluate(False):
            arguments = tuple(a.xreplace(self.expr_map) for a in o.arguments)
            kwarguments = tuple((k, v.xreplace(self.expr_map)) for k, v in o.kwarguments)
            # TODO: Re-build the call context
        return o._rebuild(arguments=arguments, kwarguments=kwarguments)

    def visit_Allocation(self, o, **kwargs):
        with evaluate(False):
            variables = tuple(v.xreplace(self.expr_map) for v in o.variables)
        return o._rebuild(variables=variables)

    def visit_Declaration(self, o, **kwargs):
        with evaluate(False):
            if o.dimensions is not None:
                dimensions = tuple(d.xreplace(self.expr_map) for d in o.dimensions)
            else:
                dimensions = None
            variables = tuple(v.xreplace(self.expr_map) for v in o.variables)
        return o._rebuild(dimensions=dimensions, variables=variables)

    def visit_TypeDef(self, o, **kwargs):
        declarations = self.visit(o.declarations)
        return o._rebuild(declarations=declarations)


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
