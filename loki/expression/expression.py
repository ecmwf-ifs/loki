from pymbolic.mapper import IdentityMapper

from loki.visitors import Visitor, Transformer
from loki.tools import flatten, as_tuple
from loki.expression.search import retrieve_symbols, retrieve_functions, retrieve_variables

__all__ = ['FindSymbols', 'FindFunctions', 'FindVariables',
           'SubstituteExpressions', 'ExpressionFinder']


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
        variables = as_tuple(flatten(self.retrieve(a) for a in o.arguments))
        variables += as_tuple(flatten(self.retrieve(a) for _, a in o.kwarguments))
        return set(variables) if self.unique else variables

    def visit_Allocation(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(a) for a in o.variables))
        return set(variables) if self.unique else variables

    def visit_Declaration(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(v) for v in o.variables))
        if o.dimensions is not None:
            variables += as_tuple(flatten(self.retrieve(d) for d in o.dimensions))
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


class LokiIdentityMapper(IdentityMapper):

    def __init__(self):
        super(LokiIdentityMapper, self).__init__()

    map_logic_literal = IdentityMapper.map_constant
    map_float_literal = IdentityMapper.map_constant
    map_int_literal = IdentityMapper.map_constant
    map_scalar = IdentityMapper.map_variable
    map_array = IdentityMapper.map_variable
    map_inline_call = IdentityMapper.map_call_with_kwargs

    def map_sum(self, expr, *args, **kwargs):
        return expr.__class__(tuple(self.rec(child, *args, **kwargs) for child in expr.children))

    def map_product(self, expr, *args, **kwargs):
        return expr.__class__(tuple(self.rec(child, *args, **kwargs) for child in expr.children))

    map_parenthesised_add = map_sum
    map_parenthesised_mul = map_product
    map_parenthesised_pow = IdentityMapper.map_power


class SubstituteExpressionsMapper(LokiIdentityMapper):
    """
    A Pymbolic expression mapper (i.e., a visitor for the expression tree) that
    defines on-the-fly handlers from a given substitution map.
    """

    def __init__(self, expr_map):
        super(SubstituteExpressionsMapper, self).__init__()

        self.expr_map = expr_map
        for expr in self.expr_map.keys():
            setattr(self, expr.mapper_method, self.map_from_expr_map)

    def map_from_expr_map(self, expr, *args, **kwargs):
        if expr in self.expr_map:
            return self.expr_map[expr]
        else:
            map_fn = getattr(super(SubstituteExpressionsMapper, self), expr.mapper_method)
            return map_fn(expr, *args, **kwargs)


class SubstituteExpressions(Transformer):
    """
    A dedicated visitor to perform expression substitution in all IR nodes.

    :param expr_map: Expression mapping to apply to all expressions in a tree.
    """

    def __init__(self, expr_map):
        super(SubstituteExpressions, self).__init__()

        self.expr_mapper = SubstituteExpressionsMapper(expr_map)

    def visit_Statement(self, o, **kwargs):
        target = self.expr_mapper(o.target)
        expr = self.expr_mapper(o.expr)
        return o._rebuild(target=target, expr=expr)

    def visit_Conditional(self, o, **kwargs):
        conditions = tuple(self.expr_mapper(e) for e in o.conditions)
        bodies = self.visit(o.bodies)
        else_body = self.visit(o.else_body)
        return o._rebuild(conditions=conditions, bodies=bodies, else_body=else_body)

    def visit_Loop(self, o, **kwargs):
        variable = self.expr_mapper(o.variable)
        bounds = tuple(b if b is None else self.expr_mapper(b) for b in o.bounds)
        body = self.visit(o.body)
        return o._rebuild(variable=variable, bounds=bounds, body=body)

    def visit_Call(self, o, **kwargs):
        arguments = tuple(self.expr_mapper(a) for a in o.arguments)
        kwarguments = tuple((k, self.expr_mapper(v)) for k, v in o.kwarguments)
        # TODO: Re-build the call context
        return o._rebuild(arguments=arguments, kwarguments=kwarguments)

    def visit_Allocation(self, o, **kwargs):
        variables = tuple(self.expr_mapper(v) for v in o.variables)
        return o._rebuild(variables=variables)

    def visit_Declaration(self, o, **kwargs):
        if o.dimensions is not None:
            dimensions = tuple(self.expr_mapper(d) for d in o.dimensions)
        else:
            dimensions = None
        variables = tuple(self.expr_mapper(v) for v in o.variables)
        return o._rebuild(dimensions=dimensions, variables=variables)

    def visit_TypeDef(self, o, **kwargs):
        declarations = self.visit(o.declarations)
        return o._rebuild(declarations=declarations)
