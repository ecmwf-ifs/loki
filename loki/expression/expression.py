from loki.ir import Node
from loki.visitors import Visitor, Transformer
from loki.tools import flatten, as_tuple
from loki.expression.symbol_types import Array, Scalar
from loki.expression.search import (
    retrieve_expressions, retrieve_variables, retrieve_inline_calls, retrieve_literals)
from loki.expression.mappers import SubstituteExpressionsMapper

__all__ = ['FindExpressions', 'FindVariables', 'FindInlineCalls', 'FindLiterals',
           'SubstituteExpressions', 'ExpressionFinder']


class ExpressionFinder(Visitor):
    """
    Base class visitor to collect specific sub-expressions,
    eg. functions or symbols, from all nodes in an IR tree.

    :param retrieve: Custom retrieval function that yields all wanted
                     sub-expressions from an expression.
    :param unique: If ``True`` the visitor will return a set of unique sub-expression
                   instead of a list of possibly repeated instances.
    :param with_expression_root: If ``True`` the visitor will return tuples which
                                 contain the sub-expression and the corresponding
                                 IR node in which the expression is contained.

    Note that :class:`FindXXX` classes are provided to find the most
    common sub-expression types, eg. symbols, functions and variables.
    """
    # pylint: disable=unused-argument

    # By default we return nothing
    retrieval_function = lambda x: ()

    def __init__(self, unique=True, retrieve=None, with_expression_root=False):
        super(ExpressionFinder, self).__init__()
        self.unique = unique
        self.with_expression_root = with_expression_root

        # Use custom retrieval function or the class default
        # TODO: This is pretty hacky, isn't it..?
        if retrieve is not None:
            type(self).retrieval_function = staticmethod(retrieve)

    def find_uniques(self, variables):
        """
        Reduces the number of matched sub-expressions to a set of unique sub-expressions,
        if self.unique is ``True``.
        Currently, two sub-expressions are considered NOT to be unique if they have the same
        - ``name``
        - ``parent.name`` (or ``None``)
        - ``dimensions`` (for :class:`Array`)
        """
        def dict_key(var):
            if isinstance(var, (Scalar, Array)):
                return (var.name,
                        var.parent.name if hasattr(var, 'parent') and var.parent else None,
                        var.dimensions if isinstance(var, Array) else None)
            return str(var)

        if self.unique:
            var_dict = {dict_key(var): var for var in variables}
            return set(var_dict.values())
        return variables

    def retrieve(self, expr):
        """
        Internal retrieval function used on expressions.
        """
        return self.retrieval_function(expr)

    def _return(self, node, expressions):
        """
        Form the return value from the found expressions.
        """
        if not expressions:
            return ()
        if self.with_expression_root:
            return (node, self.find_uniques(flatten(expressions)))
        return self.find_uniques(expressions)

    def flatten(self, expr_list):
        # Flatten the (possibly nested) list
        newlist = flatten(expr_list)
        if self.with_expression_root:
            # This did remove our tuples: restore them by running through the
            # new list and collect everything behind a ``Node`` as its expressions
            tuple_list = []
            node = None
            exprs = []
            assert not newlist or isinstance(newlist[0], Node)
            for el in newlist:
                if isinstance(el, Node):
                    if node and exprs:
                        tuple_list += [(node, exprs)]
                    node = el
                    exprs = []
                else:
                    exprs.append(el)
            if node and exprs:
                tuple_list += [(node, exprs)]
            newlist = tuple_list
        return as_tuple(newlist)

    default_retval = tuple

    def visit_tuple(self, o, **kwargs):
        variables = self.flatten(self.visit(c) for c in o)
        return self.find_uniques(variables)

    visit_list = visit_tuple

    def visit_Statement(self, o, **kwargs):
        variables = as_tuple(self.retrieve(o.target))
        variables += as_tuple(self.retrieve(o.expr))
        return self._return(o, variables)

    def visit_Conditional(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(c) for c in o.conditions))
        variables += as_tuple(flatten(self.visit(c) for c in o.bodies))
        variables += as_tuple(self.visit(o.else_body))
        return self._return(o, variables)

    def visit_Loop(self, o, **kwargs):
        variables = as_tuple(self.retrieve(o.variable)) if o.variable else ()
        variables += as_tuple(flatten(self.retrieve(c) for c in o.bounds or []
                                      if c is not None))
        variables += as_tuple(flatten(self.visit(c) for c in o.body))
        return self._return(o, variables)

    def visit_CallStatement(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(a) for a in o.arguments))
        variables += as_tuple(flatten(self.retrieve(a) for _, a in o.kwarguments or []))
        return self._return(o, variables)

    def visit_Allocation(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(a) for a in o.variables))
        return self._return(o, variables)

    def visit_Declaration(self, o, **kwargs):
        variables = as_tuple(flatten(self.retrieve(v) for v in o.variables))
        if o.dimensions is not None:
            variables += as_tuple(flatten(self.retrieve(d) for d in o.dimensions))
        return self._return(o, variables)


class FindExpressions(ExpressionFinder):
    """
    A visitor to collect all symbols (i.e., :class:`pymbolic.primitives.Expression`) in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(retrieve_expressions)


class FindVariables(ExpressionFinder):
    """
    A visitor to collect all variables (:class:`pymbolic.primitives.Variable`, which includes
    :class:`loki.Scalar` and :class:`loki.Array`) symbols used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(retrieve_variables)


class FindInlineCalls(ExpressionFinder):
    """
    A visitor to collect all :class:`loki.InlineCall` symbols used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(retrieve_inline_calls)


class FindLiterals(ExpressionFinder):
    """
    A visitor to collect all :class:`loki.FloatLiteral`, :class:`loki.IntLiteral`,
    :class:`loki.LogicLiteral`, :class:`loki.StringLiteral`.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(retrieve_literals)


class SubstituteExpressions(Transformer):
    """
    A dedicated visitor to perform expression substitution in all IR nodes.

    :param expr_map: Expression mapping to apply to all expressions in a tree.
    """
    # pylint: disable=unused-argument

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
        variable = self.expr_mapper(o.variable) if o.variable else None
        bounds = tuple(b if b is None else self.expr_mapper(b) for b in o.bounds or [])
        body = self.visit(o.body)
        return o._rebuild(variable=variable, bounds=bounds, body=body)

    def visit_CallStatement(self, o, **kwargs):
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
