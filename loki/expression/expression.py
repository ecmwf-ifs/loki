from pymbolic.primitives import Expression

from loki.ir import Node
from loki.visitors import Visitor, Transformer
from loki.tools import flatten, as_tuple
from loki.expression.symbol_types import Array, Scalar, InlineCall
from loki.expression.mappers import SubstituteExpressionsMapper, retrieve_expressions

__all__ = ['FindExpressions', 'FindVariables', 'FindInlineCalls', 'FindExpressionRoot',
           'SubstituteExpressions', 'ExpressionFinder']


class ExpressionFinder(Visitor):
    """
    Base class visitor to collect specific sub-expressions,
    eg. functions or symbols, from all nodes in an IR tree.

    :param retrieve: Custom retrieval function that yields all wanted
                     sub-expressions from an expression.
    :param unique: If ``True`` the visitor will return a set of unique sub-expression
                   instead of a list of possibly repeated instances.
    :param with_ir_node: If ``True`` the visitor will return tuples which
                         contain the sub-expression and the corresponding
                         IR node in which the expression is contained.

    Note that :class:`FindXXX` classes are provided to find the most
    common sub-expression types, eg. symbols, functions and variables.
    """
    # pylint: disable=unused-argument

    # By default we return nothing
    retrieval_function = lambda x: ()

    def __init__(self, unique=True, retrieve=None, with_ir_node=False):
        super(ExpressionFinder, self).__init__()
        self.unique = unique
        self.with_ir_node = with_ir_node

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
            assert isinstance(var, Expression)
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
        Create the return value from the found expressions.
        """
        if not expressions:
            return ()
        if self.with_ir_node:
            # A direct call to flatten() would destroy our tuples, thus we need to
            # sort through the list and single out existing tuple-value pairs and
            # plain expressions before finding uniques
            is_leaf = lambda el: isinstance(el, tuple) and len(el) == 2 and isinstance(el[0], Node)
            newlist = flatten(expressions, is_leaf=is_leaf)
            tuple_list = [el for el in newlist if is_leaf(el)]
            exprs = [el for el in newlist if not is_leaf(el)]
            if exprs:
                tuple_list += [(node, self.find_uniques(exprs))]
            return as_tuple(tuple_list)

        # Flatten the (possibly nested) list
        return self.find_uniques(as_tuple(flatten(expressions)))

    default_retval = tuple

    def visit_tuple(self, o, **kwargs):
        expressions = [self.visit(c, **kwargs) for c in o]
        return self._return(o, as_tuple(expressions))

    visit_list = visit_tuple

    def visit_Expression(self, o, **kwargs):
        return as_tuple(self.retrieve(o))

    def visit_Node(self, o, **kwargs):
        expressions = [self.visit(c, **kwargs) for c in flatten(o.children)]
        return self._return(o, as_tuple(expressions))


class FindExpressions(ExpressionFinder):
    """
    A visitor to collect all symbols (i.e., :class:`pymbolic.primitives.Expression`) in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, Expression)))


class FindVariables(ExpressionFinder):
    """
    A visitor to collect all variables (:class:`pymbolic.primitives.Variable`, which includes
    :class:`loki.Scalar` and :class:`loki.Array`) symbols used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, (Scalar, Array))))


class FindInlineCalls(ExpressionFinder):
    """
    A visitor to collect all :class:`loki.InlineCall` symbols used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, InlineCall)))


class FindExpressionRoot(ExpressionFinder):
    """
    A visitor to obtain the root node of the expression tree in which a given
    py:class:`pymbolic.primitives.Expression` is located.
    """

    def __init__(self, expr):
        super().__init__(unique=False, retrieve=(
            lambda e: e if retrieve_expressions(e, lambda _e: _e is expr) else ()))


class SubstituteExpressions(Transformer):
    """
    A dedicated visitor to perform expression substitution in all IR nodes.

    :param expr_map: Expression mapping to apply to all expressions in a tree.
    """
    # pylint: disable=unused-argument

    def __init__(self, expr_map, invalidate_source=True):
        super(SubstituteExpressions, self).__init__(invalidate_source=invalidate_source)

        self.expr_mapper = SubstituteExpressionsMapper(expr_map,
                                                       invalidate_source=invalidate_source)

    def visit_Expression(self, o, **kwargs):
        return self.expr_mapper(o)
