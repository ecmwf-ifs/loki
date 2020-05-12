from loki.ir import Node
from loki.visitors import Visitor, Transformer
from loki.tools import flatten, as_tuple
from loki.expression.symbol_types import Array, Scalar
from loki.expression.search import retrieve_expressions, retrieve_variables, retrieve_inline_calls
from loki.expression.mappers import SubstituteExpressionsMapper

__all__ = ['FindExpressions', 'FindVariables', 'FindInlineCalls',
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
                        str(var.dimensions) if isinstance(var, Array) else None)
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
        return self.find_uniques(flatten(expressions))

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
        expressions = self.flatten(self.visit(c, **kwargs) for c in o)
        return self.find_uniques(expressions)

    visit_list = visit_tuple

    def visit_Expression(self, o, **kwargs):
        return as_tuple(self.retrieve(o))

    def visit_Node(self, o, **kwargs):
        expressions = as_tuple(self.visit(c, **kwargs) for c in flatten(o.children))
        return self._return(o, expressions)


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


class SubstituteExpressions(Transformer):
    """
    A dedicated visitor to perform expression substitution in all IR nodes.

    :param expr_map: Expression mapping to apply to all expressions in a tree.
    """
    # pylint: disable=unused-argument

    def __init__(self, expr_map):
        super(SubstituteExpressions, self).__init__()

        self.expr_mapper = SubstituteExpressionsMapper(expr_map)

    def visit_Expression(self, o, **kwargs):
        return self.expr_mapper(o)
