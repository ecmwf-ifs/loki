"""
Visitor classes for traversing and transforming all expression trees in
:doc:`internal_representation`.
"""
from pymbolic.primitives import Expression

from loki.ir import Node
from loki.visitors import Visitor, Transformer
from loki.tools import flatten, as_tuple
from loki.expression.mappers import SubstituteExpressionsMapper, retrieve_expressions, AttachScopesMapper
from loki.expression.symbols import (
    Array, Scalar, InlineCall, TypedSymbol, FloatLiteral, IntLiteral, LogicLiteral,
    StringLiteral, IntrinsicLiteral, DeferredTypeSymbol
)

__all__ = [
    'FindExpressions', 'FindVariables', 'FindTypedSymbols', 'FindInlineCalls',
    'FindLiterals', 'FindExpressionRoot', 'SubstituteExpressions',
    'ExpressionFinder', 'AttachScopes'
]


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
        super().__init__()
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


class FindTypedSymbols(ExpressionFinder):
    """
    A visitor to collect all typed symbols (which includes :class:`loki.Scalar`, :class:`loki.Array`,
    and :class:`loki.ProcedureSymbol`) used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, (TypedSymbol))))


class FindVariables(ExpressionFinder):
    """
    A visitor to collect all variables used in an IR tree

    This refers to expression tree nodes :any:`Scalar`, :any:`Array` and also
    :any:`DeferredTypeSymbol`.

    See :class:`ExpressionFinder` for further details
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, (Scalar, Array, DeferredTypeSymbol))))


class FindInlineCalls(ExpressionFinder):
    """
    A visitor to collect all :class:`loki.InlineCall` symbols used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, InlineCall)))


class FindLiterals(ExpressionFinder):
    """
    A visitor to collect all literals (which includes :class:`loki.FloatLiteral`,
    :class:`loki.IntLiteral`, :class:`loki.LogicLiteral`, :class:`loki.StringLiteral`,
    and :class:`loki.IntrinsicLiteral`) used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, (
            FloatLiteral, IntLiteral, LogicLiteral, StringLiteral, IntrinsicLiteral))))


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

    def __init__(self, expr_map, invalidate_source=True, **kwargs):
        super().__init__(invalidate_source=invalidate_source, **kwargs)

        self.expr_mapper = SubstituteExpressionsMapper(expr_map,
                                                       invalidate_source=invalidate_source)

    def visit_Expression(self, o, **kwargs):
        return self.expr_mapper(o)


class AttachScopes(Visitor):

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.expr_mapper = AttachScopesMapper(fail=fail)

    @staticmethod
    def _update(o, children, **args):
        args_frozen = o.args_frozen
        args_frozen.update(args)
        o._update(*children, **args_frozen)
        return o

    def visit_object(self, o, **kwargs):
        """Return any foreign object unchanged."""
        return o

    def visit(self, o, *args, **kwargs):
        kwargs.setdefault('scope', None)
        return super().visit(o, *args, **kwargs)

    def visit_Expression(self, o, **kwargs):
        return self.expr_mapper(o, scope=kwargs['scope'])

    def visit_list(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o)

    visit_tuple = visit_list

    def visit_Node(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._update(o, children)

    def visit_Scope(self, o, **kwargs):
        kwargs['scope'] = o
        children = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._update(o, children, symbols=o.symbols)

    @staticmethod
    def _update_symbol_table_with_decls_and_imports(o):
        for v in o.variables:
            o.symbols.setdefault(v.name, v.type)
        for s in o.imported_symbols:
            o.symbols.setdefault(s.name, s.type)

    def visit_Subroutine(self, o, **kwargs):
        # First, make sure declared variables and imported symbols have an
        # entry in the scope's table
        self._update_symbol_table_with_decls_and_imports(o)

        # Then recurse to all children
        kwargs['scope'] = o
        o.spec = self.visit(o.spec, **kwargs)
        o.body = self.visit(o.body, **kwargs)
        o._members = self.visit(o.members, **kwargs)
        return o

    def visit_Module(self, o, **kwargs):
        # First, make sure declared variables and imported symbols have an
        # entry in the scope's table
        self._update_symbol_table_with_decls_and_imports(o)

        # Then recurse to all children
        kwargs['scope'] = o
        o.spec = self.visit(o.spec, **kwargs)
        o.routines = self.visit(o.routines, **kwargs)
        return o

    def visit_TypeDef(self, o, **kwargs):
        # First, make sure declared variables and imported symbols have an
        # entry in the scope's table
        self._update_symbol_table_with_decls_and_imports(o)

        # Then recurse to all children
        kwargs['scope'] = o
        body = self.visit(o.body, **kwargs)
        return self._update(o, (), body=body, symbols=o.symbols, rescope_variables=False)

    def visit_Associate(self, o, **kwargs):
        # First, make sure declared variables and imported symbols have an
        # entry in the scope's table
        self._update_symbol_table_with_decls_and_imports(o)

        # Then recurse to all children
        kwargs['scope'] = o
        associations = tuple((self.visit(var, **kwargs), self.visit(expr, **kwargs))
                             for var, expr in o.associations)
        body = self.visit(o.body, **kwargs)
        return self._update(o, (), associations=associations, body=body, symbols=o.symbols,
                            rescope_variables=False)
