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

    Parameters
    ----------
    retrieve :
        Custom retrieval function that yields all wanted sub-expressions
        from an expression.
    unique : bool, optional
        If `True` the visitor will return a `set` of unique sub-expression
        instead of a list of possibly repeated instances.
    with_ir_node : bool, optional
        If `True` the visitor will return tuples which contain the
        sub-expression and the corresponding IR node in which the
        expression is contained.

    Note that specialized ``FindXXX`` classes are provided in :any:`loki.expression`
    to find some of the most common sub-expression types, eg. symbols, functions
    and variables.
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
        if self.unique is `True`.

        Currently, two sub-expressions are considered NOT to be unique if they have the same
        - :attr:`name`
        - :attr:`parent.name` (or `None`)
        - :attr:`dimensions` (for :any:`Array`)
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
    A visitor to collect all expression tree nodes
    (i.e., :class:`pymbolic.primitives.Expression`) in an IR tree.

    See :any:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, Expression)))


class FindTypedSymbols(ExpressionFinder):
    """
    A visitor to collect all :any:`TypedSymbol` used in an IR tree.

    See :any:`ExpressionFinder`
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
    A visitor to collect all :any:`InlineCall` symbols used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, InlineCall)))


class FindLiterals(ExpressionFinder):
    """
    A visitor to collect all literals (which includes :any:`FloatLiteral`,
    :any:`IntLiteral`, :any:`LogicLiteral`, :any:`StringLiteral`,
    and :any:`IntrinsicLiteral`) used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retrieval_function = staticmethod(
        lambda expr: retrieve_expressions(expr, lambda e: isinstance(e, (
            FloatLiteral, IntLiteral, LogicLiteral, StringLiteral, IntrinsicLiteral))))


class FindExpressionRoot(ExpressionFinder):
    """
    A visitor to obtain the root node of the expression tree in which a given
    :class:`pymbolic.primitives.Expression` is located.
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
    """
    Scoping visitor that traverses the control flow tree and uses
    :any:`AttachScopesMapper` to update all :any:`TypedSymbol` expression
    tree nodes with a pointer to their corresponding scope.

    Parameters
    ----------
    fail : bool, optional
        If set to True, this lets the visitor fail if it encounters a node
        without a declaration or an entry in any of the symbol tables
        (default: False).
    """

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.expr_mapper = AttachScopesMapper(fail=fail)

    @staticmethod
    def _update(o, children, **args):
        """
        Utility routine to update the IR node
        """
        args_frozen = o.args_frozen
        args_frozen.update(args)
        o._update(*children, **args_frozen)
        return o

    def visit_object(self, o, **kwargs):
        """Return any foreign object unchanged."""
        return o

    def visit(self, o, *args, **kwargs):
        """
        Default visitor method that dispatches the node-specific handler
        """
        kwargs.setdefault('scope', None)
        return super().visit(o, *args, **kwargs)

    def visit_Expression(self, o, **kwargs):
        """
        Dispatch :any:`AttachScopesMapper` for :any:`Expression` tree nodes
        """
        return self.expr_mapper(o, scope=kwargs['scope'])

    def visit_list(self, o, **kwargs):
        """
        Visit each entry in a list and return as a tuple
        """
        return tuple(self.visit(c, **kwargs) for c in o)

    visit_tuple = visit_list

    def visit_Node(self, o, **kwargs):
        """
        Generic handler for IR :any:`Node` objects

        Recurses to children and updates the node
        """
        children = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._update(o, children)

    def visit_Scope(self, o, **kwargs):
        """
        Generic handler for :any:`Scope` objects

        Makes sure that declared variables and imported symbols have an
        entry in that node's symbol table before recursing to children with
        this node as new scope.
        """
        # First, make sure declared variables and imported symbols have an
        # entry in the scope's table
        self._update_symbol_table_with_decls_and_imports(o)

        # Then recurse to all children
        kwargs['scope'] = o
        children = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._update(o, children, symbols=o.symbols, rescope_variables=False)

    @staticmethod
    def _update_symbol_table_with_decls_and_imports(o):
        """
        Utility function to insert default entries for symbols declared or
        imported in a node
        """
        for v in getattr(o, 'variables', ()):
            o.symbols.setdefault(v.name, v.type)
        for s in getattr(o, 'imported_symbols', ()):
            o.symbols.setdefault(s.name, s.type)

    def visit_Subroutine(self, o, **kwargs):
        """
        Handler for :any:`Subroutine` nodes

        Makes sure that declared variables and imported symbols have an
        entry in the routine's symbol table before recursing to spec, body,
        and member routines with this routine as new scope.
        """
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
        """
        Handler for :any:`Module` nodes

        Makes sure that declared variables and imported symbols have an
        entry in the module's symbol table before recursing to spec,
        and member routines with this module as new scope.
        """
        # First, make sure declared variables and imported symbols have an
        # entry in the scope's table
        self._update_symbol_table_with_decls_and_imports(o)

        # Then recurse to all children
        kwargs['scope'] = o
        o.spec = self.visit(o.spec, **kwargs)
        o.routines = self.visit(o.routines, **kwargs)
        return o
