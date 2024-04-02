# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Visitor classes for traversing and transforming all expression trees in
:doc:`internal_representation`.
"""
from pymbolic.primitives import Expression

from loki.ir import Node, Visitor, Transformer
from loki.tools import flatten, as_tuple
from loki.expression.mappers import (
    SubstituteExpressionsMapper, ExpressionRetriever, AttachScopesMapper
)
from loki.expression.symbols import (
    Array, Scalar, InlineCall, TypedSymbol, FloatLiteral, IntLiteral,
    LogicLiteral, StringLiteral, IntrinsicLiteral, DeferredTypeSymbol
)

__all__ = [
    'FindExpressions', 'FindVariables', 'FindTypedSymbols', 'FindInlineCalls',
    'FindLiterals', 'SubstituteExpressions', 'ExpressionFinder', 'AttachScopes'
]


class ExpressionFinder(Visitor):
    """
    Base class visitor to collect specific sub-expressions,
    eg. functions or symbols, from all nodes in an IR tree.

    Note that specialized ``FindXXX`` classes are provided in :py:mod:`loki.expression`
    to find some of the most common sub-expression types, eg. symbols, functions
    and variables.

    Attributes
    ----------
    retriever : :class:`pymbolic.mapper.Mapper`
        An implementation of an expression mapper, e.g., :any:`ExpressionRetriever`,
        that is used to search expression trees. Note that it needs to provide a
        ``retrieve`` method to initiate the traversal and retrieve the list of expressions.

    Parameters
    ----------
    unique : bool, optional
        If `True` the visitor will return a `set` of unique sub-expression
        instead of a list of possibly repeated instances.
    with_ir_node : bool, optional
        If `True` the visitor will return tuples which contain the
        sub-expression and the corresponding IR node in which the
        expression is contained.
    """
    # pylint: disable=unused-argument

    retriever = ExpressionRetriever(lambda _: False)

    def __init__(self, unique=True, with_ir_node=False):
        super().__init__()
        self.unique = unique
        self.with_ir_node = with_ir_node

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

    @classmethod
    def retrieve(cls, expr):
        """
        Internal retrieval function used on expressions.
        """
        return cls.retriever.retrieve(expr)

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
            def is_leaf(el):
                return isinstance(el, tuple) and len(el) == 2 and isinstance(el[0], Node)
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

    def visit_TypeDef(self, o, **kwargs):
        """
        Custom handler for :any:`TypeDef` nodes that does not traverse the
        body (reason being that discovering variables used or declared
        inside the type definition would be unexpected if called on a
        containing :any:`Subroutine` or :any:`Module`)
        """
        return self._return(o, ())

    def visit_VariableDeclaration(self, o, **kwargs):
        expressions = ()
        for v in o.symbols:
            if v.type.initial is not None:
                expressions += (self.retrieve(v.type.initial),)
        return self._return(o, expressions)


class FindExpressions(ExpressionFinder):
    """
    A visitor to collect all expression tree nodes
    (i.e., :class:`pymbolic.primitives.Expression`) in an IR tree.

    See :any:`ExpressionFinder`
    """
    retriever = ExpressionRetriever(lambda e: isinstance(e, Expression))


class FindTypedSymbols(ExpressionFinder):
    """
    A visitor to collect all :any:`TypedSymbol` used in an IR tree.

    See :any:`ExpressionFinder`
    """
    retriever = ExpressionRetriever(lambda e: isinstance(e, TypedSymbol))


class FindVariables(ExpressionFinder):
    """
    A visitor to collect all variables used in an IR tree

    This refers to expression tree nodes :any:`Scalar`, :any:`Array` and also
    :any:`DeferredTypeSymbol`.

    See :class:`ExpressionFinder` for further details
    """
    retriever = ExpressionRetriever(lambda e: isinstance(e, (Scalar, Array, DeferredTypeSymbol)))


class FindInlineCalls(ExpressionFinder):
    """
    A visitor to collect all :any:`InlineCall` symbols used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retriever = ExpressionRetriever(lambda e: isinstance(e, InlineCall))


class FindLiterals(ExpressionFinder):
    """
    A visitor to collect all literals (which includes :any:`FloatLiteral`,
    :any:`IntLiteral`, :any:`LogicLiteral`, :any:`StringLiteral`,
    and :any:`IntrinsicLiteral`) used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retriever = ExpressionRetriever(lambda e: isinstance(e, (
        FloatLiteral, IntLiteral, LogicLiteral, StringLiteral, IntrinsicLiteral
    )))


class SubstituteExpressions(Transformer):
    """
    A dedicated visitor to perform expression substitution in all IR nodes

    It applies :any:`SubstituteExpressionsMapper` with the provided :data:`expr_map`
    to every expression in the traversed IR tree.

    .. note::
       No recursion is performed on substituted expression nodes, they are taken
       as-is from the map. Otherwise substitutions that involve the original node
       would result in infinite recursion - for example a replacement that wraps
       a variable in an inline call:  ``my_var -> wrapped_in_call(my_var)``.

       When there is a need to recursively apply the mapping, the mapping needs to
       be applied to itself first. A potential use-case is renaming of variables,
       which may appear as the name of an array subscript as well as in the ``dimensions``
       attribute of the same expression: ``SOME_ARR(SOME_ARR > SOME_VAL)``.
       The mapping can be applied to itself using the utility function
       :any:`recursive_expression_map_update`.

    Parameters
    ----------
    expr_map : dict
        Expression mapping to apply to the expression tree.
    invalidate_source : bool, optional
        By default the :attr:`source` property of nodes is discarded
        when rebuilding the node, setting this to `False` allows to
        retain that information
    """
    # pylint: disable=unused-argument

    def __init__(self, expr_map, invalidate_source=True, **kwargs):
        super().__init__(invalidate_source=invalidate_source, **kwargs)

        self.expr_mapper = SubstituteExpressionsMapper(expr_map,
                                                       invalidate_source=invalidate_source)

    def visit_Expression(self, o, **kwargs):
        """
        call :any:`SubstituteExpressionsMapper` for the given expression node
        """
        if kwargs.get('recurse_to_declaration_attributes'):
            return self.expr_mapper(o, recurse_to_declaration_attributes=True)
        return self.expr_mapper(o)

    def visit_Import(self, o, **kwargs):
        """
        For :any:`Import` (as well as :any:`VariableDeclaration` and :any:`ProcedureDeclaration`)
        we set ``recurse_to_declaration_attributes=True`` to make sure properties in the symbol
        table are updated during dispatch to the expression mapper
        """
        kwargs['recurse_to_declaration_attributes'] = True
        return super().visit_Node(o, **kwargs)

    visit_VariableDeclaration = visit_Import
    visit_ProcedureDeclaration = visit_Import


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
        if kwargs.get('recurse_to_declaration_attributes'):
            return self.expr_mapper(o, scope=kwargs['scope'], recurse_to_declaration_attributes=True)
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

    def visit_Import(self, o, **kwargs):
        """
        For :any:`Import` (as well as :any:`VariableDeclaration` and :any:`ProcedureDeclaration`)
        we set ``recurse_to_declaration_attributes=True`` to make sure properties in the symbol
        table are updated during dispatch to the expression mapper
        """
        kwargs['recurse_to_declaration_attributes'] = True
        return self.visit_Node(o, **kwargs)

    visit_VariableDeclaration = visit_Import
    visit_ProcedureDeclaration = visit_Import

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
        return self._update(o, children, symbol_attrs=o.symbol_attrs, rescope_symbols=False)

    @staticmethod
    def _update_symbol_table_with_decls_and_imports(o):
        """
        Utility function to insert default entries for symbols declared or
        imported in a node
        """
        for v in getattr(o, 'variables', ()):
            o.symbol_attrs.setdefault(v.name, v.type)
        for s in getattr(o, 'imported_symbols', ()):
            o.symbol_attrs.setdefault(s.name, s.type)

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
        o.contains = self.visit(o.contains, **kwargs)
        return o
