"""
Mappers for traversing and transforming the
:ref:`internal_representation:Expression tree`.
"""
import re
from itertools import zip_longest
import pymbolic.primitives as pmbl
from pymbolic.mapper import Mapper, WalkMapper, CombineMapper, IdentityMapper
from pymbolic.mapper.stringifier import (
    StringifyMapper, PREC_NONE, PREC_SUM, PREC_CALL, PREC_PRODUCT
)
from fparser.two.Fortran2003 import Intrinsic_Name

from loki.logging import debug
from loki.tools import as_tuple, flatten

__all__ = ['LokiStringifyMapper', 'ExpressionRetriever', 'ExpressionDimensionsMapper',
           'ExpressionCallbackMapper', 'SubstituteExpressionsMapper',
           'LokiIdentityMapper', 'AttachScopesMapper']

_intrinsic_fortran_names = Intrinsic_Name.function_names


class LokiStringifyMapper(StringifyMapper):
    """
    A class derived from the default :class:`StringifyMapper` that adds mappings for nodes of the
    expression tree that we added ourselves.

    This is the default pretty printer for nodes in the expression tree.
    """
    # pylint: disable=no-self-use,unused-argument,abstract-method

    _regex_string_literal = re.compile(r"((?<!')'(?:'')*(?!'))")

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return str(expr.value)

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            return '%s_%s' % (str(expr.value), str(expr.kind))
        return str(expr.value)

    map_int_literal = map_logic_literal

    def map_string_literal(self, expr, enclosing_prec, *args, **kwargs):
        return "'%s'" % self._regex_string_literal.sub(r"'\1", expr.value)

    map_intrinsic_literal = map_logic_literal

    def map_variable_symbol(self, expr, enclosing_prec, *args, **kwargs):
        if expr.parent is not None:
            parent = self.rec(expr.parent, enclosing_prec, *args, **kwargs)
            return self.format('%s%%%s', parent, expr.basename)
        return expr.name

    map_deferred_type_symbol = map_variable_symbol

    def map_meta_symbol(self, expr, enclosing_prec, *args, **kwargs):
        return self.rec(expr._symbol, enclosing_prec, *args, **kwargs)

    map_scalar = map_meta_symbol
    map_array = map_meta_symbol

    map_inline_call = StringifyMapper.map_call_with_kwargs

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        name = self.rec(expr.function, PREC_CALL, *args, **kwargs)
        expression = self.rec(expr.parameters[0], PREC_NONE, *args, **kwargs)
        if expr.kind:
            if isinstance(expr.kind, pmbl.Expression):
                kind = ', kind=' + self.rec(expr.kind, PREC_NONE, *args, **kwargs)
            else:
                kind = ', kind=' + str(expr.kind)
        else:
            kind = ''
        return self.format('%s(%s%s)', name, expression, kind)

    def map_range(self, expr, enclosing_prec, *args, **kwargs):
        children = [self.rec(child, PREC_NONE, *args, **kwargs) if child is not None else ''
                    for child in expr.children]
        if expr.step is None:
            children = children[:-1]
        return self.parenthesize_if_needed(self.join(':', children), enclosing_prec, PREC_NONE)

    map_range_index = map_range
    map_loop_range = map_range

    def map_sum(self, expr, enclosing_prec, *args, **kwargs):
        """
        Since substraction and unary minus are mapped to multiplication with (-1), we are here
        looking for such cases and determine the matching operator for the output.
        """
        def get_op_prec_expr(expr):
            from loki.expression.symbols import Product  # pylint: disable=import-outside-toplevel
            if isinstance(expr, Product) and expr.children and pmbl.is_zero(expr.children[0]+1):
                if len(expr.children) == 2:
                    # only the minus sign and the other child
                    return '-', PREC_PRODUCT, expr.children[1]
                return '-', PREC_PRODUCT, Product(expr.children[1:])
            return '+', PREC_SUM, expr

        terms = []
        for ch in expr.children:
            op, prec, expr = get_op_prec_expr(ch)
            terms += [op, self.rec(expr, prec, *args, **kwargs)]

        # Remove leading '+'
        if terms[0] == '-':
            terms[1] = '%s%s' % (terms[0], terms[1])
        terms = terms[1:]

        return self.parenthesize_if_needed(self.join(' ', terms), enclosing_prec, PREC_SUM)

    def map_product(self, expr, enclosing_prec, *args, **kwargs):
        if len(expr.children) == 2 and expr.children[0] == -1:
            # Negative values are encoded as multiplication by (-1) (constant, not IntLiteral).
            # We replace this by a minus again
            return self.parenthesize_if_needed(
                '-{}'.format(self.join_rec('*', expr.children[1:], PREC_PRODUCT, *args, **kwargs)),
                enclosing_prec, PREC_PRODUCT)
        return super().map_product(expr, enclosing_prec, *args, **kwargs)

    def map_parenthesised_add(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_sum(expr, PREC_NONE, *args, **kwargs))

    def map_parenthesised_mul(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_product(expr, PREC_NONE, *args, **kwargs))

    def map_parenthesised_pow(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_power(expr, PREC_NONE, *args, **kwargs))

    def map_string_concat(self, expr, enclosing_prec, *args, **kwargs):
        return ' // '.join(self.rec(c, enclosing_prec, *args, **kwargs) for c in expr.children)

    def map_literal_list(self, expr, enclosing_prec, *args, **kwargs):
        return '[' + ','.join(str(c) for c in expr.elements) + ']'

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        name_str = self.rec(expr.aggregate, PREC_NONE, *args, **kwargs)
        index_str = self.join_rec(', ', expr.index_tuple, PREC_NONE, *args, **kwargs)
        return '%s(%s)' % (name_str, index_str)

    def map_procedure_symbol(self, expr, enclosing_prec, *args, **kwargs):
        return expr.name


class LokiWalkMapper(WalkMapper):
    """
    A mapper that traverses the expression tree and calls :meth:`visit`
    for each visited node.

    Parameters
    ----------
    recurse_to_parent : bool, optional
        For symbols that belong to a derived type, recurse also to the
        ``parent`` of that symbol (default: `True`)
    """
    # pylint: disable=abstract-method

    def __init__(self, recurse_to_parent=True):
        super().__init__()
        self.recurse_to_parent = recurse_to_parent

    def map_variable_symbol(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        if self.recurse_to_parent and expr.parent:
            self.rec(expr.parent, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_deferred_type_symbol = map_variable_symbol
    map_procedure_symbol = map_variable_symbol

    def map_meta_symbol(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        self.rec(expr._symbol, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_scalar = map_meta_symbol
    map_array = map_meta_symbol

    def map_array_subscript(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        self.rec(expr.aggregate, *args, **kwargs)
        self.rec(expr.index, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_logic_literal = WalkMapper.map_constant
    map_string_literal = WalkMapper.map_constant
    map_intrinsic_literal = WalkMapper.map_constant

    def map_float_literal(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        if expr.kind:
            self.rec(expr.kind, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_int_literal = map_float_literal

    map_inline_call = WalkMapper.map_call_with_kwargs

    def map_cast(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        self.rec(expr.function, *args, **kwargs)
        for child in expr.parameters:
            self.rec(child, *args, **kwargs)
        if expr.kind is not None:
            self.rec(expr.kind, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_range = WalkMapper.map_slice
    map_range_index = WalkMapper.map_slice
    map_loop_range = WalkMapper.map_slice

    map_parenthesised_add = WalkMapper.map_sum
    map_parenthesised_mul = WalkMapper.map_product
    map_parenthesised_pow = WalkMapper.map_power
    map_string_concat = WalkMapper.map_sum

    def map_literal_list(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        for elem in expr.elements:
            if not isinstance(elem, str):
                # TODO: We are not representing all cases properly
                # (e.g., implicit loops) and instead retain them as plain
                # strings. Do not recurse on those for the moment...
                self.rec(elem, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)


class ExpressionRetriever(LokiWalkMapper):
    """
    A mapper for the expression tree that looks for entries specified by
    a query.

    Parameters
    ----------
    query :
        Function handle that is given each visited expression node and
        yields `True` or `False` depending on whether that expression
        should be included into the result.
    recurse_query : optional
        Optional function handle to which each visited expression node is
        given and that should return `True` or `False` depending on whether
        that expression node and its children should be visited.
    """
    # pylint: disable=abstract-method

    def __init__(self, query, recurse_query=None, **kwargs):
        super().__init__(**kwargs)

        self.query = query
        if recurse_query is not None:
            self.visit = lambda expr, *args, **kwargs: recurse_query(expr)
        self.reset()

    def post_visit(self, expr, *args, **kwargs):
        if self.query(expr):
            self.exprs.append(expr)

    def reset(self):
        self.exprs = []

    def retrieve(self, expr, *args, **kwargs):
        self.reset()
        self(expr, *args, **kwargs)
        return self.exprs


class ExpressionDimensionsMapper(Mapper):
    """
    A visitor for an expression that determines the dimensions of the expression.
    """
    # pylint: disable=no-self-use
    # pylint: disable=abstract-method

    def map_algebraic_leaf(self, expr, *args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from loki.expression.symbols import IntLiteral
        return as_tuple(IntLiteral(1))

    map_logic_literal = map_algebraic_leaf
    map_float_literal = map_algebraic_leaf
    map_int_literal = map_algebraic_leaf
    map_string_literal = map_algebraic_leaf
    map_intrinsic_literal = map_algebraic_leaf
    map_variable_symbol = map_algebraic_leaf
    map_scalar = map_algebraic_leaf

    def map_deferred_type_symbol(self, expr, *args, **kwargs):
        raise ValueError('Symbol with deferred type encountered: {}'.format(expr))

    def map_array(self, expr, *args, **kwargs):
        if not expr.dimensions:
            # We have the full array
            return expr.shape

        dims = self.rec(expr._symbol, *args, **kwargs)

        # Replace colon dimensions by the value from shape
        shape = expr.shape or [None] * len(dims)
        dims = [s if d == ':' else d for d, s in zip(dims, shape)]

        # Remove singleton dimensions
        dims = [d for d in dims if d != '1']
        return as_tuple(dims)

    def map_array_subscript(self, expr, *args, **kwargs):
        return flatten(tuple(self.rec(d, *args, **kwargs) for d in expr.index_tuple))

    def map_range_index(self, expr, *args, **kwargs):  # pylint: disable=unused-argument
        if expr.lower is None and expr.upper is None:
            # We have the full range
            return as_tuple(expr)

        lower = expr.lower.value - 1 if expr.lower is not None else 0
        step = expr.step.value if expr.step is not None else 1
        return as_tuple((expr.upper - lower) // step)

    def map_sum(self, expr, *args, **kwargs):
        dim = (1,)
        for ch in expr.children:
            child_dim = self.rec(ch, *args, **kwargs)
            if dim == (1,):
                dim = child_dim
            elif dim != child_dim and child_dim != 1:
                raise ValueError('Non-matching dimensions: {} and {}'.format(str(dim), str(child_dim)))
        return dim

    map_product = map_sum


class ExpressionCallbackMapper(CombineMapper):
    """
    A visitor for expressions that returns the combined result of a specified callback function.
    """
    # pylint: disable=abstract-method

    def __init__(self, callback, combine):
        super().__init__()
        self.callback = callback
        self.combine = combine

    def map_constant(self, expr, *args, **kwargs):
        return self.callback(expr, *args, **kwargs)

    map_logic_literal = map_constant
    map_string_literal = map_constant
    map_intrinsic_literal = map_constant

    def map_int_literal(self, expr, *args, **kwargs):
        rec_results = (self.callback(expr, *args, **kwargs), )
        if expr.kind is not None:
            rec_results += (self.rec(expr.kind, *args, **kwargs), )
        return self.combine(rec_results)

    map_float_literal = map_int_literal

    map_variable_symbol = map_constant
    map_deferred_type_symbol = map_constant

    def map_meta_symbol(self, expr, *args, **kwargs):
        rec_results = (self.callback(expr, *args, **kwargs), )
        rec_results += (self.rec(expr._symbol, *args, **kwargs), )
        return self.combine(rec_results)

    map_scalar = map_meta_symbol
    map_array = map_meta_symbol

    def map_array_subscript(self, expr, *args, **kwargs):
        rec_results = (self.rec(expr.aggregate, *args, **kwargs), )
        rec_results += (self.rec(expr.index, *args, **kwargs), )
        return self.combine(rec_results)

    map_inline_call = CombineMapper.map_call_with_kwargs

    def map_cast(self, expr, *args, **kwargs):
        rec_results = (self.rec(expr.function, *args, **kwargs), )
        rec_results += (self.rec(expr.parameters[0], *args, **kwargs), )
        if expr.kind is not None:
            rec_results += (self.rec(expr.kind, *args, **kwargs), )
        return self.combine(rec_results)

    def map_range(self, expr, *args, **kwargs):
        return self.combine(tuple(self.rec(c, *args, **kwargs)
                                  for c in expr.children if c is not None))

    map_range_index = map_range
    map_loop_range = map_range

    map_parenthesised_add = CombineMapper.map_sum
    map_parenthesised_mul = CombineMapper.map_product
    map_parenthesised_pow = CombineMapper.map_power
    map_string_concat = CombineMapper.map_sum

    def map_literal_list(self, expr, *args, **kwargs):
        return self.combine(tuple(self.rec(c, *args, **kwargs) for c in expr.elements))

    map_procedure_symbol = map_constant


class LokiIdentityMapper(IdentityMapper):
    """
    A visitor to traverse and transform an expression tree

    This can serve as basis for any transformation mappers
    that apply changes to the expression tree. Expression nodes that
    are unchanged are returned as is.

    Parameters
    ----------
    invalidate_source : bool, optional
        By default the :attr:`source` property of nodes is discarded
        when rebuilding the node, setting this to `False` allows to
        retain that information
    """

    def __init__(self, invalidate_source=True):
        super().__init__()
        self.invalidate_source = invalidate_source

    def __call__(self, expr, *args, **kwargs):
        if expr is None:
            return None
        new_expr = super().__call__(expr, *args, **kwargs)
        if new_expr is not expr and hasattr(new_expr, 'update_metadata'):
            metadata = getattr(expr, 'get_metadata', dict)()
            if self.invalidate_source:
                metadata['source'] = None
            new_expr.update_metadata(metadata)
        return new_expr

    rec = __call__

    map_algebraic_leaf = IdentityMapper.map_constant
    map_logic_literal = IdentityMapper.map_constant
    map_string_literal = IdentityMapper.map_constant
    map_intrinsic_literal = IdentityMapper.map_constant

    def map_int_literal(self, expr, *args, **kwargs):
        kind = self.rec(expr.kind, *args, **kwargs)
        if kind is expr.kind:
            return expr
        return expr.__class__(expr.value, kind=kind)

    map_float_literal = map_int_literal

    def map_variable_symbol(self, expr, *args, **kwargs):
        kind = self.rec(expr.type.kind, *args, **kwargs)
        if kind is not expr.type.kind and expr.scope:
            # Update symbol table entry for kind directly because with a scope attached
            # it does not affect the outcome of expr.clone
            expr.scope.symbols[expr.name] = expr.type.clone(kind=kind)

        parent = self.rec(expr.parent, *args, **kwargs)
        if parent is expr.parent and (kind is expr.type.kind or expr.scope):
            return expr
        return expr.clone(parent=parent, type=expr.type.clone(kind=kind))

    map_deferred_type_symbol = map_variable_symbol
    map_procedure_symbol = map_variable_symbol

    def map_meta_symbol(self, expr, *args, **kwargs):
        symbol = self.rec(expr._symbol, *args, **kwargs)
        # This is tricky as a rebuilt of the symbol will yield Scalar, Array, ProcedureSymbol etc
        # but with no rebuilt it may return VariableSymbol. Therefore we need to return the
        # original expression if the underlying symbol is unchanged
        if symbol is expr._symbol:
            return expr
        return symbol

    map_scalar = map_meta_symbol

    def map_array(self, expr, *args, **kwargs):
        symbol = self.rec(expr.symbol, *args, **kwargs)
        dimensions = self.rec(expr.dimensions, *args, **kwargs)
        shape = self.rec(symbol.type.shape, *args, **kwargs)
        if (getattr(symbol, 'symbol', symbol) is expr.symbol and
                all(d is orig_d for d, orig_d in zip_longest(dimensions or (), expr.dimensions or ())) and
                all(d is orig_d for d, orig_d in zip_longest(shape or (), symbol.type.shape or ()))):
            return expr
        return symbol.clone(dimensions=dimensions, type=symbol.type.clone(shape=shape))

    def map_array_subscript(self, expr, *args, **kwargs):  # pylint: disable=no-self-use
        raise RuntimeError('Recursion should have ended at map_array')

    map_inline_call = IdentityMapper.map_call_with_kwargs

    def map_cast(self, expr, *args, **kwargs):
        function = self.rec(expr.function, *args, **kwargs)
        parameters = self.rec(expr.parameters, *args, **kwargs)
        kind = self.rec(expr.kind, *args, **kwargs)
        if (function is expr.function and kind is expr.kind and
                all(p is orig_p for p, orig_p in zip_longest(parameters, expr.parameters))):
            return expr
        return expr.__class__(function, parameters, kind=kind)

    def map_sum(self, expr, *args, **kwargs):
        # Need to re-implement to avoid application of flattened_sum/flattened_product
        children = self.rec(expr.children, *args, **kwargs)
        if all(c is orig_c for c, orig_c in zip_longest(children, expr.children)):
            return expr
        return expr.__class__(children)

    map_parenthesised_add = map_sum
    map_product = map_sum
    map_parenthesised_mul = map_product
    map_parenthesised_pow = IdentityMapper.map_power
    map_string_concat = map_sum

    map_range = IdentityMapper.map_slice
    map_range_index = IdentityMapper.map_slice
    map_loop_range = IdentityMapper.map_slice

    def map_literal_list(self, expr, *args, **kwargs):
        values = tuple(v if isinstance(v, str) else self.rec(v, *args, **kwargs)
                       for v in expr.elements)
        if all(v is orig_v for v, orig_v in zip_longest(values, expr.elements)):
            return expr
        return expr.__class__(values)


class SubstituteExpressionsMapper(LokiIdentityMapper):
    """
    A Pymbolic expression mapper (i.e., a visitor for the expression tree) that
    defines on-the-fly handlers from a given substitution map.

    It returns a copy of the expression tree with expressions substituted according
    to the given `expr_map`.
    """
    # pylint: disable=abstract-method

    def __init__(self, expr_map, invalidate_source=True):
        super().__init__(invalidate_source=invalidate_source)

        self.expr_map = expr_map
        for expr in self.expr_map.keys():
            setattr(self, expr.mapper_method, self.map_from_expr_map)

    def map_from_expr_map(self, expr, *args, **kwargs):
        # We have to recurse here to make sure we are applying the substitution also to
        # "hidden" places (such as dimension expressions inside an array).
        # And we have to actually carry out the expression first before looking up the
        # super()-method as the node type might change.
        expr = self.expr_map.get(expr, expr)
        map_fn = getattr(super(), expr.mapper_method)
        return map_fn(expr, *args, **kwargs)


class AttachScopesMapper(LokiIdentityMapper):
    """
    A Pymbolic expression mapper (i.e., a visitor for the expression tree)
    that determines the scope of :any:`TypedSymbol` nodes and updates its
    :attr:`scope` pointer accordingly.

    Parameters
    ----------
    fail : bool, optional
        If `True`, the mapper raises :any:`RuntimeError` if the scope for a
        symbol can not be found.
    """

    def __init__(self, fail=False):
        super().__init__(invalidate_source=False)
        self.fail = fail

    def _update_symbol_scope(self, expr, scope):
        symbol_scope = scope.get_symbol_scope(expr.name)
        if symbol_scope is not None:
            if symbol_scope is not expr.scope:
                expr = expr.rescope(symbol_scope)
        elif self.fail:
            raise RuntimeError(f'AttachScopesMapper: {expr!s} was not found in any scope')
        elif expr not in _intrinsic_fortran_names:
            debug('AttachScopesMapper: %s was not found in any scopes', str(expr))
        return expr

    def map_deferred_type_symbol(self, expr, *args, **kwargs):
        expr = self._update_symbol_scope(expr, kwargs['scope'])
        return super().map_deferred_type_symbol(expr, *args, **kwargs)

    def map_variable_symbol(self, expr, *args, **kwargs):
        expr = self._update_symbol_scope(expr, kwargs['scope'])
        return super().map_variable_symbol(expr, *args, **kwargs)

    def map_procedure_symbol(self, expr, *args, **kwargs):
        expr = self._update_symbol_scope(expr, kwargs['scope'])
        return super().map_procedure_symbol(expr, *args, **kwargs)

