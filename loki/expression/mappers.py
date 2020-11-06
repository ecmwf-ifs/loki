import re
from pymbolic.primitives import Expression, Product, is_zero
from pymbolic.mapper import Mapper, WalkMapper, CombineMapper, IdentityMapper
from pymbolic.mapper.stringifier import (
    StringifyMapper, PREC_NONE, PREC_SUM, PREC_CALL, PREC_PRODUCT
)

from loki.tools import as_tuple, flatten

__all__ = ['LokiStringifyMapper', 'ExpressionRetriever', 'ExpressionDimensionsMapper',
           'ExpressionCallbackMapper', 'SubstituteExpressionsMapper', 'retrieve_expressions']


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

    def map_scalar(self, expr, enclosing_prec, *args, **kwargs):
        if expr.parent is not None:
            parent = self.rec(expr.parent, enclosing_prec, *args, **kwargs)
            return self.format('%s%%%s', parent, expr.basename)
        return expr.name

    def map_array(self, expr, enclosing_prec, *args, **kwargs):
        dims, parent = '', ''
        if expr.dimensions:
            dims = self.rec(expr.dimensions, PREC_NONE, *args, **kwargs)
        if expr.parent is not None:
            parent = self.rec(expr.parent, PREC_NONE, *args, **kwargs) + '%'
        return self.format('%s%s%s', parent, expr.basename, dims)

    map_inline_call = StringifyMapper.map_call_with_kwargs

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        name = self.rec(expr.function, PREC_CALL, *args, **kwargs)
        expression = self.rec(expr.parameters[0], PREC_NONE, *args, **kwargs)
        if expr.kind:
            if isinstance(expr.kind, Expression):
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
            if isinstance(expr, Product) and expr.children and is_zero(expr.children[0]+1):
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
        return self.parenthesize(self.map_sum(expr, enclosing_prec, *args, **kwargs))

    def map_parenthesised_mul(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_product(expr, enclosing_prec, *args, **kwargs))

    def map_parenthesised_pow(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_power(expr, enclosing_prec, *args, **kwargs))

    def map_string_concat(self, expr, enclosing_prec, *args, **kwargs):
        return ' // '.join(self.rec(c, enclosing_prec, *args, **kwargs) for c in expr.children)

    def map_literal_list(self, expr, enclosing_prec, *args, **kwargs):
        return '[' + ','.join(str(c) for c in expr.elements) + ']'

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        index_str = self.join_rec(', ', expr.index_tuple, PREC_NONE, *args, **kwargs)
        return '(%s)' % index_str

    def map_procedure_symbol(self, expr, enclosing_prec, *args, **kwargs):
        return expr.name


class ExpressionRetriever(WalkMapper):
    """
    A visitor for the expression tree that looks for entries specified by a query.

    :param query: Function handle that is given each visited expression node and
                  yields `True` or `False` depending on whether that expression
                  should be included into the result.
    :param recurse_query: Optional function handle that is given each visited
                          expression node and yields `True` or `False` depending
                          on whether that expression and its children should be
                          visited.
    """
    # pylint: disable=abstract-method

    def __init__(self, query, recurse_query=None):
        super().__init__()

        self.query = query
        self.exprs = list()

        if recurse_query is not None:
            self.visit = lambda expr, *args, **kwargs: recurse_query(expr)

    def post_visit(self, expr, *args, **kwargs):
        if self.query(expr):
            self.exprs.append(expr)

    map_scalar = WalkMapper.map_variable

    def map_array(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        if expr.dimensions:
            self.rec(expr.dimensions, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    def map_array_subscript(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        self.rec(expr.index, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_logic_literal = WalkMapper.map_constant
    map_float_literal = WalkMapper.map_constant
    map_int_literal = WalkMapper.map_constant
    map_string_literal = WalkMapper.map_constant
    map_intrinsic_literal = WalkMapper.map_constant
    map_inline_call = WalkMapper.map_call_with_kwargs

    def map_cast(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        for p in expr.parameters:
            self.rec(p, *args, **kwargs)
        if isinstance(expr.kind, Expression):
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
            self.visit(elem)
        self.post_visit(expr, *args, **kwargs)

    map_procedure_symbol = WalkMapper.map_function_symbol


def retrieve_expressions(expr, cond, recurse_cond=None):
    """
    Utility function to retrieve all expressions satisfying condition `cond`.

    Can be used with py:class:`ExpressionRetriever` to query the IR for
    expression nodes using custom conditions.

    :param cond: Function handle that is given each visited expression node and
                 yields `True` or `False` depending on whether that expression
                 should be included into the result.
    :param recurse_cond: Optional function handle that is given each visited
                         expression node and yields `True` or `False` depending
                         on whether that expression and its children should be
                         visited.
    """
    retriever = ExpressionRetriever(cond, recurse_query=recurse_cond)
    retriever(expr)
    return retriever.exprs


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
    map_scalar = map_algebraic_leaf

    def map_array(self, expr, *args, **kwargs):
        if expr.dimensions is None:
            # We have the full array
            return expr.shape

        # pylint: disable=import-outside-toplevel
        from loki.expression.symbols import RangeIndex, IntLiteral
        dims = self.rec(expr.dimensions, *args, **kwargs)
        # Replace colon dimensions by the value from shape
        shape = expr.shape or [None] * len(dims)
        dims = [s if (isinstance(d, RangeIndex) and d.lower is None and d.upper is None)
                else d for d, s in zip(dims, shape)]
        # Remove singleton dimensions
        dims = [d for d in dims if d != IntLiteral(1)]
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
    map_int_literal = map_constant
    map_float_literal = map_constant
    map_string_literal = map_constant
    map_intrinsic_literal = map_constant
    map_scalar = map_constant
    map_array = map_constant
    map_variable = map_constant

    def map_array_subscript(self, expr, *args, **kwargs):
        dimensions = self.rec(expr.index_tuple, *args, **kwargs)
        return self.combine(dimensions)

    map_inline_call = CombineMapper.map_call_with_kwargs

    def map_cast(self, expr, *args, **kwargs):
        if expr.kind and isinstance(expr.kind, Expression):
            kind = (self.rec(expr.kind, *args, **kwargs),)
        else:
            kind = tuple()
        return self.combine((self.rec(expr.function, *args, **kwargs),
                             self.rec(expr.parameters[0])) + kind)

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
    A visitor which creates a copy of the expression tree.
    """
    # pylint: disable=abstract-method

    def __init__(self, invalidate_source=True):
        super().__init__()
        self.invalidate_source = invalidate_source

    def __call__(self, expr, *args, **kwargs):
        new_expr = super().__call__(expr, *args, **kwargs)
        if new_expr is not expr and hasattr(new_expr, 'update_metadata'):
            metadata = getattr(expr, 'get_metadata', dict)()
            if self.invalidate_source:
                metadata['source'] = None
            new_expr.update_metadata(metadata)
        return new_expr

    rec = __call__

    map_logic_literal = IdentityMapper.map_constant
    map_float_literal = IdentityMapper.map_constant
    map_int_literal = IdentityMapper.map_constant
    map_string_literal = IdentityMapper.map_constant
    map_intrinsic_literal = IdentityMapper.map_constant

    def map_scalar(self, expr, *args, **kwargs):
        return expr.__class__(expr.name, expr.scope, type=expr.type, parent=expr.parent,
                              source=expr.source)

    def map_array(self, expr, *args, **kwargs):
        if expr.dimensions:
            dimensions = self.rec(expr.dimensions, *args, **kwargs)
        else:
            dimensions = None
        return expr.__class__(expr.name, expr.scope, type=expr.type, parent=expr.parent,
                              dimensions=dimensions, source=expr.source)

    def map_array_subscript(self, expr, *args, **kwargs):
        return expr.__class__(self.rec(expr.index, *args, **kwargs))

    map_inline_call = IdentityMapper.map_call_with_kwargs

    def map_cast(self, expr, *args, **kwargs):
        if isinstance(expr.kind, Expression):
            kind = self.rec(expr.kind, *args, **kwargs)
        else:
            kind = expr.kind
        return expr.__class__(self.rec(expr.function, *args, **kwargs),
                              tuple(self.rec(p, *args, **kwargs) for p in expr.parameters),
                              kind=kind)

    def map_sum(self, expr, *args, **kwargs):
        return expr.__class__(tuple(self.rec(child, *args, **kwargs) for child in expr.children))

    def map_product(self, expr, *args, **kwargs):
        return expr.__class__(tuple(self.rec(child, *args, **kwargs) for child in expr.children))

    map_parenthesised_add = map_sum
    map_parenthesised_mul = map_product
    map_parenthesised_pow = IdentityMapper.map_power
    map_string_concat = map_sum

    map_range = IdentityMapper.map_slice
    map_range_index = IdentityMapper.map_slice
    map_loop_range = IdentityMapper.map_slice
    map_procedure_symbol = IdentityMapper.map_function_symbol

    def map_literal_list(self, expr, *args, **kwargs):
        values = tuple(v if isinstance(v, str) else self.rec(v, *args, **kwargs)
                       for v in expr.elements)
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
