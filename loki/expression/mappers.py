# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Mappers for traversing and transforming the
:ref:`internal_representation:Expression tree`.
"""
from copy import deepcopy
import re
from itertools import zip_longest
import pymbolic.primitives as pmbl
from pymbolic.mapper import Mapper, WalkMapper, CombineMapper, IdentityMapper
from pymbolic.mapper.stringifier import (
    StringifyMapper, PREC_NONE, PREC_SUM, PREC_CALL, PREC_PRODUCT
)
try:
    from fparser.two.Fortran2003 import Intrinsic_Name
    _intrinsic_fortran_names = Intrinsic_Name.function_names
except ImportError:
    _intrinsic_fortran_names = ()

from loki.logging import debug
from loki.tools import as_tuple, flatten
from loki.types import SymbolAttributes, BasicType


__all__ = ['LokiStringifyMapper', 'ExpressionRetriever', 'ExpressionDimensionsMapper',
           'ExpressionCallbackMapper', 'SubstituteExpressionsMapper',
           'LokiIdentityMapper', 'AttachScopesMapper', 'DetachScopesMapper']



class LokiStringifyMapper(StringifyMapper):
    """
    A class derived from the default :class:`StringifyMapper` that adds mappings for nodes of the
    expression tree that we added ourselves.

    This is the default pretty printer for nodes in the expression tree.
    """
    # pylint: disable=unused-argument,abstract-method

    _regex_string_literal = re.compile(r"((?<!')'(?:'')*(?!'))")

    def __init__(self, *args, **kwargs):
        from loki.expression import operations as op  # pylint: disable=import-outside-toplevel,cyclic-import
        super().__init__(*args, **kwargs)

        # This should really be a class property but due to the circular dependency
        # (Pymbolic expression nodes requiring `LokiStringifyMapper` for `make_stringifier`)
        # we cannot perform the relevant import on a module level
        self.parenthesised_multiplicative_primitives = (
            op.ParenthesisedAdd, op.ParenthesisedMul,
            op.ParenthesisedDiv, op.ParenthesisedPow
        )

    def rec_with_force_parens_around(self, expr, *args, **kwargs):
        # Re-implement here to add no_force_parens_around
        force_parens_around = kwargs.pop("force_parens_around", ())
        no_force_parens_around = kwargs.pop("no_force_parens_around",
                                            self.parenthesised_multiplicative_primitives)

        result = self.rec(expr, *args, **kwargs)

        if isinstance(expr, force_parens_around) and not isinstance(expr, no_force_parens_around):
            result = f"({result})"

        return result

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return str(expr.value)

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            return f'{str(expr.value)}_{str(expr.kind)}'
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
    map_procedure_symbol = map_variable_symbol

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
        from loki.expression.operations import ParenthesisedMul  # pylint: disable=import-outside-toplevel,cyclic-import
        def get_op_prec_expr(expr):
            if isinstance(expr, pmbl.Product) and not isinstance(expr, ParenthesisedMul):
                if pmbl.is_zero(expr.children[0]+1):
                    if len(expr.children) == 2:
                        # only the minus sign and the other child
                        return '-', PREC_PRODUCT, expr.children[1]
                    return '-', PREC_PRODUCT, expr.__class__(expr.children[1:])
            return '+', PREC_SUM, expr

        terms = []
        for ch in expr.children:
            op, prec, expr = get_op_prec_expr(ch)
            terms += [op, self.rec(expr, prec, *args, **kwargs)]

        # Remove leading '+'
        if terms[0] == '-':
            terms[1] = f'{terms[0]}{terms[1]}'
        terms = terms[1:]

        return self.parenthesize_if_needed(self.join(' ', terms), enclosing_prec, PREC_SUM)

    def map_product(self, expr, enclosing_prec, *args, **kwargs):
        if len(expr.children) == 2 and expr.children[0] == -1:
            # Negative values are encoded as multiplication by (-1) (constant, not IntLiteral).
            # We replace this by a minus again
            return self.parenthesize_if_needed(
                f'-{self.join_rec("*", expr.children[1:], PREC_PRODUCT, *args, **kwargs)}',
                enclosing_prec, PREC_PRODUCT)
        # Make Pymbolic's default bracketing less conservative by not enforcing
        # parenthesis around products nested in a product, which can cause
        # round-off deviations for agressively optimising compilers
        kwargs['force_parens_around'] = (pmbl.FloorDiv, pmbl.Remainder)
        return self.parenthesize_if_needed(
                self.join_rec("*", expr.children, PREC_PRODUCT, *args, **kwargs),
                enclosing_prec, PREC_PRODUCT)

    def map_quotient(self, expr, enclosing_prec, *args, **kwargs):
        # Similar to products we drop the conservative parenthesis around products and
        # quotients for the numerator
        kwargs['force_parens_around'] = (pmbl.FloorDiv, pmbl.Remainder)
        numerator = self.rec_with_force_parens_around(expr.numerator, PREC_PRODUCT, *args, **kwargs)
        kwargs['force_parens_around'] = self.multiplicative_primitives
        denominator = self.rec_with_force_parens_around(expr.denominator, PREC_PRODUCT, *args, **kwargs)
        return self.parenthesize_if_needed(self.format('%s / %s', numerator, denominator),
                                           enclosing_prec, PREC_PRODUCT)

    def map_parenthesised_add(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_sum(expr, PREC_NONE, *args, **kwargs))

    def map_parenthesised_mul(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_product(expr, PREC_NONE, *args, **kwargs))

    def map_parenthesised_div(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_quotient(expr, PREC_NONE, *args, **kwargs))

    def map_parenthesised_pow(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize(self.map_power(expr, PREC_NONE, *args, **kwargs))

    def map_string_concat(self, expr, enclosing_prec, *args, **kwargs):
        return ' // '.join(self.rec(c, enclosing_prec, *args, **kwargs) for c in expr.children)

    def map_literal_list(self, expr, enclosing_prec, *args, **kwargs):
        values = ', '.join(self.rec(c, PREC_NONE, *args, **kwargs) for c in expr.elements)
        if expr.dtype is not None:
            return f'[ {str(expr.dtype)} :: {values} ]'
        return f'[ {values} ]'

    def map_inline_do(self, expr, enclosing_prec, *args, **kwargs):
        assert len(expr.values) == 1
        values = self.rec(expr.values[0], PREC_NONE, *args, **kwargs)
        variable = self.rec(expr.variable, PREC_NONE, *args, **kwargs)
        bounds = self.rec(expr.bounds, PREC_NONE, *args, **kwargs)
        return f'( {values}, {variable} = {bounds} )'

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        name_str = self.rec(expr.aggregate, PREC_NONE, *args, **kwargs)
        index_str = self.join_rec(', ', expr.index_tuple, PREC_NONE, *args, **kwargs)
        return f'{name_str}({index_str})'

    map_string_subscript = map_array_subscript


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

    map_string_subscript = map_array_subscript

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
    map_parenthesised_div = WalkMapper.map_quotient
    map_parenthesised_pow = WalkMapper.map_power
    map_string_concat = WalkMapper.map_sum

    def map_literal_list(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        for elem in expr.elements:
            if not isinstance(elem, str):
                # TODO: We are not representing all cases properly
                # (e.g., implied loops) and instead retain them as plain
                # strings. Do not recurse on those for the moment...
                self.rec(elem, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    def map_inline_do(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        self.rec(expr.values, *args, **kwargs)
        self.rec(expr.variable, *args, **kwargs)
        self.rec(expr.bounds, *args, **kwargs)
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
    # pylint: disable=abstract-method

    def map_algebraic_leaf(self, expr, *args, **kwargs):
        # pylint: disable=import-outside-toplevel,cyclic-import
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
        raise ValueError(f'Symbol with deferred type encountered: {expr}')

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

    map_string_subscript = map_algebraic_leaf

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
            elif child_dim not in (dim, 1):
                raise ValueError(f'Non-matching dimensions: {str(dim)} and {str(child_dim)}')
        return dim

    map_product = map_sum

    def map_inline_do(self, expr, *args, **kwargs):
        return self.rec(expr.bounds, *args, **kwargs)


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

    map_string_subscript = map_array_subscript

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
    map_parenthesised_div = CombineMapper.map_quotient
    map_parenthesised_pow = CombineMapper.map_power
    map_string_concat = CombineMapper.map_sum

    def map_literal_list(self, expr, *args, **kwargs):
        return self.combine(tuple(self.rec(c, *args, **kwargs) for c in expr.elements))

    def map_inline_do(self, expr, *args, **kwargs):
        return self.combine(tuple(
            self.rec(expr.values, *args, **kwargs),
            self.rec(expr.variable, *args, **kwargs),
            self.rec(expr.bounds, *args, **kwargs)
        ))

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
        kwargs.setdefault('recurse_to_declaration_attributes', False)
        new_expr = super().__call__(expr, *args, **kwargs)
        if getattr(expr, 'source', None):
            if isinstance(new_expr, tuple):
                for e in new_expr:
                    if self.invalidate_source:
                        e.source = None
                    else:
                        e.source = deepcopy(expr.source)
            else:
                if self.invalidate_source:
                    new_expr.source = None
                else:
                    new_expr.source = deepcopy(expr.source)
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
        # When updating declaration attributes, which are stored in the symbol table,
        # we need to disable `recurse_to_declaration_attributes` to avoid infinite
        # recursion because of the various ways that Fortran allows to use the declared
        # symbol also inside the declaration expression
        recurse_to_declaration_attributes = kwargs['recurse_to_declaration_attributes'] or expr.scope is None
        kwargs['recurse_to_declaration_attributes'] = False

        if recurse_to_declaration_attributes:
            old_type = expr.type
            kind = self.rec(old_type.kind, *args, **kwargs)

            if expr.scope and expr.name == old_type.initial:
                # FIXME: This is a hack to work around situations where a constant
                # symbol (from a parent scope) with the same name as the declared
                # variable is used as initializer. This hands down the correct scope
                # (in case this traversal is part of ``AttachScopesMapper``) and thus
                # interrupts an otherwise infinite recursion (see LOKI-52).
                _kwargs = kwargs.copy()
                _kwargs['scope'] = expr.scope.parent
                initial = self.rec(old_type.initial, *args, **_kwargs)
            else:
                initial = self.rec(old_type.initial, *args, **kwargs)

            if old_type.bind_names:
                bind_names = ()
                for bind_name in old_type.bind_names:
                    if bind_name == expr.name:
                        # FIXME: This is a hack to work around situations where an
                        # explicit interface is used with the same name as the
                        # type bound procedure. This hands down the correct scope.
                        _kwargs = kwargs.copy()
                        _kwargs['scope'] = expr.scope.parent
                        bind_names += (self.rec(bind_name, *args, **_kwargs),)
                    else:
                        bind_names += (self.rec(bind_name, *args, **kwargs),)
            else:
                bind_names = None

            is_type_changed = (
                kind is not old_type.kind or initial is not old_type.initial or
                any(new is not old for new, old in zip_longest(as_tuple(bind_names), as_tuple(old_type.bind_names)))
            )
            if is_type_changed:
                new_type = old_type.clone(kind=kind, initial=initial, bind_names=bind_names)
                if expr.scope:
                    # Update symbol table entry
                    expr.scope.symbol_attrs[expr.name] = new_type

        parent = self.rec(expr.parent, *args, **kwargs)
        if expr.scope is None:
            if parent is expr.parent and not is_type_changed:
                return expr
            return expr.clone(parent=parent, type=new_type)

        if parent is expr.parent:
            return expr
        return expr.clone(parent=parent)

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
        from loki.expression.symbols import ProcedureSymbol, InlineCall  # pylint: disable=import-outside-toplevel
        symbol = self.rec(expr.symbol, *args, **kwargs)
        parent = self.rec(expr.parent, *args, **kwargs) if expr.parent else None
        dimensions = self.rec(expr.dimensions, *args, **kwargs)
        if isinstance(symbol, ProcedureSymbol):
            # Workaround for frontend deficiencies: Fparser may wrongfully
            # classify an inline call as an array, which may later on be
            # corrected thanks to type information in the symbol table.
            # When this happens, we need to convert this to an inline call here
            # and make sure we don't loose the call parameters (aka dimensions)
            return InlineCall(function=symbol.clone(parent=parent), parameters=dimensions)

        if kwargs['recurse_to_declaration_attributes']:
            _kwargs = kwargs.copy()
            _kwargs['recurse_to_declaration_attributes'] = False
            shape = self.rec(symbol.type.shape, *args, **_kwargs)
        else:
            shape = symbol.type.shape

        if (getattr(symbol, 'symbol', symbol) is expr.symbol and
                all(d is orig_d for d, orig_d in zip_longest(dimensions or (), expr.dimensions or ())) and
                all(d is orig_d for d, orig_d in zip_longest(shape or (), symbol.type.shape or ()))):
            return expr
        return symbol.clone(dimensions=dimensions, type=symbol.type.clone(shape=shape), parent=parent)

    def map_array_subscript(self, expr, *args, **kwargs):
        raise RuntimeError('Recursion should have ended at map_array')

    def map_string_subscript(self, expr, *args, **kwargs):
        symbol = self.rec(expr.symbol, *args, **kwargs)
        index_tuple = self.rec(expr.index_tuple, *args, **kwargs)
        return expr.__class__(symbol, index_tuple)

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

    def map_quotient(self, expr, *args, **kwargs):
        return expr.__class__(self.rec(expr.numerator, *args, **kwargs),
                              self.rec(expr.denominator, *args, **kwargs))

    map_parenthesised_add = map_sum
    map_product = map_sum
    map_parenthesised_mul = map_product
    map_parenthesised_div = map_quotient
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
        return expr.__class__(values, dtype=expr.dtype)

    def map_inline_do(self, expr, *args, **kwargs):
        values = self.rec(expr.values, *args, **kwargs)
        variable = self.rec(expr.variable, *args, **kwargs)
        bounds = self.rec(expr.bounds, *args, **kwargs)
        return expr.__class__(values, variable, bounds)


class SubstituteExpressionsMapper(LokiIdentityMapper):
    """
    A Pymbolic expression mapper (i.e., a visitor for the expression tree) that
    defines on-the-fly handlers from a given substitution map.

    It returns a copy of the expression tree with expressions substituted according
    to the given :data:`expr_map`. If an expression node is encountered that is
    found in :data:`expr_map`, it is replaced with the corresponding expression from
    the map. For any other nodes, traversal is performed via :any:`LokiIdentityMapper`.

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
    # pylint: disable=abstract-method

    def __init__(self, expr_map, invalidate_source=True):
        super().__init__(invalidate_source=invalidate_source)

        self.expr_map = expr_map
        for expr in self.expr_map.keys():
            setattr(self, expr.mapper_method, self.map_from_expr_map)

    def map_from_expr_map(self, expr, *args, **kwargs):
        """
        Replace an expr with its substitution, if found in the :attr:`expr_map`,
        otherwise continue tree traversal
        """
        if expr in self.expr_map:
            return self.expr_map[expr]
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
        """
        Find the scope of :data:`expr` and, if it is different,
        attach the new scope and return the symbol
        """
        symbol_scope = scope.get_symbol_scope(expr.name)
        if symbol_scope is None and '%' in expr.name:
            symbol_scope = scope.get_symbol_scope(expr.name_parts[0])
        if symbol_scope is not None:
            if symbol_scope is not expr.scope:
                expr = expr.rescope(symbol_scope)
        elif self.fail:
            raise RuntimeError(f'AttachScopesMapper: {expr!s} was not found in any scope')
        elif expr not in _intrinsic_fortran_names:
            debug('AttachScopesMapper: %s was not found in any scopes', str(expr))
        return expr

    def map_variable_symbol(self, expr, *args, **kwargs):
        """
        Handler for :class:`VariableSymbol`

        This updates the symbol's scope via :meth:`_update_symbol_scope`
        and then calls the parent class handler routine

        Note: this may be a different handler as attaching the scope and therefore
        type may change a symbol's type, e.g. from :class:`DeferredTypeSymbol` to :class:`Scalar`
        """
        new_expr = self._update_symbol_scope(expr, kwargs['scope'])
        if new_expr.scope and new_expr.scope is not kwargs['scope']:
            # We call the parent handler to take care of properties like initial value, kind etc.,
            # all of which should be declared at or above the scope of the expression
            kwargs['scope'] = new_expr.scope
        map_fn = getattr(super(), new_expr.mapper_method)

        # If we cannot resolve scope or type of an expression, we mark it as deferred
        if not new_expr.scope and not new_expr.type:
            new_expr.type = SymbolAttributes(dtype=BasicType.DEFERRED)

        return map_fn(new_expr, *args, **kwargs)

    map_deferred_type_symbol = map_variable_symbol
    map_procedure_symbol = map_variable_symbol


class DetachScopesMapper(LokiIdentityMapper):
    """
    A Pymbolic expression mapper (i.e., a visitor for the expression tree)
    that rebuilds an expression unchanged but with the scope for every
    :any:`TypedSymbol` detached.

    This will ensure that type information is stored locally on the object
    itself, which is useful when storing information for inter-procedural
    analysis passes.
    """

    def __init__(self):
        super().__init__(invalidate_source=False)

    def map_variable_symbol(self, expr, *args, **kwargs):
        new_expr = super().map_variable_symbol(expr, *args, **kwargs)
        new_expr = new_expr.clone(scope=None)
        return new_expr

    map_deferred_type_symbol = map_variable_symbol
    map_procedure_symbol = map_variable_symbol
