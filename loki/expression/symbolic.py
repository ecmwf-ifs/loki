# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
import enum
from functools import reduce
from math import gcd
import operator as _op
import numpy as np
import pymbolic.primitives as pmbl

from loki.expression.mappers import LokiIdentityMapper
import loki.expression.symbols as sym
from loki.tools import as_tuple

__all__ = [
    'is_constant', 'symbolic_op', 'simplify', 'accumulate_polynomial_terms',
    'Simplification', 'SimplifyMapper', 'is_dimension_constant'
]


def is_minus_prefix(expr):
    """
    Return `True` if the given expression prefixes a nested expression with a minus sign,
    else return `False`.

    It essentially means that `expr == Product((-1, ...))`.
    """
    if isinstance(expr, sym.Product) and expr.children:
        return pmbl.is_zero(expr.children[0]+1)
    return False


def strip_minus_prefix(expr):
    """
    Return the expression without the minus prefix.

    Raises a `ValueError` if the expression is not prefixed by a minus.
    """
    if not is_minus_prefix(expr):
        raise ValueError('Given expression does not have a minus prefix.')
    children = expr.children[1:]
    if len(children) == 1:
        return children[0]
    return sym.Product(as_tuple(children))


def is_constant(expr):
    """
    Return `True` if the given expression reduces to a constant value, else return `False`.
    """
    if is_minus_prefix(expr):
        return is_constant(strip_minus_prefix(expr))
    return pmbl.is_constant(expr)


def is_dimension_constant(d):
    """Establish if a given dimension symbol is a compile-time constant"""
    if isinstance(d, sym.IntLiteral):
        return True

    if isinstance(d, sym.RangeIndex):
        if d.lower:
            return is_dimension_constant(d.lower) and is_dimension_constant(d.upper)
        return is_dimension_constant(d.upper)

    if isinstance(d, sym.Scalar) and isinstance(d.initial , sym.IntLiteral):
        return True

    return False


def symbolic_op(expr1, op, expr2):
    """
    Evaluate `expr1 <op> expr2` (or equivalently, `op(expr1, expr2)`) and
    return the result.

    `op` can be any binary operation such as the rich comparison operators
    from the `operator` library.

    While calling this function largely equivalent to applying the operator
    directly, it is to be understood as a convenience layer that applies,
    depending on the operator, a number of symbolically neutral manipulations.
    Currently, this only applies to comparison operators (such as `eq`, `ne`,
    `lt, `le`, `gt`, `ge`). Since expression nodes do not imply an order,
    such comparisons would fail even if a symbolic meaning can be derived.

    For that reason, these operations are reformulated as the difference
    between the two expressions and compared against `0`. For example:
    ```
    # n < n + 1
    Scalar('n') < Sum((Scalar('n'), IntLiteral(1)))
    ```
    raises a `TypeError` but
    ```
    # n < n + 1
    symbolic_op(Scalar('n'), operator.lt, Sum((Scalar('n'), IntLiteral(1))))
    ```
    returns `True`.

    This is done by transforming this expression into
    ```
    # n - (n + 1) < 0
    Sum((Scalar('n'), Product(-1, Sum((Scalar('n'), IntLiteral(1)))))) < 0
    ```
    and then calling `simplify` on the left hand side to obtain
    ```
    # -1 < 0
    Product(-1, IntLiteral(1)) < 0
    ```
    In combination with stripping the minus prefix this yields the result.
    """
    if op in (_op.eq, _op.ne, _op.lt, _op.le, _op.gt, _op.ge):
        expr1, expr2 = simplify(expr1 - expr2), 0
        if is_minus_prefix(expr1):
            # Strip minus prefix to possibly yield constant expression
            if op in (_op.eq, _op.ne):
                return symbolic_op(strip_minus_prefix(expr1), op, expr2)
            return not symbolic_op(strip_minus_prefix(expr1), op, expr2)
    return op(expr1, expr2)


def distribute_product(expr):
    """
    Flatten (nested) products into a sum of products.

    This converts for example `a * (b + c) * (d + e)` to
    `a * b * d + a * c * d + a * b * e + a * c * e`.
    """
    def _retval(numerator, denominator):
        if not denominator:
            return numerator
        if len(denominator) == 1:
            return sym.Quotient(numerator, denominator[0])
        return sym.Quotient(numerator, sym.Product(as_tuple(denominator)))

    if not isinstance(expr, sym.Product):
        return expr

    queue = list(expr.children)
    denominator = []
    done = [[]]

    while queue:
        item = queue.pop(0)

        if isinstance(item, sym.IntLiteral) and item.value == 1:
            continue

        if isinstance(item, sym.Product):
            # Prepend children to maintain order of operands
            queue = list(item.children) + queue
        elif isinstance(item, sym.Quotient):
            # Enqueue the numerator and save the denominator for later
            queue = [item.numerator] + queue
            denominator += [item.denominator]
        elif isinstance(item, sym.Sum):
            # This is the distribution part
            old_done, done = done, []
            for child in item.children:
                done += [l + [child] for l in old_done]
        else:
            # Some other factor that we simply carry over
            done = [l + [item] for l in done]

    if not done:
        return _retval(sym.IntLiteral(1), denominator)

    # Form the new products, eliminating multiple `-1` in the process
    children = []
    for components in done:
        is_neg = False
        if -1 in components:
            is_neg = sum(1 for v in components if v == -1) % 2 == 1
            components = [v for v in components if v != -1]

        if not components:
            components = sym.IntLiteral(1)
        elif len(components) == 1:
            components = components[0]
        else:
            components = sym.Product(as_tuple(components))

        if is_neg:
            components = sym.Product((-1, components))
        children.append(components)

    if len(children) == 1:
        return _retval(children[0], denominator)
    return _retval(sym.Sum(as_tuple(children)), denominator)


def distribute_quotient(expr):
    """
    Flatten (nested) quotients into a sum of quotients.

    This converts for example `(a/b + c) / d` to `a / (b*d) + c / d`.
    """
    if not isinstance(expr, sym.Quotient):
        return expr

    if is_minus_prefix(expr.numerator):
        q = sym.Quotient(strip_minus_prefix(expr.numerator), expr.denominator)
        return sym.Product((-1, distribute_quotient(q)))

    if is_minus_prefix(expr.denominator):
        q = sym.Quotient(expr.numerator, strip_minus_prefix(expr.denominator))
        return sym.Product((-1, distribute_quotient(q)))

    queue = [expr.numerator]
    done = []

    while queue:
        item = queue.pop(0)

        if isinstance(item, sym.IntLiteral) and item.value == 0:
            continue

        if isinstance(item, sym.Sum):
            # Prepend children to maintain order of operands
            queue = list(item.children) + queue
        elif isinstance(item, sym.Quotient):
            done += [distribute_quotient(sym.Quotient(item.numerator, item.denominator * expr.denominator))]
        else:
            # Convert to a quotient
            done += [sym.Quotient(item, expr.denominator)]

    if not done:
        return sym.IntLiteral(1)
    if len(done) == 1:
        return done[0]
    return sym.Sum(as_tuple(done))


def flatten_expr(expr):
    """
    Flatten an expression by flattening any sub-sums and distributing products and quotients.

    This converts for example `a + (b - (c + d) * e)` to `a + b - c * e - d * e`.

    This is an (enhanced) re-implementation of the original `flattened_sum` routine from
    Pymbolic to account for the Loki-specific expression nodes and expand the flattening to
    distributing products.
    """
    queue = list(as_tuple(expr))
    done = []

    while queue:
        item = queue.pop(0)

        if pmbl.is_zero(item):
            continue

        if isinstance(item, sym.Product):
            item = distribute_product(item)

        if isinstance(item, sym.Quotient):
            item = distribute_quotient(item)

        if isinstance(item, sym.Sum):
            # Prepend children to maintain order of operands
            queue = list(item.children) + queue
        else:
            done.append(item)

    if not done:
        return sym.IntLiteral(0)
    if len(done) == 1:
        return done[0]
    return sym.Sum(as_tuple(done))


def sum_int_literals(expr):
    """
    Sum up the values of all `IntLiteral` in the sum and return the reduced sum.
    """
    def _process(child):
        if isinstance(child, sym.IntLiteral):
            return child.value, None
        if is_minus_prefix(child):
            value, stripped_child = _process(strip_minus_prefix(child))
            if value != 0:
                return -value, stripped_child
        return 0, child

    if not isinstance(expr, sym.Sum):
        return expr

    transformed_components = list(zip(*[_process(child) for child in expr.children]))
    value = sum(transformed_components[0])
    remaining_components = [ch for ch in transformed_components[1] if ch is not None]
    if value != 0:
        remaining_components = [sym.IntLiteral(value)] + remaining_components

    if not remaining_components:
        return sym.IntLiteral(0)
    if len(remaining_components) == 1:
        return remaining_components[0]
    return sym.Sum(as_tuple(remaining_components))


def separate_coefficients(expr):
    """
    Helper routine that separates components of a product into constant coefficients
    and remaining factors.

    :param sym.Product expr: the product comprising constant and non-constant sub-expressions.
    :returns: the constant coefficient and remaining non-constant sub-expressions.
    :rtype: (int, list)
    """
    def _process(child):
        if isinstance(child, (int, np.integer)):
            return child, None
        if isinstance(child, sym.IntLiteral):
            return child.value, None
        if is_minus_prefix(child):
            # We recurse here as products that are only there to change the sign
            # should not introduce a layer in the expression tree.
            value, component = _process(child.children[1])
            return -value, component
        return 1, child

    if isinstance(expr, sym.IntLiteral):
        return expr.value, []
    if not isinstance(expr, sym.Product):
        return 1, [expr]

    if is_minus_prefix(expr):
        value, remaining_components = separate_coefficients(strip_minus_prefix(expr))
        return -value, remaining_components

    transformed_components = list(zip(*[_process(child) for child in expr.children]))
    value = reduce(_op.mul, transformed_components[0], 1)
    remaining_components = [ch for ch in transformed_components[1] if ch is not None]
    return value, remaining_components


def mul_int_literals(expr):
    """
    Multiply all `IntLiteral` in the given `Product` and return the reduced expression.
    """
    if not isinstance(expr, sym.Product):
        return expr

    value, remaining_components = separate_coefficients(expr)
    if value == 0:
        return sym.IntLiteral(0)
    if abs(value) != 1:
        remaining_components = [sym.IntLiteral(abs(value))] + remaining_components

    if not remaining_components:
        ret = sym.IntLiteral(1)
    elif len(remaining_components) == 1:
        ret = remaining_components[0]
    else:
        ret = sym.Product(as_tuple(remaining_components))

    if value < 0:
        return sym.Product((-1, ret))
    return ret


def div_int_literals(expr):
    """
    Reduce fractions where the denominator is a `IntLiteral`.
    """
    if not isinstance(expr, sym.Quotient):
        return expr

    if is_minus_prefix(expr.numerator):
        q = sym.Quotient(strip_minus_prefix(expr.numerator), expr.denominator)
        return sym.Product((-1, div_int_literals(q)))

    if is_minus_prefix(expr.denominator):
        q = sym.Quotient(expr.numerator, strip_minus_prefix(expr.denominator))
        return sym.Product((-1, div_int_literals(q)))

    if not isinstance(expr.denominator, sym.IntLiteral):
        return expr

    if isinstance(expr.numerator, sym.IntLiteral):
        div = gcd(expr.numerator.value, expr.denominator.value)
        numerator = sym.IntLiteral(expr.numerator.value / div)
        denominator = sym.IntLiteral(expr.denominator.value / div)

    elif isinstance(expr.numerator, sym.Product):
        value, remaining_components = separate_coefficients(expr.numerator)
        div = gcd(value, expr.denominator.value)
        numerator = mul_int_literals(sym.Product((sym.IntLiteral(value / div), *remaining_components)))
        denominator = sym.IntLiteral(expr.denominator.value / div)

    else:
        numerator, denominator = expr.numerator, expr.denominator

    if denominator == 1:
        return numerator
    return sym.Quotient(numerator, denominator)


def accumulate_polynomial_terms(expr):
    """
    Collect all occurences of each base and determine the constant coefficient
    in a list of expressions.

    Note that this works for any non-constant sub-expression as "base" for summands and thus
    this can be applied not only to polynomials.

    :param list components: list of expressions, e.g., components of a :py:class:`sym.Sum`.
    :returns: mapping of base and corresponding coefficient
    :rtype: dict
    """
    if isinstance(expr, sym.Sum):
        components = expr.children
    else:
        components = as_tuple(expr)

    summands = defaultdict(int)  # map (base, coefficient) pairs
    for item in components:
        if isinstance(item, sym.Product):
            value, remaining_components = separate_coefficients(item)
            if value == 0:
                continue
            if not remaining_components:
                summands[1] += value
            else:
                # We sort the components using their string representation
                summands[as_tuple(sorted(remaining_components, key=str))] += value
        elif isinstance(item, (int, np.integer)):
            summands[1] += item
        elif isinstance(item, sym.IntLiteral):
            summands[1] += item.value
        else:
            summands[as_tuple(item)] += 1

    return dict(summands)


def collect_coefficients(expr):
    """
    Simplify a polynomial-type expression by combining all occurences of a non-constant
    subexpression into a single summand.

    :param list components: list of expressions, e.g., components of a :py:class:`sym.Sum`.
    :returns: reduced list of expressions.
    :rtype: list
    """
    def _get_coefficient(value):
        if value == 1:
            return []
        if value == -1:
            return [-1]
        if value < 0:
            return [-1, sym.IntLiteral(abs(value))]
        return [sym.IntLiteral(abs(value))]

    summands = accumulate_polynomial_terms(expr)
    components = []

    # Treat the constant part separately to make sure this is flat
    constant = summands.pop(1, 0)
    if constant < 0:
        components += [sym.Product((-1, sym.IntLiteral(abs(constant))))]
    elif constant > 0:
        components += [sym.IntLiteral(constant)]

    # Insert the remaining summands
    for base, factor in summands.items():
        if factor == 0:
            continue
        if factor == 1 and len(base) == 1:
            components.append(base[0])
        else:
            components.append(sym.Product(as_tuple(_get_coefficient(factor) + list(base))))

    if not components:
        return sym.IntLiteral(0)
    if len(components) == 1:
        return components[0]
    return sym.Sum(as_tuple(components))


class Simplification(enum.Flag):
    """
    The selection of available simplification techniques that can be used to simplify expressions.
    Multiple techniques can be combined using bitwise logical operations, for example:
    ```
    Flatten | IntegerArithmetic
    ALL & ~Flatten
    ```

    Attributes:
        Flatten             Flatten sub-sums and distribute products.
        IntegerArithmetic   Perform arithmetic on integer literals (addition and multiplication).
        CollectCoefficients Combine summands as far as possible.
        ALL                 All of the above.
    """
    Flatten = enum.auto()
    IntegerArithmetic = enum.auto()
    CollectCoefficients = enum.auto()

    ALL = Flatten | IntegerArithmetic | CollectCoefficients  # pylint: disable=unsupported-binary-operation


class SimplifyMapper(LokiIdentityMapper):
    """
    A mapper that attempts to symbolically simplify an expression.

    It applies all enabled simplifications from `Simplification` to a expression.
    """
    # pylint: disable=abstract-method

    def __init__(self, enabled_simplifications=Simplification.ALL, invalidate_source=True):
        super().__init__(invalidate_source=invalidate_source)

        self.enabled_simplifications = enabled_simplifications

    def map_sum(self, expr, *args, **kwargs):
        new_expr = sym.Sum(as_tuple([self.rec(child, *args, **kwargs) for child in expr.children]))

        if self.enabled_simplifications & Simplification.Flatten:
            new_expr = flatten_expr(new_expr)

        if self.enabled_simplifications & Simplification.IntegerArithmetic:
            new_expr = sum_int_literals(new_expr)

        if self.enabled_simplifications & Simplification.CollectCoefficients:
            new_expr = collect_coefficients(new_expr)

        if new_expr != expr:
            return self.rec(new_expr, *args, **kwargs)
        return expr

    def map_product(self, expr, *args, **kwargs):
        new_expr = sym.Product(as_tuple([self.rec(child, *args, **kwargs) for child in expr.children]))

        if self.enabled_simplifications & Simplification.Flatten:
            new_expr = flatten_expr(new_expr)

        if self.enabled_simplifications & Simplification.IntegerArithmetic:
            new_expr = mul_int_literals(new_expr)

        if new_expr != expr:
            return self.rec(new_expr, *args, **kwargs)
        return expr

    def map_quotient(self, expr, *args, **kwargs):
        numerator = self.rec(expr.numerator, *args, **kwargs)
        denominator = self.rec(expr.denominator, *args, **kwargs)
        new_expr = sym.Quotient(numerator, denominator)

        if self.enabled_simplifications & Simplification.Flatten:
            new_expr = flatten_expr(new_expr)

        if self.enabled_simplifications & Simplification.IntegerArithmetic:
            new_expr = div_int_literals(new_expr)

        if new_expr != expr:
            return self.rec(new_expr, *args, **kwargs)
        return expr

    map_parenthesised_add = map_sum
    map_parenthesised_mul = map_product
    map_parenthesised_div = map_quotient


def simplify(expr, enabled_simplifications=Simplification.ALL):
    """
    Simplify the given expression by applying selected simplifications.
    """
    return SimplifyMapper(enabled_simplifications=enabled_simplifications)(expr)
