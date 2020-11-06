from collections import defaultdict
import enum
from functools import reduce
import operator
import numpy as np
import pymbolic.primitives as pmbl

from loki.expression import symbols as sym, LokiIdentityMapper
from loki.tools import as_tuple

__all__ = ['Simplification', 'SimplifyMapper', 'simplify', 'accumulate_polynomial_terms']


def is_minus_prefix(expr):
    """
    Return `True` if the given expression prefixes a nested expression with a minus sign,
    else return `False`.

    It essentially means that `expr == Product((-1, other_expr))`.
    """
    if isinstance(expr, sym.Product) and expr.children and len(expr.children) == 2:
        return pmbl.is_zero(expr.children[0]+1)
    return False


def distribute_product(expr):
    """
    Flatten (nested) products into a sum of products.

    This converts for example `a * (b + c) * (d + e)` to
    `a * b * d + a * c * d + a * b * e + a * c * e`.
    """
    if not isinstance(expr, sym.Product):
        return expr

    queue = list(expr.children)
    done = [[]]

    while queue:
        item = queue.pop(0)

        if isinstance(item, sym.IntLiteral) and item.value == 1:
            continue

        if isinstance(item, sym.Product):
            # Prepend children to maintain order of operands
            queue = list(item.children) + queue
        elif isinstance(item, sym.Sum):
            # This is the distribution part
            old_done, done = done, []
            for child in item.children:
                done += [l + [child] for l in old_done]
        else:
            # Some other factor that we simply carry over
            done = [l + [item] for l in done]

    if not done:
        return sym.IntLiteral(1)

    # Form the new products, eliminating multiple `-1` in the process
    children = []
    for components in done:
        if -1 in components:
            is_neg = sum(1 for v in components if v == -1) % 2 == 1
            components = [v for v in components if v != -1]
            if not components:
                components = [sym.IntLiteral(1)]
            if is_neg:
                components = [-1] + components
        if not components:
            components = [sym.IntLiteral(1)]
        if len(components) == 1:
            children.append(components[0])
        else:
            children.append(sym.Product(as_tuple(components)))

    if len(children) == 1:
        return children[0]
    return sym.Sum(as_tuple(children))


def flatten_expr(expr):
    """
    Flatten an expression by flattening any sub-sums and distributing products.

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


def sum_int_literals(components):
    """
    Sum up the values of all `IntLiteral` in the given list of components of a sum
    and return the accumulated value as an `IntLiteral` in a list together with the
    remaining (not summed-up) components.
    """
    def _process(child):
        if isinstance(child, sym.IntLiteral):
            return child.value, None
        if is_minus_prefix(child) and isinstance(child.children[1], sym.IntLiteral):
            return -child.children[1].value, None
        return 0, child

    transformed_components = list(zip(*[_process(child) for child in components]))
    value = sum(transformed_components[0])
    remaining_components = [ch for ch in transformed_components[1] if ch is not None]
    if value == 0 and remaining_components:
        return remaining_components
    return [sym.IntLiteral(value)] + remaining_components


def separate_coefficients(components):
    """
    Helper routine that separates components of a product into constant coefficients
    and remaining factors.

    This is used in `mul_int_literals` and `collect_coefficients`.

    :param list components: the list of components containing constant and
                            non-constant sub-expressions.
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

    transformed_components = list(zip(*[_process(child) for child in components]))
    value = reduce(operator.mul, transformed_components[0], 1)
    remaining_components = [ch for ch in transformed_components[1] if ch is not None]

    return value, remaining_components


def mul_int_literals(components):
    """
    Multiply the values of all `IntLiteral` in the given list of expressions and return the
    resulting value as an `IntLiteral` together with the remaining (not multiplied) components.
    """
    value, remaining_components = separate_coefficients(components)
    if value == 0:
        return [sym.IntLiteral(0)]
    if value == 1 and remaining_components:
        return remaining_components
    if value < 0:
        value = abs(value)
        return [sym.Product((-1, sym.IntLiteral(value)))] + remaining_components
    return [sym.IntLiteral(value)] + remaining_components


def accumulate_polynomial_terms(components):
    """
    Collect all occurences of each base and determine the constant coefficient
    in a list of expressions.

    Note that this works for any non-constant sub-expression as "base" for summands and thus
    this can be applied not only to polynomials.

    :param list components: list of expressions, e.g., components of a :py:class:`sym.Sum`.
    :returns: mapping of base and corresponding coefficient
    :rtype: dict
    """
    summands = defaultdict(int)  # map (base, coefficient) pairs

    if isinstance(components, sym.Sum):
        components = components.children

    for item in as_tuple(components):
        if isinstance(item, sym.Product):
            value, remaining_components = separate_coefficients(item.children)
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


def collect_coefficients(components):
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

    summands = accumulate_polynomial_terms(components)
    result = []

    # Treat the constant part separately to make sure this is flat
    constant = summands.pop(1, 0)
    if constant < 0:
        result += [sym.Product((-1, sym.IntLiteral(abs(constant))))]
    elif constant > 0:
        result += [sym.IntLiteral(constant)]

    # Insert the remaining summands
    for base, factor in summands.items():
        if factor == 0:
            continue
        if factor == 1 and len(base) == 1:
            result.append(base[0])
        else:
            result.append(sym.Product(as_tuple(_get_coefficient(factor) + list(base))))

    return result or [sym.IntLiteral(0)]


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

    ALL = Flatten | IntegerArithmetic | CollectCoefficients


class SimplifyMapper(LokiIdentityMapper):
    """
    A mapper that attempts to symbolically simplify an expression.
    """
    # pylint: disable=abstract-method

    def __init__(self, enabled_simplifications=Simplification.ALL, invalidate_source=True):
        super().__init__(invalidate_source=invalidate_source)

        self.enabled_simplifications = enabled_simplifications

    def map_sum(self, expr, *args, **kwargs):
        children = [self.rec(child, *args, **kwargs) for child in expr.children]

        if self.enabled_simplifications & Simplification.Flatten:
            flat_sum = flatten_expr(sym.Sum(as_tuple(children)))
            if not isinstance(flat_sum, sym.Sum):
                return flat_sum
            children = flat_sum.children

        if self.enabled_simplifications & Simplification.IntegerArithmetic:
            children = sum_int_literals(children)

        if self.enabled_simplifications & Simplification.CollectCoefficients:
            children = collect_coefficients(children)

        if len(children) == 1:
            return children[0]
        return type(expr)(as_tuple(children))

    def map_product(self, expr, *args, **kwargs):
        children = [self.rec(child, *args, **kwargs) for child in expr.children]

        if self.enabled_simplifications & Simplification.Flatten:
            flat_prod = flatten_expr(sym.Product(as_tuple(children)))
            if not isinstance(flat_prod, sym.Product):
                return self.rec(flat_prod, *args, **kwargs)
            children = flat_prod.children

        if self.enabled_simplifications & Simplification.IntegerArithmetic:
            children = mul_int_literals(children)

        if len(children) == 1:
            return children[0]
        return type(expr)(as_tuple(children))

    map_parenthesised_add = map_sum
    map_parenthesised_mul = map_product


def simplify(expr, enabled_simplifications=Simplification.ALL):
    """
    Simplify the given expression by applying selected simplifications.
    """
    return SimplifyMapper(enabled_simplifications=enabled_simplifications)(expr)
