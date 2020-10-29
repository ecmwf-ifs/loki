import enum
from functools import reduce
import operator
import pymbolic.primitives as pmbl

from loki.expression import symbols as sym, LokiIdentityMapper
from loki.tools import as_tuple

__all__ = ['Simplification', 'SimplifyMapper', 'simplify']


def is_minus_prefix(expr):
    """
    Return `True` if the given expression prefixes an expression with a minus sign,
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
    Sum up the values of all `IntLiteral` in the given list of expressions and return the accumulated
    value as an `IntLiteral` together with the remaining (not summed-up) components.
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


def mul_int_literals(components):
    """
    Multiply the values of all `IntLiteral` in the given list of expressions and return the
    resulting value as an `IntLiteral` together with the remaining (not multiplied) components.
    """
    def _process(child):
        if isinstance(child, int):
            return child, None
        if isinstance(child, sym.IntLiteral):
            return child.value, None
        if is_minus_prefix(child) and isinstance(child.children[1], sym.IntLiteral):
            return -child.children[1].value, None
        return 1, child

    transformed_components = list(zip(*[_process(child) for child in components]))
    value = reduce(operator.mul, transformed_components[0], 1)
    remaining_components = [ch for ch in transformed_components[1] if ch is not None]
    if value == 0:
        return [sym.IntLiteral(0)]
    if value == 1 and remaining_components:
        return remaining_components
    if value < 0:
        value = abs(value)
        return [sym.Product((-1, sym.IntLiteral(value)))] + remaining_components
    return [sym.IntLiteral(value)] + remaining_components


class Simplification(enum.Flag):
    """
    The selection of available simplification techniques that can be used to simplify expressions.
    """
    Flatten = enum.auto()
    IntegerArithmetic = enum.auto()

    ALL = Flatten | IntegerArithmetic


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
