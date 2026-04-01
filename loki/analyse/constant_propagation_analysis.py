# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import itertools
import math
import operator
from copy import deepcopy
from typing import Any

from loki import Transformer
from loki.analyse.abstract_dfa import AbstractDataflowAnalysis
from loki.expression import (
    Array, DeferredTypeSymbol, FloatLiteral, IntLiteral, LogicLiteral,
    LokiIdentityMapper, LoopRange, Product, RangeIndex, StringLiteral
)
from loki.expression.symbolic import get_pyrange
from loki.expression.symbols import _Literal

__all__ = ['ConstantPropagationAnalysis']


class ConstantPropagationAnalysis(AbstractDataflowAnalysis):
    """Scaffolding for constant-propagation analysis over Loki IR."""

    class Attacher(Transformer):
        """Attach placeholder constant maps without mutating the IR."""

        def __init__(self, parent, **kwargs):
            self.parent = parent
            super().__init__(inplace=False, invalidate_source=False, **kwargs)

        def visit_Node(self, o, **kwargs):
            constants_map = deepcopy(kwargs.get('constants_map', {}))
            o._update(_constants_map=constants_map)
            return super().visit_Node(o, **kwargs)

    class Detacher(Transformer):
        """Remove transient constant-propagation metadata from IR nodes."""

        def __init__(self, **kwargs):
            super().__init__(inplace=True, invalidate_source=False, **kwargs)

        def visit_Node(self, o, **kwargs):
            o._update(_constants_map=None)
            return super().visit_Node(o, **kwargs)

    class ConstPropMapper(LokiIdentityMapper):
        """Mapper for expression-level constant replacement and folding."""

        def __init__(self, fold_floats=True):
            self.fold_floats = fold_floats
            super().__init__()

        def map_array(self, expr, *args, **kwargs):
            constants_map = kwargs.get('constants_map', {})
            return constants_map.get((expr.basename, getattr(expr, 'dimensions', ())), expr)

        map_scalar = map_array
        map_deferred_type_symbol = map_array

        def map_constant(self, expr, *args, **kwargs):
            if isinstance(expr, int):
                return IntLiteral(expr)
            if isinstance(expr, float):
                return FloatLiteral(str(expr))
            if isinstance(expr, bool):
                return LogicLiteral(expr)
            return expr

        def map_sum(self, expr, *args, **kwargs):
            return self.binary_num_op_helper(expr, sum, math.fsum, *args, **kwargs)

        def map_product(self, expr, *args, **kwargs):
            mapped_product = self.binary_num_op_helper(expr, math.prod, math.prod, *args, **kwargs)
            children = getattr(mapped_product, 'children', None)
            if children and children[0] == IntLiteral(-1):
                mapped_product = Product((-1, children[1]))
            return mapped_product

        def map_quotient(self, expr, *args, **kwargs):
            return self.binary_num_op_helper(
                expr, operator.floordiv, operator.truediv,
                left_attr='numerator', right_attr='denominator', *args, **kwargs
            )

        def map_power(self, expr, *args, **kwargs):
            return self.binary_num_op_helper(
                expr, operator.pow, operator.pow,
                left_attr='base', right_attr='exponent', *args, **kwargs
            )

        def binary_num_op_helper(self, expr, int_op, float_op, *args, left_attr=None, right_attr=None, **kwargs):
            left = right = None
            lr_fields = not (left_attr is None and right_attr is None)
            if lr_fields:
                left = self.rec(getattr(expr, left_attr), *args, **kwargs)
                right = self.rec(getattr(expr, right_attr), *args, **kwargs)
                children = (left, right)
            else:
                children = self.rec(expr.children, *args, **kwargs)

            literals, non_literals = ConstantPropagationAnalysis._separate_literals(children)
            if not non_literals:
                if any(isinstance(v, FloatLiteral) for v in literals):
                    if self.fold_floats:
                        if lr_fields:
                            return FloatLiteral(str(float_op(float(left.value), float(right.value))))
                        return FloatLiteral(str(float_op([float(c.value) for c in children])))
                    return expr
                if lr_fields:
                    return IntLiteral(int_op(left.value, right.value))
                return IntLiteral(int_op([c.value for c in children]))

            if lr_fields:
                return expr.__class__(left, right)
            return expr.__class__(children)

        def map_logical_and(self, expr, *args, **kwargs):
            return self.binary_bool_op_helper(expr, lambda x, y: x and y, True, *args, **kwargs)

        def map_logical_or(self, expr, *args, **kwargs):
            return self.binary_bool_op_helper(expr, lambda x, y: x or y, False, *args, **kwargs)

        def binary_bool_op_helper(self, expr, bool_op, initial, *args, **kwargs):
            if LogicLiteral(not initial) in expr.children:
                return LogicLiteral(not initial)

            children = tuple(self.rec(c, *args, **kwargs) for c in expr.children)
            if LogicLiteral(not initial) in children:
                return LogicLiteral(not initial)

            _, non_literals = ConstantPropagationAnalysis._separate_literals(children)
            if not non_literals:
                return LogicLiteral(functools.reduce(bool_op, [c.value for c in children], initial))

            return expr.__class__(children)

        def map_logical_not(self, expr, *args, **kwargs):
            child = self.rec(expr.child, *args, **kwargs)
            _, non_literals = ConstantPropagationAnalysis._separate_literals([child])
            if not non_literals:
                return LogicLiteral(not child.value)
            return expr.__class__(child)

        def map_comparison(self, expr, *args, **kwargs):
            left = self.rec(expr.left, *args, **kwargs)
            right = self.rec(expr.right, *args, **kwargs)
            _, non_literals = ConstantPropagationAnalysis._separate_literals([left, right])
            if not non_literals:
                operators_map = {
                    'lt': operator.lt,
                    'le': operator.le,
                    'eq': operator.eq,
                    'ne': operator.ne,
                    'ge': operator.ge,
                    'gt': operator.gt,
                }
                operator_str = expr.operator if expr.operator in operators_map else expr.operator_to_name[expr.operator]
                return LogicLiteral(operators_map[operator_str](left.value, right.value))
            return expr.__class__(left, expr.operator, right)

        def map_loop_range(self, expr, *args, **kwargs):
            start = self.rec(expr.start, *args, **kwargs)
            stop = self.rec(expr.stop, *args, **kwargs)
            step = self.rec(expr.step, *args, **kwargs)
            return expr.__class__((start, stop, step))

        def map_string_concat(self, expr, *args, **kwargs):
            children = tuple(self.rec(c, *args, **kwargs) for c in expr.children)
            _, non_literals = ConstantPropagationAnalysis._separate_literals(children)
            if not non_literals:
                return StringLiteral(''.join(c.value for c in children))
            return expr.__class__(children)

    def __init__(self, fold_floats=True, unroll_loops=True, apply_transform=False):
        self.fold_floats = fold_floats
        self.unroll_loops = unroll_loops
        self.apply_transform = apply_transform

    def get_attacher(self) -> Any:
        return self.Attacher(self)

    def get_detacher(self) -> Any:
        return self.Detacher()

    def attach_dataflow_analysis(self, module_or_routine):
        constants_map = self.generate_declarations_map(module_or_routine)
        attacher = self.get_attacher()
        if hasattr(module_or_routine, 'spec'):
            attacher.visit(module_or_routine.spec, constants_map=deepcopy(constants_map))
        if hasattr(module_or_routine, 'body'):
            attacher.visit(module_or_routine.body, constants_map=deepcopy(constants_map))
        elif not hasattr(module_or_routine, 'spec'):
            attacher.visit(module_or_routine, constants_map=deepcopy(constants_map))

    def detach_dataflow_analysis(self, module_or_routine):
        detacher = self.get_detacher()
        if hasattr(module_or_routine, 'spec'):
            detacher.visit(module_or_routine.spec)
        if hasattr(module_or_routine, 'body'):
            detacher.visit(module_or_routine.body)
        elif not hasattr(module_or_routine, 'spec'):
            detacher.visit(module_or_routine)

    def generate_declarations_map(self, routine):
        """Build the initial constant map from declaration-time initializers."""

        def index_initial_elements(indices, element):
            if len(indices) == 1:
                return element.elements[indices[0].value - 1]
            return index_initial_elements(indices[1:], element.elements[indices[0].value - 1])

        declarations_map = {}
        for symbol in getattr(routine, 'symbols', ()):
            if isinstance(symbol, DeferredTypeSymbol) or symbol.initial is None:
                continue

            if isinstance(symbol, Array):
                declarations_map.update({
                    (symbol.basename, indices): index_initial_elements(indices, symbol.initial)
                    for indices in self._array_indices_to_accesses(
                        [RangeIndex((None, None, None))] * len(symbol.shape), symbol.shape
                    )
                })
            else:
                declarations_map[(symbol.basename, ())] = symbol.initial
        return declarations_map

    @staticmethod
    def _array_indices_to_accesses(dimensions, shape):
        accesses = functools.partial(itertools.product)
        for count, dimension in enumerate(dimensions):
            if isinstance(dimension, RangeIndex):
                start = dimension.start if dimension.start is not None else IntLiteral(1)
                stop = dimension.stop if dimension.stop is not None else shape[count]
                accesses = functools.partial(
                    accesses,
                    [IntLiteral(value) for value in get_pyrange(LoopRange((start, stop, dimension.step)))]
                )
            else:
                accesses = functools.partial(accesses, [dimension])
        return accesses()

    @staticmethod
    def _separate_literals(children):
        separated = ([], [])
        for child in children:
            if isinstance(child, _Literal):
                separated[0].append(child)
            else:
                separated[1].append(child)
        return separated
