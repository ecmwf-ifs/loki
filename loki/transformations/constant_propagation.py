# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import itertools
import operator

import math

from loki import FindNodes, LokiIdentityMapper, dataflow_analysis_attached, get_pyrange, DeferredTypeSymbol, Product
from loki.ir import Loop, Transformer, Conditional, Assignment
from loki.tools import as_tuple
from loki.transformations.transform_loop import LoopUnrollTransformer
from loki.expression.symbols import (
    _Literal, Array, RangeIndex, IntLiteral, FloatLiteral, LogicLiteral, StringLiteral, LoopRange
)

__all__ = ['ConstantPropagator']


class ConstantPropagator(Transformer):
    class ConstPropMapper(LokiIdentityMapper):
        def __init__(self, fold_floats=True):
            self.fold_floats = fold_floats
            super().__init__()

        def map_array(self, expr, *args, **kwargs):
            constants_map = kwargs.get('constants_map', dict())
            return constants_map.get((expr.basename, getattr(expr, 'dimensions', ())), expr)

        map_scalar = map_array
        map_deferred_type_symbol = map_array

        def map_constant(self, expr, *args, **kwargs):
            if isinstance(expr, int):
                return IntLiteral(expr)
            elif isinstance(expr, float):
                return FloatLiteral(str(expr))
            elif isinstance(expr, bool):
                return LogicLiteral(expr)

        def map_sum(self, expr, *args, **kwargs):
            return self.binary_num_op_helper(expr, sum, math.fsum, *args, **kwargs)

        def map_product(self, expr, *args, **kwargs):
            mapped_product = self.binary_num_op_helper(expr, math.prod, math.prod, *args, **kwargs)
            # Only way to get this here is if loki transformed `-expr` to `-1 * expr`, but couldn't const prop for expr
            if getattr(mapped_product, 'children', (False,))[0] == IntLiteral(-1):
                mapped_product = Product((-1, mapped_product.children[1]))
            return mapped_product

        def map_quotient(self, expr, *args, **kwargs):
            return self.binary_num_op_helper(expr, operator.floordiv, operator.truediv,
                                             left_attr='numerator', right_attr='denominator', *args, **kwargs)

        def map_power(self, expr, *args, **kwargs):
            return self.binary_num_op_helper(expr, operator.pow, operator.pow,
                                             left_attr='base', right_attr='exponent', *args, **kwargs)

        def binary_num_op_helper(self, expr, int_op, float_op, left_attr=None, right_attr=None, *args, **kwargs):
            left = right = None
            lr_fields = not (left_attr is None and right_attr is None)
            if lr_fields:
                left = self.rec(getattr(expr, left_attr), *args, **kwargs)
                right = self.rec(getattr(expr, right_attr), *args, **kwargs)
                # Just to make easier use of code below
                children = [left, right]
            else:
                children = self.rec(expr.children, *args, **kwargs)

            literals, non_literals = ConstantPropagator._separate_literals(children)
            if len(non_literals) == 0:
                if any([isinstance(v, FloatLiteral) for v in literals]):
                    # Strange rounding possibility
                    if self.fold_floats:
                        if lr_fields:
                            return FloatLiteral(str(float_op(float(left.value), float(right.value))))
                        else:
                            return FloatLiteral(str(float_op([float(c.value) for c in children])))
                else:
                    if lr_fields:
                        return IntLiteral(int_op(left.value, right.value))
                    else:
                        return IntLiteral(int_op([c.value for c in children]))

            if lr_fields:
                return expr.__class__(left, right)
            else:
                return expr.__class__(children)

        def map_logical_and(self, expr, *args, **kwargs):
            return self.binary_bool_op_helper(expr, lambda x, y: x and y, True, *args, **kwargs)

        def map_logical_or(self, expr, *args, **kwargs):
            return self.binary_bool_op_helper(expr, lambda x, y: x or y, False, *args, **kwargs)

        def binary_bool_op_helper(self, expr, bool_op, initial, *args, **kwargs):
            children = tuple([self.rec(c, *args, **kwargs) for c in expr.children])

            literals, non_literals = ConstantPropagator._separate_literals(children)
            if len(non_literals) == 0:
                return LogicLiteral(functools.reduce(bool_op, [c.value for c in children], initial))

            return expr.__class__(children)

        def map_logical_not(self, expr, *args, **kwargs):
            child = self.rec(expr.child, **kwargs)

            literals, non_literals = ConstantPropagator._separate_literals([child])
            if len(non_literals) == 0:
                return LogicLiteral(not child.value)

            return expr.__class__(child)

        def map_comparison(self, expr, *args, **kwargs):
            left = self.rec(expr.left, *args, **kwargs)
            right = self.rec(expr.right, *args, **kwargs)

            literals, non_literals = ConstantPropagator._separate_literals([left, right])
            if len(non_literals) == 0:
                # TODO: This should be a match statement >=3.10
                operators_map = {
                    'lt': operator.lt,
                    'le': operator.le,
                    'eq': operator.eq,
                    'ne': operator.ne,
                    'ge': operator.ge,
                    'gt': operator.gt,
                }
                operator_str = expr.operator if expr.operator in operators_map.keys() else expr.operator_to_name[expr.operator]
                return LogicLiteral(operators_map[operator_str](left.value, right.value))

            return expr.__class__(left, expr.operator, right)

        def map_loop_range(self, expr, *args, **kwargs):
            start = self.rec(expr.start, *args, **kwargs)
            stop = self.rec(expr.stop, *args, **kwargs)
            step = self.rec(expr.step, *args, **kwargs)
            return expr.__class__((start, stop, step))

        def map_string_concat(self, expr, *args, **kwargs):
            children = tuple([self.rec(c, *args, **kwargs) for c in expr.children])

            literals, non_literals = ConstantPropagator._separate_literals(children)
            if len(non_literals) == 0:
                return StringLiteral(''.join([c.value for c in children]))

            return expr.__class__(children)

    def __init__(self, fold_floats=True, unroll_loops=True):
        self.fold_floats = fold_floats
        self.unroll_loops = unroll_loops
        super().__init__()

    @staticmethod
    def _separate_literals(children):
        separated = ([], [])
        for c in children:
            # is_constant only covers int, float, & complex
            if isinstance(c, _Literal):
                separated[0].append(c)
            else:
                separated[1].append(c)
        return separated

    @staticmethod
    def _array_indices_to_accesses(dimensions, shape):
        accesses = functools.partial(itertools.product)
        for (count, dimension) in enumerate(dimensions):
            if isinstance(dimension, RangeIndex):
                start = dimension.start if dimension.start is not None else IntLiteral(1)
                # TODO: shape[] might not be as nice as we want
                stop = dimension.stop if dimension.stop is not None else shape[count]
                accesses = functools.partial(accesses, [IntLiteral(v) for v in
                                                        get_pyrange(LoopRange((start, stop, dimension.step)))])
            else:
                accesses = functools.partial(accesses, [dimension])

        return accesses()

    @staticmethod
    def generate_declarations_map(routine):
        def index_initial_elements(i, e):
            if len(i) == 1:
                return e.elements[i[0].value - 1]
            else:
                return index_initial_elements(i[1:], e.elements[i[0].value - 1])

        declarations_map = dict()
        with dataflow_analysis_attached(routine):
            for s in routine.symbols:
                if isinstance(s, DeferredTypeSymbol) or s.initial is None:
                    continue
                if isinstance(s, Array):
                    declarations_map.update({(s.basename, i): index_initial_elements(i, s.initial) for i in
                                             ConstantPropagator._array_indices_to_accesses(
                                                 [RangeIndex((None, None, None))] * len(s.shape), s.shape
                                             )})
                else:
                    declarations_map[(s.basename, ())] = s.initial
        return declarations_map

    def _pop_array_accesses(self, o, **kwargs):
        # Clear out the unknown dimensions
        constants_map = kwargs.get('constants_map', dict())

        # If the shape is unknown, then for now, just pop everything
        if o.lhs.shape is None:
            keys = constants_map.keys()
            for key in keys:
                if key[0] == o.lhs.name:
                    constants_map.pop(key)
            return
        literal_mask = [isinstance(d, _Literal) for d in o.lhs.dimensions]
        masked_accesses = [o.lhs.dimensions[i] if m else RangeIndex((None, None, None)) for i, m in
                           enumerate(literal_mask)]
        possible_accesses = self._array_indices_to_accesses(masked_accesses, self.ConstPropMapper(self.fold_floats)(o.lhs.shape, **kwargs))
        for access in possible_accesses:
            constants_map.pop((o.lhs.basename, access), None)

    def visit_Assignment(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', dict())

        new_rhs = self.ConstPropMapper(self.fold_floats)(o.rhs, **kwargs)
        o = Assignment(
            o.lhs,
            new_rhs,
            o.ptr,
            o.comment
        )

        # What if the lhs isn't a scalar shape?
        if isinstance(o.lhs, Array):
            new_dimensions = [self.ConstPropMapper(self.fold_floats)(d, **kwargs) for d in o.lhs.dimensions]
            _, new_d_non_literals = self._separate_literals(new_dimensions)

            new_lhs = Array(o.lhs.name, o.lhs.scope, o.lhs.type, as_tuple(new_dimensions))
            o = Assignment(
                new_lhs,
                o.rhs,
                o.ptr,
                o.comment
            )
            if len(new_d_non_literals) != 0:
                self._pop_array_accesses(o, **kwargs)
                return o

        literals, non_literals = self._separate_literals([new_rhs])
        if len(non_literals) == 0:
            if isinstance(o.lhs, Array):
                for access in self._array_indices_to_accesses(o.lhs.dimensions, o.lhs.shape):
                    constants_map[(o.lhs.basename, access)] = new_rhs
            else:
                constants_map[(o.lhs.basename, ())] = new_rhs
        else:
            # TODO: What if it's a pointer
            if isinstance(o.lhs, Array):
                for access in self._array_indices_to_accesses(o.lhs.dimensions, o.lhs.shape):
                    constants_map.pop((o.lhs.basename, access), None)
            else:
                constants_map.pop((o.lhs.basename, ()), None)

        return o

    def visit(self, o, *args, **kwargs):
        constants_map = kwargs.pop('constants_map', dict())
        return super().visit(o, *args, constants_map=constants_map, **kwargs)

    def visit_Conditional(self, o, **kwargs):
        constants_map = kwargs.pop('constants_map', dict())
        new_condition = self.ConstPropMapper(self.fold_floats)(o.condition, constants_map=constants_map, **kwargs)
        body_constants_map = constants_map.copy()
        else_body_constants_map = constants_map.copy()
        new_body = self.visit(o.body, constants_map=body_constants_map, **kwargs)
        new_else_body = self.visit(o.else_body, constants_map=else_body_constants_map, **kwargs)

        o = Conditional(
            condition=new_condition,
            body=new_body,
            else_body=new_else_body,
            inline=o.inline,
            has_elseif=o.has_elseif,
            name=o.name
        )

        for key in set(body_constants_map.keys()).union(else_body_constants_map):
            if body_constants_map.get(key, None) == else_body_constants_map.get(key, None):
                constants_map[key] = body_constants_map[key]
            else:
                constants_map.pop(key, None)

        return o

    def visit_Loop(self, o, **kwargs):
        constants_map = kwargs.pop('constants_map', dict())
        constants_map.pop((o.variable.basename, ()), None)

        new_bounds = self.ConstPropMapper(self.fold_floats)(o.bounds, constants_map=constants_map, **kwargs)
        o = Loop(
            variable=o.variable,
            body=o.body,
            bounds=new_bounds
        )

        if self.unroll_loops:
            unrolled = LoopUnrollTransformer(warn_iterations_length=False).visit(o)
            return self.visit(unrolled, constants_map=constants_map, **kwargs)

        # TODO: If the last assignment to a variable is only derived from loop invariant variables,
        # then we know the value coming out of the loop. I.e. a form of invariant code motion with
        # reaching definition analysis
        for a in FindNodes(Assignment).visit(o.body):
            if isinstance(a.lhs, Array):
                self._pop_array_accesses(a, constants_map=constants_map, **kwargs)
            else:
                constants_map.pop((a.lhs.basename, ()), None)

        new_body = self.visit(o.body, constants_map=constants_map, **kwargs)

        return Loop(
            variable=o.variable,
            body=new_body,
            bounds=o.bounds
        )
