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

from loki.expression.symbols import (
    _Literal, Array, RangeIndex, IntLiteral, FloatLiteral, LogicLiteral, LoopRange, StringLiteral
)
from loki.ir import (
    Loop, NestedTransformer, Conditional, Assignment
)
from loki.tools import as_tuple, flatten

from loki.transformations.transform_loop import LoopUnrollTransformer

__all__ = ['ConstantPropagator']


class ConstantPropagator(NestedTransformer):

    def __init__(self, fold_floats=True, unroll_loops=True):
        self.fold_floats = fold_floats
        self.unroll_loops = unroll_loops
        super().__init__()

    # TODO: Static method
    def separate_literals(self, children):
        separated = ([], [])
        for c in children:
            if isinstance(c, _Literal):
                separated[0].append(c)
            else:
                separated[1].append(c)
        return separated

    def visit_Assignment(self, o, **kwargs):
        new_rhs = self.visit(o.rhs, **kwargs)
        # o.rhs = new_rhs
        o = Assignment(
            o.lhs,
            new_rhs,
            o.ptr,
            o.comment
        )

        constants_map = kwargs.get('constants_map', dict())
        # What if the lhs isn't a scalar shape?
        if isinstance(o.lhs, Array):
            new_dimensions = [self.visit(d, **kwargs) for d in o.lhs.dimensions]
            new_d_literals, new_d_non_literals = self.separate_literals(new_dimensions)

            new_lhs = Array(o.lhs.name, o.lhs.scope, o.lhs.type, as_tuple(new_dimensions))
            o.lhs = new_lhs
            if len(new_d_non_literals) != 0:
                # Clear out the unknown dimensions

                # If the shape is unknown, then for now, just pop everything
                if o.lhs.shape is None:
                    keys = constants_map.keys()
                    for key in keys:
                        if key[0] == o.lhs.name:
                            constants_map.pop(key)
                    return o

                literal_mask = [isinstance(d, _Literal) for d in new_dimensions]

                masked_accesses = [new_dimensions[i] if m else RangeIndex((None, None, None)) for i, m in enumerate(literal_mask)]
                possible_accesses = self.array_indices_to_accesses(masked_accesses, self.visit(o.lhs.shape, **kwargs))

                for access in possible_accesses:
                    constants_map.pop((o.lhs.basename, access), None)
                return o

        literals, non_literals = self.separate_literals([new_rhs])
        if len(non_literals) == 0:
            if isinstance(o.lhs, Array):
                for access in self.array_indices_to_accesses(o.lhs.dimensions, o.lhs.shape):
                    constants_map[(o.lhs.basename, access)] = new_rhs
            else:
                constants_map[(o.lhs.basename, ())] = new_rhs
        else:
            # TODO: What if it's a pointer
            if isinstance(o.lhs, Array):
                for access in self.array_indices_to_accesses(o.lhs.dimensions, o.lhs.shape):
                    constants_map.pop((o.lhs.basename, access), None)
            else:
                constants_map.pop((o.lhs.basename, ()), None)

        return o

    def array_indices_to_accesses(self, dimensions, shape):
        accesses = functools.partial(itertools.product)
        for (count, dimension) in enumerate(dimensions):
            if isinstance(dimension, RangeIndex):
                start = dimensions[count].start if dimensions[count].start is not None else IntLiteral(1)
                # TODO: shape[] might not be as nice as we want
                stop = dimensions[count].stop if dimensions[count].stop is not None else shape[count]
                step = dimensions[count].step if dimensions[count].step is not None else IntLiteral(1)
                accesses = functools.partial(accesses, [IntLiteral(v) for v in
                                                        range(start.value, stop.value + 1, step.value)])
            else:
                accesses = functools.partial(accesses, [dimension])

        return accesses()

    def visit_Array(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', dict())
        return constants_map.get((o.basename, getattr(o, 'dimensions', ())), o)

    def visit_Scalar(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', dict())
        return constants_map.get((o.basename, ()), o)

    def visit_DeferredTypeSymbol(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', dict())
        return constants_map.get((o.basename, ()), o)

    def visit_Sum(self, o, **kwargs):
        new_children = tuple([self.visit(c, **kwargs) for c in o.children])
        o.children = new_children

        literals, non_literals = self.separate_literals(new_children)
        if len(non_literals) == 0:
            if any([isinstance(v, FloatLiteral) for v in literals]):
                # Strange rounding possibility
                if self.fold_floats:
                    return FloatLiteral(str(math.fsum([float(c.value) for c in new_children])))
                else:
                    return o
            else:
                return IntLiteral(sum([c.value for c in new_children]))

        return o

    def visit_Product(self, o, **kwargs):
        new_children = tuple([self.visit(c, **kwargs) for c in o.children])
        o.children = new_children

        literals, non_literals = self.separate_literals(new_children)
        if len(non_literals) == 0:
            if any([isinstance(v, FloatLiteral) for v in literals]):
                # Strange rounding possibility
                if self.fold_floats:
                    return FloatLiteral(str(math.prod([float(c.value) for c in new_children])))
                else:
                    return o
            else:
                return IntLiteral(math.prod([c.value for c in new_children]))

        return o

    def visit_Quotient(self, o, **kwargs):
        new_numerator = self.visit(o.numerator, **kwargs)
        new_denominator = self.visit(o.denominator, **kwargs)
        o.numerator = new_numerator
        o.denominator = new_denominator

        literals, non_literals = self.separate_literals([new_denominator, new_numerator])
        if len(non_literals) == 0:
            if any([isinstance(v, FloatLiteral) for v in literals]):
                # Strange rounding possibility
                if self.fold_floats:
                    # TODO: This could be a zero
                    return FloatLiteral(str(float(new_numerator.value) / float(new_denominator.value)))
                else:
                    return o
            else:
                # TODO: This could be a zero
                return IntLiteral(new_numerator.value // new_denominator.value)

        return o

    def visit_Power(self, o, **kwargs):
        new_base = self.visit(o.base, **kwargs)
        new_exponent = self.visit(o.exponent, **kwargs)
        o.base = new_base
        o.exponent = new_exponent

        literals, non_literals = self.separate_literals([new_base, new_exponent])
        if len(non_literals) == 0:
            if any([isinstance(v, FloatLiteral) for v in literals]):
                # Strange rounding possibility
                if self.fold_floats:
                    return FloatLiteral(new_base.value ** new_exponent.value)
                else:
                    return o
            else:
                return IntLiteral(new_base.value ** new_exponent.value)

    def visit_Comparison(self, o, **kwargs):
        new_left = self.visit(o.left, **kwargs)
        new_right = self.visit(o.right, **kwargs)

        literals, non_literals = self.separate_literals([new_left, new_right])
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
            operator_str = o.operator if o.operator in operators_map.keys() else o.operator_to_name[o.operator]
            return LogicLiteral(operators_map[operator_str](new_left.value, new_right.value))

        o.left = new_left
        o.right = new_right
        return o

    def visit_LogicalAnd(self, o, **kwargs):
        new_children = tuple([self.visit(c, **kwargs) for c in o.children])

        literals, non_literals = self.separate_literals(new_children)
        if len(non_literals) == 0:
            return LogicLiteral(functools.reduce(lambda x, y: x and y, [c.value for c in new_children], True))

        o.children = new_children
        return o

    def visit_LogicalOr(self, o, **kwargs):
        new_children = tuple([self.visit(c, **kwargs) for c in o.children])

        literals, non_literals = self.separate_literals(new_children)
        if len(non_literals) == 0:
            return LogicLiteral(functools.reduce(lambda x, y: x or y, [c.value for c in new_children], False))

        o.children = new_children
        return o

    def visit_LogicalNot(self, o, **kwargs):
        new_child = self.visit(o.child, **kwargs)

        literals, non_literals = self.separate_literals([new_child])
        if len(non_literals) == 0:
            return LogicLiteral(not new_child.value)

        o.child = new_child
        return o

    def visit(self, o, *args, **kwargs):
        constants_map = kwargs.pop('constants_map', dict())

        # TODO: Not sure I like this
        if 'declarations' in kwargs:
            declarations = kwargs.pop('declarations', dict())
            declarations = [c for d in declarations for c in d.children if c is not None]
            dec_const_map = {(d.basename, ()): d.initial for d in flatten(declarations) if d.initial is not None}
            constants_map.update(dec_const_map)

        return super().visit(o, *args, constants_map=constants_map, **kwargs)

    def visit_Conditional(self, o, **kwargs):
        new_condition = self.visit(o.condition, **kwargs)
        constants_map = kwargs.pop('constants_map', dict())
        body_constants_map = constants_map.copy()
        else_body_constants_map = constants_map.copy()
        new_body = self.visit(o.body, constants_map=body_constants_map, **kwargs)
        new_else_body = self.visit(o.else_body, constants_map=else_body_constants_map, **kwargs)

        o = Conditional(
            condition=new_condition,
            bosy=new_body,
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

        new_bounds = self.visit(o.bounds, constants_map=constants_map, **kwargs)
        # o.bounds = new_bounds
        o = Loop(
            variable=o.variable,
            body=o.body,
            bounds=new_bounds
        )

        if self.unroll_loops:
            unrolled = LoopUnrollTransformer().visit(o)
            return self.visit(unrolled, constants_map=constants_map, **kwargs)
        # TODO: The no unrolling version might act strangely
        new_body = self.visit(o.body, constants_map=constants_map, **kwargs)
        o.body = new_body

        return o

    def visit_LoopRange(self, o, **kwargs):
        new_start = self.visit(o.start, **kwargs)
        new_stop = self.visit(o.stop, **kwargs)
        new_step = self.visit(o.step, **kwargs)

        return LoopRange((new_start, new_stop, new_step))

    def visit_int(self, o, **kwargs):
        return IntLiteral(o)

    def visit_StringConcat(self, o, **kwargs):
        new_children = tuple([self.visit(c, **kwargs) for c in o.children])
        o.children = new_children

        literals, non_literals = self.separate_literals(new_children)
        if len(non_literals) == 0:
            return StringLiteral(''.join([c.value for c in new_children]))

        return o
