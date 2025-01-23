# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import math
import operator
import functools
from copy import deepcopy

from loki import Transformer, Array, DeferredTypeSymbol, RangeIndex, \
    IntLiteral, get_pyrange, LoopRange, as_tuple, Assignment, FindNodes, LokiIdentityMapper, FindVariables, Loop, \
    is_constant
from loki.expression.symbols import _Literal, FloatLiteral, LogicLiteral, Product, StringLiteral
from loki.analyse.data_flow_analysis import DataFlowAnalysis
from loki.analyse.abstract_dfa import AbstractDataflowAnalysis
from loki.transformations.transform_loop import LoopUnrollTransformer

__all__ = [
    'ConstantPropagationAnalysis'
]

class ConstantPropagationAnalysis(AbstractDataflowAnalysis):
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

            literals, non_literals = ConstantPropagationAnalysis._separate_literals(children)
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

            literals, non_literals = ConstantPropagationAnalysis._separate_literals(children)
            if len(non_literals) == 0:
                return LogicLiteral(functools.reduce(bool_op, [c.value for c in children], initial))

            return expr.__class__(children)

        def map_logical_not(self, expr, *args, **kwargs):
            child = self.rec(expr.child, **kwargs)

            literals, non_literals = ConstantPropagationAnalysis._separate_literals([child])
            if len(non_literals) == 0:
                return LogicLiteral(not child.value)

            return expr.__class__(child)

        def map_comparison(self, expr, *args, **kwargs):
            left = self.rec(expr.left, *args, **kwargs)
            right = self.rec(expr.right, *args, **kwargs)

            literals, non_literals = ConstantPropagationAnalysis._separate_literals([left, right])
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
                operator_str = expr.operator if expr.operator in operators_map.keys() else expr.operator_to_name[
                    expr.operator]
                return LogicLiteral(operators_map[operator_str](left.value, right.value))

            return expr.__class__(left, expr.operator, right)

        def map_loop_range(self, expr, *args, **kwargs):
            start = self.rec(expr.start, *args, **kwargs)
            stop = self.rec(expr.stop, *args, **kwargs)
            step = self.rec(expr.step, *args, **kwargs)
            return expr.__class__((start, stop, step))

        def map_string_concat(self, expr, *args, **kwargs):
            children = tuple([self.rec(c, *args, **kwargs) for c in expr.children])

            literals, non_literals = ConstantPropagationAnalysis._separate_literals(children)
            if len(non_literals) == 0:
                return StringLiteral(''.join([c.value for c in children]))

            return expr.__class__(children)


    class _Attacher(Transformer):

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
            possible_accesses = ConstantPropagationAnalysis._array_indices_to_accesses(masked_accesses,
                                                                ConstantPropagationAnalysis.ConstPropMapper(False)(o.lhs.shape,
                                                                                                       **kwargs))
            for access in possible_accesses:
                constants_map.pop((o.lhs.basename, access), None)

        def __init__(self, parent, **kwargs):
            self.parent = parent
            super().__init__(inplace=not self.parent._apply_transform, invalidate_source=self.parent._apply_transform, **kwargs)

        def visit_Assignment(self, o, **kwargs):
            constants_map = kwargs.get('constants_map', dict())
            # Create a deep copy of the constants map when we enter this node. This is so that the node
            # has the constants as they were before it mutated them, which is probably more useful
            constants_map_in = deepcopy(constants_map)
            o._update(_constants_map=constants_map_in)

            new_rhs = ConstantPropagationAnalysis.ConstPropMapper(self.parent.fold_floats)(o.rhs, **kwargs)
            if self.parent._apply_transform:
                o._update(rhs=new_rhs)
            # Work with this lhs in case we're not applying transforms & can't modify o.lhs
            lhs = o.lhs

            # What if the lhs isn't a scalar shape?
            if isinstance(lhs, Array):
                new_dimensions = [ConstantPropagationAnalysis.ConstPropMapper(self.parent.fold_floats)(d, **kwargs) for d in lhs.dimensions]
                _, new_d_non_literals = ConstantPropagationAnalysis._separate_literals(new_dimensions)

                new_lhs = Array(lhs.name, lhs.scope, lhs.type, as_tuple(new_dimensions))
                if self.parent._apply_transform:
                    o._update(lhs=new_lhs)
                lhs = new_lhs
                if len(new_d_non_literals) != 0:
                    self._pop_array_accesses(o, **kwargs)
                    return o

            literals, non_literals = ConstantPropagationAnalysis._separate_literals([new_rhs])
            if len(non_literals) == 0:
                if isinstance(lhs, Array):
                    for access in ConstantPropagationAnalysis._array_indices_to_accesses(lhs.dimensions, lhs.shape):
                        constants_map[(lhs.basename, access)] = new_rhs
                else:
                    constants_map[(lhs.basename, ())] = new_rhs
            else:
                # TODO: What if it's a pointer
                if isinstance(lhs, Array):
                    for access in ConstantPropagationAnalysis._array_indices_to_accesses(lhs.dimensions, lhs.shape):
                        constants_map.pop((lhs.basename, access), None)
                else:
                    constants_map.pop((lhs.basename, ()), None)

            return o

        def visit(self, o, *args, **kwargs):
            constants_map = kwargs.pop('constants_map', dict())
            return super().visit(o, *args, constants_map=constants_map, **kwargs)

        def visit_Conditional(self, o, **kwargs):
            constants_map = kwargs.pop('constants_map', dict())
            constants_map_in = deepcopy(constants_map)
            o._update(_constants_map=constants_map_in)

            new_condition = ConstantPropagationAnalysis.ConstPropMapper(self.parent.fold_floats)(o.condition, constants_map=constants_map, **kwargs)
            body_constants_map = deepcopy(constants_map)
            else_body_constants_map = deepcopy(constants_map)
            new_body = self.visit(o.body, constants_map=body_constants_map, **kwargs)
            new_else_body = self.visit(o.else_body, constants_map=else_body_constants_map, **kwargs)

            if self.parent._apply_transform:
                o._update(
                    condition=new_condition,
                    body=new_body,
                    else_body=new_else_body
                )

            if  isinstance(new_condition, LogicLiteral):
                if new_condition.value:
                    constants_map.update(body_constants_map)
                else:
                    constants_map.update(else_body_constants_map)
            else:
                for key in set(body_constants_map.keys()).union(else_body_constants_map):
                    if body_constants_map.get(key, None) == else_body_constants_map.get(key, None):
                        constants_map[key] = body_constants_map[key]
                    else:
                        constants_map.pop(key, None)

            return o

        def visit_Loop(self, o, **kwargs):
            constants_map = kwargs.pop('constants_map', dict())
            constants_map_in = deepcopy(constants_map)
            o._update(_constants_map=constants_map_in)

            constants_map.pop((o.variable.basename, ()), None)

            new_bounds = ConstantPropagationAnalysis.ConstPropMapper(self.parent.fold_floats)(o.bounds, constants_map=constants_map, **kwargs)
            if self.parent._apply_transform:
                o._update(bounds=new_bounds)

            if self.parent.unroll_loops:
                temp_loop = o.clone()
                temp_loop._update(bounds=new_bounds)
                unrolled = LoopUnrollTransformer(warn_iterations_length=False).visit(temp_loop)
                # If we cannot unroll, then we need to fall back to the no unroll analysis
                if not isinstance(unrolled, Loop):
                    if self.parent._apply_transform:
                        o = self.visit(unrolled, constants_map=constants_map, **kwargs)
                        # TODO: _update each node in the new body with the const map
                    return o

            # TODO: could also be mutating subroutine, not just an assign
            lhs_vars = {o.variable}
            lhs_vars.update([l.variable for l in FindNodes(Loop).visit(o.body)])

            # Build a set of invariants
            assignments = FindNodes(Assignment).visit(o.body)
            for a in assignments:
                lhs_vars.add(a.lhs)

            # Then figure out which lhs are generated from only invariants
            for a in assignments:
                # if rhs of a has no lhs vars from loop body (i.e. consists solely of loop invariant var),
                # and all bounds are const (i.e. we can guarantee we'll take the loop at least once)
                if (len(set(FindVariables().visit(a.rhs)).intersection(lhs_vars)) == 0 and
                        is_constant(new_bounds.start) and is_constant(new_bounds.stop) and (is_constant(new_bounds.step) or new_bounds.step is None)):
                    # Pass to visit_assignment
                    self.visit_Assignment(a, constants_map=constants_map, **kwargs)
                else:
                    if isinstance(a.lhs, Array):
                        self._pop_array_accesses(a, constants_map=constants_map, **kwargs)
                    else:
                        constants_map.pop((a.lhs.basename, ()), None)

            new_body = self.visit(o.body, constants_map=deepcopy(constants_map), **kwargs)
            if self.parent._apply_transform:
                o._update(body=new_body)

            return o

    class _Detacher(Transformer):
        """
        Remove in-place any dataflow analysis properties.
        """

        def __init__(self, **kwargs):
            super().__init__(inplace=True, invalidate_source=False, **kwargs)

        def visit_Node(self, o, **kwargs):
            o._update(_constants_map=None)
            return super().visit_Node(o, **kwargs)

    def __init__(self, fold_floats, unroll_loops, _apply_transform=False):
        self.fold_floats = fold_floats
        self.unroll_loops = unroll_loops
        self._apply_transform = _apply_transform
        super().__init__()

    def get_attacher(self):
        return self._Attacher(self)

    def attach_dataflow_analysis(self, module_or_routine):
        constants_map = self.generate_declarations_map(module_or_routine)

        # TODO: Implement
        # if hasattr(module_or_routine, 'spec'):
        #     (self.get_attacher().visit(module_or_routine.spec, constants_map=constants_map))

        if hasattr(module_or_routine, 'body'):
            (self.get_attacher().visit(module_or_routine.body, constants_map=constants_map))

    def generate_declarations_map(self, routine):
        def index_initial_elements(i, e):
            if len(i) == 1:
                return e.elements[i[0].value - 1]
            else:
                return index_initial_elements(i[1:], e.elements[i[0].value - 1])

        declarations_map = dict()
        # TODO: What if there's a context already?
        with DataFlowAnalysis().dataflow_analysis_attached(routine):
            for s in routine.symbols:
                if isinstance(s, DeferredTypeSymbol) or s.initial is None:
                    continue
                if isinstance(s, Array):
                    declarations_map.update({(s.basename, i): index_initial_elements(i, s.initial) for i in
                                             self._array_indices_to_accesses(
                                                 [RangeIndex((None, None, None))] * len(s.shape), s.shape
                                             )})
                else:
                    declarations_map[(s.basename, ())] = s.initial
        return declarations_map

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
    def _separate_literals(children):
        separated = ([], [])
        for c in children:
            # is_constant only covers int, float, & complex
            if isinstance(c, _Literal):
                separated[0].append(c)
            else:
                separated[1].append(c)
        return separated
