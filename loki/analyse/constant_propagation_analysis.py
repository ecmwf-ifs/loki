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

from loki.analyse.abstract_dfa import AbstractDataflowAnalysis
from loki.expression import (
    Array, DeferredTypeSymbol, FloatLiteral, IntLiteral, LogicLiteral,
    LokiIdentityMapper, LoopRange, Product, RangeIndex, StringLiteral
)
from loki.expression.symbolic import get_pyrange, is_constant, SimplifyMapper
from loki.expression.symbols import _Literal
from loki.ir import FindNodes, Assignment, FindVariables, Loop, Transformer
from loki.tools import as_tuple

__all__ = ['ConstantPropagationMapper', 'ConstantPropagationAnalysis']


class ConstantPropagationMapper(SimplifyMapper):
    """ Mapper for expression-level constant replacement and folding. """

    def __init__(self, fold_floats=True):
        self.fold_floats = fold_floats
        super().__init__()

    def map_array(self, expr, *args, **kwargs):
        constants_map = kwargs.get('constants_map', {})
        return constants_map.get((expr.basename, getattr(expr, 'dimensions', ())), expr)

    def map_quotient(self, expr, *args, **kwargs):
        """ Always force-evaluate integer-division """
        if isinstance(expr.numerator, IntLiteral) and isinstance(expr.denominator, IntLiteral):
            return IntLiteral(float(expr.numerator.value) / float(expr.denominator.value))
        return super().map_quotient(expr, *args, **kwargs)

    map_scalar = map_array
    map_deferred_type_symbol = map_array


class ConstantPropagationAnalysis(AbstractDataflowAnalysis):
    """Scaffolding for constant-propagation analysis over Loki IR."""

    class Attacher(Transformer):
        """Attach placeholder constant maps without mutating the IR."""

        def _pop_array_accesses(self, lhs, **kwargs):
            constants_map = kwargs.get('constants_map', {})
            new_shape = ConstantPropagationMapper(self.parent.fold_floats)(
                lhs.shape, constants_map=constants_map
            )

            literal_mask = [is_constant(dimension) for dimension in lhs.dimensions]
            computable_dimension_mask = [is_constant(extent) for extent in new_shape]

            masked_indices = []
            ignore_mask = []
            partial = False
            for literal, computable, dimension in zip(literal_mask, computable_dimension_mask, lhs.dimensions):
                if literal:
                    masked_indices.append(dimension)
                    ignore_mask.append(False)
                elif computable:
                    masked_indices.append(RangeIndex((None, None, None)))
                    ignore_mask.append(False)
                else:
                    partial = True
                    masked_indices.append(-1)
                    ignore_mask.append(True)

            possible_accesses = ConstantPropagationAnalysis._array_indices_to_accesses(masked_indices, new_shape)
            keys = tuple(constants_map.keys())
            for access in possible_accesses:
                if partial:
                    for key in keys:
                        if key[0] == lhs.name and all(
                                current == candidate or ignore
                                for current, candidate, ignore in zip(key[1], access, ignore_mask)
                        ):
                            constants_map.pop(key, None)
                else:
                    constants_map.pop((lhs.basename, access), None)

        def __init__(self, parent, **kwargs):
            self.parent = parent
            super().__init__(inplace=True, invalidate_source=False, **kwargs)

        def visit_Node(self, o, **kwargs):
            constants_map = deepcopy(kwargs.get('constants_map', {}))
            o._update(_constants_map=constants_map)
            return super().visit_Node(o, **kwargs)

        def visit_Assignment(self, o, **kwargs):
            constants_map = kwargs.get('constants_map', {})
            mapper = ConstantPropagationMapper(self.parent.fold_floats)
            mapper_kwargs = dict(kwargs)
            mapper_kwargs['constants_map'] = constants_map
            incoming_constants_map = deepcopy(constants_map)

            new_rhs = mapper(o.rhs, **mapper_kwargs)
            new_lhs = o.lhs

            if isinstance(o.lhs, Array):
                new_dimensions = tuple(mapper(d, **mapper_kwargs) for d in o.lhs.dimensions)
                new_lhs = o.lhs.clone(dimensions=new_dimensions)

                _, non_literal_dimensions = ConstantPropagationAnalysis._separate_literals(new_dimensions)
                if non_literal_dimensions:
                    self._pop_array_accesses(new_lhs, constants_map=constants_map)
                    o._update(lhs=new_lhs, rhs=new_rhs, _constants_map=incoming_constants_map)
                    return o

            _, non_literals = ConstantPropagationAnalysis._separate_literals((new_rhs,))
            if not non_literals and not isinstance(new_lhs, Array):
                self.parent.update_constants_map(new_lhs, new_rhs, constants_map)
            else:
                self.parent.invalidate_constants_map(new_lhs, constants_map)

            o._update(lhs=new_lhs, rhs=new_rhs, _constants_map=incoming_constants_map)
            return o

        def visit_Conditional(self, o, **kwargs):
            constants_map = kwargs.get('constants_map', {})
            mapper = ConstantPropagationMapper(self.parent.fold_floats)
            mapper_kwargs = dict(kwargs)
            mapper_kwargs['constants_map'] = constants_map
            incoming_constants_map = deepcopy(constants_map)
            body_kwargs = dict(kwargs)
            body_kwargs['constants_map'] = deepcopy(constants_map)
            else_kwargs = dict(kwargs)
            else_kwargs['constants_map'] = deepcopy(constants_map)

            new_condition = mapper(o.condition, **mapper_kwargs)

            new_body = self.visit(o.body, **body_kwargs)
            new_else_body = self.visit(o.else_body, **else_kwargs)
            body_constants_map = body_kwargs['constants_map']
            else_constants_map = else_kwargs['constants_map']

            merged_constants_map = deepcopy(incoming_constants_map)
            all_keys = set(body_constants_map) | set(else_constants_map)
            for key in all_keys:
                if (
                        key in body_constants_map and key in else_constants_map
                        and body_constants_map[key] == else_constants_map[key]
                ):
                    merged_constants_map[key] = body_constants_map[key]
                else:
                    merged_constants_map.pop(key, None)

            constants_map.clear()
            constants_map.update(merged_constants_map)

            o._update(
                condition=new_condition,
                body=new_body,
                else_body=new_else_body,
                _constants_map=incoming_constants_map,
            )
            return o

        def visit_Loop(self, o, **kwargs):
            constants_map = kwargs.get('constants_map', {})
            mapper = ConstantPropagationMapper(self.parent.fold_floats)
            mapper_kwargs = dict(kwargs)
            mapper_kwargs['constants_map'] = constants_map
            incoming_constants_map = deepcopy(constants_map)

            new_bounds = mapper(o.bounds, **mapper_kwargs)
            new_loop = o.clone(bounds=new_bounds)

            step = new_bounds.step if new_bounds.step is not None else IntLiteral(1)
            can_unroll = self.parent.apply_transform and self.parent.unroll_loops
            can_unroll = can_unroll and all(is_constant(expr) for expr in (new_bounds.start, new_bounds.stop, step))

            if can_unroll:
                from loki.transformations.transform_loop import LoopUnrollTransformer  # pylint: disable=import-outside-toplevel
                unrolled = LoopUnrollTransformer(warn_iterations_length=False).visit(new_loop)
                if not isinstance(unrolled, type(o)):
                    unrolled_body = self.visit(as_tuple(unrolled), **kwargs)
                    if not unrolled_body:
                        return None
                    return as_tuple(unrolled_body)

            body_kwargs = dict(kwargs)
            body_kwargs['constants_map'] = deepcopy(constants_map)
            body_kwargs['constants_map'].pop((o.variable.basename, ()), None)
            new_body = self.visit(o.body, **body_kwargs)

            lhs_vars = {o.variable}
            lhs_vars.update(loop.variable for loop in FindNodes(Loop).visit(o.body))

            assignments = FindNodes(Assignment).visit(new_body)
            for assign in assignments:
                lhs_vars.add(assign.lhs)

            bounds_are_const = (
                is_constant(new_bounds.start)
                and is_constant(new_bounds.stop)
                and (is_constant(new_bounds.step) or new_bounds.step is None)
            )
            bounds_has_steps = bounds_are_const and len(
                get_pyrange(LoopRange((new_bounds.start, new_bounds.stop, new_bounds.step)))
            ) > 0

            if bounds_are_const:
                if bounds_has_steps:
                    loop_constants_map = constants_map
                else:
                    loop_constants_map = deepcopy(constants_map)

                for assign in assignments:
                    if not set(FindVariables().visit(assign.rhs)).intersection(lhs_vars):
                        assign_kwargs = dict(kwargs)
                        assign_kwargs['constants_map'] = loop_constants_map
                        self.visit_Assignment(assign, **assign_kwargs)
            else:
                for assign in assignments:
                    self.parent.invalidate_constants_map(assign.lhs, constants_map)

            self.parent.invalidate_constants_map(o.variable, constants_map)

            o._update(bounds=new_bounds, body=new_body, _constants_map=incoming_constants_map)
            return o

    class Detacher(Transformer):
        """Remove transient constant-propagation metadata from IR nodes."""

        def __init__(self, **kwargs):
            super().__init__(inplace=True, invalidate_source=False, **kwargs)

        def visit_Node(self, o, **kwargs):
            o._update(_constants_map=None)
            return super().visit_Node(o, **kwargs)

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

    def update_constants_map(self, lhs, value, constants_map):
        constants_map[(lhs.basename, ())] = value

    def invalidate_constants_map(self, lhs, constants_map):
        if isinstance(lhs, Array):
            for access in tuple(key for key in constants_map if key[0] == lhs.basename):
                constants_map.pop((lhs.basename, access), None)
            return

        constants_map.pop((lhs.basename, ()), None)

    @staticmethod
    def _separate_literals(children):
        separated = ([], [])
        for child in children:
            if isinstance(child, _Literal):
                separated[0].append(child)
            else:
                separated[1].append(child)
        return separated
