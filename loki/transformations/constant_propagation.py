# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import deepcopy

import itertools

from loki.expression import (
    symbols as sym, get_pyrange, is_constant, SimplifyMapper
)
from loki.ir import nodes as ir, FindNodes, FindVariables, FindInlineCalls, Transformer
from loki.subroutine import Subroutine
from loki.tools import dict_override
from loki.types import BasicType, ProcedureType

from loki.transformations.transform_loop import LoopUnrollTransformer


__all__ = [
    'do_constant_propagation', 'ConstantPropagationMapper',
    'ConstantPropagationTransformer'
]


def _pop_array_accesses(lhs, **kwargs):
    constants_map = kwargs.get('constants_map', {})
    new_shape = ConstantPropagationMapper()(lhs.shape, constants_map=constants_map)

    literal_mask = [is_constant(dimension) for dimension in lhs.dimensions]
    computable_dimension_mask = [is_constant(extent) for extent in new_shape]

    masked_indices = []
    for literal, computable, dimension in zip(literal_mask, computable_dimension_mask, lhs.dimensions):
        if literal:
            masked_indices.append(dimension)
        elif computable:
            masked_indices.append(sym.RangeIndex((None, None, None)))
        else:
            masked_indices.append(-1)

    possible_accesses = _array_indices_to_accesses(masked_indices, new_shape)
    for access in possible_accesses:
        constants_map.pop((lhs.basename, access), None)


def update_constants_map(lhs, value, constants_map):
    constants_map[(lhs.basename, ())] = value


def invalidate_constants_map(lhs, constants_map):
    if isinstance(lhs, sym.Array):
        for key in tuple(key for key in constants_map if key[0] == lhs.basename):
            constants_map.pop(key, None)
        return

    constants_map.pop((lhs.basename, ()), None)


def _separate_literals(children):
    separated = ([], [])
    for child in children:
        if isinstance(child, sym._Literal):
            separated[0].append(child)
        else:
            separated[1].append(child)
    return separated


def _array_indices_to_accesses(dimensions, shape):
    arg_lists = []
    for count, dimension in enumerate(dimensions):
        if isinstance(dimension, sym.RangeIndex):
            start = dimension.start if dimension.start else sym.IntLiteral(1)
            stop = dimension.stop if dimension.stop else shape[count]
            arg_lists.append([
                sym.IntLiteral(v) for v in get_pyrange(sym.LoopRange((start, stop, dimension.step)))
            ])
        else:
            arg_lists.append([dimension])
    return itertools.product(*arg_lists)


class ConstantPropagationMapper(SimplifyMapper):
    """ Mapper for expression-level constant replacement and folding. """

    def map_array(self, expr, *args, **kwargs):
        constants_map = kwargs.get('constants_map', {})
        return constants_map.get((expr.basename, getattr(expr, 'dimensions', ())), expr)

    def map_quotient(self, expr, *args, **kwargs):
        """ Always force-evaluate integer-division """
        if isinstance(expr.numerator, sym.IntLiteral) and isinstance(expr.denominator, sym.IntLiteral):
            return sym.IntLiteral(float(expr.numerator.value) / float(expr.denominator.value))
        return super().map_quotient(expr, *args, **kwargs)

    map_scalar = map_array
    map_deferred_type_symbol = map_array


class ConstantPropagationTransformer(Transformer):
    """Apply constant-propagation analysis as a transformation driver."""

    def invalidate_call_constants(self, node, constants_map):
        """
        Invalidate constants potentially modified by calls in an IR subtree.

        Pure procedures only invalidate writable actual arguments. Unknown or
        impure procedures invalidate all constants, including non-argument state.
        Inline intrinsics are treated as side-effect-free; intrinsic subroutine
        calls are conservative barriers.

        Parameters
        ----------
        node : :any:`Node` or expression
            Subtree to inspect before substituting constants.
        constants_map : dict
            Known constants, updated in place.

        Returns
        -------
        bool
            Whether any call may modify state.
        """
        has_effects = False
        calls = itertools.chain(FindNodes(ir.CallStatement).visit(node), FindInlineCalls().visit(node))
        for call in calls:
            if (
                isinstance(call, sym.InlineCall)
                and isinstance(call.procedure_type, ProcedureType)
                and call.procedure_type.is_intrinsic
            ):
                continue

            routine = call.routine
            prefix = tuple(p.lower() for p in getattr(routine, 'prefix', ()) or ())
            is_pure = 'pure' in prefix or ('elemental' in prefix and 'impure' not in prefix)
            if routine is BasicType.DEFERRED or not is_pure:
                constants_map.clear()
                return True

            for dummy, actual in call.arg_iter():
                if dummy.type.intent != 'in':
                    has_effects = True
                    for variable in FindVariables().visit(actual):
                        invalidate_constants_map(variable, constants_map)
        return has_effects

    def visit_Assignment(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', {})
        mapper = ConstantPropagationMapper()

        if self.invalidate_call_constants(o, constants_map):
            invalidate_constants_map(o.lhs, constants_map)
            return o
        rhs_symbols = FindVariables().visit(o.rhs)
        if kwargs.get('within_loop', False) and o.lhs in rhs_symbols:
            # In loop bodies, skip "increment" updates to the LHS value
            return o

        # Resolve known constants on the RHS
        new_rhs = mapper(o.rhs, constants_map=constants_map)
        new_lhs = o.lhs

        if isinstance(o.lhs, sym.Array):
            new_dimensions = tuple(mapper(d, constants_map=constants_map) for d in o.lhs.dimensions)
            new_lhs = o.lhs.clone(dimensions=new_dimensions)

            _, non_literal_dimensions = _separate_literals(new_dimensions)
            if non_literal_dimensions:
                _pop_array_accesses(new_lhs, constants_map=constants_map)
                return o._rebuild(lhs=new_lhs, rhs=new_rhs)

        _, non_literals = _separate_literals((new_rhs,))
        if not non_literals and not isinstance(new_lhs, sym.Array):
            update_constants_map(new_lhs, new_rhs, constants_map)
        else:
            invalidate_constants_map(new_lhs, constants_map)

        return o._rebuild(lhs=new_lhs, rhs=new_rhs)

    def visit_CallStatement(self, o, **kwargs):
        """Account for call effects without replacing writable actual arguments."""
        self.invalidate_call_constants(o, kwargs.get('constants_map', {}))
        return o

    def visit_Conditional(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', {})
        mapper = ConstantPropagationMapper()

        call_effects = self.invalidate_call_constants(o.condition, constants_map)
        new_condition = o.condition if call_effects else mapper(o.condition, constants_map=constants_map)

        # Pass two copies of the constants map forward ...
        with dict_override(kwargs, {'constants_map': deepcopy(constants_map)}):
            new_body = self.visit(o.body, **kwargs)
            body_constants_map = kwargs['constants_map']
        with dict_override(kwargs, {'constants_map': deepcopy(constants_map)}):
            new_else_body = self.visit(o.else_body, **kwargs)
            else_constants_map = kwargs['constants_map']

        # ... then merge the maps, removing all non-shared entries
        merged_constants_map = {}
        all_keys = set(body_constants_map) | set(else_constants_map)
        for key in all_keys:
            if (
                    key in body_constants_map and key in else_constants_map
                    and body_constants_map[key] == else_constants_map[key]
            ):
                merged_constants_map[key] = body_constants_map[key]
            else:
                merged_constants_map.pop(key, None)

        # Update the shared constants map with the merged result
        constants_map.clear()
        constants_map.update(merged_constants_map)

        return o._rebuild(condition=new_condition, body=new_body, else_body=new_else_body)

    def visit_Loop(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', {})
        mapper = ConstantPropagationMapper()

        bounds_call_effects = self.invalidate_call_constants(o.bounds, constants_map)
        new_bounds = o.bounds if bounds_call_effects else mapper(o.bounds, constants_map=constants_map)
        # Body calls may change incoming constants on earlier loop iterations.
        call_effects = self.invalidate_call_constants(o.body, constants_map)

        # When recursing into loops, send a flag down to trigger detection
        # of loop-variant assignments ("increment" updates to variables).
        with dict_override(kwargs, {
                'within_loop': True, 'constants_map': deepcopy(constants_map)
        }):
            kwargs['constants_map'].pop((o.variable.basename, ()), None)
            new_body = self.visit(o.body, **kwargs)

        lhs_vars = {o.variable}
        lhs_vars.update(loop.variable for loop in FindNodes(ir.Loop).visit(o.body))

        assignments = FindNodes(ir.Assignment).visit(new_body)
        for assign in assignments:
            lhs_vars.add(assign.lhs)

        bounds_are_const = (
            is_constant(new_bounds.start)
            and is_constant(new_bounds.stop)
            and (is_constant(new_bounds.step) or new_bounds.step is None)
        )

        if bounds_are_const and not call_effects:
            loop_constants_map = constants_map

            for assign in assignments:
                if set(FindVariables().visit(assign.rhs)).intersection(lhs_vars):
                    invalidate_constants_map(assign.lhs, loop_constants_map)
                    continue

                assign_kwargs = dict(kwargs)
                assign_kwargs['constants_map'] = loop_constants_map
                self.visit_Assignment(assign, **assign_kwargs)
        else:
            for assign in assignments:
                invalidate_constants_map(assign.lhs, constants_map)

        self.invalidate_call_constants(o.body, constants_map)
        invalidate_constants_map(o.variable, constants_map)

        return o._rebuild(bounds=new_bounds, body=new_body)

    def visit_WhileLoop(self, o, **kwargs):
        """Invalidate loop-carried writes before rewriting a while loop."""
        constants_map = kwargs.get('constants_map', {})
        self.invalidate_call_constants(o, constants_map)
        for assign in FindNodes(ir.Assignment).visit(o.body):
            invalidate_constants_map(assign.lhs, constants_map)

        with dict_override(kwargs, {
                'within_loop': True, 'constants_map': deepcopy(constants_map)
        }):
            call_effects = self.invalidate_call_constants(o.condition, kwargs['constants_map'])
            condition = o.condition if call_effects else ConstantPropagationMapper()(
                o.condition, constants_map=kwargs['constants_map']
            )
            body = self.visit(o.body, **kwargs)
        return o._rebuild(condition=condition, body=body)

    def generate_declarations_map(self, routine):
        """Build the initial constant map from declaration-time initializers."""

        def index_initial_elements(indices, element):
            if len(indices) == 1:
                return element.elements[indices[0].value - 1]
            return index_initial_elements(indices[1:], element.elements[indices[0].value - 1])

        declarations_map = {}
        for symbol in getattr(routine, 'symbols', ()):
            if not isinstance(symbol, (sym.Scalar, sym.Array)) or symbol.initial is None:
                continue

            if isinstance(symbol, sym.Array):
                declarations_map.update({
                    (symbol.basename, indices): index_initial_elements(indices, symbol.initial)
                    for indices in _array_indices_to_accesses(
                        [sym.RangeIndex((None, None, None))] * len(symbol.shape), symbol.shape
                    )
                })
            else:
                declarations_map[(symbol.basename, ())] = symbol.initial
        return declarations_map


def do_constant_propagation(routine, unroll_loops=False):
    """ Apply constant-propagation over the body of a :any:`Subroutine`. """

    assert isinstance(routine, Subroutine), \
        f'[Loki] Constant propagation can only be applied to Subroutine, but found {routine}'

    const_prop = ConstantPropagationTransformer(inplace=True, invalidate_source=False)
    declarations_map = const_prop.generate_declarations_map(routine)

    if routine.spec:
        routine.spec = const_prop.visit(routine.spec, constants_map=declarations_map)
    if routine.body:
        routine.body = const_prop.visit(routine.body, constants_map=declarations_map)

    if unroll_loops:
        routine.body = LoopUnrollTransformer().visit(routine.body)

        # If loop unrolling is requested, do another forward propagation pass
        routine.body = const_prop.visit(routine.body, constants_map=declarations_map)

    return routine
