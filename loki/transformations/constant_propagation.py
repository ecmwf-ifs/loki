# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import deepcopy

import itertools

from loki import BasicType, ProcedureSymbol, as_tuple
from loki.expression import (
    symbols as sym, get_pyrange, is_constant, SimplifyMapper
)
from loki.ir import nodes as ir, FindNodes, FindVariables, Transformer
from loki.subroutine import Subroutine
from loki.tools import dict_override

from loki.transformations.transform_loop import LoopUnrollTransformer


__all__ = [
    'do_constant_propagation', 'ConstantPropagationMapper',
    'ConstantPropagationTransformer'
]


def get_possible_array_accesses(lhs, **kwargs):
    constants_map = kwargs.get('constants_map', {})
    new_shape = ConstantPropagationMapper()(lhs.shape, constants_map=constants_map)
    is_constant_shape = new_shape is not None and all(is_constant(extent) for extent in new_shape)

    # Find the maximum dimension of the array.
    # Try to get it from static info, but fall back to the slow search of the constants map.
    max_dimension = 0
    if len(lhs.dimensions) == 0 and not is_constant_shape:
        # Whole-array, unknown shape
        for (basename, key_index) in constants_map.keys():
            if basename == lhs.basename:
                max_dimension = max(max_dimension, len(key_index))
    elif not is_constant_shape:
        # Subscripted, unknown shape
        max_dimension = len(lhs.dimensions)
    elif len(lhs.dimensions) == 0:
        # Whole-array, known shape
        max_dimension = len(new_shape)
    else:
        # Subscripted, known shape
        max_dimension = max(len(lhs.dimensions), len(new_shape))

    # Padded for the zip below, so that we can handle the case where we have a shape but no dimensions (or vice versa)
    literal_mask = [is_constant(dimension) for dimension in lhs.dimensions]
    literal_mask.extend([False] * (max_dimension - len(literal_mask)))

    if all(literal_mask):
        # If we know the exact dimensions, we can just return that one access
        return (lhs.dimensions,)

    # Pad the dimensions for same reasoning as above
    dimensions = list(lhs.dimensions)
    dimensions.extend([sym.RangeIndex((None, None, None))] * (max_dimension - len(dimensions)))

    # Apply the mask to get a list of indices that are either literal, or computable for each dimension.
    masked_indices = tuple(dimension if is_literal else sym.RangeIndex((None, None, None))
                            for is_literal, dimension in zip(literal_mask, dimensions))

    possible_accesses = []
    if is_constant_shape:
        # If the shape is literal, we can conservatively generate all possible accesses for the array
        possible_accesses = array_indices_to_accesses(masked_indices, new_shape)
    else:
        # Else, if we don't know the shape, we search the constants map and conservatively return all accesses
        # that match the known dimensions of the array.
        first_masked_index = next((index for (index, is_literal) in enumerate(literal_mask) if not is_literal))
        for (basename, key_index) in constants_map.keys():
            if basename == lhs.basename and key_index[:first_masked_index] == masked_indices[:first_masked_index]:
                possible_accesses.append(key_index)

    return as_tuple(possible_accesses)


def array_indices_to_accesses(indices, shape):
    partial_indices = []
    # Use the masked indices to generate all possible accesses for the array,
    # using the literal indices to minimise the access space.
    for count, index in enumerate(indices):
        if isinstance(index, sym.RangeIndex):
            start = index.start if index.start is not None else sym.IntLiteral(1)
            stop = index.stop if index.stop is not None else shape[count]
            partial_indices.append([
                sym.IntLiteral(v) for v in get_pyrange(sym.LoopRange((start, stop, index.step)))
            ])
        else:
            partial_indices.append([index])
    return list(itertools.product(*partial_indices))


def update_constants_map(lhs, value, constants_map):
    if isinstance(lhs, sym.Array):
        for access in get_possible_array_accesses(lhs, constants_map=constants_map):
            constants_map[(lhs.basename, access)] = value
    else:
        constants_map[(lhs.basename, ())] = value


def invalidate_constants_map(lhs, constants_map):
    if isinstance(lhs, sym.Array):
        for access in get_possible_array_accesses(lhs, constants_map=constants_map):
            constants_map.pop((lhs.basename, access), None)
    else:
        constants_map.pop((lhs.basename, ()), None)


def separate_literals(children):
    separated = ([], [])
    for child in children:
        if isinstance(child, sym._Literal):
            separated[0].append(child)
        else:
            separated[1].append(child)
    return separated


def pop_procedure_accesses(procedure, *args, **kwargs):
    constants_map = kwargs.get('constants_map', {})

    if procedure.procedure_type == BasicType.DEFERRED or procedure.procedure_type.is_intrinsic:
        # If we can't get the intent, be conservative
        arg_list = list(procedure.arguments)
        arg_list.extend([arg for (kw, arg) in procedure.kwarguments])
        for arg in arg_list:
            if isinstance(arg, (sym.Scalar, sym.Array)):
                invalidate_constants_map(arg, constants_map)
        return procedure.arguments, procedure.kwarguments

    arg_iter = procedure.arg_iter()
    split = len(procedure.arguments)
    args_pairs = list(itertools.islice(arg_iter, split))
    kwargs_pairs = list(arg_iter)

    args_list = tuple(call_kwarg for (_, call_kwarg) in process_procedure_args(args_pairs, *args, **kwargs))
    kwargs_list = tuple((dummy_kwarg.basename, call_kwarg) for (dummy_kwarg, call_kwarg)
                        in process_procedure_args(kwargs_pairs, *args, **kwargs))
    return args_list, kwargs_list


def process_procedure_args(args_list, *args, **kwargs):
    constants_map = kwargs.get('constants_map', {})
    mapper = ConstantPropagationMapper()

    mapped_args = []
    for (dummy_arg, call_arg) in args_list:
        if dummy_arg.type.intent == 'in':
            mapped_args.append((dummy_arg, mapper(call_arg, *args, **kwargs)))
        else:
            # Else invalidate only arguments that are not explicitly marked as intent `in` (i.e. read-only)
            invalidate_constants_map(call_arg, constants_map)
            mapped_args.append((dummy_arg, call_arg))
    return mapped_args


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

    def map_inline_call(self, expr, *args, **kwargs):

        args_list, kwargs_list = pop_procedure_accesses(expr, *args, **kwargs)
        return expr.clone(parameters=args_list, kw_parameters=dict(kwargs_list))

    map_scalar = map_array
    map_deferred_type_symbol = map_array


class ConstantPropagationTransformer(Transformer):
    """Apply constant-propagation analysis as a transformation driver."""

    def visit_CallStatement(self, o, **kwargs):

        args_list, kwargs_list = pop_procedure_accesses(o, **kwargs)

        return o._rebuild(arguments=args_list, kwarguments=kwargs_list)

    def visit_Assignment(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', {})
        mapper = ConstantPropagationMapper()

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

            _, non_literal_dimensions = separate_literals(new_dimensions)
            if non_literal_dimensions:
                invalidate_constants_map(new_lhs, constants_map)
                return o._rebuild(lhs=new_lhs, rhs=new_rhs)

        _, non_literals = separate_literals((new_rhs,))
        if non_literals:
            invalidate_constants_map(new_lhs, constants_map)
        else:
            update_constants_map(new_lhs, new_rhs, constants_map)

        return o._rebuild(lhs=new_lhs, rhs=new_rhs)

    def visit_Conditional(self, o, **kwargs):
        constants_map = kwargs.get('constants_map', {})
        mapper = ConstantPropagationMapper()

        new_condition = mapper(o.condition, constants_map=constants_map)

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

        new_bounds = mapper(o.bounds, constants_map=constants_map)

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

        if bounds_are_const:
            loop_constants_map = constants_map

            for assign in assignments:
                if not set(FindVariables().visit(assign.rhs)).intersection(lhs_vars):
                    assign_kwargs = dict(kwargs)
                    assign_kwargs['constants_map'] = loop_constants_map
                    self.visit_Assignment(assign, **assign_kwargs)
        else:
            for assign in assignments:
                invalidate_constants_map(assign.lhs, constants_map)

        invalidate_constants_map(o.variable, constants_map)

        return o._rebuild(bounds=new_bounds, body=new_body)

    def generate_declarations_map(self, routine):
        """Build the initial constant map from declaration-time initializers."""

        def index_initial_elements(indices, element, lower_bounds):
            offset = indices[0].value - lower_bounds[0]
            if len(indices) == 1:
                return element.elements[offset]
            return index_initial_elements(indices[1:], element.elements[offset], lower_bounds[1:])

        def is_range_const(index_range):
            return (index_range.start is not None and is_constant(index_range.start) and
                    index_range.stop is not None and is_constant(index_range.stop))

        declarations_map = {}
        arrays = []
        for symbol in getattr(routine, 'symbols', ()):
            if (
                    isinstance(symbol, (sym.DeferredTypeSymbol , ProcedureSymbol))
                    or symbol.initial is None
            ):
                continue

            if isinstance(symbol, sym.Array):
                # Process later so that we have the scalars available
                arrays.append(symbol)
            else:
                declarations_map[(symbol.basename, ())] = symbol.initial
        for array in arrays:
            new_shape = ConstantPropagationMapper()(array.shape, constants_map=declarations_map)
            if new_shape is None or not isinstance(array.initial, sym.LiteralList):
                continue

            if all(is_constant(extent) for extent in new_shape):
                accesses = array_indices_to_accesses([sym.RangeIndex((None, None, None))] * len(new_shape), new_shape)
                for index in accesses:
                    declarations_map[(array.basename, index)] = ConstantPropagationMapper()(
                        index_initial_elements(index, array.initial, [1] * len(new_shape)),
                        constants_map=declarations_map
                    )
            elif isinstance(new_shape[0], sym.RangeIndex) and is_range_const(new_shape[0]):
                # Only works for the simple case of 1D arrays.
                # Needs more logic to handle multi-dimensional arrays due to the reshape() call
                if len(new_shape) != 1:
                    continue

                lower_bounds = list(map(lambda x: min(x).value, zip(*array_indices_to_accesses(new_shape, new_shape))))
                for index in array_indices_to_accesses(new_shape, new_shape):
                    declarations_map[(array.basename, index)] = (ConstantPropagationMapper()(
                        index_initial_elements(index, array.initial, lower_bounds),
                        constants_map=declarations_map)
                    )

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
