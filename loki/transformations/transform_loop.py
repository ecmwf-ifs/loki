# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utility routines that provide loop transformations.

"""
import functools
from collections import defaultdict
import operator as op
import numpy as np

from loki.analyse import (
    dataflow_analysis_attached, read_after_write_vars,
    loop_carried_dependencies
)
from loki.expression import (
    symbols as sym, simplify, is_constant, symbolic_op, parse_expr,
    IntLiteral, FloatLiteral
)
from loki.ir import (
    Loop, Conditional, Comment, Pragma, FindNodes, Transformer,
    NestedMaskedTransformer, is_parent_of, is_loki_pragma,
    get_pragma_parameters, pragmas_attached, SubstituteExpressions,
    FindVariables
)
from loki.logging import info, warning
from loki.tools import (
    flatten, as_tuple, CaseInsensitiveDict, binary_insertion_sort,
    optional
)
from loki.transformations.array_indexing import (
    promotion_dimensions_from_loop_nest, promote_nonmatching_variables
)


__all__ = ['loop_interchange', 'loop_fusion', 'loop_fission', 'loop_unroll']


from loki.analyse.util_polyhedron import Polyhedron

def eliminate_variable(polyhedron, index_or_variable):
    """
    Eliminate a variable from the polyhedron.

    Mathematically, this is a projection of the polyhedron onto the hyperplane
    H:={x|x_j=0} with x_j the dimension corresponding to the eliminated variable.

    This is an implementation of the Fourier-Motzkin elimination.

    :param :class:``Polyhedron`` polyhedron: the polyhedron to be reduced in dimension.
    :param index_or_variable: the index, name, or expression symbol that is to be
                              eliminated.
    :type index_or_variable: int or str or sym.Array or sym.Scalar

    :return: the reduced polyhedron.
    :rtype: :class:``Polyhedron``
    """
    if isinstance(index_or_variable, int):
        j = index_or_variable
    else:
        j = polyhedron.variable_to_index(index_or_variable)

    # Indices of lower bounds on x_j
    L = [i for i in range(polyhedron.A.shape[0]) if polyhedron.A[i,j] < 0]
    # Indices of upper bounds on x_j
    U = [i for i in range(polyhedron.A.shape[0]) if polyhedron.A[i,j] > 0]
    # Indices of constraints not involving x_j
    Z = [i for i in range(polyhedron.A.shape[0]) if i not in L+U]
    # Cartesian product of lower and upper bounds
    R = [(l, u) for l in L for u in U]

    # Project polyhedron onto hyperplane H:={x|x_j = 0}
    A = np.zeros(polyhedron.A.shape, dtype=np.dtype(int))
    b = np.zeros(polyhedron.b.shape, dtype=np.dtype(int))
    next_constraint = 0
    for idx in Z:
        A[next_constraint,:] = polyhedron.A[idx,:]
        b[next_constraint] = polyhedron.b[idx]
        next_constraint += 1
    for l, u in R:
        A[next_constraint,:] = polyhedron.A[u,j] * polyhedron.A[l,:] - polyhedron.A[l,j] * polyhedron.A[u,:]
        b[next_constraint] = polyhedron.A[u,j] * polyhedron.b[l] - polyhedron.A[l,j] * polyhedron.b[u]
        next_constraint += 1

    # TODO: normalize rows

    # Trim matrix and right hand side, eliminate j-th column
    A = np.delete(A[:next_constraint,:], j, axis=1)
    b = b[:next_constraint]
    variables = polyhedron.variables
    if variables is not None:
        variables = variables[:j] + variables[j+1:]
    return Polyhedron(A, b, variables)


def generate_loop_bounds(iteration_space, iteration_order):
    """
    Generate loop bounds according to a changed iteration order.

    This creates a new polyhedron representing the iteration space for the
    provided iteration order.

    :param :class:``Polyhedron`` iteration_space: the iteration space that
            should be reordered.
    :param list iteration_order: the new iteration order as a list of
            indices of iteration variables.

    :return: the reordered iteration space.
    :rtype: :class:``Polyhedron``
    """
    assert iteration_space.variables is not None
    assert len(iteration_order) <= len(iteration_space.variables)

    lower_bounds= [None] * len(iteration_order)
    upper_bounds= [None] * len(iteration_order)
    index_map = list(range(len(iteration_order)))
    reduced_polyhedron = iteration_space

    # Find projected loop bounds
    constraint_count = 0
    for var_idx in reversed(iteration_order):
        # Get index of variable in reduced polyhedron
        idx = index_map[var_idx]
        assert idx is not None
        # Store bounds for variable
        lower_bounds[var_idx] = reduced_polyhedron.lower_bounds(idx)
        upper_bounds[var_idx] = reduced_polyhedron.upper_bounds(idx)
        constraint_count += len(lower_bounds[var_idx]) + len(upper_bounds[var_idx])
        # Eliminate variable from polyhedron
        reduced_polyhedron = eliminate_variable(reduced_polyhedron, idx)
        # Update index map after variable elimination
        index_map[var_idx] = None
        index_map[var_idx+1:] = [i-1 for i in index_map[var_idx+1:]]

    # Build new iteration space polyhedron
    variables = [iteration_space.variables[i] for i in iteration_order]
    variables += iteration_space.variables[len(iteration_order):]
    A = np.zeros([constraint_count, len(variables)], dtype=np.dtype(int))
    b = np.zeros([constraint_count], dtype=np.dtype(int))
    next_constraint = 0
    for new_idx, var_idx in enumerate(iteration_order):
        # TODO: skip lower/upper bounds already fulfilled
        for bound in lower_bounds[var_idx]:
            lhs, rhs = Polyhedron.generate_entries_for_lower_bound(bound, variables, new_idx)
            A[next_constraint,:] = lhs
            b[next_constraint] = rhs
            next_constraint += 1
        for bound in upper_bounds[var_idx]:
            lhs, rhs = Polyhedron.generate_entries_for_lower_bound(bound, variables, new_idx)
            A[next_constraint,:] = -lhs
            b[next_constraint] = -rhs
            next_constraint += 1
    A = A[:next_constraint,:]
    b = b[:next_constraint]
    return Polyhedron(A, b, variables)


def get_nested_loops(loop, depth):
    """
    Helper routine to extract all loops in a loop nest.
    """
    loops = [loop]
    for _ in range(1, depth):
        loops_in_body = [node for node in loop.body if isinstance(node, Loop)]
        assert len(loops_in_body) == 1
        loop = loops_in_body[0]
        loops += [loop]
    return as_tuple(loops)


def get_loop_components(loops):
    """
    Helper routine to extract loop variables, ranges and bodies of list of loops.
    """
    loop_variables, loop_ranges, loop_bodies = zip(*[(loop.variable, loop.bounds, loop.body) for loop in loops])
    return (as_tuple(loop_variables), as_tuple(loop_ranges), as_tuple(loop_bodies))


def loop_interchange(routine, project_bounds=False):
    """
    Search for loops annotated with the `loki loop-interchange` pragma and attempt
    to reorder them.

    Note that this effectively just exchanges variable and bounds for each of the loops,
    leaving the rest (including bodies, pragmas, etc.) intact.
    """
    with pragmas_attached(routine, Loop):
        loop_map = {}
        for loop_nest in FindNodes(Loop).visit(routine.body):
            if not is_loki_pragma(loop_nest.pragma, starts_with='loop-interchange'):
                continue

            # Get variable order from pragma
            var_order = get_pragma_parameters(loop_nest.pragma).get('loop-interchange', None)
            if var_order:
                var_order = [var.strip().lower() for var in var_order.split(',')]
                depth = len(var_order)
            else:
                depth = 2

            # Extract loop nest
            loops = get_nested_loops(loop_nest, depth)
            loop_variables, loop_ranges, *_ = get_loop_components(loops)

            # Find the loop order from the variable order
            if var_order is None:
                var_order = [str(var).lower() for var in reversed(loop_variables)]
            loop_variable_names = [var.name.lower() for var in loop_variables]
            loop_order = [loop_variable_names.index(var) for var in var_order]

            # Project iteration space
            if project_bounds:
                iteration_space = Polyhedron.from_loop_ranges(loop_variables, loop_ranges)
                iteration_space = generate_loop_bounds(iteration_space, loop_order)

            # Rebuild loops starting with innermost
            inner_loop_map = None
            for idx, (loop, loop_idx) in enumerate(zip(reversed(loops), reversed(loop_order))):
                if project_bounds:
                    new_idx = len(loop_order) - idx - 1
                    ignore_variables = list(range(new_idx+1, len(loop_order)))
                    lower_bounds = iteration_space.lower_bounds(new_idx, ignore_variables)
                    upper_bounds = iteration_space.upper_bounds(new_idx, ignore_variables)

                    if len(lower_bounds) == 1:
                        lower_bounds = lower_bounds[0]
                    else:
                        fct_symbol = sym.ProcedureSymbol('max', scope=routine)
                        lower_bounds = sym.InlineCall(fct_symbol, parameters=as_tuple(lower_bounds))

                    if len(upper_bounds) == 1:
                        upper_bounds = upper_bounds[0]
                    else:
                        fct_symbol = sym.ProcedureSymbol('min', scope=routine)
                        upper_bounds = sym.InlineCall(fct_symbol, parameters=as_tuple(upper_bounds))

                    bounds = sym.LoopRange((lower_bounds, upper_bounds))
                else:
                    bounds = loop_ranges[loop_idx]

                outer_loop = loop.clone(variable=loop_variables[loop_idx], bounds=bounds)
                if inner_loop_map is not None:
                    outer_loop = Transformer(inner_loop_map).visit(outer_loop)
                inner_loop_map = {loop: outer_loop}

            # Annotate loop-interchange in a comment
            old_vars = ', '.join(loop_variable_names)
            new_vars = ', '.join(var_order)
            comment = Comment(f'! Loki loop-interchange ({old_vars} <--> {new_vars})')

            # Strip loop-interchange pragma and register new loop nest in map
            pragmas = tuple(p for p in as_tuple(loops[0].pragma)
                            if not is_loki_pragma(p, starts_with='loop-interchange'))
            loop_map[loop_nest] = (comment, outer_loop.clone(pragma=pragmas))

        # Apply loop-interchange mapping
        if loop_map:
            routine.body = Transformer(loop_map).visit(routine.body)
            info('%s: interchanged %d loop nest(s)', routine.name, len(loop_map))


def pragma_ranges_to_loop_ranges(parameters, scope):
    """
    Convert loop ranges given in the pragma parameters from string to a tuple of `LoopRange`
    objects.
    """
    if 'range' not in parameters:
        return None
    ranges = []
    for item in parameters['range'].split(','):
        bounds = [parse_expr(bound, scope=scope) for bound in item.split(':')]
        ranges += [sym.LoopRange(as_tuple(bounds))]

    return as_tuple(ranges)


def loop_fusion(routine):
    """
    Search for loops annotated with the `loki loop-fusion` pragma and attempt
    to fuse them into a single loop.
    """
    fusion_groups = defaultdict(list)
    loop_map = {}
    with pragmas_attached(routine, Loop):
        # Extract all annotated loops and sort them into fusion groups
        for loop in FindNodes(Loop).visit(routine.body):
            if is_loki_pragma(loop.pragma, starts_with='loop-fusion'):
                parameters = get_pragma_parameters(loop.pragma, starts_with='loop-fusion')
                group = parameters.get('group', 'default')
                fusion_groups[group] += [(loop, parameters)]

        if not fusion_groups:
            return

        # Merge loops in each group and put them in the position of the group's first loop
        #  UNLESS 'insert' location is specified for at least one of the group's fusion
        #  pragmas, in this case the position is the first occurence of 'insert' for each group
        for group, loop_parameter_lists in fusion_groups.items():
            loop_list, parameters = zip(*loop_parameter_lists)

            # First, determine the collapse depth and extract user-annotated loop ranges from pragmas
            collapse = [param.get('collapse', None) for param in parameters]
            insert_locs = [param.get('insert', False) for param in parameters]
            if collapse != [collapse[0]] * len(collapse):
                raise RuntimeError(f'Conflicting collapse values in group "{group}"')
            collapse = int(collapse[0]) if collapse[0] is not None else 1

            pragma_ranges = [pragma_ranges_to_loop_ranges(param, routine) for param in parameters]

            # If we have a pragma somewhere with an explicit loop range, we use that for the fused loop
            range_set = {r for r in pragma_ranges if r is not None}
            if len(range_set) not in (0, 1):
                raise RuntimeError(f'Pragma-specified loop ranges in group "{group}" do not match')

            fusion_ranges = None
            if range_set:
                fusion_ranges = range_set.pop()

            # Next, extract loop ranges for all loops in group and convert to iteration space
            # polyhedrons for easier alignment
            loop_variables, loop_ranges, loop_bodies = \
                    zip(*[get_loop_components(get_nested_loops(loop, collapse)) for loop in loop_list])
            iteration_spaces = [Polyhedron.from_loop_ranges(variables, ranges)
                                for variables, ranges in zip(loop_variables, loop_ranges)]

            # Find the fused iteration space (if not given by a pragma)
            if fusion_ranges is None:
                fusion_ranges = []
                for level in range(collapse):
                    lower_bounds, upper_bounds = [], []
                    ignored_variables = list(range(level+1, collapse))

                    for p in iteration_spaces:
                        for bound in p.lower_bounds(level, ignored_variables):
                            # Decide if we learn something new from this bound, which could be because:
                            # (1) we don't have any bounds, yet
                            # (2) bound is smaller than existing lower bounds (i.e. diff < 0)
                            # (3) bound is not constant and none of the existing bounds are lower (i.e. diff >= 0)
                            diff = [simplify(bound - b) for b in lower_bounds]
                            is_any_negative = any(is_constant(d) and symbolic_op(d, op.lt, 0) for d in diff)
                            is_any_not_negative = any(is_constant(d) and symbolic_op(d, op.ge, 0) for d in diff)
                            is_new_bound = (not lower_bounds or is_any_negative or
                                            (not is_constant(bound) and not is_any_not_negative))
                            if is_new_bound:
                                # Remove any lower bounds made redundant by bound:
                                lower_bounds = [b for b, d in zip(lower_bounds, diff)
                                                if not (is_constant(d) and symbolic_op(d, op.lt, 0))]
                                lower_bounds += [bound]

                        for bound in p.upper_bounds(level, ignored_variables):
                            # Decide if we learn something new from this bound, which could be because:
                            # (1) we don't have any bounds, yet
                            # (2) bound is larger than existing upper bounds (i.e. diff > 0)
                            # (3) bound is not constant and none of the existing bounds are larger (i.e. diff <= 0)
                            diff = [simplify(bound - b) for b in upper_bounds]
                            is_any_positive = any(is_constant(d) and symbolic_op(d, op.gt, 0) for d in diff)
                            is_any_not_positive = any(is_constant(d) and symbolic_op(d, op.le, 0) for d in diff)
                            is_new_bound = (not upper_bounds or is_any_positive or
                                            (not is_constant(bound) and not is_any_not_positive))
                            if is_new_bound:
                                # Remove any lower bounds made redundant by bound:
                                upper_bounds = [b for b, d in zip(upper_bounds, diff)
                                                if not (is_constant(d) and symbolic_op(d, op.gt, 0))]
                                upper_bounds += [bound]

                    if len(lower_bounds) == 1:
                        lower_bounds = lower_bounds[0]
                    else:
                        fct_symbol = sym.DeferredTypeSymbol(name='min', scope=routine)
                        lower_bounds = sym.InlineCall(fct_symbol, parameters=as_tuple(lower_bounds))

                    if len(upper_bounds) == 1:
                        upper_bounds = upper_bounds[0]
                    else:
                        fct_symbol = sym.DeferredTypeSymbol(name='max', scope=routine)
                        upper_bounds = sym.InlineCall(fct_symbol, parameters=as_tuple(upper_bounds))

                    fusion_ranges += [sym.LoopRange((lower_bounds, upper_bounds))]

            # Align loop ranges and collect bodies
            fusion_bodies = []
            fusion_variables = loop_variables[0]
            for idx, (variables, ranges, bodies, p) in enumerate(
                    zip(loop_variables, loop_ranges, loop_bodies, iteration_spaces)):
                # TODO: This throws away anything that is not in the inner-most loop body.
                body = flatten([Comment(f'! Loki loop-fusion - body {idx} begin'),
                                bodies[-1],
                                Comment(f'! Loki loop-fusion - body {idx} end')])

                # Replace loop variables if necessary
                var_map = {}
                for loop_variable, fusion_variable in zip(variables, fusion_variables):
                    if loop_variable != fusion_variable:
                        var_map.update({var: fusion_variable for var in FindVariables().visit(body)
                                        if var.name.lower() == loop_variable.name})
                if var_map:
                    body = SubstituteExpressions(var_map).visit(body)

                # Wrap in conditional if loop bounds are different
                conditions = []
                for loop_range, fusion_range, variable in zip(ranges, fusion_ranges, fusion_variables):
                    if symbolic_op(loop_range.start, op.ne, fusion_range.start):
                        conditions += [sym.Comparison(variable, '>=', loop_range.start)]
                    if symbolic_op(loop_range.stop, op.ne, fusion_range.stop):
                        conditions += [sym.Comparison(variable, '<=', loop_range.stop)]
                if conditions:
                    if len(conditions) == 1:
                        condition = conditions[0]
                    else:
                        condition = sym.LogicalAnd(as_tuple(conditions))
                    body = Conditional(condition=condition, body=as_tuple(body), else_body=())

                fusion_bodies += [body]

            # Create the nested fused loop and replace original loops
            fusion_loop = flatten(fusion_bodies)
            for fusion_variable, fusion_range in zip(reversed(fusion_variables), reversed(fusion_ranges)):
                fusion_loop = Loop(variable=fusion_variable, body=as_tuple(fusion_loop), bounds=fusion_range)

            comment = Comment(f'! Loki loop-fusion group({group})')
            insert_loc = insert_locs.index(None) if None in insert_locs else 0
            loop_map[loop_list[insert_loc]] = (comment, Pragma(keyword='loki',
                content=f'fused-loop group({group})'), fusion_loop)
            comment = Comment(f'! Loki loop-fusion group({group}) - loop hoisted')
            loop_map.update({loop: comment for i_loop, loop in enumerate(loop_list) if i_loop != insert_loc})

        # Apply transformation
        routine.body = Transformer(loop_map).visit(routine.body)
        info('%s: fused %d loops in %d groups.', routine.name,
             sum(len(loop_list) for loop_list in fusion_groups.values()), len(fusion_groups))


class FissionTransformer(NestedMaskedTransformer):
    """
    Bespoke transformer that splits loops or loop nests at
    ``!$loki loop-fission`` pragmas.

    For that, the subtree that makes up the loop body is traversed multiple,
    times capturing everything before, after or in-between fission pragmas
    in each traversal, using :class:``NestedMaskedTransformer``.
    Any intermediate nodes that define sections (e.g. conditionals) are
    reproduced in each subtree traversal.

    This works also for nested loops with individually different fission
    annotations.

    Parameters
    ----------
    loop_pragmas : dict of (:any:`Loop`, list of :any:`Pragma`)
        Mapping of all loops to the list of contained
        ``loop-fission`` pragmas at which they should be split.
    """

    def __init__(self, loop_pragmas, active=True, **kwargs):
        super().__init__(active=active, require_all_start=True, greedy_stop=True, **kwargs)
        self.loop_pragmas = loop_pragmas
        self.split_loops = {}

    def visit_Loop(self, o, **kwargs):
        if o not in self.loop_pragmas:
            # loops that are not marked for fission can be handled as
            # in the regular NestedMaskedTransformer
            return super().visit_InternalNode(o, **kwargs)

        if not (self.active or self.start):
            # this happens if we encounter a loop marked for fission while
            # already traversing the subtree of an enclosing fission loop.
            # no more macros are marked to make this subtree active, thus
            # we can bail out here
            return None

        # Recurse for all children except the body
        body_index = o._traversable.index('body')
        visited = tuple(self.visit(c, **kwargs) for i, c in enumerate(o.children) if i != body_index)

        # Save current state so we can restore for each subtree
        _start, _stop, _active = self.start, self.stop, self.active

        def rebuild_fission_branch(start_node, stop_node, **kwargs):
            if start_node is None:
                # This subtree is either active already or we have a fission pragma
                # with collapse in _start from an enclosing loop
                self.active = _active
                self.start = _start.copy()
            else:
                # We build a subtree after a fission pragma. Make sure that all
                # pragmas have been encountered before processing the subtree
                self.active = False
                self.start = _start.copy() | {start_node}
                self.mapper[start_node] = None
            # we stop when encountering this or any previously defined stop nodes
            self.stop = _stop.copy() | set(as_tuple(stop_node))
            body = flatten(self.visit(o.body, **kwargs))
            if start_node is not None:
                self.mapper.pop(start_node)
            if not body:
                return [()]
            # inject a comment to mark where the loop was split
            comment = [] if start_node is None else [Comment(f'! Loki - {start_node.content}')]
            return comment + [self._rebuild(o, visited[:body_index] + (body,) + visited[body_index:])]

        # Use masked transformer to build subtrees from/to pragma
        rebuilt = rebuild_fission_branch(None, self.loop_pragmas[o][0], **kwargs)
        for start, stop in zip(self.loop_pragmas[o][:-1], self.loop_pragmas[o][1:]):
            rebuilt += rebuild_fission_branch(start, stop, **kwargs)
        rebuilt += rebuild_fission_branch(self.loop_pragmas[o][-1], None, **kwargs)

        # Register the new loops in the mapping
        loops = [l for l in rebuilt if isinstance(l, Loop)]
        self.split_loops.update({pragma: loops[i:] for i, pragma in enumerate(self.loop_pragmas[o])})

        # Restore original state (except for the active status because this has potentially
        # been changed when traversing the loop body)
        self.start, self.stop = _start, _stop

        return as_tuple(i for i in rebuilt if i)


def loop_fission(routine, promote=True, warn_loop_carries=True):
    """
    Search for ``!$loki loop-fission`` pragmas in loops and split them.

    The expected pragma syntax is
    ``!$loki loop-fission [collapse(n)] [promote(var-name, var-name, ...)]``
    where ``collapse(n)`` gives the loop nest depth to be split (defaults to n=1)
    and ``promote`` optionally specifies a list of variable names to be promoted
    by the split iteration space dimensions.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which loop fission is to be applied.
    promote : bool, optional
        Try to automatically detect read-after-write across fission points
        and promote corresponding variables. Note that this does not affect
        promotion of variables listed directly in the pragma's ``promote``
        option.
    warn_loop_carries : bool, optional
        Try to automatically detect loop-carried dependencies and warn
        when the fission point sits after the initial read and before the
        final write.
    """
    promotion_vars_dims = CaseInsensitiveDict()

    pragma_loops = defaultdict(list)  # List of enclosing loops per fission pragmas
    loop_pragmas = defaultdict(list)  # List of pragmas splitting a loop
    promotion_vars_dims = {}  # Variables to promote with new dimension
    promotion_vars_index = {}  # Variable subscripts to promote with new indices
    loop_carried_vars = {}  # List of loop carried dependencies in original loop

    # First, find the loops enclosing each pragma
    for loop in FindNodes(Loop).visit(routine.body):
        for pragma in FindNodes(Pragma).visit(loop.body):
            if is_loki_pragma(pragma, starts_with='loop-fission'):
                pragma_loops[pragma] += [loop]

    if not pragma_loops:
        return

    with optional(promote or warn_loop_carries, dataflow_analysis_attached, routine):
        for pragma in pragma_loops:
            # Now, sort the loops enclosing each pragma from outside to inside and
            # keep only the ones relevant for fission
            loops = binary_insertion_sort(pragma_loops[pragma], lt=is_parent_of)
            collapse = int(get_pragma_parameters(pragma).get('collapse', 1))
            pragma_loops[pragma] = loops[-collapse:]

            # Attach the pragma to the list of pragmas to be processed for the
            # outermost loop
            loop_pragmas[loops[-collapse]] += [pragma]

            # Promote variables given in promotion list
            promote_vars = [var.strip().lower()
                            for var in get_pragma_parameters(pragma).get('promote', '').split(',') if var]

            # Automatically determine promotion variables
            if promote:
                promote_vars += [v.name.lower() for v in read_after_write_vars(loops[-1].body, pragma)
                                 if v.name.lower() not in promote_vars]
            promotion_vars_dims, promotion_vars_index = promotion_dimensions_from_loop_nest(
                promote_vars, pragma_loops[pragma], promotion_vars_dims, promotion_vars_index)

            # Store loop-carried dependencies for later analysis
            if warn_loop_carries:
                loop_carried_vars[pragma] = loop_carried_dependencies(pragma_loops[pragma][0])

    fission_trafo = FissionTransformer(loop_pragmas)
    routine.body = fission_trafo.visit(routine.body)
    info('%s: split %d loop(s) at %d loop-fission pragma(s).', routine.name, len(loop_pragmas), len(pragma_loops))

    # Warn about broken loop-carried dependencies
    if warn_loop_carries:
        with dataflow_analysis_attached(routine):
            for pragma, loop_carries in loop_carried_vars.items():
                loop, *remainder = fission_trafo.split_loops[pragma]
                if not remainder:
                    continue

                # The loop before the pragma has to read the variable ...
                broken_loop_carries = loop_carries & loop.uses_symbols
                # ... but it is written after the pragma
                broken_loop_carries &= set.union(*[l.defines_symbols for l in remainder])

                if broken_loop_carries:
                    if pragma.source and pragma.source.lines:
                        line_info = f' at l. {pragma.source.lines[0]}'
                    else:
                        line_info = ''
                    warning(f'Loop-fission{line_info} potentially breaks loop-carried dependencies' +
                            f'for variables: {", ".join(str(v) for v in broken_loop_carries)}')

    promote_nonmatching_variables(routine, promotion_vars_dims, promotion_vars_index)


class LoopUnrollTransformer(Transformer):
    """
    Transformer that unrolls loops or loop nests at
    ``!$loki loop-unroll`` pragmas.

    For loops to be unrolled, they must have literal bounds and step.
    If not, then they are simply ignored.

    This works also for nested loops with individually different unroll
    annotations. However, a child nested loop with a more restrictive depth
    will not be able to override its parent's depth.
    """

    def __init__(self, warn_iterations_length=True):
        self.warn_iterations_length = warn_iterations_length
        super().__init__()

    # depth is treated as an option of some depth or none, i.e. unroll all
    def visit_Loop(self, o, depth=None):
        """
        Apply this :class:`Transformer` to an IR tree.

        Parameters
        ----------
        o : :any:`Node`
            The node to visit.
        depth : 'Int', optional
            How deep down a loop nest unrolling should be applied.
        """

        # If the step isn't explicitly given, then it's implicitly 1
        step = o.bounds.step if o.bounds.step is not None else IntLiteral(1)
        start, stop = o.bounds.start, o.bounds.stop

        depth = depth - 1 if depth is not None else None

        # Only unroll if we have all literal bounds and step
        if isinstance(start, (IntLiteral, FloatLiteral)) and\
                isinstance(stop, (IntLiteral, FloatLiteral)) and\
                isinstance(step, (IntLiteral, FloatLiteral)):

            #  int() to truncate any floats - which are not invalid in all specs!
            unroll_range = range(int(start), int(stop) + 1, int(step))
            if self.warn_iterations_length and len(unroll_range) > 32:
                warning(f"Unrolling loop over 32 iterations ({len(unroll_range)}), this may take a long time & "
                        f"provide few performance benefits.")

            acc = functools.reduce(op.add,
                                   [
                                       # Create a copy of the loop body for every value of the iterator
                                       SubstituteExpressions({o.variable: sym.IntLiteral(i)}).visit(o.body)
                                       for i in unroll_range
                                   ],
                                   ())

            if depth is None or depth >= 1:
                acc = [self.visit(a, depth=depth) for a in acc]

            return as_tuple(flatten(acc))

        return Loop(
            variable=o.variable,
            body=self.visit(o.body, depth=depth),
            bounds=o.bounds
                    )


def loop_unroll(routine, warn_iterations_length=True):
    """
    Search for ``!$loki loop-unroll`` pragmas in loops and unroll them.

    The expected pragma syntax is
    ``!$loki loop-unroll [depth(n)]``
    where ``depth(n)`` controls the unrolling of nested loops. For instance,
    ``depth(1)`` will only unroll the top most loop of a set of nested loops.
    However, a child nested loop with a more restrictive depth will not be
    able to override its parent's depth. If ``depth(n)`` is not specified,
    then all loops nested under a parent with this pragma will be unrolled.
    E.g. The code sample below will only unroll A and B, but not C:

    ! Loop A
    !$loki loop-unroll depth(1)
    DO a = 1, 10
        ! Loop B
        !$loki loop-unroll
        DO b = 1, 10
            ...
        END DO
        ! Loop C - will not be unrolled
        DO c = 1, 10
            ...
        END DO
    END DO

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which loop unrolling is to be applied.
    warn_iterations_length : 'Boolean', optional
        This specifies if warnings should be generated when unrolling
        loops with a large number of iterations (32). It's mainly to
        disable warnings when loops are being unrolled for internal
        transformations and analysis.
    """

    class PragmaLoopUnrollTransformer(Transformer):
        def __init__(self, warn_iterations_length=True):
            self.warn_iterations_length = warn_iterations_length
            super().__init__()

        def visit_Loop(self, o, *args, **kwargs):
            # Check for pragmas
            if is_loki_pragma(o.pragma, starts_with='loop-unroll'):
                parameters = get_pragma_parameters(o.pragma, starts_with='loop-unroll')

                # Get the depth
                param = parameters.get('depth', None)
                depth = int(param) if param is not None else None

                # Unroll and recurse
                unrolled_loop = LoopUnrollTransformer(self.warn_iterations_length).visit(o, depth=depth)

                # unrolled_loop could be either an unrollable Loop() or a Tuple() of Nodes
                try:
                    return as_tuple(flatten([self.visit(a) for a in as_tuple(flatten(unrolled_loop))]))
                # Loop() is not iterable
                except TypeError:
                    return self.visit(unrolled_loop, *args, **kwargs)

            return super().visit_Node(o, *args, **kwargs)

    with pragmas_attached(routine, Loop):
        routine.body = PragmaLoopUnrollTransformer(warn_iterations_length=warn_iterations_length).visit(routine.body)
