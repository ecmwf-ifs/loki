"""
Collection of utility routines that provide loop transformations.

"""
from collections import defaultdict
import operator as op
import numpy as np
from pymbolic.primitives import Variable

from loki.expression import (
    symbols as sym, SubstituteExpressions, FindVariables,
    accumulate_polynomial_terms, simplify, is_constant, symbolic_op
)
from loki.frontend.fparser import parse_fparser_expression
from loki.ir import Loop, Conditional, Comment, Pragma
from loki.logging import info
from loki.pragma_utils import is_loki_pragma, get_pragma_parameters, pragmas_attached
from loki.transform.transform_array_indexing import promote_variables
from loki.tools import flatten, as_tuple, CaseInsensitiveDict, binary_insertion_sort
from loki.visitors import FindNodes, Transformer, MaskedTransformer, NestedMaskedTransformer, is_parent_of

__all__ = ['loop_interchange', 'loop_fusion', 'loop_fission', 'Polyhedron', 'section_hoist']


class Polyhedron:
    """
    Halfspace representation of a (convex) polyhedron.

    A polyhedron `P c R^d` is described by a set of inequalities, in matrix form
    ```
    P = { x=[x1,...,xd]^T c R^d | Ax <= b }
    ```
    with n-by-d matrix `A` and d-dimensional right hand side `b`.

    In loop transformations, polyhedrons are used to represent iteration spaces of
    d-deep loop nests.

    :param np.array A: the representation matrix A.
    :param np.array b: the right hand-side vector b.
    :param list variables: list of variables representing the dimensions in the polyhedron.
    """

    def __init__(self, A, b, variables=None):
        A = np.array(A, dtype=np.dtype(int))
        b = np.array(b, dtype=np.dtype(int))
        assert A.ndim == 2 and b.ndim == 1
        assert A.shape[0] == b.shape[0]
        self.A = A
        self.b = b

        self.variables = None
        self.variable_names = None
        if variables is not None:
            assert len(variables) == A.shape[1]
            self.variables = variables
            self.variable_names = [v.name.lower() for v in self.variables]

    def variable_to_index(self, variable):
        if self.variable_names is None:
            raise RuntimeError('No variables list associated with polyhedron.')
        if isinstance(variable, (sym.Array, sym.Scalar)):
            variable = variable.name.lower()
        assert isinstance(variable, str)
        return self.variable_names.index(variable)

    @staticmethod
    def _to_literal(value):
        if value < 0:
            return sym.Product((-1, sym.IntLiteral(abs(value))))
        return sym.IntLiteral(value)

    def lower_bounds(self, index_or_variable):
        """
        Return all lower bounds imposed on a variable.

        Lower bounds for variable `j` are given by the index set
        ```
        L = {i in {0,...,d-1} | A_ij < 0}
        ```

        :param index_or_variable: the index, name, or expression symbol for which the
                    lower bounds are produced.
        :type index_or_variable: int or str or sym.Array or sym.Scalar

        :returns list: the bounds for that variable.
        """
        if isinstance(index_or_variable, int):
            j = index_or_variable
        else:
            j = self.variable_to_index(index_or_variable)

        bounds = []
        for i in range(self.A.shape[0]):
            if self.A[i,j] < 0:
                components = [self._to_literal(self.A[i,k]) * self.variables[k]
                              for k in range(self.A.shape[1]) if k != j and self.A[i,k] != 0]
                if not components:
                    lhs = sym.IntLiteral(0)
                elif len(components) == 1:
                    lhs = components[0]
                else:
                    lhs = sym.Sum(as_tuple(components))
                bounds += [simplify(sym.Quotient(self._to_literal(self.b[i]) - lhs,
                                                 self._to_literal(self.A[i,j])))]
        return bounds

    def upper_bounds(self, index_or_variable):
        """
        Return all upper bounds imposed on a variable.

        Upper bounds for variable `j` are given by the index set
        ```
        U = {i in {0,...,d-1} | A_ij > 0}
        ```

        :param index_or_variable: the index, name, or expression symbol for which the
                    upper bounds are produced.
        :type index_or_variable: int or str or sym.Array or sym.Scalar

        :returns list: the bounds for that variable.
        """
        if isinstance(index_or_variable, int):
            j = index_or_variable
        else:
            j = self.variable_to_index(index_or_variable)

        bounds = []
        for i in range(self.A.shape[0]):
            if self.A[i,j] > 0:
                components = [self._to_literal(self.A[i,k]) * self.variables[k]
                              for k in range(self.A.shape[1]) if k != j and self.A[i,k] != 0]
                if not components:
                    lhs = sym.IntLiteral(0)
                elif len(components) == 1:
                    lhs = components[0]
                else:
                    lhs = sym.Sum(as_tuple(components))
                bounds += [simplify(sym.Quotient(self._to_literal(self.b[i]) - lhs,
                                                 self._to_literal(self.A[i,j])))]
        return bounds

    @classmethod
    def from_loop_ranges(cls, loop_variables, loop_ranges):
        """
        Create polyhedron from a list of loop ranges and associated variables.
        """
        assert len(loop_ranges) == len(loop_variables)

        # Add any variables that are not loop variables to the vector of variables
        variables = list(loop_variables)
        variable_names = [v.name.lower() for v in variables]
        for v in sorted(FindVariables().visit(loop_ranges), key=lambda v: v.name.lower()):
            if v.name.lower() not in variable_names:
                variables += [v]
                variable_names += [v.name.lower()]

        n = 2 * len(loop_ranges)
        d = len(variables)
        A = np.zeros([n, d], dtype=np.dtype(int))
        b = np.zeros([n], dtype=np.dtype(int))

        for i, (loop_variable, loop_range) in enumerate(zip(loop_variables, loop_ranges)):
            assert loop_range.step is None or loop_range.step == '1'
            j = variables.index(loop_variable.name.lower())

            # Create inequality from lower bound
            lower_bound = simplify(loop_range.start)
            if not (is_constant(lower_bound) or
                    isinstance(lower_bound, (Variable, sym.Sum, sym.Product))):
                raise ValueError('Cannot derive inequality from bound {}'.format(str(lower_bound)))

            summands = accumulate_polynomial_terms(lower_bound)
            b[2*i] = -summands.pop(1, 0)
            A[2*i, j] = -1
            for base, coef in summands.items():
                if not len(base) == 1:
                    raise ValueError('Non-affine lower bound {}'.format(str(lower_bound)))
                A[2*i, variables.index(base[0].name.lower())] = coef

            # Create inequality from upper bound
            upper_bound = simplify(loop_range.stop)
            if not (is_constant(upper_bound) or
                    isinstance(upper_bound, (Variable, sym.Sum, sym.Product))):
                raise ValueError('Cannot derive inequality from bound {}'.format(str(upper_bound)))

            summands = accumulate_polynomial_terms(upper_bound)
            b[2*i+1] = summands.pop(1, 0)
            A[2*i+1, j] = 1
            for base, coef in summands.items():
                if not len(base) == 1:
                    raise ValueError('Non-affine upper bound {}'.format(str(upper_bound)))
                A[2*i+1, variable_names.index(base[0].name.lower())] = -coef

        return cls(A, b, variables)


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
    if project_bounds:
        # For that we need Fourier-Motzkin Elimination in the Polyhedron first
        raise NotImplementedError

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

            # Extract loop nest and determine iteration space
            loops = get_nested_loops(loop_nest, depth)
            loop_variables, loop_ranges, *_ = get_loop_components(loops)
            if project_bounds:
                iteration_space = Polyhedron.from_loop_ranges(loop_variables, loop_ranges)

            if var_order is None:
                var_order = [str(var).lower() for var in reversed(loop_variables)]

            # Find the loop order from the variable order
            loop_variable_names = [var.name.lower() for var in loop_variables]
            loop_order = [loop_variable_names.index(var) for var in var_order]

            # Rebuild loops starting with innermost
            inner_loop_map = None
            for loop, loop_idx in zip(reversed(loops), reversed(loop_order)):
                if project_bounds:
                    # TODO: project iteration spaces
                    lower_bound = iteration_space.lower_bounds(loop_idx)
                    upper_bound = iteration_space.upper_bounds(loop_idx)
                    bounds = sym.LoopRange((lower_bound, upper_bound))
                else:
                    bounds = loop_ranges[loop_idx]

                outer_loop = loop.clone(variable=loop_variables[loop_idx], bounds=bounds)
                if inner_loop_map is not None:
                    outer_loop = Transformer(inner_loop_map).visit(outer_loop)
                inner_loop_map = {loop: outer_loop}

            # Annotate loop-interchange in a comment
            old_vars = ', '.join(loop_variable_names)
            new_vars = ', '.join(var_order)
            comment = Comment('! Loki loop-interchange ({} <--> {})'.format(old_vars, new_vars))

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
        bounds = [parse_fparser_expression(bound, scope=scope) for bound in item.split(':')]
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
    for group, loop_parameter_lists in fusion_groups.items():
        loop_list, parameters = zip(*loop_parameter_lists)

        # First, determine the collapse depth and extract user-annotated loop ranges from pragmas
        collapse = [param.get('collapse', None) for param in parameters]
        if collapse != [collapse[0]] * len(collapse):
            raise RuntimeError('Conflicting collapse values in group "{}"'.format(group))
        collapse = int(collapse[0]) if collapse[0] is not None else 1

        pragma_ranges = [pragma_ranges_to_loop_ranges(param, routine.scope) for param in parameters]

        # If we have a pragma somewhere with an explicit loop range, we use that for the fused loop
        range_set = {r for r in pragma_ranges if r is not None}
        if len(range_set) not in (0, 1):
            raise RuntimeError('Pragma-specified loop ranges in group "{}" do not match'.format(group))

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
                for p in iteration_spaces:
                    for bound in p.lower_bounds(level):
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

                    for bound in p.upper_bounds(level):
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
                    fct_symbol = sym.ProcedureSymbol('min', scope=routine.scope)
                    lower_bounds = sym.InlineCall(fct_symbol, parameters=as_tuple(lower_bounds))

                if len(upper_bounds) == 1:
                    upper_bounds = upper_bounds[0]
                else:
                    fct_symbol = sym.ProcedureSymbol('max', scope=routine.scope)
                    upper_bounds = sym.InlineCall(fct_symbol, parameters=as_tuple(lower_bounds))

                fusion_ranges += [sym.LoopRange((lower_bounds, upper_bounds))]

        # Align loop ranges and collect bodies
        fusion_bodies = []
        fusion_variables = loop_variables[0]
        for idx, (variables, ranges, bodies, p) in enumerate(
                zip(loop_variables, loop_ranges, loop_bodies, iteration_spaces)):
            # TODO: This throws away anything that is not in the inner-most loop body.
            body = flatten([Comment('! Loki loop-fusion - body {} begin'.format(idx)),
                            bodies[-1],
                            Comment('! Loki loop-fusion - body {} end'.format(idx))])

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
                    conditions = conditions[0]
                else:
                    conditions = sym.LogicalAnd(as_tuple(conditions))
                body = Conditional(conditions=[conditions], bodies=[body], else_body=())

            fusion_bodies += [body]

        # Create the nested fused loop and replace original loops
        fusion_loop = flatten(fusion_bodies)
        for fusion_variable, fusion_range in zip(reversed(fusion_variables), reversed(fusion_ranges)):
            fusion_loop = Loop(variable=fusion_variable, body=as_tuple(fusion_loop), bounds=fusion_range)

        comment = Comment('! Loki loop-fusion group({})'.format(group))
        loop_map[loop_list[0]] = (comment, fusion_loop)
        comment = Comment('! Loki loop-fusion group({}) - loop hoisted'.format(group))
        loop_map.update({loop: comment for loop in loop_list[1:]})

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

    :param dict loop_pragmas:
        a dictionary that maps all loops to the list of contained
        ``loop-fission`` pragmas on which they should be split.
    """

    def __init__(self, loop_pragmas, active=True, **kwargs):
        super().__init__(active=active, require_all_start=True, greedy_stop=True, **kwargs)
        self.loop_pragmas = loop_pragmas

    def visit_Loop(self, o, **kwargs):
        if o not in self.loop_pragmas:
            # loops that are not marked for fission can be handled as
            # in the regular NestedMaskedTransformer
            return super().visit_Loop(o, **kwargs)

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
            body = self.visit(o.body, **kwargs)
            if start_node is not None:
                self.mapper.pop(start_node)
            if not body:
                return None
            return self._rebuild(o, visited[:body_index] + (body,) + visited[body_index:])

        # Use masked transformer to build subtrees from/to pragma
        rebuilt = [rebuild_fission_branch(None, self.loop_pragmas[o][0], **kwargs)]
        for start, stop in zip(self.loop_pragmas[o][:-1], self.loop_pragmas[o][1:]):
            rebuilt += [rebuild_fission_branch(start, stop, **kwargs)]
        rebuilt += [rebuild_fission_branch(self.loop_pragmas[o][-1], None, **kwargs)]

        # Restore original state (except for the active status because this has potentially
        # been changed when traversing the loop body)
        self.start, self.stop = _start, _stop

        return as_tuple(i for i in rebuilt if i is not None)


def loop_fission(routine):
    """
    Search for ``!$loki loop-fission`` pragmas inside loops and to split them
    into multiple loops.

    The pragma syntax is
    ``!$loki loop-fission [collapse(n)] [promote(var-name, var-name, ...)]``
    where ``collapse(n)`` gives the loop nest depth to be split (defaults to n=1)
    and ``promote`` specifies a list of variable names to be promoted by the
    split iteration space dimensions.
    """
    promotion_vars_dims = CaseInsensitiveDict()

    pragma_loops = defaultdict(list)  # List of enclosing loops per fission pragmas
    loop_pragmas = defaultdict(list)  # List of pragmas splitting a loop
    promotion_vars_dims = {}  # Variables to promote with new dimension
    promotion_vars_index = {}  # Variable subscripts to promote with new indices

    # First, find the loops enclosing each pragma
    for loop in FindNodes(Loop).visit(routine.body):
        for pragma in FindNodes(Pragma).visit(loop.body):
            if is_loki_pragma(pragma, starts_with='loop-fission'):
                pragma_loops[pragma] += [loop]

    if not pragma_loops:
        return

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
        loop_lengths = [simplify(loop.bounds.stop - loop.bounds.start + sym.IntLiteral(1))
                        for loop in reversed(pragma_loops[pragma])]
        loop_index = [loop.variable for loop in reversed(pragma_loops[pragma])]
        promote_vars = {var.strip().lower()
                        for var in get_pragma_parameters(pragma).get('promote', '').split(',') if var}

        for var_name in promote_vars:
            # Check if we have already marked this variable for promotion: let's make sure the added
            # dimensions are large enough for this loop (nest)
            if var_name not in promotion_vars_dims:
                promotion_vars_dims[var_name] = loop_lengths
                promotion_vars_index[var_name] = loop_index
            else:
                if len(promotion_vars_dims[var_name]) != len(loop_lengths):
                    raise RuntimeError('Conflicting promotion dimensions for "{}"'.format(var_name))
                for i, (loop_length, index) in enumerate(zip(loop_lengths, loop_index)):
                    if index != promotion_vars_index[var_name][i]:
                        raise RuntimeError('Loop variable "{}" does not match previous index "{}" for "{}"'.format(
                            str(index), str(promotion_vars_index[var_name][i]), var_name))
                    if symbolic_op(promotion_vars_dims[var_name][i], op.lt, loop_length):
                        promotion_vars_dims[var_name][i] = loop_length

    routine.body = FissionTransformer(loop_pragmas).visit(routine.body)

    if promotion_vars_dims:
        for var_name, size in promotion_vars_dims.items():
            promote_variables(routine, [var_name], -1, index=promotion_vars_index[var_name], size=size)
        info('%s: promoted variable(s): %s', routine.name, ', '.join(promotion_vars_dims.keys()))


def section_hoist(routine):
    """
    Hoist one or multiple code sections and insert them at a specified target location.
    """
    hoist_groups = defaultdict(list)

    # Find all section-hoist directives
    for pragma in FindNodes(Pragma).visit(routine.body):
        if is_loki_pragma(pragma, starts_with='section-hoist'):
            parameters = get_pragma_parameters(pragma, starts_with='section-hoist')
            group = parameters.get('group', 'default')
            hoist_groups[group] += [pragma]

    hoist_map = {}
    starts, stops = [], []
    for group, pragmas in hoist_groups.items():
        # Extract sections
        section_starts = [pragma for pragma in pragmas if 'begin' in get_pragma_parameters(pragma)]
        section_stops = [pragma for pragma in pragmas if 'end' in get_pragma_parameters(pragma)]
        assert len(section_starts) == len(section_stops)

        hoist_body = ()
        for start, stop in zip(section_starts, section_stops):
            reach = get_pragma_parameters(start).get('reach', None)
            if reach is not None:
                raise NotImplementedError

            # Extract the section to hoist
            section = MaskedTransformer(start=start, stop=stop, mapper={start: None}).visit(routine.body)

            # Append it to the group's body, wrapped in comments
            begin_comment = Comment('! Loki section-hoist group({}) - begin section'.format(group))
            end_comment = Comment('! Loki section-hoist group({}) - end section'.format(group))
            hoist_body += as_tuple(flatten([begin_comment, section, end_comment]))

            # Register start and end nodes for transformer mask
            starts += [stop]
            stops += [start]

            # Replace end pragma by comment
            comment = Comment('! Loki section-hoist group({}) - section hoisted'.format(group))
            hoist_map[stop] = comment

        # Find the target for this hoisting group
        target = [pragma for pragma in pragmas if 'target' in get_pragma_parameters(pragma)]
        if not target:
            raise RuntimeError('No target found for section-hoist group {}'.format(group))
        if len(target) > 1:
            raise RuntimeError('Multiple targets given for section-hoist group {}'.format(group))
        target = target[0]

        # Insert target <-> hoisted sections into map
        hoist_map[target] = hoist_body

    if hoist_map:
        routine.body = MaskedTransformer(active=True, start=starts, stop=stops, mapper=hoist_map).visit(routine.body)

        num_sections = sum((len(group) - 1) // 2 for group in hoist_groups.values())
        info('%s: hoisted %d section(s) in %d group(s)', routine.name, num_sections, len(hoist_groups))
