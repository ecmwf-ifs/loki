"""
Collection of utility routines that provide loop transformations.

"""
from collections import defaultdict
import operator as op
import numpy as np
from pymbolic.primitives import Variable

from loki.expression import (
    symbols as sym, SubstituteExpressions, FindVariables,
    accumulate_polynomial_terms, simplify, is_constant, symbolic_op)
from loki.frontend.fparser import parse_fparser_expression
from loki.ir import Loop, WhileLoop, Conditional, Comment, Pragma, CallStatement
from loki.logging import info
from loki.tools import (
    is_loki_pragma, get_pragma_parameters, flatten, as_tuple, CaseInsensitiveDict)
from loki.visitors import FindNodes, Transformer, MaskedTransformer

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
    return as_tuple(loop_variables), as_tuple(loop_ranges), as_tuple(loop_bodies)


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
        loop_variables, loop_ranges, _ = get_loop_components(loops)
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


def loop_fusion(routine):
    """
    Search for loops annotated with the `loki loop-fusion` pragma and attempt
    to fuse them into a single loop.
    """
    # Extract all annotated loops and sort them into fusion groups
    fusion_groups = defaultdict(list)
    for loop in FindNodes(Loop).visit(routine.body):
        if is_loki_pragma(loop.pragma, starts_with='loop-fusion'):
            parameters = get_pragma_parameters(loop.pragma, starts_with='loop-fusion')
            group = parameters.get('group', 'default')
            fusion_groups[group] += [loop]

    if not fusion_groups:
        return

    def _pragma_ranges_to_loop_ranges(parameters):
        """
        Convert loop ranges given in the pragma parameters from string to a tuple of `LoopRange`
        objects.
        """
        if 'range' not in parameters:
            return None
        ranges = []
        for item in parameters['range'].split(','):
            bounds = [parse_fparser_expression(bound, scope=routine.scope) for bound in item.split(':')]
            ranges += [sym.LoopRange(as_tuple(bounds))]

        return as_tuple(ranges)

    # Merge loops in each group and put them in the position of the group's first loop
    loop_map = {}
    for group, loop_list in fusion_groups.items():
        parameters = [get_pragma_parameters(loop.pragma, starts_with='loop-fusion') for loop in loop_list]

        # First, determine the collapse depth and extract user-annotated loop ranges from pragmas
        collapse = [param.get('collapse', None) for param in parameters]
        if collapse != [collapse[0]] * len(collapse):
            raise RuntimeError('Conflicting collapse values in group "{}"'.format(group))
        collapse = int(collapse[0]) if collapse[0] is not None else 1

        pragma_ranges = [_pragma_ranges_to_loop_ranges(param) for param in parameters]

        # If we have a pragma somewhere with an explicit loop range, we use that for the fused loop
        range_set = {r for r in pragma_ranges if r is not None}
        if len(range_set) not in (0, 1):
            raise RuntimeError('Pragma-specified loop ranges in group "{}" do not match'.format(group))

        fusion_ranges = None
        if range_set:
            fusion_ranges = range_set.pop()

        # Next, extract loop ranges for all loops in group and convert to iteration space
        # polyhedrons for easier alignment
        loop_variables, loop_ranges, loop_bodies = zip(*[get_loop_components(get_nested_loops(loop, collapse))
                                                         for loop in loop_list])
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
        for variables, ranges, bodies, p in zip(loop_variables, loop_ranges, loop_bodies, iteration_spaces):
            # TODO: This throws away anything that is not in the inner-most loop body.
            body = bodies[-1]

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

        loop_map[loop_list[0]] = (Comment('! Loki transformation loop-fusion group({})'.format(group)),
                                  fusion_loop)
        loop_map.update({loop: None for loop in loop_list[1:]})

    # Apply transformation
    routine.body = Transformer(loop_map).visit(routine.body)
    info('%s: fused %d loops in %d groups.', routine.name,
         sum(len(loop_list) for loop_list in fusion_groups.values()), len(fusion_groups))


def loop_fission(routine):
    """
    Search for `loki loop-fission` pragmas inside loops and attempt to split them into
    multiple loops.
    """
    comment = Comment('! Loki transformation loop-fission')
    variable_map = routine.variable_map
    loop_map = {}
    promotion_vars_dims = CaseInsensitiveDict()

    # Run over all loops and look for pragmas in each loop's body instead of searching for pragmas
    # directly as we would otherwise need a second pass to find the loop that pragma refers to.
    # Moreover, this makes it easier to apply multiple fission steps for a loop at the same time.
    for loop in FindNodes(Loop).visit(routine.body):

        # Look for all loop-fission pragmas inside this routine's body
        # Note that we need to store (node, pragma) pairs as pragmas can also be attached to nodes
        pragmas = {}
        for ch in loop.body:
            if isinstance(ch, Pragma) and is_loki_pragma(ch, starts_with='loop-fission'):
                pragmas[ch] = ch
            elif (hasattr(ch, 'pragma') and ch.pragma is not None and
                  is_loki_pragma(ch.pragma, starts_with='loop-fission')):
                pragmas[ch] = ch.pragma

        if not pragmas:
            continue

        # Create the bodies for the split loops
        bodies = []
        current_body = []
        for ch in loop.body:
            if ch in pragmas:
                bodies += [current_body]
                if isinstance(ch, Pragma):
                    current_body = []
                else:
                    current_body = [ch.clone(pragma=None)]
            else:
                current_body += [ch]
        if current_body:
            bodies += [current_body]

        # Promote variables given in promotion list
        loop_length = simplify(loop.bounds.stop - loop.bounds.start + sym.IntLiteral(1))
        promote_vars = {var.strip().lower() for pragma in pragmas.values()
                        for var in get_pragma_parameters(pragma).get('promote', '').split(',') if var}
        for var_name in promote_vars:
            var = variable_map[var_name]

            # Promoted variable shape
            if var_name in promotion_vars_dims:
                # We have already marked this variable for promotion: let's make sure the added
                # dimension is large enough for this loop
                shape = promotion_vars_dims[var_name]
                if symbolic_op(shape[-1], op.lt, loop_length):
                    shape = shape[:-1] + (loop_length,)
            else:
                shape = getattr(var, 'shape', ()) + (loop_length,)
            promotion_vars_dims[var_name] = shape

            # Insert loop variable as last subscript dimension
            var_maps = []
            for body in bodies:
                var_uses = [v for v in FindVariables().visit(body) if v.name.lower() == var_name]
                var_dimensions = [getattr(v, 'dimensions', sym.ArraySubscript(())).index + (loop.variable,)
                                  for v in var_uses]
                var_maps += [{v: v.clone(dimensions=dim) for v, dim in zip(var_uses, var_dimensions)}]
            bodies = [SubstituteExpressions(var_map).visit(body) if var_map else body
                      for var_map, body in zip(var_maps, bodies)]

        # Finally, create the new loops
        loop_map[loop] = [(comment, Loop(variable=loop.variable, bounds=loop.bounds, body=body))
                          for body in bodies]

    # Apply loop maps and shape promotion
    if loop_map:
        routine.body = Transformer(loop_map).visit(routine.body)
        info('%s: split %d loop(s) into %d loops.', routine.name, len(loop_map),
             sum(len(loop_list) for loop_list in loop_map.values()))
    if promotion_vars_dims:
        var_map = {}
        for var_name, shape in promotion_vars_dims.items():
            var = variable_map[var_name]
            var_map[var] = var.clone(type=var.type.clone(shape=shape), dimensions=shape)
        routine.spec = SubstituteExpressions(var_map).visit(routine.spec)
        info('%s: promoted variable(s): %s', routine.name, ', '.join(promotion_vars_dims.keys()))


def pragma_replacement_mapper(pragma, node, replacement=None):
    """
    Utility function that produces a mapper that can be used with a :class:`Transformer` to
    replace a pragma.

    :param :class:`ir.Pragma` pragma: the pragma to be replaced.
    :param :class:`ir.Node` node: the pragma node or a node in which the pragma is inlined.
    :param replacement: the replacement for the pragma. Can be `None` to simply remove it.
    """
    if isinstance(node, Pragma):
        # Given node is the pragma
        assert node == pragma
        return {node: replacement}

    # Pragma is inlined in the given node
    assert hasattr(node, 'pragma') and pragma in node.pragma
    pragmas = [p for p in node.pragma if p != pragma]

    if replacement is not None:
        return {node: as_tuple(flatten([replacement, node.clone(pragma=pragmas)]))}
    return {node: node.clone(pragma=pragmas)}


def section_hoist(routine):
    """
    Hoist one or multiple code sections and insert them at a specified target location.
    """
    hoist_groups = defaultdict(list)

    # We need to look for Pragma nodes as well as inlined pragmas to find all section-hoist directives
    node_types = (Loop, WhileLoop, CallStatement)
    for node in FindNodes((Pragma,) + node_types).visit(routine.body):
        if isinstance(node, Pragma):
            pragmas = as_tuple(node)
        else:
            pragmas = as_tuple(node.pragma)

        for pragma in pragmas:
            if is_loki_pragma(pragma, starts_with='section-hoist'):
                parameters = get_pragma_parameters(pragma, starts_with='loop-fusion')
                group = parameters.get('group', 'default')
                hoist_groups[group] += [(pragma, node)]

    hoist_map = {}
    starts, stops = [], []
    for group, pragma_node_pairs in hoist_groups.items():

        # Extract sections
        section_starts = [(pragma, node) for pragma, node in pragma_node_pairs
                          if 'begin' in get_pragma_parameters(pragma)]
        section_ends = [(pragma, node) for pragma, node in pragma_node_pairs
                        if 'end' in get_pragma_parameters(pragma)]
        assert len(section_starts) == len(section_ends)

        hoist_body = ()
        for (start_pragma, start_node), (end_pragma, end_node) in zip(section_starts, section_ends):
            reach = get_pragma_parameters(start_pragma).get('reach', None)
            if reach is not None:
                raise NotImplementedError

            # For some reason this always produces a Section object which we avoid here
            # by passing routine.body.body
            mapper = pragma_replacement_mapper(start_pragma, start_node)
            section = MaskedTransformer(start=start_node, stop=end_node, mapper=mapper).visit(routine.body.body)
            start_comment = Comment('! Loki section-hoist group({}) - begin section'.format(group))
            end_comment = Comment('! Loki section-hoist group({}) - end section'.format(group))
            hoist_body += as_tuple(flatten([start_comment, section, end_comment]))

            # Register start and end nodes for transformer mask
            starts += [end_node]
            stops += [start_node]

            # Replace end pragma by comment
            comment = Comment('! Loki section-hoist group({}) - section hoisted'.format(group))
            hoist_map.update(pragma_replacement_mapper(end_pragma, end_node, comment))

        # Find the target for this hoisting group
        target = [(pragma, node) for pragma, node in pragma_node_pairs
                  if 'target' in get_pragma_parameters(pragma)]
        if not target:
            raise RuntimeError('No target found for section-hoist group {}'.format(group))
        if len(target) > 1:
            raise RuntimeError('Multiple targets given for section-hoist group {}'.format(group))
        target_pragma, target_node = target[0]

        # Insert target <-> hoisted sections into map
        hoist_map.update(pragma_replacement_mapper(target_pragma, target_node, hoist_body))

    routine.body = MaskedTransformer(active=True, start=starts, stop=stops, mapper=hoist_map).visit(routine.body)
