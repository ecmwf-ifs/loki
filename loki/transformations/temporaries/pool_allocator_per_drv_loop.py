# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Pool allocator variant that operates per driver loop.

This module provides :any:`TemporariesPoolAllocatorPerDrvLoopTransformation`,
a subclass of :any:`TemporariesPoolAllocatorTransformation` that creates
independent pool allocator infrastructure for each driver loop in a routine.

This is required by the small-kernels pipeline where multiple independent
driver loops may exist in a single routine, each calling different kernels
with different temporary allocation requirements.
"""

from loki.expression import (
    IntLiteral, InlineCall, Literal, Variable, Sum, Product,
    Cast
)
from loki.ir import (
    FindNodes, Transformer, Assignment,
    Loop, pragmas_attached
)
from loki.logging import warning, debug
from loki.tools import as_tuple

from loki.transformations.temporaries.pool_allocator import TemporariesPoolAllocatorTransformation
from loki.transformations.utilities import find_driver_loops


__all__ = ['TemporariesPoolAllocatorPerDrvLoopTransformation']


class TemporariesPoolAllocatorPerDrvLoopTransformation(TemporariesPoolAllocatorTransformation):
    """
    Pool allocator transformation that creates one stack per driver loop.

    Unlike the parent :any:`TemporariesPoolAllocatorTransformation` which
    creates a single pool allocator per driver routine, this variant
    identifies all driver loops (via :func:`find_driver_loops`) and injects
    independent stack pointer initialisation into each one.

    For "nested driver" kernels (those with ``LowerBlockIndex`` trafo_data),
    driver loops are also identified and processed, enabling pool allocation
    within kernel-level block loops created by
    ``SCCBlockSectionToLoopTransformation``.

    The aggregate stack size across all driver loops is used for the single
    ``ALLOCATE(ZSTACK(ISTSZ, nb))`` statement, wrapped in ``MAX(..., 1)``
    to prevent zero-sized allocations.

    Parameters
    ----------
    Same as :any:`TemporariesPoolAllocatorTransformation`.

    Notes
    -----
    ``check_bounds`` defaults to ``True``, providing runtime safety against
    zero-sized or overflowing stack allocations.
    """

    # Reuse the parent's trafo_data key so that successor stack sizes
    # stored by the parent's kernel-side logic are visible to this
    # transformation's driver-side logic.
    _key = 'TemporariesPoolAllocatorTransformation'

    def transform_subroutine(self, routine, **kwargs):
        """
        Dispatch pool allocator logic per driver loop.

        For kernels, temporaries are replaced with Cray-pointer stack
        allocations (delegated to the parent's
        :meth:`apply_pool_allocator_to_temporaries`). For drivers and
        "nested driver" kernels (those with ``LowerBlockIndex``
        trafo_data), each driver loop gets its own stack pointer
        initialisation.
        """
        role = kwargs['role']
        item = kwargs.get('item', None)
        ignore = item.ignore if item else ()
        targets = as_tuple(kwargs.get('targets', None))

        if item:
            item.trafo_data[self._key] = {'kind_imports': {}}

        self.import_c_sizeof(routine)
        self.import_real64(routine)

        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = as_tuple(sub_sgraph.successors(item)) if sub_sgraph is not None else ()

        with pragmas_attached(routine, Loop):
            # Determine driver loops: for drivers always, for kernels
            # only when they have LowerBlockIndex trafo_data (nested drivers)
            if role == 'driver' or (item and 'LowerBlockIndex' in item.trafo_data):
                driver_loops = find_driver_loops(section=routine.body, targets=targets)
            else:
                driver_loops = []

            if role == 'kernel':
                stack_size = self.apply_pool_allocator_to_temporaries(routine, item=item)
                if item:
                    stack_size = self._determine_stack_size(
                        routine, successors, stack_size, item=item
                    )
                    item.trafo_data[self._key]['stack_size'] = stack_size
            elif item:
                stack_size = self._determine_stack_size(routine, successors, item=item)
                item.trafo_data[self._key]['stack_size'] = stack_size

            if item:
                self.import_allocation_types(routine, item)

            if driver_loops:
                self.add_driver_imports(routine)

                # Compute per-loop stack sizes and aggregate with MAX
                per_loop_sizes = [
                    self._determine_stack_size(
                        routine, successors, item=item, drv_loop=drv_loop
                    )
                    for drv_loop in driver_loops
                ]
                aggregate_stack_size = self._aggregate_stack_sizes(per_loop_sizes)

                # Inject pool allocator into each driver loop
                drv_loop_map = {}
                for drv_loop in driver_loops:
                    drv_loop_map[drv_loop] = self._create_pool_allocator_in_drv_loop(
                        routine, aggregate_stack_size, drv_loop
                    )

                if drv_loop_map:
                    routine.body = Transformer(drv_loop_map).visit(routine.body)

            # Inject stack arguments into all targeted calls in the routine
            self.inject_pool_allocator_into_calls(
                routine, targets, ignore, driver=(role == 'driver')
            )

    def get_block_index(self, routine, variable_map):
        """
        Resolve the block index, including ``local_``-prefixed variants.

        ``SCCBlockSectionToLoopTransformation`` creates block-dimension
        variables with a ``local_`` prefix. This override checks for both
        the original name and the prefixed form.
        """
        # Try standard resolution first
        block_index = super().get_block_index(routine, variable_map)
        if block_index is not None:
            return block_index

        # Try with local_ prefix
        local_index = f'local_{self.block_dim.index}'
        if local_index.lower() in {k.lower() for k in variable_map}:
            return variable_map.get(local_index, None)

        # Try local_ prefix on compound indices
        for index in self.block_dim.indices:
            local_name = f'local_{index.split("%", maxsplit=1)[0]}'
            if local_name.lower() in {k.lower() for k in variable_map}:
                return routine.resolve_typebound_var(
                    f'local_{index}', variable_map
                )

        return None

    @staticmethod
    def _aggregate_stack_sizes(sizes):
        """
        Combine per-loop stack-size expressions into a single aggregate.

        Unwraps nested ``MAX(...)`` calls, removes zero literals and
        duplicates, then returns ``Literal(0)`` (all zero), a single
        expression, or ``MAX(...)`` of all unique non-zero expressions.

        Parameters
        ----------
        sizes : list of :any:`Expression`
            Per-loop stack-size expressions.

        Returns
        -------
        :any:`Expression`
            The aggregated stack size (maximum across all inputs).
        """
        flat = []
        for s in sizes:
            if isinstance(s, InlineCall) and s.function == 'MAX':
                flat.extend(s.parameters)
            else:
                flat.append(s)

        seen = set()
        non_zero = []
        for s in flat:
            if isinstance(s, (Literal, IntLiteral)) and int(s) == 0:
                continue
            key = str(s)
            if key not in seen:
                seen.add(key)
                non_zero.append(s)

        if not non_zero:
            return Literal(0)
        if len(non_zero) == 1:
            return non_zero[0]
        return InlineCall(
            function=Variable(name='MAX'),
            parameters=as_tuple(non_zero),
            kw_parameters=()
        )

    def _create_pool_allocator_in_drv_loop(self, routine, stack_size, drv_loop):
        """
        Inject pool allocator stack pointer initialisation into a driver loop.

        This creates the ``ISTSZ``/``ZSTACK`` variables (on first call) and
        inserts ``YLSTACK_L = LOC(ZSTACK(1, block_index))`` and the
        corresponding end-pointer assignment into the loop body.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The driver routine.
        stack_size : :any:`Expression`
            The aggregate stack size expression.
        drv_loop : :any:`Loop`
            The driver loop to inject into.

        Returns
        -------
        :any:`Loop`
            The modified driver loop with stack pointer assignments.
        """
        stack_storage, stack_size_var = self._get_stack_storage_and_size_var(routine, stack_size)
        stack_ptr = self._get_stack_ptr(routine)
        stack_end = self._get_stack_end(routine)

        # Find position for stack pointer assignment
        assignments = FindNodes(Assignment).visit(drv_loop.body)
        block_index = self.get_block_index(routine, routine.variable_map)
        if block_index is None:
            warning(
                f'{self.__class__.__name__}: '
                f'Could not resolve block index in {routine.name}; '
                f'no stack pointer assignment inserted!'
            )
            return drv_loop

        # Build set of block-dim index names (including local_ prefix)
        block_indices = set()
        for idx in self.block_dim.indices:
            block_indices.add(idx.lower())
            block_indices.add(f'local_{idx}'.lower())
            if '%' in idx:
                block_indices.add(idx.split('%')[-1].lower())
                block_indices.add(f'local_{idx.split("%")[-1]}'.lower())

        if drv_loop.variable == block_index or str(drv_loop.variable).lower() == str(block_index).lower():
            # Block variable is the loop variable
            assign_pos = -1
        else:
            # Look for the exact assignment that defines the resolved block index
            assign_pos = None
            for assignment in assignments:
                if assignment.lhs == block_index or str(assignment.lhs).lower() == str(block_index).lower():
                    assert assignment in drv_loop.body
                    assign_pos = drv_loop.body.index(assignment)
                    break

            # Fall back to broad block-index matching only if the exact symbol
            # is not found directly in the top-level loop body.
            if assign_pos is None:
                for assignment in assignments:
                    if str(assignment.lhs).lower() in block_indices:
                        assert assignment in drv_loop.body
                        assign_pos = drv_loop.body.index(assignment)
                        break

            if assign_pos is None:
                warning(
                    f'{self.__class__.__name__}: '
                    f'Could not find a block dimension for loop with variable '
                    f'{drv_loop.variable} and bounds {drv_loop.bounds} in '
                    f'{routine.name}; no stack pointer assignment inserted!'
                )
                return drv_loop

        # Check for existing pointer assignment
        stack_ptr_name = f'{self.stack_local_var_name}_{self.stack_ptr_name}'
        if any(str(a.lhs).lower() == stack_ptr_name.lower() for a in assignments):
            debug(
                f'{self.__class__.__name__}: '
                f'Stack pointer already exists within loop with variable '
                f'{drv_loop.variable} in {routine.name}; skipping.'
            )
            return drv_loop

        # Build pointer assignment
        if self.cray_ptr_loc_rhs:
            ptr_assignment = Assignment(lhs=stack_ptr, rhs=IntLiteral(1))
        else:
            ptr_assignment = Assignment(
                lhs=stack_ptr, rhs=InlineCall(
                    function=Variable(name='LOC'),
                    parameters=(
                        stack_storage.clone(
                            dimensions=(Literal(1), block_index)
                        ),
                    ),
                    kw_parameters=None
                )
            )

        # Build stack end assignment
        new_assignments = (ptr_assignment,)
        if self.check_bounds:
            _kind = routine.imported_symbol_map.get('REAL64')
            if self.cray_ptr_loc_rhs:
                stack_incr = Assignment(
                    lhs=stack_end, rhs=Sum((stack_ptr, stack_size_var))
                )
            else:
                _real_size_bytes = Cast(name='REAL', expression=Literal(1), kind=_kind)
                _real_size_bytes = InlineCall(
                    Variable(name='C_SIZEOF'),
                    parameters=as_tuple(_real_size_bytes)
                )
                stack_incr = Assignment(
                    lhs=stack_end,
                    rhs=Sum((stack_ptr, Product((stack_size_var, _real_size_bytes))))
                )
            new_assignments += (stack_incr,)

        return drv_loop.clone(
            body=drv_loop.body[:assign_pos + 1] + new_assignments + drv_loop.body[assign_pos + 1:]
        )
