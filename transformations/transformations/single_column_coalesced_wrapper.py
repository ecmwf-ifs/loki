# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import (
     Transformation, resolve_associates, ir, NestedTransformer, FindNodes, demote_variables,
     Transformer, pragmas_attached, flatten, FindScopes, as_tuple
)
from transformations.single_column_coalesced_vector import (
     SCCDevectorTransformation, SCCRevectorTransformation, SCCDemoteTransformation
)
from transformations.single_column_coalesced import (
     SCCBaseTransformation, SCCAnnotateTransformation, SCCHoistTransformation
)

__all__ = ['SingleColumnCoalescedTransformation']


class SingleColumnCoalescedTransformation(Transformation):
    """
    Single Column Coalesced: Direct CPU-to-GPU transformation for
    block-indexed gridpoint routines.

    This transformation will remove individiual CPU-style
    vectorization loops from "kernel" routines and either either
    re-insert the vector loop at the highest possible level (without
    interfering with subroutine calls), or completely strip it and
    promote the index variable to the driver if
    ``hoist_column_arrays`` is set.

    Unlike the CLAW-targetting SCA extraction, this will leave the
    block-based array passing structure in place, but pass a
    thread-local array index into any "kernel" routines. The
    block-based argument passing should map well to coalesced memory
    accesses on GPUs.

    Note, this requires preprocessing with the
    :class:`DerivedTypeArgumentsTransformation`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    vertical : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the vertical dimension, as needed to decide array privatization.
    block_dim : :any:`Dimension`
        Optional ``Dimension`` object to define the blocking dimension
        to use for hoisted column arrays if hoisting is enabled.
    directive : string or None
        Directives flavour to use for parallelism annotations; either
        ``'openacc'`` or ``None``.
    hoist_column_arrays : bool
        Flag to trigger the more aggressive "column array hoisting"
        optimization.
    """

    def __init__(self, horizontal, vertical=None, block_dim=None, directive=None,
                 demote_local_arrays=True, hoist_column_arrays=True):
        self.horizontal = horizontal
        self.vertical = vertical
        self.block_dim = block_dim

        assert directive in [None, 'openacc']
        self.directive = directive

        self.demote_local_arrays = demote_local_arrays
        self.hoist_column_arrays = hoist_column_arrays

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply transformation to convert a :any:`Subroutine` to SCC format.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; either
            ``"driver"`` or ``"kernel"``
        targets : list of strings
            Names of all kernel routines that are to be considered "active"
            in this call tree and should thus be processed accordingly.
        """

        role = kwargs['role']
        item = kwargs.get('item', None)
        targets = kwargs.get('targets', None)

        if role == 'driver':
            self.process_driver(routine, targets=targets, item=item)

        if role == 'kernel':
            demote_locals = self.demote_local_arrays
            if item:
                demote_locals = item.config.get('demote_locals', self.demote_local_arrays)
            self.process_kernel(routine, demote_locals=demote_locals)

    def process_kernel(self, routine, demote_locals=True):
        """
        Applies the SCC loop layout transformation to a "kernel"
        subroutine. This will primarily strip the innermost vector
        loops and either re-insert the vector loop at the highest
        possible level (without interfering with subroutine calls),
        or completely strip it and promote the index variable to the
        driver if ``hoist_column_arrays`` is set.

        In both cases argument arrays are left fully dimensioned,
        allowing us to use them in recursive subroutine invocations.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Bail if routine is marked as sequential or routine has already been processed
        if SCCBaseTransformation.check_routine_pragmas(routine, self.directive):
            return

        # check for horizontal loop bounds in subroutine symbol table
        SCCBaseTransformation.check_horizontal_var(routine, self.horizontal)

        # Find the iteration index variable for the specified horizontal
        v_index = SCCBaseTransformation.get_integer_variable(routine, name=self.horizontal.index)

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        resolve_associates(routine)

        # Resolve WHERE clauses
        SCCBaseTransformation.resolve_masked_stmts(routine, loop_variable=v_index)

        # Resolve vector notation, eg. VARIABLE(KIDIA:KFDIA)
        SCCBaseTransformation.resolve_vector_dimension(routine, loop_variable=v_index, bounds=self.horizontal.bounds)

        # Remove all vector loops over the specified dimension
        SCCDevectorTransformation.kernel_remove_vector_loops(routine, self.horizontal)

        # Replace sections with marked Section node
        section_mapper = {s: ir.Section(body=s, label='vector_section')
                         for s in SCCDevectorTransformation.extract_vector_sections(routine.body.body, self.horizontal)}
        routine.body = NestedTransformer(section_mapper).visit(routine.body)

        # Find vector sections marked in the SCCDevectorTransformation
        sections = [s for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section']

        # Extract the local variables to demote after we wrap the sections in vector loops.
        # We do this, because need the section blocks to determine which local arrays
        # may carry buffered values between them, so that we may not demote those!
        to_demote = SCCDemoteTransformation.kernel_get_locals_to_demote(routine, sections, self.horizontal)
        # Demote all private local variables that do not buffer values between sections
        if demote_locals:
            variables = tuple(v.name for v in to_demote)
            if variables:
                demote_variables(routine, variable_names=variables, dimensions=self.horizontal.size)

        if not self.hoist_column_arrays:
            # Promote vector loops to be the outermost loop dimension in the kernel
            mapper = {s.body: SCCRevectorTransformation.wrap_vector_section(s.body, routine, self.horizontal)
                              for s in FindNodes(ir.Section).visit(routine.body)
                              if s.label == 'vector_section'}
            routine.body = NestedTransformer(mapper).visit(routine.body)

        # Remove section wrappers
        section_mapper = {s: s.body for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section'}
        if section_mapper:
            routine.body = Transformer(section_mapper).visit(routine.body)

        if self.hoist_column_arrays:
            # Promote all local arrays with column dimension to arguments
            # TODO: Should really delete and re-insert in spec, to prevent
            # issues with shared declarations.
            column_locals = SCCHoistTransformation.get_column_locals(routine, vertical=self.vertical)
            promoted = [v.clone(type=v.type.clone(intent='INOUT')) for v in column_locals]
            routine.arguments += as_tuple(promoted)

            # Add loop index variable
            if v_index not in routine.arguments:
                SCCHoistTransformation.add_loop_index_to_args(v_index, routine)

        if self.directive == 'openacc':
            SCCAnnotateTransformation.insert_annotations(routine, self.horizontal, self.vertical,
                                                         self.hoist_column_arrays)

    def process_driver(self, routine, targets=None, item=None):
        """
        Process the "driver" routine by inserting the other level
        parallel loops, and optionally hoisting temporary column
        arrays.

        Note that if ``hoist_column_arrays`` is set, the driver needs
        to be processed before any kernels are trnasformed. This is
        due to the use of an interprocedural analysis forward pass
        needed to collect the list of "column arrays".

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        item : :any:`Item`
            Scheduler work item corresponding to routine.
        """

        # Resolve associates, since the PGI compiler cannot deal with
        # implicit derived type component offload by calling device
        # routines.
        resolve_associates(routine)

        column_locals = []
        if item:
            item.trafo_data['SCCHoistTransformation'] = {'column_locals': []}

        # Apply hoisting of temporary "column arrays"
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if not call.name in targets:
                continue

            if self.hoist_column_arrays:
                SCCHoistTransformation.hoist_temporary_column_arrays(routine, call, self.horizontal,
                                                                     self.vertical, self.block_dim,
                                                                     item=item)
                # Get list of hoisted column locals
                if item:
                    column_locals = item.trafo_data['SCCHoistTransformation'].get('column_locals', None)
                if self.directive == 'openacc':
                    SCCAnnotateTransformation.device_alloc_column_locals(routine, column_locals)

        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):

            for call in FindNodes(ir.CallStatement).visit(routine.body):
                if not call.name in targets:
                    continue

                # Find the driver loop by checking the call's heritage
                ancestors = flatten(FindScopes(call).visit(routine.body))
                loops = [a for a in ancestors if isinstance(a, ir.Loop)]
                if not loops:
                    # Skip if there are no driver loops
                    continue
                driver_loop = loops[0]
                kernel_loop = [l for l in loops if l.variable == self.horizontal.index]
                if kernel_loop:
                    kernel_loop = kernel_loop[0]

                assert not driver_loop == kernel_loop

                # Mark driver loop as "gang parallel".
                SCCAnnotateTransformation.annotate_driver(self.directive, driver_loop, kernel_loop,
                                                          self.block_dim, column_locals)
