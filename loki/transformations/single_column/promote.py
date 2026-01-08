# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import is_dimension_constant, Array
from loki.ir import nodes as ir, FindNodes, FindInlineCalls
from loki.tools import flatten, as_tuple

from loki.transformations.array_indexing import demote_variables, promote_variables
from loki.transformations.utilities import get_local_arrays, get_integer_variable


__all__ = ['SCCPromoteTransformation']


class SCCPromoteTransformation(Transformation):
    """
    A set of utilities to determine which local arrays can be safely demoted in a
    :any:`Subroutine` as part of a transformation pass.

    Unless the option `demote_local_arrays` is set to `False`, this transformation will demote
    local arrays that do not buffer values between vector loops. Specific arrays in individual
    routines can also be marked for preservation by assigning them to the `preserve_arrays` list
    in the :any:`SchedulerConfig`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def __init__(self, horizontal, block_dim, promote_local_arrays=True):
        self.horizontal = horizontal
        self.block_dim = block_dim

        self.promote_local_arrays = promote_local_arrays

    @classmethod
    def get_locals_to_promote(cls, routine, sections, horizontal):
        """
        Create a list of local temporary arrays after checking that
        demotion is safe.

        Demotion is considered safe if the temporary is only used
        within one coherent vector-section (see
        :any:`extract_vector_sections`).

        Local temporaries get demoted if they have:
        * Only one dimension, which is the ``horizontal``
        * Have the ``horizontal`` as the innermost dimension, with all
          other dimensions being declared constant parameters.

        """
        # Create a list of local temporary arrays to filter down
        candidates = get_local_arrays(routine, routine.spec)

        # # Only demote local arrays with the horizontal as fast dimension
        # candidates = [
        #     v for v in candidates if v.shape and
        #     v.shape[0] in horizontal.sizes
        # ]
        # # Also demote arrays whose remaning dimensions are known constants
        # candidates = [
        #     v for v in candidates
        #     if all(is_dimension_constant(d) for d in v.shape[1:])
        # ]

        # Create an index into all variable uses per vector-level section
        vars_per_section = {
            s: set(
                v.name.lower() for v in get_local_arrays(routine, s, unique=False)
            ) for s in sections
        }

        # Count in how many sections each temporary is used
        counts = {}
        for arr in candidates:
            counts[arr] = sum(
                1 if arr.name.lower() in v else 0
                for v in vars_per_section.values()
            )

        # Demote temporaries that are only used in one section or not at all
        to_promote = [k for k, v in counts.items() if v > 1]

        # # Get InlineCall args containing a horizontal array section
        # icalls = FindInlineCalls().visit(routine.body)
        # _params = flatten([call.parameters + as_tuple(call.kw_parameters.values()) for call in icalls])
        # _params = [p for p in _params if isinstance(p, Array)]

        # call_args = [
        #     p.clone(dimensions=None) for p in _params
        #     if any(s in (p.dimensions or p.shape) for s in horizontal.size_expressions)
        # ]

        # # Filter out variables that we will pass down the call tree
        # calls = FindNodes(ir.CallStatement).visit(routine.body)
        # call_args += flatten(call.arguments for call in calls)
        # call_args += flatten(list(dict(call.kwarguments).values()) for call in calls)
        # to_demote = [v for v in to_demote if v.name not in call_args]

        return set(to_promote)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCDemote utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']
        item = kwargs.get('item', None)

        if role == 'kernel':
            promote_locals = self.promote_local_arrays
            preserve_arrays = []
            if item:
                promote_locals = item.config.get('promote_locals', self.promote_local_arrays)
                preserve_arrays = item.config.get('preserve_arrays', [])
                self.process_kernel(routine, promote_locals=promote_locals, preserve_arrays=preserve_arrays, item=item)

    def process_kernel(self, routine, promote_locals=True, preserve_arrays=None, item=None):
        """
        Applies the SCCDemote utilities to a "kernel" and demotes all suitable local arrays.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Find vector sections marked in the SCCDevectorTransformation
        sections = [
            s for s in FindNodes(ir.Section).visit(routine.body)
            if s.label == 'vector_section'
        ]

        # Extract the local variables to demote after we wrap the sections in vector loops.
        # We do this, because need the section blocks to determine which local arrays
        # may carry buffered values between them, so that we may not demote those!
        to_promote = self.get_locals_to_promote(routine, sections, self.horizontal)

        # Filter out arrays marked explicitly for preservation
        if preserve_arrays:
            to_promote = [v for v in to_promote if not v.name in preserve_arrays]

        block_loop_variable = item.trafo_data['LowerBlockLoop'].get('variable', None)
        block_loop_upper = item.trafo_data['LowerBlockLoop'].get('upper', None)

        block_loop_variable = get_integer_variable(routine, self.block_dim.index)
        block_loop_upper = get_integer_variable(routine, self.block_dim.size)

        print(f"[SCC PROMOTE] block_loop_variable: {block_loop_variable}")
        print(f"[SCC PROMOTE] block_loop_upper: {block_loop_upper}")

        # Demote all private local variables that do not buffer values between sections
        if promote_locals:
            variables = tuple(v.name for v in to_promote)
            print(f"promote_locals {routine} - variables: {variables}")
            if variables:
                promote_variables(
                    routine, variable_names=variables,
                    # dimensions=self.horizontal.sizes
                    pos=-1, # index=self.block_dim.index,
                    # size=self.block_dim.size
                    index=block_loop_variable, size=block_loop_upper,
                    ignore_index_undefined=True
                )

        variables_ = f", ".join(v for v in variables)
        print(f"variables_: {variables_}")
        routine.body.prepend(ir.Pragma(keyword='loki', content=f'unstructured-data create({variables_})'))
        routine.body.append(ir.Pragma(keyword='loki', content=f'exit unstructured-data delete({variables_})'))
