# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import is_dimension_constant, Array
from loki.ir import nodes as ir, FindNodes, FindInlineCalls
from loki.tools import flatten, as_tuple, OrderedSet

# from loki.transformations.array_indexing import demote_variables
from loki.transformations.array_indexing import promote_variables
from loki.transformations.utilities import get_local_arrays, get_integer_variable

__all__ = ['SCCPromoteTransformation']


class SCCPromoteTransformation(Transformation):
    """
    """

    def __init__(self, block_dim, promote_local_arrays=True):
        self.block_dim = block_dim

        self.promote_local_arrays = promote_local_arrays

    @classmethod
    def get_locals_to_promote(cls, routine, sections, block_dim):
        """
        """
        # Create a list of local temporary arrays to filter down
        candidates = get_local_arrays(routine, routine.spec)

        # # Only  local arrays with the horizontal as fast dimension
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
            s: OrderedSet(
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
        # to_promote = [k for k, v in counts.items() if v > 1]
        to_promote = candidates

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
        # to_promote = [v for v in to_promote if v.name not in call_args]

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
                # preserve_arrays = item.config.get('preserve_arrays', [])
                preseve_arrays = []
            self.process_kernel(routine, promote_locals=promote_locals, preserve_arrays=preserve_arrays)

    def process_kernel(self, routine, promote_locals=True, preserve_arrays=None):
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
            if s.label == 'block_section'
        ]

        # Extract the local variables to demote after we wrap the sections in vector loops.
        # We do this, because need the section blocks to determine which local arrays
        # may carry buffered values between them, so that we may not demote those!
        to_promote = self.get_locals_to_promote(routine, sections, self.block_dim)

        variable_map = routine.variable_map
        block_index = None
        block_size = None
        for index in self.block_dim.indices:
            if index.split('%', maxsplit=1)[0] in variable_map:
                block_index = get_integer_variable(routine, index)
                break
        for _size in self.block_dim.sizes:
            if _size.split('%', maxsplit=1)[0] in variable_map:
                block_size = get_integer_variable(routine, _size)


        # Demote all private local variables that do not buffer values between sections
        if promote_locals:
            variables = tuple(v.name for v in to_promote)
            # if variables:
            #     print(f"[PROMOTE ROUTINE {routine}]: {variables}")
            variables = tuple(v.name for v in to_promote)
            promote_variables(
                    routine, variable_names=variables,
                    pos=-1, index=block_index, size=block_size,
                    ignore_index_undefined=True
            )
            # if variables:
            #     demote_variables(
            #         routine, variable_names=variables,
            #         dimensions=self.horizontal.sizes
            #     )
