# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.ir import nodes as ir, FindNodes

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

        to_promote = candidates

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
            promote_variables(
                    routine, variable_names=variables,
                    pos=-1, index=block_index, size=block_size,
                    ignore_index_undefined=True
            )
