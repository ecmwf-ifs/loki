# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import Array
from loki.ir import nodes as ir, FindNodes

from loki.transformations.array_indexing import promote_variables
from loki.transformations.utilities import get_local_variables, get_integer_variable

__all__ = ['CPUPromoteTransformation']


class CPUPromoteTransformation(Transformation):
    """
    Promote local scalar variables that are assigned inside horizontal
    loops to arrays with the horizontal dimension, so that they can
    be vectorised correctly.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the horizontal data
        dimension and iteration space.
    promote_local_arrays : bool, optional
        Whether to promote local scalars. Default ``True``.
    """

    def __init__(self, horizontal, promote_local_arrays=True):
        self.horizontal = horizontal
        self.promote_local_arrays = promote_local_arrays

    @classmethod
    def get_locals_to_promote(cls, routine, horizontal):
        """
        Identify local scalar variables that are assigned inside
        horizontal loops and should be promoted to arrays.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to analyse.
        horizontal : :any:`Dimension`
            The horizontal dimension descriptor.

        Returns
        -------
        set
            Set of variables to promote.
        """
        # Create a list of local temporary variables
        candidates = get_local_variables(routine, routine.spec)

        # Filter those being arrays with horizontal dim as fast dimension
        candidates = [
            v for v in candidates if not isinstance(v, Array) or v.shape and
            v.shape[0] not in horizontal.sizes
        ]

        # Exclude loop index variables
        loops = FindNodes(ir.Loop).visit(routine.body)
        indices = {loop.variable.name.lower() for loop in loops}
        candidates = sorted(
            [v for v in candidates if v.name.lower() not in indices],
            key=lambda x: x.name
        )

        # Find variables assigned inside horizontal loops
        hor_loops = [
            loop for loop in loops
            if loop.variable.name.lower() in [_.lower() for _ in horizontal.indices]
        ]
        assignments = []
        for hor_loop in hor_loops:
            _assigns = FindNodes(ir.Assignment).visit(hor_loop.body)
            assignments.extend(_assigns)
        defined = {assign.lhs.name.lower() for assign in assignments}

        candidates = [v for v in candidates if v.name.lower() in defined]

        return set(candidates)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply scalar promotion to a :any:`Subroutine`.

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
        Promote local scalars that are assigned inside horizontal loops
        to arrays with the horizontal dimension.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        promote_locals : bool
            Whether to actually perform promotion.
        preserve_arrays : list, optional
            Array names to exclude from promotion.
        """
        to_promote = self.get_locals_to_promote(routine, self.horizontal)

        variable_map = routine.variable_map
        hor_index = None
        hor_size = None
        for index in self.horizontal.indices:
            if index.split('%', maxsplit=1)[0] in variable_map:
                hor_index = get_integer_variable(routine, index)
                break
        for _size in self.horizontal.sizes:
            if _size.split('%', maxsplit=1)[0] in variable_map:
                hor_size = get_integer_variable(routine, _size)

        if promote_locals and to_promote:
            variables = tuple(v.name for v in to_promote)
            promote_variables(routine, variable_names=variables,
                    pos=0, index=hor_index, size=hor_size)
