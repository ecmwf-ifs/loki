# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki import Transformation, CaseInsensitiveDict, as_tuple
from transformations.scc_base import SCCBaseTransformation

__all__ = ['SCCHoistTransformation']

class SCCHoistTransformation(Transformation):
    """
    A transformation to promote all local arrays with column dimensions to arguments.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    hoist_column_arrays : bool
        Flag to trigger the more aggressive "column array hoisting"
        optimization.
    """

    def __init__(self, horizontal, hoist_column_arrays):
        self.horizontal = horizontal
        self.hoist_column_arrays = hoist_column_arrays

    @classmethod
    def get_column_locals(cls, routine, vertical):
        """
        List of array variables that include a `vertical` dimension and
        thus need to be stored in shared memory.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in the vector loops should be removed.
        vertical: :any:`Dimension`
            The dimension object specifying the vertical dimension
        """
        variables = list(routine.variables)

        # Filter out purely local array variables
        argument_map = CaseInsensitiveDict({a.name: a for a in routine.arguments})
        variables = [v for v in variables if not v.name in argument_map]
        variables = [v for v in variables if isinstance(v, sym.Array)]

        variables = [v for v in variables if any(vertical.size in d for d in v.shape)]

        return variables

    @classmethod
    def add_loop_index_to_args(cls, v_index, routine):
        """
        Add loop index to routine arguments.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to modify.
        v_index : :any:`Scalar`
            The induction variable for the promoted horizontal loops.
        """

        new_v = v_index.clone(type=v_index.type.clone(intent='in'))
        # Remove original variable first, since we need to update declaration
        routine.variables = as_tuple(v for v in routine.variables if v != v_index)
        routine.arguments += as_tuple(new_v)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCHoist utilities to a :any:`Subroutine`.

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
            self.process_kernel(routine)

    def process_kernel(self, routine):
        """
        Applies the SCCHoist utilities to a "kernel" and promote all local arrays with column
        dimension to arguments.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # TODO: we only need this here because we cannot mark routines to skip at the scheduler level
        # Bail if routine is marked as sequential or have already been processed
        if SCCBaseTransformation.check_routine_pragmas(routine, self.directive):
            return

        # Find the iteration index variable for the specified horizontal
        v_index = SCCBaseTransformation.get_integer_variable(routine, name=self.horizontal.index)

        column_locals = self.get_column_locals(routine, vertical=self.vertical)
        promoted = [v.clone(type=v.type.clone(intent='INOUT')) for v in column_locals]
        routine.arguments += as_tuple(promoted)

        # Add loop index variable
        if v_index not in routine.arguments:
            self.add_loop_index_to_args(v_index, routine)
