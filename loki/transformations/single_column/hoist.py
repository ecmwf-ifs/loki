# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki.ir import nodes as ir

from loki.transformations.hoist_variables import HoistVariablesTransformation
from loki.transformations.single_column.base import SCCBaseTransformation


__all__ = ['SCCHoistTemporaryArraysTransformation']


class SCCHoistTemporaryArraysTransformation(HoistVariablesTransformation):
    """
    **Specialisation** for the *Synthesis* part of the hoist variables
    transformation that uses automatic arrays in the driver layer to
    allocate hoisted temporaries.

    This flavour of the hoisting synthesis will add a blocking dimension
    to the allocation and add OpenACC directives to the driver routine
    to trigger device side-allocation of the hoisted temporaries.

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension
        to use for hoisted array arguments on the driver side.
    """

    def __init__(self, block_dim=None, **kwargs):
        self.block_dim = block_dim
        super().__init__(**kwargs)

    def driver_variable_declaration(self, routine, variables):
        """
        Adds driver-side declarations of full block-size arrays to
        pass to kernels. It also adds the OpenACC pragmas for
        driver-side allocation/deallocation.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        variables : tuple of :any:`Variable`
            The array to be declared, allocated and de-allocated.
        """
        if not self.block_dim:
            raise RuntimeError(
                '[Loki] SingleColumnCoalescedTransform: No blocking dimension found '
                'for array argument hoisting.'
            )

        block_var = SCCBaseTransformation.get_integer_variable(routine, self.block_dim.size)
        routine.variables += tuple(
            v.clone(
                dimensions=v.dimensions + (block_var,),
                type=v.type.clone(shape=v.shape + (block_var,))
            ) for v in variables
        )

        # Add explicit device-side allocations/deallocations for hoisted temporaries
        vnames = ', '.join(v.name for v in variables)
        pragma = ir.Pragma(keyword='acc', content=f'enter data create({vnames})')
        pragma_post = ir.Pragma(keyword='acc', content=f'exit data delete({vnames})')

        # Add comments around standalone pragmas to avoid false attachment
        routine.body.prepend((ir.Comment(''), pragma, ir.Comment('')))
        routine.body.append((ir.Comment(''), pragma_post, ir.Comment('')))

    def driver_call_argument_remapping(self, routine, call, variables):
        """
        Adds hoisted sub-arrays to the kernel call from a driver routine.

        This assumes that the hoisted temporaries have been allocated with
        a blocking dimension and are device-resident. The remapping will then
        add the block-index as the last index to each passed array argument.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to add the variable declaration to.
        call : :any:`CallStatement`
            Call object to which hoisted arrays will be added.
        variables : tuple of :any:`Variable`
            The array to be declared, allocated and de-allocated.
        """
        if not self.block_dim:
            raise RuntimeError(
                '[Loki] SingleColumnCoalescedTransform: No blocking dimension found '
                'for array argument hoisting.'
            )
        idx_var = SCCBaseTransformation.get_integer_variable(routine, self.block_dim.index)
        if self.as_kwarguments:
            new_kwargs = tuple(
                (a.name, v.clone(dimensions=tuple(sym.RangeIndex((None, None))
                for _ in v.dimensions) + (idx_var,))) for (a, v) in variables
            )
            kwarguments = call.kwarguments if call.kwarguments is not None else ()
            return call.clone(kwarguments=kwarguments + new_kwargs)
        new_args = tuple(
            v.clone(dimensions=tuple(sym.RangeIndex((None, None)) for _ in v.dimensions) + (idx_var,))
            for v in variables
        )
        return call.clone(arguments=call.arguments + new_args)
