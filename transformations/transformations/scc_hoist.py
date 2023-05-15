# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki import(
          Transformation, CaseInsensitiveDict, as_tuple, BasicType,
          SubstituteExpressions, info, ir, Transformer
)
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

    @classmethod
    def hoist_temporary_column_arrays(cls, routine, call, horizontal, vertical, block_dim, directive):
        """
        Hoist temporary column arrays to the driver level. This
        includes allocating them as local arrays on the host and on
        the device via ``!$acc enter create``/ ``!$acc exit delete``
        directives.

        Note that this employs an interprocedural analysis pass
        (forward), and thus needs to be executed for the calling
        routine before any of the callees are processed.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        call : :any:`CallStatement`
            Call to subroutine from which we hoist the column arrays.
        horizontal: :any:`Dimension`
            The dimension object specifying the horizontal vector dimension
        vertical: :any:`Dimension`
            The dimension object specifying the vertical loop dimension
        block_dim : :any:`Dimension`
            Optional ``Dimension`` object to define the blocking dimension
            to use for hoisted column arrays if hoisting is enabled.
        directive : string or None
            Directives flavour to use for parallelism annotations; either
            ``'openacc'`` or ``None``.
        """

        if call.not_active or call.routine is BasicType.DEFERRED:
            raise RuntimeError(
                '[Loki] SingleColumnCoalescedTransform: Target kernel is not attached '
                'to call in driver routine.'
            )

        if not block_dim:
            raise RuntimeError(
                '[Loki] SingleColumnCoalescedTransform: No blocking dimension found '
                'for column hoisting.'
            )

        kernel = call.routine
        call_map = {}

        column_locals = SCCHoistTransformation.get_column_locals(kernel, vertical=vertical)
        arg_map = dict(call.arg_iter())
        arg_mapper = SubstituteExpressions(arg_map)

        # Create a driver-level buffer variable for all promoted column arrays
        # TODO: Note that this does not recurse into the kernels yet!
        block_var = SCCBaseTransformation.get_integer_variable(routine, block_dim.size)
        arg_dims = [v.shape + (block_var,) for v in column_locals]
        # Translate shape variables back to caller's namespace
        routine.variables += as_tuple(v.clone(dimensions=arg_mapper.visit(dims), scope=routine)
                                      for v, dims in zip(column_locals, arg_dims))

        # Add explicit OpenACC statements for creating device variables
        if directive == 'openacc' and column_locals:
            vnames = ', '.join(v.name for v in column_locals)
            pragma = ir.Pragma(keyword='acc', content=f'enter data create({vnames})')
            pragma_post = ir.Pragma(keyword='acc', content=f'exit data delete({vnames})')
            # Add comments around standalone pragmas to avoid false attachment
            routine.body.prepend((ir.Comment(''), pragma, ir.Comment('')))
            routine.body.append((ir.Comment(''), pragma_post, ir.Comment('')))

        # Add a block-indexed slice of each column variable to the call
        idx = SCCBaseTransformation.get_integer_variable(routine, block_dim.index)
        new_args = [v.clone(
            dimensions=as_tuple([sym.RangeIndex((None, None)) for _ in v.shape]) + (idx,),
            scope=routine
        ) for v in column_locals]
        new_call = call.clone(arguments=call.arguments + as_tuple(new_args))

        info(f'[Loki-SCC] Hoisted variables in call {routine.name} => {call.name}:'
             f'{[v.name for v in column_locals]}')

        # Find the iteration index variable for the specified horizontal
        v_index = SCCBaseTransformation.get_integer_variable(routine, name=horizontal.index)
        if v_index.name not in routine.variable_map:
            routine.variables += as_tuple(v_index)

        # Append new loop variable to call signature
        new_call._update(kwarguments=new_call.kwarguments + ((horizontal.index, v_index),))

        # Now create a vector loop around the kernel invocation
        pragma = ()
        if directive == 'openacc':
            pragma = ir.Pragma(keyword='acc', content='loop vector')
        v_start = arg_map[kernel.variable_map[horizontal.bounds[0]]]
        v_end = arg_map[kernel.variable_map[horizontal.bounds[1]]]
        bounds = sym.LoopRange((v_start, v_end))
        vector_loop = ir.Loop(
            variable=v_index, bounds=bounds, body=(new_call,), pragma=as_tuple(pragma)
        )
        call_map[call] = vector_loop

        routine.body = Transformer(call_map).visit(routine.body)

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
        targets = kwargs.get('targets', None)

        if role == 'kernel':
            self.process_kernel(routine)

        if role == 'driver':
            self.process_driver(routine, targets=targets)

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

    def process_driver(self, routine, targets=None):
        """
        Hoist temporary column arrays.

        Note that if ``hoist_column_arrays`` is set, the driver needs
        to be processed before any kernels are transformed. This is
        due to the use of an interprocedural analysis forward pass
        needed to collect the list of "column arrays".

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """

        # Apply hoisting of temporary "column arrays"
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if not call.name in targets:
                continue

            if self.hoist_column_arrays:
                self.hoist_temporary_column_arrays(routine, call, self.horizontal, self.vertical,
                                                   self.block_dim, self.directive)
