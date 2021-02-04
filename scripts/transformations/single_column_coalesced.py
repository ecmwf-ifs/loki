from loki import (
    Transformation, FindNodes, Transformer, CallStatement, Loop,
    Variable, LoopRange, SymbolAttributes, BasicType, CaseInsensitiveDict,
    as_tuple
)


__all__ = ['SingleColumnCoalescedTransformation']


def get_integer_variable(routine, name):
    """
    Find a local variable in the routine, or create an integer-typed one.
    """
    if name in routine.variable_map:
        v_index = routine.variable_map[name]
    else:
        dtype = SymbolAttributes(BasicType.INTEGER)
        v_index = Variable(name=name, type=dtype, scope=routine)
    return v_index


class SingleColumnCoalescedTransformation(Transformation):
    """
    Single Column Coalesced: Direct CPU-to-GPU trnasformation for
    block-indexed gridpoint routines.

    This transformation will remove individiual CPU-style
    vectorization loops from "kernel" routines and re-insert the
    a single horizontal vector loop in the "driver" routine.

    Unlike the CLAW-targetting SCA extraction, this will leave the
    block-based array passing structure in place, but pass a
    thread-local array index into any "kernel" routines. The
    block-based argument passing should map well to coalesced memory
    accesses on GPUs.

    Note, this requires preprocessing with the
    `DerivedTypeArgumentsTransformation`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in existing
        code to define the horizontal data dimension and iteration space.
    """

    def __init__(self, horizontal):
        self.horizontal = horizontal

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

        role = kwargs.get('role')
        targets = kwargs.get('targets', None)

        if role == 'driver':
            self.process_driver(routine, targets=targets)

        if role == 'kernel':
            self.process_kernel(routine)

    def process_kernel(self, routine):
        """
        Remove all vector loops that match the stored ``dimension``
        and promote the index variable to an argument, leaving fully
        dimensioned array arguments intact.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Remove all vector loops over the specified dimension
        loop_map = {}
        for loop in FindNodes(Loop).visit(routine.body):
            if loop.variable == self.horizontal.index:
                loop_map[loop] = loop.body
        routine.body = Transformer(loop_map).visit(routine.body)

        # Promote the index variable to an argument (prepend)
        v_index = get_integer_variable(routine, name=self.horizontal.index)
        if v_index not in routine.arguments:
            routine.arguments += as_tuple(v_index)

    def process_driver(self, routine, targets=None):
        """
        Remove insert the parallel iteration loop around calls to
        kernel routines and insert the parallelisation boilerplate.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list of strings
            Names of all kernel routines that are to be considered "active"
            in this call tree and should thus be processed accordingly.
        """
        targets = as_tuple(str(t).lower() for t in targets)

        # Find the iteration index variable for the horizontal
        # dimension and add to local variables
        v_index = get_integer_variable(routine, name=self.horizontal.index)
        routine.variables += as_tuple(v_index)

        call_map = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in targets:
                if call.context is None or not call.context.active:
                    raise RuntimeError(
                        "[Loki-SCC] Need call context for processing call to {}".format(call.name)
                    )

                # Append iteration variable to the subroutine call via keyword args
                kwarguments = dict(call.kwarguments)
                kwarguments[v_index] = v_index
                new_call = call.clone(kwarguments=as_tuple(kwarguments.items()))

                # Find the local variables corresponding to the kernels dimension
                arg_map = CaseInsensitiveDict([(p.name, a) for p, a in call.context.arg_iter(call)])
                v_start = arg_map[self.horizontal.bounds[0]]
                v_end = arg_map[self.horizontal.bounds[1]]

                # Create a loop over the specified iteration dimemsion
                bounds = LoopRange((v_start, v_end))
                parallel_loop = Loop(variable=v_index, bounds=bounds, body=[new_call])

                # Replace the call with a parallel loop over the iteration dimemsion
                call_map[call] = parallel_loop
        routine.body = Transformer(call_map).visit(routine.body)
