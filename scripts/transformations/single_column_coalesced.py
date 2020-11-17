from loki import (
    Transformation, FindNodes, FindVariables, Transformer,
    SubstituteExpressions, CallStatement, Loop, Variable, Scalar,
    Array, LoopRange, RangeIndex, SymbolAttributes, Pragma, BasicType,
    CaseInsensitiveDict, as_tuple, pragmas_attached
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
    directive : string or None
        Directives flavour to use for parallelism annotations; either
        ``'openacc'`` or ``None``.
    """

    def __init__(self, horizontal, directive=None):
        self.horizontal = horizontal

        assert directive in [None, 'openacc']
        self.directive = directive

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


        # Demote all local variables
        self.demote_locals(routine)

        if self.directive == 'openacc':

            with pragmas_attached(routine, Loop):
                # Mark all remaining loops as seq (in place)
                for loop in FindNodes(Loop).visit(routine.body):
                    # Skip loops explicitly marked with `!$loki/claw nodep`
                    if loop.pragma and any('nodep' in p.content.lower() for p in as_tuple(loop.pragma)):
                        continue

                    loop._update(pragma=Pragma(keyword='acc', content='loop seq'))

            # Mark routine as `!$acc routine seq` to make it device-callable
            routine.body.prepend(Pragma(keyword='acc', content='routine seq'))

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

    def demote_locals(self, routine):
        """
        Demotes all local variables (not in routine.arguments).
        """

        # Establish the new dimensions and shapes first, before cloning the variables
        # The reason for this is that shapes of all variable instances are linked
        # via caching, meaning we can easily void the shape of an unprocessed variable.
        variables = list(routine.variables)
        variables += list(FindVariables(unique=False).visit(routine.body))

        # Filter out purely local array variables
        argument_map = CaseInsensitiveDict({a.name: a for a in routine.arguments})
        variables = [v for v in variables if isinstance(v, Array) and v.shape is not None]
        variables = [v for v in variables if not v.name in argument_map]
        # Record original array shapes
        shape_map = CaseInsensitiveDict({v.name: v.shape for v in variables})

        # Demote local variables
        vmap = {}
        for v in variables:
            old_shape = shape_map[v.name]
            new_shape = as_tuple(s for s in old_shape if s not in self.horizontal.size_expressions)

            if old_shape and old_shape[0] in self.horizontal.size_expressions:
                new_type = v.type.clone(shape=new_shape)
                if len(old_shape) > 1:
                    vmap[v] = v.clone(dimensions=v.dimensions[1:], type=new_type)
                else:
                    vmap[v] = Scalar(name=v.name, parent=v.parent, type=new_type, scope=routine)

        routine.body = SubstituteExpressions(vmap).visit(routine.body)
        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
