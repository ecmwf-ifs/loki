from loki import (
    Transformation, FindNodes, FindVariables, Transformer,
    SubstituteExpressions, CallStatement, Loop, Variable, Scalar,
    Array, LoopRange, RangeIndex, SymbolAttributes, Pragma, BasicType,
    CaseInsensitiveDict, as_tuple, pragmas_attached,
    JoinableStringList, FindScopes
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
                 hoist_column_arrays=True):
        self.horizontal = horizontal
        self.vertical = vertical
        self.block_dim = block_dim

        assert directive in [None, 'openacc']
        self.directive = directive
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

        role = kwargs.get('role')
        targets = kwargs.get('targets', None)

        if role == 'driver':
            self.process_driver(routine, targets=targets)

        if role == 'kernel':
            self.process_kernel(routine)

    def get_column_locals(self, routine):
        """
        List of array variables that include a `vertical` dimension and
        thus need to be stored in shared memory.
        """
        variables = list(routine.variables)

        # Filter out purely local array variables
        argument_map = CaseInsensitiveDict({a.name: a for a in routine.arguments})
        variables = [v for v in variables if not v.name in argument_map]
        variables = [v for v in variables if isinstance(v, Array)]

        variables = [v for v in variables if v.shape is not None]
        variables = [v for v in variables if any(self.vertical.size in d for d in v.shape)]

        return variables

    def process_kernel(self, routine):
        """
        Remove all vector loops that match the stored ``horizontal``
        and promote the index variable to an argument, leaving fully
        dimensioned array arguments intact.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """
        # Find the iteration index variable for the specified horizontal
        v_index = get_integer_variable(routine, name=self.horizontal.index)

        # Remove all vector loops over the specified dimension
        loop_map = {}
        for loop in FindNodes(Loop).visit(routine.body):
            if loop.variable == self.horizontal.index:
                loop_map[loop] = loop.body
        routine.body = Transformer(loop_map).visit(routine.body)

        # Demote all private local variables
        self.demote_private_locals(routine)

        if self.hoist_column_arrays:
            # Promote all local arrays with column dimension to arguments
            # TODO: Should really delete and re-insert in spec, to prevent
            # issues with shared declarations.
            column_locals = self.get_column_locals(routine)
            promoted = [v.clone(type=v.type.clone(intent='INOUT')) for v in column_locals]
            routine.arguments += as_tuple(promoted)

            # Add loop index variable
            if v_index not in routine.arguments:
                routine.arguments += as_tuple(v_index)

        if self.directive == 'openacc':

            with pragmas_attached(routine, Loop):
                # Mark all remaining loops as seq (in place)
                for loop in FindNodes(Loop).visit(routine.body):
                    # Skip loops explicitly marked with `!$loki/claw nodep`
                    if loop.pragma and any('nodep' in p.content.lower() for p in as_tuple(loop.pragma)):
                        continue

                    if self.directive == 'openacc':
                        loop._update(pragma=Pragma(keyword='acc', content='loop seq'))


        if not self.hoist_column_arrays:
            # If we're not hoisting column arrays, the kernel remains mapped
            # to the OpenACC "vector" level. For this, we insert an all-encompassing
            # vector loop around the kernel and privatize all remaining local arrays
            # (after domting the horizontal). This encompasses all arrays that are
            # do not have vertical dimension.

            # Add vector-level loops at the highest level in the kernel
            self.kernel_add_vector_loops(routine)

        if self.directive == 'openacc':
            if not self.hoist_column_arrays:
                # Mark routine as `!$acc routine seq` to make it device-callable
                routine.body.prepend(Pragma(keyword='acc', content='routine vector'))
            else:
                # Mark routine as `!$acc routine seq` to make it device-callable
                routine.body.prepend(Pragma(keyword='acc', content='routine seq'))

    def process_driver(self, routine, targets=None):
        """
        Process the "driver" routine by inserting the other level parallel loops,
        and optionally hoisting temporary column routines.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """
        with pragmas_attached(routine, Loop, attach_pragma_post=True):

            for call in FindNodes(CallStatement).visit(routine.body):
                if not call.name in targets:
                    continue

                # Find the driver loop by checking the call's heritage
                ancestors = flatten(FindScopes(call).visit(routine.body))
                loops = [a for a in ancestors if isinstance(a, Loop)]
                if not loops:
                    # Skip if there are no driver loops
                    continue
                loop = loops[0]

                # Mark driver loop as "gang parallel".
                if self.directive == 'openacc':
                    if loop.pragma is None:
                        loop._update(pragma=Pragma(keyword='acc', content='parallel loop gang'))
                        loop._update(pragma_post=Pragma(keyword='acc', content='end parallel loop'))

                # Apply hoisting of temporary "column arrays"
                if self.hoist_column_arrays:
                    self.hoist_temporary_column_arrays(routine, call)

    def hoist_temporary_column_arrays(self, routine, call):
        """
        Hoist temporary column arrays to the driver level. This includes allocating
        them as local arrays on the host and on the device via ``!$acc enter create``/
        ``!$acc exit delete`` directives.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """
        if call.context is None or not call.context.active:
            raise RuntimeError(
                '[Loki] SingleColumnCoalescedTransform: Target kernel is not attached '
                'to call in driver routine.'
            )

        if not self.block_dim:
            raise RuntimeError(
                '[Loki] SingleColumnCoalescedTransform: No blocking dimension found '
                'for column hoisting.'
            )

        kernel = call.context.routine
        call_map = {}

        column_locals = self.get_column_locals(kernel)
        arg_map = dict(call.context.arg_iter(call))
        arg_mapper = SubstituteExpressions(arg_map)

        # Create a driver-level buffer variable for all promoted column arrays
        # TODO: Note that this does not recurse into the kernels yet!
        block_var = get_integer_variable(routine, self.block_dim.size)
        arg_dims = [v.shape + (block_var,) for v in column_locals]
        # Translate shape variables back to caller's namespace
        routine.variables += as_tuple(v.clone(dimensions=arg_mapper.visit(dims), scope=routine)
                                      for v, dims in zip(column_locals, arg_dims))

        # Add explicit OpenACC statements for creating device variables
        def _pragma_string(items):
            return str(JoinableStringList(items, cont=' &\n!$acc &   ', sep=', ', width=72))

        if self.directive == 'openacc':
            vnames = _pragma_string(v.name for v in column_locals)
            pragma = Pragma(keyword='acc', content='enter data create({})'.format(vnames))
            pragma_post = Pragma(keyword='acc', content='exit data delete({})'.format(vnames))
            routine.body.prepend(pragma)
            routine.body.append(pragma_post)

        # Add a block-indexed slice of each column variable to the call
        idx = get_integer_variable(routine, self.block_dim.index)
        new_args = [v.clone(
            dimensions=as_tuple([RangeIndex((None, None)) for _ in v.shape]) + (idx,),
            scope=routine
        ) for v in column_locals]
        new_call = call.clone(arguments=call.arguments + as_tuple(new_args))

        # Find the iteration index variable for the specified horizontal
        v_index = get_integer_variable(routine, name=self.horizontal.index)
        if v_index.name not in routine.variable_map:
            routine.variables += as_tuple(v_index)

        # Append new loop variable to call signature
        new_call._update(kwarguments=new_call.kwarguments + ((self.horizontal.index, v_index),))

        # Now create a vector loop around the kerne invocation
        pragma = None
        if self.directive == 'openacc':
            pragma = Pragma(keyword='acc', content='loop vector')
        v_start = arg_map[kernel.variable_map[self.horizontal.bounds[0]]]
        v_end = arg_map[kernel.variable_map[self.horizontal.bounds[1]]]
        bounds = LoopRange((v_start, v_end))
        vector_loop = Loop(variable=v_index, bounds=bounds, body=[new_call], pragma=pragma)
        call_map[call] = vector_loop

        routine.body = Transformer(call_map).visit(routine.body)

    def demote_private_locals(self, routine):
        """
        Demotes all local variables that can be privatized at the `acc loop vector`
        level.

        Array variables whose dimensions include only the vector dimension
        or known (short) constant dimensions (eg. local vector or matrix arrays)
        can be privatized without requiring shared GPU memory. Array variables
        with unknown (at compile time) dimensions (eg. the vertical dimension)
        cannot be privatized at the vector loop level and should therefore not
        be demoted here.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Establish the new dimensions and shapes first, before cloning the variables
        # The reason for this is that shapes of all variable instances are linked
        # via caching, meaning we can easily void the shape of an unprocessed variable.
        variables = list(routine.variables)
        variables += list(FindVariables(unique=False).visit(routine.body))

        # Filter out purely local array variables
        argument_map = CaseInsensitiveDict({a.name: a for a in routine.arguments})
        variables = [v for v in variables if not v.name in argument_map]
        variables = [v for v in variables if isinstance(v, Array)]

        # Find all arrays with shapes that do not include the vertical
        # dimension and can thus be privatized.
        variables = [v for v in variables if v.shape is not None]
        variables = [v for v in variables if not any(self.vertical.size in d for d in v.shape)]

        # Record original array shapes
        shape_map = CaseInsensitiveDict({v.name: v.shape for v in variables})

        # Demote private local variables
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

    def kernel_add_vector_loops(self, routine):
        """
        Insert the "vector" loop in GPU format around the entire body
        of the kernel routine. If directives are specified this also
        derived the set of private local arrays and marks them in the
        "vector"-level loop.

        Note that this assumes that the body has no nested
        vector-level kernel calls. For nested cases, we need to apply
        this to all non-nested code chunks in a kernel routine
        instead. **THIS IS NOT YET IMPLEMENTED!**

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Find local arrays that need explicitly privatization
        argument_map = CaseInsensitiveDict({a.name: a for a in routine.arguments})
        private_arrays = [v for v in routine.variables if not v.name in argument_map]
        private_arrays = [v for v in private_arrays if isinstance(v, Array)]
        private_arrays = [v for v in private_arrays if not any(self.vertical.size in d for d in v.shape)]

        # Construct pragma and wrap entire body in vector loop
        private_arrs = ', '.join(v.name for v in private_arrays)
        pragma = None
        if self.directive == 'openacc':
            pragma = Pragma(keyword='acc', content='loop vector private({})'.format(private_arrs))

        v_start = routine.variable_map[self.horizontal.bounds[0]]
        v_end = routine.variable_map[self.horizontal.bounds[1]]
        bounds = LoopRange((v_start, v_end))
        index = get_integer_variable(routine, self.horizontal.index)
        vector_loop = Loop(variable=index, bounds=bounds, body=[routine.body.body], pragma=pragma)
        routine.body.body = (vector_loop,)
