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
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    vertical : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the vertical dimension, as needed to decide array privatization.
    directive : string or None
        Directives flavour to use for parallelism annotations; either
        ``'openacc'`` or ``None``.
    """

    def __init__(self, horizontal, vertical=None, directive=None):
        self.horizontal = horizontal
        self.vertical = vertical

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

        # Demote all private local variables
        self.demote_private_locals(routine)

        if self.directive == 'openacc':

            with pragmas_attached(routine, Loop):
                # Mark all remaining loops as seq (in place)
                for loop in FindNodes(Loop).visit(routine.body):
                    # Skip loops explicitly marked with `!$loki/claw nodep`
                    if loop.pragma and any('nodep' in p.content.lower() for p in as_tuple(loop.pragma)):
                        continue

                    loop._update(pragma=Pragma(keyword='acc', content='loop seq'))

        # Add vector-level loops at the highest level in the kernel
        self.kernel_add_vector_loops(routine)

        if self.directive == 'openacc':
            # Mark routine as `!$acc routine seq` to make it device-callable
            routine.body.prepend(Pragma(keyword='acc', content='routine vector'))


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

        # TODO: For now, now change on the driver side is needed.
        pass

    def demote_private_locals(self, routine):
        """
        Demotes all local variables that can be privatized at the `acc loop vector`
        level.

        Array variablesthat whose dimensions include only the vector dimension
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
