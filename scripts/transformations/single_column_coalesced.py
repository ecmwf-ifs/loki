from more_itertools import pairwise

from loki import (
    Transformation, FindNodes, FindVariables, Transformer,
    SubstituteExpressions, CallStatement, Loop, Variable, Scalar,
    Array, LoopRange, RangeIndex, SymbolAttributes, Pragma, BasicType,
    CaseInsensitiveDict, as_tuple, pragmas_attached,
    JoinableStringList, FindScopes, Comment, NestedMaskedTransformer,
    flatten, resolve_associates, Assignment, Conditional,
    FindExpressions, RangeIndex, MaskedStatement, RangeIndex,
    LoopRange
)


__all__ = ['SingleColumnCoalescedTransformation']


def get_integer_variable(routine, name):
    """
    Find a local variable in the routine, or create an integer-typed one.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to find the variable
    name : string
        Name of the variable to find the in the routine.
    """
    if name in routine.variable_map:
        v_index = routine.variable_map[name]
    else:
        dtype = SymbolAttributes(BasicType.INTEGER)
        v_index = Variable(name=name, type=dtype, scope=routine)
    return v_index


def kernel_remove_vector_loops(routine, horizontal):
    """
    Remove all vector loops over the specified dimension.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal : :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """
    loop_map = {}
    for loop in FindNodes(Loop).visit(routine.body):
        if loop.variable == horizontal.index:
            loop_map[loop] = loop.body
    routine.body = Transformer(loop_map).visit(routine.body)


def kernel_promote_vector_loops(routine, horizontal):
    """
    Promote vector loops to be the outermost loop in the kernel.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """
    def _construct_loop(body):
        """
        Create a single loop around the horizontal from a given body
        """
        v_start = routine.variable_map[horizontal.bounds[0]]
        v_end = routine.variable_map[horizontal.bounds[1]]
        bounds = LoopRange((v_start, v_end))
        index = get_integer_variable(routine, horizontal.index)
        # Ensure we clone all body nodes, to avoid recursion issues
        return Loop(variable=index, bounds=bounds, body=Transformer().visit(body))

    def _sections_from_nodes(nodes, section):
        """
        Extract a list of code sub-sections from a section and separator nodes.
        """
        nodes = [None, *nodes, None]
        sections = []
        for start, stop in pairwise(nodes):
            t = NestedMaskedTransformer(start=start, stop=stop, active=start is None, inplace=True)
            sec = as_tuple(t.visit(section))
            if start is not None:
                sec = sec[1:]  # Strip `start` node
            sections.append(sec)
        return sections

    def _wrap_local_section(section, mapper):
        """
        Wrap vector loops around calls that we cannot promot around
        """
        calls = FindNodes(CallStatement).visit(section)
        for s in _sections_from_nodes(calls, section):
            if len(s) > 0 and len(FindNodes(Assignment).visit(s)) > 0:
                # Add a comment before the pragma-annotated loop to ensure
                # not overlap with neighbouring pragmas
                vector_loop = _construct_loop(s)
                mapper[s] = (Comment(''), vector_loop)

    def _process_outer_section(section, mapper):
        """
        For any nodes that define sections span that recursive calls,
        we cannot promote vector loops past the section boundaries.
        So, we need to find those constrained sections, and deal with
        them separately, and recursively if they are nested.
        """

        # Identify outer "scopes" (loops/conditionals) constrained by recursive routine calls
        outer_scopes = []
        calls = FindNodes(CallStatement).visit(section)
        for call in calls:
            ancestors = flatten(FindScopes(call).visit(section))
            ancestor_scopes = [a for a in ancestors if isinstance(a, (Loop, Conditional))]
            if len(ancestor_scopes) > 0:
                outer_scopes.append(ancestor_scopes[0])

        # Insert outer vector loops around call-free sections outside of
        # the constrained scopes.
        outer_sections = _sections_from_nodes(outer_scopes, section)
        for section in outer_sections:
            # Apply wrapping around the unbroken outer sections
            _wrap_local_section(section, mapper)

        # Now recursively deal with constrained scopes explicitly by
        # applying the same mechanism to the loop or conditional body.
        for scope in outer_scopes:
            if isinstance(scope, Loop):
                _process_outer_section(scope.body, mapper)

            if isinstance(scope, Conditional):
                _process_outer_section(scope.body, mapper)
                _process_outer_section(scope.else_body, mapper)


    if horizontal.bounds[0] not in routine.variable_map:
        raise RuntimeError('No horizontal start variable found in {}'.format(routine.name))
    if horizontal.bounds[1] not in routine.variable_map:
        raise RuntimeError('No horizontal end variable found in {}'.format(routine.name))

    mapper = {}

    # Now re-wrap individual sections with vector loops at the
    # appropriate level, as defined by "scope" nodes (loops,
    # conditionals) that might be constrined by recursive function
    # calls.
    _process_outer_section(routine.body.body, mapper)

    routine.body = Transformer(mapper).visit(routine.body)


def kernel_demote_private_locals(routine, horizontal, vertical):
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
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    vertical: :any:`Dimension`
        The dimension object specifying the vertical loop dimension
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
    variables = [v for v in variables if not any(vertical.size in d for d in v.shape)]

    # Filter out variables that we will pass down the call tree
    calls = FindNodes(CallStatement).visit(routine.body)
    call_args = flatten(call.arguments for call in calls)
    call_args += flatten(list(dict(call.kwarguments).values()) for call in calls)
    variables = [v for v in variables if v.name not in call_args]

    # Record original array shapes
    shape_map = CaseInsensitiveDict({v.name: v.shape for v in variables})

    # TODO: We need to ensure that we only demote things that we do not use
    # to buffer things across two sections. With the extended loop promotion,
    # we now need to check that we only demote in distinct vector loops, rather
    # than across entire routines....

    # Demote private local variables
    vmap = {}
    for v in variables:
        old_shape = shape_map[v.name]
        new_shape = as_tuple(s for s in old_shape if s not in horizontal.size_expressions)

        if old_shape and old_shape[0] in horizontal.size_expressions:
            new_type = v.type.clone(shape=new_shape)
            if len(old_shape) > 1:
                new_dims = v.dimensions[1:] if v.dimensions else None
                vmap[v] = v.clone(dimensions=new_dims, type=new_type)
            else:
                vmap[v] = Scalar(name=v.name, parent=v.parent, type=new_type, scope=routine)

    routine.body = SubstituteExpressions(vmap).visit(routine.body)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)


def kernel_annotate_vector_loops_openacc(routine, horizontal, vertical):
    """
    Insert ``!$acc loop vector`` annotations around horizontal vector
    loops, including the necessary private variable declarations.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    vertical: :any:`Dimension`
        The dimension object specifying the vertical loop dimension
    """

    # Find any local arrays that need explicitly privatization
    argument_map = CaseInsensitiveDict({a.name: a for a in routine.arguments})
    private_arrays = [v for v in routine.variables if not v.name in argument_map]
    private_arrays = [v for v in private_arrays if isinstance(v, Array)]
    private_arrays = [v for v in private_arrays if not any(vertical.size in d for d in v.shape)]

    with pragmas_attached(routine, Loop):
        mapper = {}
        for loop in FindNodes(Loop).visit(routine.body):
            if loop.variable == horizontal.index:
                # Construct pragma and wrap entire body in vector loop
                private_arrs = ', '.join(v.name for v in private_arrays)
                pragma = None
                private_clause = '' if not private_arrays else ' private({})'.format(private_arrs)
                pragma = Pragma(keyword='acc', content='loop vector{}'.format(private_clause))
                mapper[loop] = loop.clone(pragma=pragma)

        routine.body = Transformer(mapper).visit(routine.body)


def kernel_annotate_sequential_loops_openacc(routine, horizontal):
    """
    Insert ``!$acc loop seq`` annotations around all loops that
    are not horizontal vector loops.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    """
    with pragmas_attached(routine, Loop):

        for loop in FindNodes(Loop).visit(routine.body):
            # Skip loops explicitly marked with `!$loki/claw nodep`
            if loop.pragma and any('nodep' in p.content.lower() for p in as_tuple(loop.pragma)):
                continue

            if loop.variable != horizontal.index:
                # Perform pragma addition in place to avoid nested loop replacements
                loop._update(pragma=Pragma(keyword='acc', content='loop seq'))


def resolve_masked_stmts(routine, loop_variable):
    """
    Resolve :any:`MaskedStatement` (WHERE statement) objects to an
    explicit combination of :any:`Loop` and :any:`Conditional` combination.
    """
    mapper = {}
    for masked in FindNodes(MaskedStatement).visit(routine.body):
        ranges = [e for e in FindExpressions().visit(masked.condition) if isinstance(e, RangeIndex)]
        exprmap = {r: loop_variable for r in ranges}
        assert len(ranges) > 0
        assert all(r == ranges[0] for r in ranges)
        bounds = LoopRange((ranges[0].start, ranges[0].stop, ranges[0].step))
        cond = Conditional(condition=masked.condition, body=masked.body, else_body=masked.default)
        loop = Loop(variable=loop_variable, bounds=bounds, body=cond)
        # Substitute the loop ranges with the loop index and add to mapper
        mapper[masked] = SubstituteExpressions(exprmap).visit(loop)

    routine.body = Transformer(mapper).visit(routine.body)


def resolve_vector_dimension(routine, loop_variable, bounds):
    """
    Resolve vector notation for a given dimension only. The dimension
    is defined by a loop variable and the bounds of the given range.

    TODO: Consolidate this with the internal
    `loki.transform.transform_array_indexing.resolve_vector_notation`.
    """
    bounds_str = '{}:{}'.format(bounds[0], bounds[1])

    mapper = {}
    for stmt in FindNodes(Assignment).visit(routine.body):
        ranges = [e for e in FindExpressions().visit(stmt)
                  if isinstance(e, RangeIndex) and e == bounds_str]
        if ranges:
            exprmap = {r: loop_variable for r in ranges}
            loop = Loop(variable=loop_variable, bounds=LoopRange(bounds),
                        body=SubstituteExpressions(exprmap).visit(stmt))
            mapper[stmt] = loop

    routine.body = Transformer(mapper).visit(routine.body)


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
            self.process_kernel(routine, targets=targets)

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

    def process_kernel(self, routine, targets=None):
        """
        Remove all vector loops that match the stored ``horizontal``
        and promote the index variable to an argument, leaving fully
        dimensioned array arguments intact.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        resolve_associates(routine)

        # Find the iteration index variable for the specified horizontal
        v_index = get_integer_variable(routine, name=self.horizontal.index)

        # Resolve WHERE clauses
        lvar = Variable(name=self.horizontal.index, scope=routine)
        resolve_masked_stmts(routine, loop_variable=lvar)

        # Resolve vector notation, eg. VARIABLE(KIDIA:KFDIA)
        resolve_vector_dimension(routine, loop_variable=lvar, bounds=self.horizontal.bounds)

        # Remove all vector loops over the specified dimension
        kernel_remove_vector_loops(routine, self.horizontal)

        if not self.hoist_column_arrays:
            # Promote vector loops to be the outermost loop dimension in the kernel 
            kernel_promote_vector_loops(routine, self.horizontal)

        # Demote all private local variables
        kernel_demote_private_locals(routine, self.horizontal, self.vertical)

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
            # Mark all non-parallel loops as `!$acc loop seq`
            kernel_annotate_sequential_loops_openacc(routine, self.horizontal)

            # Mark all parallel vector loops as `!$acc loop vector`
            kernel_annotate_vector_loops_openacc(routine, self.horizontal, self.vertical)

            if self.hoist_column_arrays:
                # Mark routine as `!$acc routine seq` to make it device-callable
                routine.body.prepend(Pragma(keyword='acc', content='routine seq'))

            else:
                # Mark routine as `!$acc routine vector` to make it device-callable
                routine.body.prepend(Pragma(keyword='acc', content='routine vector'))

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
            # Add comments around standalone pragmas to avoid false attachment
            routine.body.prepend((Comment(''), pragma, Comment('')))
            routine.body.append((Comment(''), pragma_post, Comment('')))

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
