from more_itertools import pairwise

from loki.expression import symbols as sym
from loki.transform import resolve_associates
from loki import ir
from loki import (
    Transformation, FindNodes, FindScopes, FindVariables,
    FindExpressions, Transformer, NestedMaskedTransformer,
    SubstituteExpressions, SymbolAttributes, BasicType, DerivedType,
    pragmas_attached, CaseInsensitiveDict, as_tuple, flatten
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
        v_index = sym.Variable(name=name, type=dtype, scope=routine)
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
    for loop in FindNodes(ir.Loop).visit(routine.body):
        if loop.variable == horizontal.index:
            loop_map[loop] = loop.body
    routine.body = Transformer(loop_map).visit(routine.body)


def kernel_sections_from_nodes(nodes, section):
    """
    Extract a list of code sub-sections from a section tuple and separator nodes.
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


def wrap_vector_section(section, routine, horizontal):
    """
    Wrap a section of nodes in a vector-level loop across the horizontal.

    Parameters
    ----------
    section : tuple of :any:`Node`
        A section of nodes to be wrapped in a vector-level loop
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """

    # Create a single loop around the horizontal from a given body
    v_start = routine.variable_map[horizontal.bounds[0]]
    v_end = routine.variable_map[horizontal.bounds[1]]
    index = get_integer_variable(routine, horizontal.index)
    bounds = sym.LoopRange((v_start, v_end))

    # Ensure we clone all body nodes, to avoid recursion issues
    vector_loop = ir.Loop(variable=index, bounds=bounds, body=Transformer().visit(section))

    # Add a comment before the pragma-annotated loop to ensure
    # we do not overlap with neighbouring pragmas
    return (ir.Comment(''), vector_loop)


def extract_vector_sections(section, horizontal):
    """
    Extract a contiguous sections of nodes that contains vector-level
    computations and are not interrupted by recursive subroutine calls
    or nested control-flow structures.

    Parameters
    ----------
    section : tuple of :any:`Node`
        A section of nodes from which to extract vector-level sub-sections
    horizontal: :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """

    # Identify outer "scopes" (loops/conditionals) constrained by recursive routine calls
    separator_nodes = []
    calls = FindNodes(ir.CallStatement).visit(section)
    for call in calls:
        if call in section:
            # If the call is at the current section's level, it's a separator
            separator_nodes.append(call)

        else:
            # If the call is deeper in the IR tree, it's highest ancestor is used
            ancestors = flatten(FindScopes(call).visit(section))
            ancestor_scopes = [a for a in ancestors if isinstance(a, (ir.Loop, ir.Conditional))]
            if len(ancestor_scopes) > 0:
                separator_nodes.append(ancestor_scopes[0])

    subsections = kernel_sections_from_nodes(separator_nodes, section)

    # Filter sub-sections that do not use the horizontal loop index variable
    subsections = [s for s in subsections if horizontal.index in list(FindVariables().visit(s))]

    # Recurse on all separator nodes that might contain further vector sections
    for separator in separator_nodes:

        if isinstance(separator, ir.Loop):
            subsec_body = extract_vector_sections(separator.body, horizontal)
            if subsec_body:
                subsections += subsec_body

        if isinstance(separator, ir.Conditional):
            subsec_body = extract_vector_sections(separator.body, horizontal)
            if subsec_body:
                subsections += subsec_body
            subsec_else = extract_vector_sections(separator.else_body, horizontal)
            if subsec_else:
                subsections += subsec_else

    return subsections


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
    variables = [v for v in variables if isinstance(v, sym.Array)]

    # Find all arrays with shapes that do not include the vertical
    # dimension and can thus be privatized.
    variables = [v for v in variables if v.shape is not None]
    variables = [v for v in variables if not any(vertical.size in d for d in v.shape)]

    # Filter out variables that we will pass down the call tree
    calls = FindNodes(ir.CallStatement).visit(routine.body)
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
            new_type = v.type.clone(shape=new_shape or None)
            new_dims = v.dimensions[1:] or None
            vmap[v] = v.clone(dimensions=new_dims, type=new_type)

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
    private_arrays = [v for v in private_arrays if isinstance(v, sym.Array)]
    private_arrays = [v for v in private_arrays if not any(vertical.size in d for d in v.shape)]

    with pragmas_attached(routine, ir.Loop):
        mapper = {}
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if loop.variable == horizontal.index:
                # Construct pragma and wrap entire body in vector loop
                private_arrs = ', '.join(v.name for v in private_arrays)
                pragma = None
                private_clause = '' if not private_arrays else f' private({private_arrs})'
                pragma = ir.Pragma(keyword='acc', content=f'loop vector{private_clause}')
                mapper[loop] = loop.clone(pragma=pragma)

        routine.body = Transformer(mapper).visit(routine.body)


def kernel_annotate_sequential_loops_openacc(routine, horizontal):
    """
    Insert ``!$acc loop seq`` annotations around all loops that
    are not horizontal vector loops.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to annotate sequential loops
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    """
    with pragmas_attached(routine, ir.Loop):

        for loop in FindNodes(ir.Loop).visit(routine.body):
            # Skip loops explicitly marked with `!$loki/claw nodep`
            if loop.pragma and any('nodep' in p.content.lower() for p in as_tuple(loop.pragma)):
                continue

            if loop.variable != horizontal.index:
                # Perform pragma addition in place to avoid nested loop replacements
                loop._update(pragma=ir.Pragma(keyword='acc', content='loop seq'))


def kernel_annotate_subroutine_present_openacc(routine):
    """
    Insert ``!$acc data present`` annotations around the body of a subroutine.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine to which annotations will be added
    """

    # Get the names of all array and derived type arguments
    args = [a for a in routine.arguments if isinstance(a, sym.Array)]
    args += [a for a in routine.arguments if isinstance(a.type.dtype, DerivedType)]
    argnames = [str(a.name) for a in args]

    routine.body.prepend(ir.Pragma(keyword='acc', content=f'data present({", ".join(argnames)})'))
    routine.body.append(ir.Pragma(keyword='acc', content='end data'))


def resolve_masked_stmts(routine, loop_variable):
    """
    Resolve :any:`MaskedStatement` (WHERE statement) objects to an
    explicit combination of :any:`Loop` and :any:`Conditional` combination.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to resolve masked statements
    loop_variable : :any:`Scalar`
        The induction variable for the created loops.
    """
    mapper = {}
    for masked in FindNodes(ir.MaskedStatement).visit(routine.body):
        ranges = [e for e in FindExpressions().visit(masked.condition) if isinstance(e, sym.RangeIndex)]
        exprmap = {r: loop_variable for r in ranges}
        assert len(ranges) > 0
        assert all(r == ranges[0] for r in ranges)
        bounds = sym.LoopRange((ranges[0].start, ranges[0].stop, ranges[0].step))
        cond = ir.Conditional(condition=masked.condition, body=masked.body, else_body=masked.default)
        loop = ir.Loop(variable=loop_variable, bounds=bounds, body=cond)
        # Substitute the loop ranges with the loop index and add to mapper
        mapper[masked] = SubstituteExpressions(exprmap).visit(loop)

    routine.body = Transformer(mapper).visit(routine.body)


def resolve_vector_dimension(routine, loop_variable, bounds):
    """
    Resolve vector notation for a given dimension only. The dimension
    is defined by a loop variable and the bounds of the given range.

    TODO: Consolidate this with the internal
    `loki.transform.transform_array_indexing.resolve_vector_notation`.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to resolve vector notation usage.
    loop_variable : :any:`Scalar`
        The induction variable for the created loops.
    bounds : tuple of :any:`Scalar`
        Tuple defining the iteration space of the inserted loops.
    """
    bounds_str = f'{bounds[0]}:{bounds[1]}'

    mapper = {}
    for stmt in FindNodes(ir.Assignment).visit(routine.body):
        ranges = [e for e in FindExpressions().visit(stmt)
                  if isinstance(e, sym.RangeIndex) and e == bounds_str]
        if ranges:
            exprmap = {r: loop_variable for r in ranges}
            loop = ir.Loop(variable=loop_variable, bounds=sym.LoopRange(bounds),
                           body=SubstituteExpressions(exprmap).visit(stmt))
            mapper[stmt] = loop

    routine.body = Transformer(mapper).visit(routine.body)


def get_column_locals(routine, vertical):
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


class SingleColumnCoalescedTransformation(Transformation):
    """
    Single Column Coalesced: Direct CPU-to-GPU transformation for
    block-indexed gridpoint routines.

    This transformation will remove individiual CPU-style
    vectorization loops from "kernel" routines and either either
    re-insert the vector loop at the highest possible level (without
    interfering with subroutine calls), or completely strip it and
    promote the index variable to the driver if
    ``hoist_column_arrays`` is set.

    Unlike the CLAW-targetting SCA extraction, this will leave the
    block-based array passing structure in place, but pass a
    thread-local array index into any "kernel" routines. The
    block-based argument passing should map well to coalesced memory
    accesses on GPUs.

    Note, this requires preprocessing with the
    :any:`DerivedTypeArgumentsTransformation`.

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
                 demote_local_arrays=True, hoist_column_arrays=True):
        self.horizontal = horizontal
        self.vertical = vertical
        self.block_dim = block_dim

        assert directive in [None, 'openacc']
        self.directive = directive

        self.demote_local_arrays = demote_local_arrays
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

    def process_kernel(self, routine, targets=None):
        """
        Applies the SCC loop layout transformation to a "kernel"
        subroutine. This will primarily strip the innermost vector
        loops and either re-insert the vector loop at the highest
        possible level (without interfering with subroutine calls),
        or completely strip it and promote the index variable to the
        driver if ``hoist_column_arrays`` is set.

        In both cases argument arrays are left fully dimensioned,
        allowing us to use them in recursive subroutine invocations.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list of strings
            Names of all kernel routines that are to be considered "active"
            in this call tree and should thus be processed accordingly.
        """

        if self.horizontal.bounds[0] not in routine.variable_map:
            raise RuntimeError(f'No horizontal start variable found in {routine.name}')
        if self.horizontal.bounds[1] not in routine.variable_map:
            raise RuntimeError(f'No horizontal end variable found in {routine.name}')

        # Find the iteration index variable for the specified horizontal
        v_index = get_integer_variable(routine, name=self.horizontal.index)

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        resolve_associates(routine)

        # Resolve WHERE clauses
        resolve_masked_stmts(routine, loop_variable=v_index)

        # Resolve vector notation, eg. VARIABLE(KIDIA:KFDIA)
        resolve_vector_dimension(routine, loop_variable=v_index, bounds=self.horizontal.bounds)

        # Remove all vector loops over the specified dimension
        kernel_remove_vector_loops(routine, self.horizontal)

        # Extract vector-level compute sections from the kernel
        sections = extract_vector_sections(routine.body.body, self.horizontal)

        if not self.hoist_column_arrays:
            # Promote vector loops to be the outermost loop dimension in the kernel
            mapper = dict((s, wrap_vector_section(s, routine, self.horizontal)) for s in sections)
            routine.body = Transformer(mapper).visit(routine.body)

        # Demote all private local variables
        if self.demote_local_arrays:
            kernel_demote_private_locals(routine, self.horizontal, self.vertical)

        if self.hoist_column_arrays:
            # Promote all local arrays with column dimension to arguments
            # TODO: Should really delete and re-insert in spec, to prevent
            # issues with shared declarations.
            column_locals = get_column_locals(routine, vertical=self.vertical)
            promoted = [v.clone(type=v.type.clone(intent='INOUT')) for v in column_locals]
            routine.arguments += as_tuple(promoted)

            # Add loop index variable
            if v_index not in routine.arguments:
                new_v = v_index.clone(type=v_index.type.clone(intent='in'))
                # Remove original variable first, since we need to update declaration
                routine.variables = as_tuple(v for v in routine.variables if v != v_index)
                routine.arguments += as_tuple(new_v)

        if self.directive == 'openacc':
            # Mark all non-parallel loops as `!$acc loop seq`
            kernel_annotate_sequential_loops_openacc(routine, self.horizontal)

            # Mark all parallel vector loops as `!$acc loop vector`
            kernel_annotate_vector_loops_openacc(routine, self.horizontal, self.vertical)

            # Wrap the routine body in `!$acc data present` markers
            # to ensure device-resident data is used for array and struct arguments.
            kernel_annotate_subroutine_present_openacc(routine)

            if self.hoist_column_arrays:
                # Mark routine as `!$acc routine seq` to make it device-callable
                routine.body.prepend(ir.Pragma(keyword='acc', content='routine seq'))

            else:
                # Mark routine as `!$acc routine vector` to make it device-callable
                routine.body.prepend(ir.Pragma(keyword='acc', content='routine vector'))

    def process_driver(self, routine, targets=None):
        """
        Process the "driver" routine by inserting the other level
        parallel loops, and optionally hoisting temporary column
        arrays.

        Note that if ``hoist_column_arrays`` is set, the driver needs
        to be processed before any kernels are trnasformed. This is
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
        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):

            for call in FindNodes(ir.CallStatement).visit(routine.body):
                if not call.name in targets:
                    continue

                # Find the driver loop by checking the call's heritage
                ancestors = flatten(FindScopes(call).visit(routine.body))
                loops = [a for a in ancestors if isinstance(a, ir.Loop)]
                if not loops:
                    # Skip if there are no driver loops
                    continue
                loop = loops[0]

                # Mark driver loop as "gang parallel".
                if self.directive == 'openacc':
                    if loop.pragma is None:
                        loop._update(pragma=ir.Pragma(keyword='acc', content='parallel loop gang'))
                        loop._update(pragma_post=ir.Pragma(keyword='acc', content='end parallel loop'))

                # Apply hoisting of temporary "column arrays"
                if self.hoist_column_arrays:
                    self.hoist_temporary_column_arrays(routine, call)

    def hoist_temporary_column_arrays(self, routine, call):
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

        column_locals = get_column_locals(kernel, vertical=self.vertical)
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
        if self.directive == 'openacc':
            vnames = ', '.join(v.name for v in column_locals)
            pragma = ir.Pragma(keyword='acc', content=f'enter data create({vnames})')
            pragma_post = ir.Pragma(keyword='acc', content=f'exit data delete({vnames})')
            # Add comments around standalone pragmas to avoid false attachment
            routine.body.prepend((ir.Comment(''), pragma, ir.Comment('')))
            routine.body.append((ir.Comment(''), pragma_post, ir.Comment('')))

        # Add a block-indexed slice of each column variable to the call
        idx = get_integer_variable(routine, self.block_dim.index)
        new_args = [v.clone(
            dimensions=as_tuple([sym.RangeIndex((None, None)) for _ in v.shape]) + (idx,),
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
            pragma = ir.Pragma(keyword='acc', content='loop vector')
        v_start = arg_map[kernel.variable_map[self.horizontal.bounds[0]]]
        v_end = arg_map[kernel.variable_map[self.horizontal.bounds[1]]]
        bounds = sym.LoopRange((v_start, v_end))
        vector_loop = ir.Loop(variable=v_index, bounds=bounds, body=[new_call], pragma=pragma)
        call_map[call] = vector_loop

        routine.body = Transformer(call_map).visit(routine.body)
