# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from loki.expression import symbols as sym
from loki.transform import resolve_associates

from loki import (
    Transformation, FindNodes, Transformer, info,
    pragmas_attached, as_tuple, flatten, ir, FindExpressions,
    SymbolAttributes, BasicType, SubstituteExpressions, DerivedType,
    FindVariables, CaseInsensitiveDict, pragma_regions_attached,
    PragmaRegion, is_loki_pragma, HoistVariablesTransformation
)

__all__ = [
    'SCCBaseTransformation', 'SCCAnnotateTransformation',
    'SCCHoistTemporaryArraysTransformation'
]


class SCCBaseTransformation(Transformation):
    """
    A basic set of utilities used in the SCC transformation. These utilities
    can either be used as a transformation in their own right, or the contained
    class methods can be called directly.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    directive : string or None
        Directives flavour to use for parallelism annotations; either
        ``'openacc'`` or ``None``.
    """

    def __init__(self, horizontal, directive=None):
        self.horizontal = horizontal

        assert directive in [None, 'openacc']
        self.directive = directive

    # TODO: correct "definition" of a pure/elemental routine (take e.g. loki serial into account ...)
    @staticmethod
    def is_elemental(routine):
        """
        Check whether :any:`Subroutine` ``routine`` is an elemental routine.

        Need for distinguishing elemental and non-elemental function to transform
        those in a different way.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to check whether elemental
        """
        for prefix in routine.prefix:
            if prefix.lower() == 'elemental':
                return True
        return False

    @staticmethod
    def check_array_dimensions_in_calls(routine):
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        for call in calls:
            for arg in call.arguments:
                if isinstance(arg, sym.Array):
                    if any(dim == sym.RangeIndex((None, None)) for dim in arg.dimensions):
                        return False
        return True

    @staticmethod
    def remove_dimensions(routine):
        arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
        array_map = {}
        for array in arrays:
            if all(dim == sym.RangeIndex((None, None)) for dim in array.dimensions):
                new_dimensions = None
                array_map[array] = array.clone(dimensions=new_dimensions)
        routine.body = SubstituteExpressions(array_map).visit(routine.body)

    @staticmethod
    def explicit_dimensions(routine):
        arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
        array_map = {}
        for array in arrays:
            if not array.dimensions:
                new_dimensions = (sym.RangeIndex((None, None)),) * len(array.shape)
                array_map[array] = array.clone(dimensions=new_dimensions)
        routine.body = SubstituteExpressions(array_map).visit(routine.body)

    @classmethod
    def check_routine_pragmas(cls, routine, directive):
        """
        Check if routine is marked as sequential or has already been processed.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to perform checks on.
        directive: string or None
            Directives flavour to use for parallelism annotations; either
            ``'openacc'`` or ``None``.
        """

        pragmas = FindNodes(ir.Pragma).visit(routine.ir)
        routine_pragmas = [p for p in pragmas if p.keyword.lower() in ['loki', 'acc']]
        routine_pragmas = [p for p in routine_pragmas if 'routine' in p.content.lower()]

        seq_pragmas = [r for r in routine_pragmas if 'seq' in r.content.lower()]
        if seq_pragmas:
            loki_seq_pragmas = [r for r in routine_pragmas if 'loki' == r.keyword.lower()]
            if loki_seq_pragmas:
                if directive == 'openacc':
                    # Mark routine as acc seq
                    mapper = {seq_pragmas[0]: None}
                    routine.spec = Transformer(mapper).visit(routine.spec)
                    routine.body = Transformer(mapper).visit(routine.body)

                    # Append the acc pragma to routine.spec, regardless of where the corresponding
                    # loki pragma is found
                    routine.spec.append(ir.Pragma(keyword='acc', content='routine seq'))
            return True

        vec_pragmas = [r for r in routine_pragmas if 'vector' in r.content.lower()]
        if vec_pragmas:
            if directive == 'openacc':
                return True

        return False

    @classmethod
    def check_horizontal_var(cls, routine, horizontal):
        """
        Check for horizontal loop bounds in a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to perform checks on.
        horizontal : :any:`Dimension`
            :any:`Dimension` object describing the variable conventions used in code
            to define the horizontal data dimension and iteration space.
        """

        if horizontal.bounds[0] not in routine.variable_map:
            raise RuntimeError(f'No horizontal start variable found in {routine.name}')
        if horizontal.bounds[1] not in routine.variable_map:
            raise RuntimeError(f'No horizontal end variable found in {routine.name}')

    @classmethod
    def get_integer_variable(cls, routine, name):
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

    @classmethod
    def resolve_masked_stmts(cls, routine, loop_variable):
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
            # TODO: Currently limited to simple, single-clause WHERE stmts
            assert len(masked.conditions) == 1 and len(masked.bodies) == 1
            ranges = [e for e in FindExpressions().visit(masked.conditions[0]) if isinstance(e, sym.RangeIndex)]
            exprmap = {r: loop_variable for r in ranges}
            assert len(ranges) > 0
            assert all(r == ranges[0] for r in ranges)
            bounds = sym.LoopRange((ranges[0].start, ranges[0].stop, ranges[0].step))
            cond = ir.Conditional(condition=masked.conditions[0], body=masked.bodies[0], else_body=masked.default)
            loop = ir.Loop(variable=loop_variable, bounds=bounds, body=(cond,))
            # Substitute the loop ranges with the loop index and add to mapper
            mapper[masked] = SubstituteExpressions(exprmap).visit(loop)

        routine.body = Transformer(mapper).visit(routine.body)

        # if loops have been inserted, check if loop variable is declared
        if mapper and loop_variable not in routine.variables:
            routine.variables += as_tuple(loop_variable)

    @classmethod
    def resolve_vector_dimension(cls, routine, loop_variable, bounds):
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

        bounds_v = (sym.Variable(name=bounds[0]), sym.Variable(name=bounds[1]))

        mapper = {}
        for stmt in FindNodes(ir.Assignment).visit(routine.body):
            ranges = [e for e in FindExpressions().visit(stmt)
                      if isinstance(e, sym.RangeIndex) and e == bounds_str]
            if ranges:
                exprmap = {r: loop_variable for r in ranges}
                loop = ir.Loop(
                    variable=loop_variable, bounds=sym.LoopRange(bounds_v),
                    body=as_tuple(SubstituteExpressions(exprmap).visit(stmt))
                )
                mapper[stmt] = loop

        routine.body = Transformer(mapper).visit(routine.body)

        # if loops have been inserted, check if loop variable is declared
        if mapper and loop_variable not in routine.variables:
            routine.variables += as_tuple(loop_variable)

    @staticmethod
    def is_driver_loop(loop, targets):
        """
        Test/check whether a given loop is a *driver loop*.

        Parameters
        ----------
        loop : :any: `Loop`
            The loop to test if it is a *driver loop*.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """
        if loop.pragma:
            for pragma in loop.pragma:
                if pragma.keyword.lower() == "loki" and pragma.content.lower() == "driver-loop":
                    return True
        for call in FindNodes(ir.CallStatement).visit(loop.body):
            if call.name in targets:
                return True
        return False

    @classmethod
    def find_driver_loops(cls, routine, targets):
        """
        Find and return all driver loops of a given `routine`.

        A *driver loop* is specified either by a call to a routine within
        `targets` or by the pragma `!$loki driver-loop`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in which to find the driver loops.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """

        driver_loops = []
        nested_driver_loops = []
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if loop in nested_driver_loops:
                continue

            if not cls.is_driver_loop(loop, targets):
                continue

            driver_loops.append(loop)
            loops = FindNodes(ir.Loop).visit(loop.body)
            nested_driver_loops.extend(loops)
        return driver_loops

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCBase utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']

        if role == 'kernel':
            self.process_kernel(routine)
        if role == 'driver':
            self.process_driver(routine)

    def process_kernel(self, routine):
        """
        Applies the SCCBase utilities to a "kernel". This consists simply
        of resolving associations, masked statements and vector notation.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Bail if routine is marked as sequential or routine has already been processed
        if self.check_routine_pragmas(routine, self.directive):
            return

        # check for horizontal loop bounds in subroutine symbol table
        self.check_horizontal_var(routine, self.horizontal)

        # Find the iteration index variable for the specified horizontal
        v_index = self.get_integer_variable(routine, name=self.horizontal.index)

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        resolve_associates(routine)

        # Resolve WHERE clauses
        self.resolve_masked_stmts(routine, loop_variable=v_index)

        # Resolve vector notation, eg. VARIABLE(KIDIA:KFDIA)
        self.resolve_vector_dimension(routine, loop_variable=v_index, bounds=self.horizontal.bounds)

    def process_driver(self, routine):
        """
        Applies the SCCBase utilities to a "driver". This consists simply
        of resolving associations.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Resolve associates, since the PGI compiler cannot deal with
        # implicit derived type component offload by calling device
        # routines.
        resolve_associates(routine)


class SCCAnnotateTransformation(Transformation):
    """
    A set of utilities to insert offload directives. This includes both :any:`Loop` and
    :any:`Subroutine` level annotations.

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
    """

    def __init__(self, horizontal, vertical, directive, block_dim):
        self.horizontal = horizontal
        self.vertical = vertical
        self.directive = directive
        self.block_dim = block_dim

    @classmethod
    def kernel_annotate_vector_loops_openacc(cls, routine, horizontal, vertical):
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
        private_arrays = [v for v in private_arrays if not any(horizontal.size in d for d in v.shape)]

        if private_arrays:
            # Log private arrays in vector regions, as these can impact performance
            info(
                f'[Loki-SCC::Annotate] Marking private arrays in {routine.name}: '
                f'{[a.name for a in private_arrays]}'
            )

        mapper = {}
        with pragma_regions_attached(routine):
            for region in FindNodes(PragmaRegion).visit(routine.body):
                if is_loki_pragma(region.pragma, starts_with='vector-reduction'):
                    if (reduction_clause := re.search(r'reduction\([\w:0-9 \t]+\)', region.pragma.content)):

                        loops = FindNodes(ir.Loop).visit(region)
                        assert len(loops) == 1
                        pragma = ir.Pragma(keyword='acc', content=f'loop vector {reduction_clause[0]}')
                        mapper[loops[0]] = loops[0].clone(pragma=(pragma,))
                        mapper[region.pragma] = None
                        mapper[region.pragma_post] = None

        with pragmas_attached(routine, ir.Loop):
            for loop in FindNodes(ir.Loop).visit(routine.body):
                if loop.variable == horizontal.index and not loop in mapper:
                    # Construct pragma and wrap entire body in vector loop
                    private_arrs = ', '.join(v.name for v in private_arrays)
                    pragma = ()
                    private_clause = '' if not private_arrays else f' private({private_arrs})'
                    pragma = ir.Pragma(keyword='acc', content=f'loop vector{private_clause}')
                    mapper[loop] = loop.clone(pragma=(pragma,))

            routine.body = Transformer(mapper).visit(routine.body)

    @classmethod
    def kernel_annotate_sequential_loops_openacc(cls, routine, horizontal, block_dim=None, ignore=()):
        """
        Insert ``!$acc loop seq`` annotations around all loops that
        are not horizontal vector loops.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in which to annotate sequential loops
        horizontal: :any:`Dimension`
            The dimension object specifying the horizontal vector dimension
        block_dim: :any: `Dimension`
            The dimension object specifying the blocking dimension
        ignore: list or tuple
            Loops to be ignored for annotation
        """
        block_dim_index = None if block_dim is None else block_dim.index
        with pragmas_attached(routine, ir.Loop):

            for loop in FindNodes(ir.Loop).visit(routine.body):
                # Skip loops explicitly marked with `!$loki/claw nodep`
                if loop.pragma and any('nodep' in p.content.lower() for p in as_tuple(loop.pragma)):
                    continue

                if loop.variable != horizontal.index and loop.variable != block_dim_index and loop not in ignore:
                    # Perform pragma addition in place to avoid nested loop replacements
                    loop._update(pragma=(ir.Pragma(keyword='acc', content='loop seq'),))

                # Warn if we detect vector insisde sequential loop nesting
                nested_loops = FindNodes(ir.Loop).visit(loop.body)
                loop_pragmas = flatten(as_tuple(l.pragma) for l in as_tuple(nested_loops))
                if any('loop vector' in pragma.content for pragma in loop_pragmas):
                    info(f'[Loki-SCC::Annotate] Detected vector loop in sequential loop in {routine.name}')

    @classmethod
    def kernel_annotate_subroutine_present_openacc(cls, routine):
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
        # Add comment to prevent false-attachment in case it is preceded by an "END DO" statement
        routine.body.append((ir.Comment(text=''), ir.Pragma(keyword='acc', content='end data')))

    @classmethod
    def insert_annotations(cls, routine, horizontal, vertical):

        # Mark all parallel vector loops as `!$acc loop vector`
        cls.kernel_annotate_vector_loops_openacc(routine, horizontal, vertical)

        # Mark all non-parallel loops as `!$acc loop seq`
        cls.kernel_annotate_sequential_loops_openacc(routine, horizontal)

        # Wrap the routine body in `!$acc data present` markers
        # to ensure device-resident data is used for array and struct arguments.
        cls.kernel_annotate_subroutine_present_openacc(routine)

        # Mark routine as `!$acc routine vector` to make it device-callable
        routine.spec.append(ir.Pragma(keyword='acc', content='routine vector'))

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCAnnotate utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """

        role = kwargs['role']
        targets = as_tuple(kwargs.get('targets'))

        if role == 'kernel':
            self.process_kernel(routine)
        if role == 'driver':
            self.process_driver(routine, targets=targets)

    def process_kernel(self, routine):
        """
        Applies the SCCAnnotate utilities to a "kernel". This consists of inserting the relevant
        ``'openacc'`` annotations at the :any:`Loop` and :any:`Subroutine` level.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Bail if routine is marked as sequential
        if SCCBaseTransformation.check_routine_pragmas(routine, self.directive):
            return

        if self.directive == 'openacc':
            self.insert_annotations(routine, self.horizontal, self.vertical)

        # Remove the vector section wrappers
        # These have been inserted by SCCDevectorTransformation
        section_mapper = {s: s.body for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section'}
        if section_mapper:
            routine.body = Transformer(section_mapper).visit(routine.body)

    def process_driver(self, routine, targets=None):
        """
        Apply the relevant ``'openacc'`` annotations to the driver loop.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """

        # For the thread block size, find the horizontal size variable that is available in
        # the driver
        num_threads = None
        symbol_map = routine.symbol_map
        for size_expr in self.horizontal.size_expressions:
            if size_expr in symbol_map:
                num_threads = size_expr
                break

        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            driver_loops = SCCBaseTransformation.find_driver_loops(routine=routine, targets=targets)
            for loop in driver_loops:
                loops = FindNodes(ir.Loop).visit(loop.body)
                kernel_loops = [l for l in loops if l.variable == self.horizontal.index]
                if kernel_loops:
                    assert not loop == kernel_loops[0]
                self.annotate_driver(
                    self.directive, loop, kernel_loops, self.block_dim, num_threads
                )

            if self.directive == 'openacc':
                # Mark all non-parallel loops as `!$acc loop seq`
                self.kernel_annotate_sequential_loops_openacc(routine, self.horizontal, self.block_dim,
                                                              ignore=driver_loops)

        # Remove the vector section wrappers
        # These have been inserted by SCCDevectorTransformation
        section_mapper = {s: s.body for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section'}
        if section_mapper:
            routine.body = Transformer(section_mapper).visit(routine.body)

    @classmethod
    def device_alloc_column_locals(cls, routine, column_locals):
        """
        Add explicit OpenACC statements for creating device variables for hoisted column locals.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        column_locals : list
            List of column locals to be hoisted to driver layer
        """

        if column_locals:
            vnames = ', '.join(v.name for v in column_locals)
            pragma = ir.Pragma(keyword='acc', content=f'enter data create({vnames})')
            pragma_post = ir.Pragma(keyword='acc', content=f'exit data delete({vnames})')
            # Add comments around standalone pragmas to avoid false attachment
            routine.body.prepend((ir.Comment(''), pragma, ir.Comment('')))
            routine.body.append((ir.Comment(''), pragma_post, ir.Comment('')))

    @classmethod
    def annotate_driver(cls, directive, driver_loop, kernel_loops, block_dim, num_threads):
        """
        Annotate driver block loop with ``'openacc'`` pragmas.

        Parameters
        ----------
        directive : string or None
            Directives flavour to use for parallelism annotations; either
            ``'openacc'`` or ``None``.
        driver_loop : :any:`Loop`
            Driver ``Loop`` to wrap in ``'opencc'`` pragmas.
        kernel_loops : list of :any:`Loop`
            Vector ``Loop`` to wrap in ``'opencc'`` pragmas if hoisting is enabled.
        block_dim : :any:`Dimension`
            Optional ``Dimension`` object to define the blocking dimension
            to detect hoisted temporary arrays and excempt them from marking.
        num_threads : str
            The size expression that determines the number of threads per thread block
        """

        # Mark driver loop as "gang parallel".
        if directive == 'openacc':
            arrays = FindVariables(unique=True).visit(driver_loop)
            arrays = [v for v in arrays if isinstance(v, sym.Array)]
            arrays = [v for v in arrays if not v.type.intent]
            arrays = [v for v in arrays if not v.type.pointer]

            # Filter out arrays that are explicitly allocated with block dimension
            sizes = block_dim.size_expressions
            arrays = [v for v in arrays if not any(d in sizes for d in as_tuple(v.shape))]
            private_arrays = ', '.join(set(v.name for v in arrays))
            private_clause = '' if not private_arrays else f' private({private_arrays})'
            vector_length_clause = '' if not num_threads else f' vector_length({num_threads})'

            # Annotate vector loops with OpenACC pragmas
            if kernel_loops:
                for loop in as_tuple(kernel_loops):
                    loop._update(pragma=(ir.Pragma(keyword='acc', content='loop vector'),))

            if driver_loop.pragma is None or (len(driver_loop.pragma) == 1 and
                                              driver_loop.pragma[0].keyword.lower() == "loki" and
                                              driver_loop.pragma[0].content.lower() == "driver-loop"):
                p_content = f'parallel loop gang{private_clause}{vector_length_clause}'
                driver_loop._update(pragma=(ir.Pragma(keyword='acc', content=p_content),))
                driver_loop._update(pragma_post=(ir.Pragma(keyword='acc', content='end parallel loop'),))

            # add acc parallel loop gang if the only existing pragma is acc data
            elif len(driver_loop.pragma) == 1:
                if (driver_loop.pragma[0].keyword == 'acc' and
                    driver_loop.pragma[0].content.lower().lstrip().startswith('data ')):
                    p_content = f'parallel loop gang{private_clause}{vector_length_clause}'
                    driver_loop._update(pragma=(driver_loop.pragma[0], ir.Pragma(keyword='acc', content=p_content)))
                    driver_loop._update(pragma_post=(ir.Pragma(keyword='acc', content='end parallel loop'),
                                              driver_loop.pragma_post[0]))


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
    key : str, optional
        Access identifier/key for the ``item.trafo_data`` dictionary.
    """

    def __init__(self, key=None, block_dim=None, **kwargs):
        self.block_dim = block_dim
        super().__init__(key=key, **kwargs)

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
        new_args = tuple(
            v.clone(dimensions=tuple(sym.RangeIndex((None, None)) for _ in v.dimensions) + (idx_var,))
            for v in variables
        )
        return call.clone(arguments=call.arguments + new_args)
