# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import (
    symbols as sym, FindVariables, is_dimension_constant
)
from loki.ir import (
    nodes as ir, FindNodes, Transformer, pragmas_attached,
    is_loki_pragma, get_pragma_parameters
)
from loki.logging import info
from loki.tools import as_tuple, flatten
from loki.types import DerivedType

from loki.transformations.utilities import (
    find_driver_loops, get_local_arrays, check_routine_pragmas
)


__all__ = ['SCCAnnotateTransformation']


class SCCAnnotateTransformation(Transformation):
    """
    A set of utilities to insert offload directives. This includes both :any:`Loop` and
    :any:`Subroutine` level annotations.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    block_dim : :any:`Dimension`
        Optional ``Dimension`` object to define the blocking dimension
        to use for hoisted column arrays if hoisting is enabled.
    directive : string or None
        Directives flavour to use for parallelism annotations; either
        ``'openacc'`` or ``None``.
    """

    def __init__(self, horizontal, directive, block_dim):
        self.horizontal = horizontal
        self.directive = directive
        self.block_dim = block_dim

    @classmethod
    def kernel_annotate_vector_loops_openacc(cls, routine):
        """
        Insert ``!$acc loop vector`` annotations around horizontal vector
        loops, including the necessary private variable declarations.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in the vector loops should be removed.
        """

        # Find any local arrays that need explicitly privatization
        private_arrays = get_local_arrays(routine, section=routine.spec)
        private_arrays = [
            v for v in private_arrays
            if all(is_dimension_constant(d) for d in v.shape)
        ]

        if private_arrays:
            # Log private arrays in vector regions, as these can impact performance
            info(
                f'[Loki-SCC::Annotate] Marking private arrays in {routine.name}: '
                f'{[a.name for a in private_arrays]}'
            )

        with pragmas_attached(routine, ir.Loop):
            for loop in FindNodes(ir.Loop).visit(routine.body):
                for pragma in as_tuple(loop.pragma):
                    if is_loki_pragma(pragma, starts_with='loop vector reduction'):
                        # Turn reduction pragmas into `!$acc` equivalent
                        pragma._update(keyword='acc')
                        continue

                    if is_loki_pragma(pragma, starts_with='loop vector'):
                        # Turn general vector pragmas into `!$acc` and add private clause
                        private_arrs = ', '.join(v.name for v in private_arrays)
                        private_clause = '' if not private_arrays else f' private({private_arrs})'
                        pragma._update(keyword='acc', content=f'loop vector{private_clause}')

    @classmethod
    def kernel_annotate_sequential_loops_openacc(cls, routine):
        """
        Insert ``!$acc loop seq`` annotations for all loops previously
        marked with ``!$loki loop seq``.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in which to annotate sequential loops
        """
        with pragmas_attached(routine, ir.Loop):
            for loop in FindNodes(ir.Loop).visit(routine.body):
                if not is_loki_pragma(loop.pragma, starts_with='loop seq'):
                    continue

                # Replace internal `!$loki loop seq`` pragam with `!$acc` equivalent
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
    def insert_annotations(cls, routine, horizontal):

        # Mark all parallel vector loops as `!$acc loop vector`
        cls.kernel_annotate_vector_loops_openacc(routine)

        # Mark all non-parallel loops as `!$acc loop seq`
        cls.kernel_annotate_sequential_loops_openacc(routine)

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
        if check_routine_pragmas(routine, self.directive):
            return

        if self.directive == 'openacc':
            self.insert_annotations(routine, self.horizontal)

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

        # Mark all parallel vector loops as `!$acc loop vector`
        self.kernel_annotate_vector_loops_openacc(routine)

        # Mark all non-parallel loops as `!$acc loop seq`
        self.kernel_annotate_sequential_loops_openacc(routine)

        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            driver_loops = find_driver_loops(routine=routine, targets=targets)
            for loop in driver_loops:
                self.annotate_driver(self.directive, loop, self.block_dim)

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
    def annotate_driver(cls, directive, driver_loop, block_dim):
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

            for pragma in as_tuple(driver_loop.pragma):
                if is_loki_pragma(pragma, starts_with='loop driver'):
                    # Replace `!$loki loop driver` pragma with OpenACC equivalent
                    params = get_pragma_parameters(driver_loop.pragma, starts_with='loop driver')
                    vlength = params.get('vector_length')
                    vlength_clause = f' vector_length({vlength})' if vlength else ''

                    content = f'parallel loop gang{private_clause}{vlength_clause}'
                    pragma_new = ir.Pragma(keyword='acc', content=content)
                    pragma_post = ir.Pragma(keyword='acc', content='end parallel loop')

                    # Replace existing loki pragma and add post-pragma
                    loop_pragmas = tuple(p for p in as_tuple(driver_loop.pragma) if p is not pragma)
                    driver_loop._update(
                        pragma=loop_pragmas + (pragma_new,),
                        pragma_post=(pragma_post,) + as_tuple(driver_loop.pragma_post)
                    )
