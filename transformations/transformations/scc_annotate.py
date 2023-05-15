# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki import(
           Transformation, ir, Transformer, FindNodes, pragmas_attached,
           CaseInsensitiveDict, DerivedType
)
from transformations.scc_base import SCCBaseTransformation

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

    def __init__(self, horizontal, vertical, directive, block_dim, hoist_column_arrays):
        self.horizontal = horizontal
        self.vertical = vertical
        self.directive = directive
        self.block_dim = block_dim
        self.hoist_column_arrays = hoist_column_arrays

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

        with pragmas_attached(routine, ir.Loop):
            mapper = {}
            for loop in FindNodes(ir.Loop).visit(routine.body):
                if loop.variable == horizontal.index:
                    # Construct pragma and wrap entire body in vector loop
                    private_arrs = ', '.join(v.name for v in private_arrays)
                    pragma = ()
                    private_clause = '' if not private_arrays else f' private({private_arrs})'
                    pragma = ir.Pragma(keyword='acc', content=f'loop vector{private_clause}')
                    mapper[loop] = loop.clone(pragma=(pragma,))

            routine.body = Transformer(mapper).visit(routine.body)

    @classmethod
    def kernel_annotate_sequential_loops_openacc(cls, routine, horizontal):
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
                    loop._update(pragma=(ir.Pragma(keyword='acc', content='loop seq'),))

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
    def insert_annotations(cls, routine, horizontal, vertical, hoist_column_arrays):

        # Mark all non-parallel loops as `!$acc loop seq`
        cls.kernel_annotate_sequential_loops_openacc(routine, horizontal)

        # Mark all parallel vector loops as `!$acc loop vector`
        cls.kernel_annotate_vector_loops_openacc(routine, horizontal, vertical)

        # Wrap the routine body in `!$acc data present` markers
        # to ensure device-resident data is used for array and struct arguments.
        cls.kernel_annotate_subroutine_present_openacc(routine)

        if hoist_column_arrays:
            # Mark routine as `!$acc routine seq` to make it device-callable
            routine.spec.append(ir.Pragma(keyword='acc', content='routine seq'))

        else:
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

        if role == 'kernel':
            self.process_kernel(routine)

    def process_kernel(self, routine):
        """
        Applies the SCCAnnotate utilities to a "kernel". This consists of inserting the relevant
        ``'openacc'`` annotations at the :any:`Loop` and :any:`Subroutine` level.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # TODO: we only need this here because we cannot mark routines to skip at the scheduler level
        # Bail if routine is marked as sequential or have already been processed
        if SCCBaseTransformation.check_routine_pragmas(routine, self.directive):
            return

        if self.directive == 'openacc':
            self.insert_annotations(routine, self.horizontal, self.vertical,
                                    self.hoist_column_arrays)
