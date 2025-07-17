# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from loki.batch import Transformation
from loki.expression import symbols as sym, is_dimension_constant
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer,
    pragmas_attached, is_loki_pragma, get_pragma_parameters,
    pragma_regions_attached
)
from loki.logging import info, warning
from loki.tools import as_tuple, flatten
from loki.types import DerivedType

from loki.transformations.utilities import (
    find_driver_loops, get_local_arrays
)


__all__ = ['SCCAnnotateTransformation']


class SCCAnnotateTransformation(Transformation):
    """
    A set of utilities to insert generic Loki directives. This includes both :any:`Loop` and
    :any:`Subroutine` level annotations.

    Parameters
    ----------
    block_dim : :any:`Dimension`
        Optional ``Dimension`` object to define the blocking dimension
        to use for hoisted column arrays if hoisting is enabled.
    privatise_derived_types : bool, default: False
        Flag to enable privatising derived-type objects in driver loops.
    """

    def __init__(self, block_dim, privatise_derived_types=False):
        self.block_dim = block_dim
        self.privatise_derived_types = privatise_derived_types

    def annotate_vector_loops(self, routine):
        """
        Insert ``!$loki loop vector`` for previously marked loops,
        including addition of the necessary private variable declarations.

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
                    if not is_loki_pragma(pragma, starts_with='loop vector'):
                        continue
                    if not private_arrays:
                        continue
                    if 'reduction' not in (pragma_params := get_pragma_parameters(pragma, starts_with='loop vector')):
                        # Add private clause
                        pragma_params['private'] = ', '.join(
                            v.name
                            for v in pragma_params.get('private', []) + private_arrays
                        )
                        pragma_content = [f'{kw}({val})' if val else kw for kw, val in pragma_params.items()]
                        pragma._update(content=f'loop vector {" ".join(pragma_content)}'.strip())

    def warn_vec_within_seq_loops(self, routine):
        """
        Check for vector inside sequential loops and print warning.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in which to check for vector inside sequential loops
        """
        with pragmas_attached(routine, ir.Loop):
            for loop in FindNodes(ir.Loop).visit(routine.body):
                if not is_loki_pragma(loop.pragma, starts_with='loop seq'):
                    continue
                # Warn if we detect vector insisde sequential loop nesting
                nested_loops = FindNodes(ir.Loop).visit(loop.body)
                loop_pragmas = flatten(as_tuple(l.pragma) for l in as_tuple(nested_loops))
                if any('loop vector' in pragma.content for pragma in loop_pragmas):
                    info(f'[Loki-SCC::Annotate] Detected vector loop in sequential loop in {routine.name}')

    def annotate_kernel_routine(self, routine):
        """
        Insert ``!$loki routine seq/vector`` directives and wrap
        subroutine body in ``!$loki device-present`` directives.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to which annotations will be added
        """

        # Move `!$loki routine seq/vector` pragmas to spec
        routine_pragmas = [
            pragma for pragma in FindNodes(ir.Pragma).visit(routine.body)
            if is_loki_pragma(pragma, starts_with='routine')
        ]
        routine.spec.append(routine_pragmas)
        routine.body = Transformer({pragma: None for pragma in routine_pragmas}).visit(routine.body)

        # Get the names of all array and derived type arguments
        args = [a for a in routine.arguments if isinstance(a, sym.Array)]
        args += [a for a in routine.arguments if isinstance(a.type.dtype, DerivedType)]
        argnames = [str(a.name) for a in args]

        if argnames:
            # Add comment to prevent false-attachment in case it is preceded by an "END DO" statement
            content = f'device-present vars({", ".join(argnames)})'
            routine.body.prepend(ir.Pragma(keyword='loki', content=content))
            # Add comment to prevent false-attachment in case it is preceded by an "END DO" statement
            content = 'end device-present'
            routine.body.append((ir.Comment(text=''), ir.Pragma(keyword='loki', content=content)))

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply pragma annotations according to ``!$loki`` placeholder
        directives.

        This routine effectively adds ``!$loki device-present``
        clauses around kernel routine bodies and adds
        ``private`` clauses to loop annotations.

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
            # Bail if this routine has been processed before
            for p in FindNodes(ir.Pragma).visit(routine.ir):
                # Check if `!$acc routine` has already been added,
                #  e.g., this transformation has already been applied
                if p.keyword.lower() == 'acc' and 'routine' in p.content.lower():
                    return

            # Mark all parallel vector loops as `!$loki loop vector`
            self.annotate_vector_loops(routine)

            # Check for sequential loops within vector loops
            self.warn_vec_within_seq_loops(routine)

            # Wrap the routine body in `!$loki device-present vars(...)` markers to
            # ensure all arguments are device-resident.
            self.annotate_kernel_routine(routine)


        if role == 'driver':
            # Mark all parallel vector loops as `!$loki loop vector`
            self.annotate_vector_loops(routine)

            # Check for sequential loops within vector loops
            self.warn_vec_within_seq_loops(routine)

            with pragma_regions_attached(routine):
                with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
                    # Find variables with existing OpenACC data declarations
                    acc_vars = self.find_acc_vars(routine, targets)

                    driver_loops = find_driver_loops(section=routine.body, targets=targets)
                    for loop in driver_loops:
                        self.annotate_driver_loop(loop, acc_vars.get(loop, []))

    def find_acc_vars(self, routine, targets):
        """
        Find variables already specified in loki/acc data clauses.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """

        acc_vars = defaultdict(list)

        for region in FindNodes(ir.PragmaRegion).visit(routine.body):
            pragma_keyword = region.pragma.keyword.lower()
            if pragma_keyword in ['loki', 'acc']:
                if pragma_keyword == 'acc':
                    parameters = get_pragma_parameters(region.pragma, starts_with='data', only_loki_pragmas=False)
                else:
                    parameters = get_pragma_parameters(region.pragma, starts_with='structured-data',
                            only_loki_pragmas=False)
                if parameters is not None:
                    driver_loops = find_driver_loops(section=region.body, targets=targets)
                    if not driver_loops:
                        continue

                    # When a key is given multiple times, get_pragma_parameters returns a list
                    # We merge them here into single entries to make our life easier below
                    parameters = {key: ', '.join(as_tuple(value)) for key, value in parameters.items()}
                    if (default := parameters.get('default', None)):
                        if not 'none' in [p.strip().lower() for p in default.split(',')]:
                            for loop in driver_loops:

                                _vars = [var.name.lower() for var in FindVariables(unique=True).visit(loop)]
                                acc_vars[loop] += _vars
                    else:
                        _vars = [
                            p.strip().lower()
                            for category in ('present', 'copy', 'copyin', 'copyout', 'deviceptr')
                            for p in parameters.get(category, '').split(',')
                        ]

                        for loop in driver_loops:
                            acc_vars[loop] += _vars

        return acc_vars

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
            pragma = ir.Pragma(keyword='loki', content=f'unstructured-data create({vnames})')
            pragma_post = ir.Pragma(keyword='loki', content=f'end unstructured-data delete({vnames})')
            # Add comments around standalone pragmas to avoid false attachment
            routine.body.prepend((ir.Comment(''), pragma, ir.Comment('')))
            routine.body.append((ir.Comment(''), pragma_post, ir.Comment('')))

    def annotate_driver_loop(self, loop, acc_vars):
        """
        Annotate driver block loop with generic Loki pragmas.

        Parameters
        ----------
        loop : :any:`Loop`
            Driver :any:`Loop` to wrap in generic Loki pragmas.
        acc_vars : list
            Variables already declared in generic Loki data directives.
        """
        sizes = self.block_dim.size_expressions

        # Mark driver loop as "gang parallel".
        loop_vars = FindVariables(unique=True).visit(loop)
        arrays = [v for v in loop_vars if isinstance(v, sym.Array)]
        arrays = [v for v in arrays if not v.type.intent]
        arrays = [v for v in arrays if not v.type.pointer]
        arrays = [v for v in arrays if not v.name_parts[0].lower() in acc_vars]
        arrays = [v for v in arrays if not any(d in sizes for d in as_tuple(v.shape))]
        private_sym = arrays

        if self.privatise_derived_types:
            # Derived-types are classified as "aggregate variables" in the OpenACC and OpenMP offload
            # standards and have the same implicit data attributes as arrays. Therefore, local derived-type
            # scalars must also be privatised.
            structs = [v for v in loop_vars if isinstance(v.type.dtype, sym.DerivedType)]
            structs = [v for v in structs if not v.name_parts[0].lower() in acc_vars]
            structs = [v for v in structs if not v.type.intent]
            structs = [v for v in structs if not v in arrays]
            if (dynamic_structs := [v.name for v in structs if (v.type.pointer or v.type.allocatable)]):
                warning(f'[Loki-SCC::Annotate] dynamically allocated structs are being privatised: {dynamic_structs}')

            # Filter out arrays that are explicitly allocated with block dimension
            private_sym +=  [
                v for v in structs
                if not any(d in sizes for d in as_tuple(getattr(v, 'shape', [])))
            ]

        private_vars = ', '.join(dict.fromkeys(v.name for v in private_sym))
        private_clause = '' if not private_vars else f' private({private_vars})'

        for pragma in as_tuple(loop.pragma):
            if is_loki_pragma(pragma, starts_with='loop driver'):
                # Replace `!$loki loop driver` pragma with OpenACC equivalent
                params = get_pragma_parameters(loop.pragma, starts_with='loop driver')
                vlength = params.get('vector_length')
                asynchronous = params.get('async')
                vlength_clause = f' vlength({vlength})' if vlength else ''
                asynchronous_clause = f' async({asynchronous})' if asynchronous else ''

                content = f'loop gang{private_clause}{vlength_clause}{asynchronous_clause}'
                pragma_new = ir.Pragma(keyword='loki', content=content)
                pragma_post = ir.Pragma(keyword='loki', content='end loop gang')

                # Replace existing loki pragma and add post-pragma
                loop_pragmas = tuple(p for p in as_tuple(loop.pragma) if p is not pragma)
                loop._update(
                    pragma=loop_pragmas + (pragma_new,),
                    pragma_post=(pragma_post,) + as_tuple(loop.pragma_post)
                )
