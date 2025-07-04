# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import symbols as sym
from loki.frontend import HAVE_FP
from loki.ir import (
    nodes as ir, FindNodes, Transformer, FindExpressions,
    SubstituteExpressions
)
from loki.tools import as_tuple

from loki.transformations.array_indexing import resolve_vector_dimension
from loki.transformations.sanitise import do_resolve_associates
from loki.transformations.utilities import (
    get_integer_variable, check_routine_sequential, rename_variables
)
if HAVE_FP:
    from fparser.two import Fortran2003


__all__ = ['SCCBaseTransformation']


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
    """

    def __init__(self, horizontal):
        self.horizontal = horizontal
        self.rename_indices = False

    @staticmethod
    def rename_index_aliases(routine, dimension):
        """
        Rename index aliases: map all index aliases ``dimension.indices`` to
        ``dimension.index``.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to rename index aliases.
        horizontal : :any:`Dimension`
            :any:`Dimension` object to rename the index aliases
            to the first/former index.
        """
        if len(dimension.indices) > 1:
            symbol_map = {index: dimension.index for index in dimension.indices[1:]}
            rename_variables(routine, symbol_map)

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
        item = kwargs.get('item', None)
        rename_indices = kwargs.get('rename_index_aliases', self.rename_indices)
        if item:
            rename_indices = item.config.get('rename_index_aliases', rename_indices)

        if role == 'kernel':
            self.process_kernel(routine, rename_indices=rename_indices)
        if role == 'driver':
            self.process_driver(routine)

    def process_kernel(self, routine, rename_indices=False):
        """
        Applies the SCCBase utilities to a "kernel". This consists simply
        of resolving associations, masked statements and vector notation.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Bail if routine is marked as sequential or routine has already been processed
        if check_routine_sequential(routine):
            return

        # Bail if routine is elemental
        if self.is_elemental(routine):
            return

        if rename_indices:
            self.rename_index_aliases(routine, dimension=self.horizontal)

        # Find the iteration index variable for the specified horizontal
        v_index = get_integer_variable(routine, name=self.horizontal.index)

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        do_resolve_associates(routine)

        # Resolve WHERE clauses
        self.resolve_masked_stmts(routine, loop_variable=v_index)

        # Resolve vector notation, eg. VARIABLE(KIDIA:KFDIA)
        resolve_vector_dimension(routine, dimension=self.horizontal)

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
        do_resolve_associates(routine)
