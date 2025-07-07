# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation

from loki.transformations.array_indexing import (
    resolve_masked_statements, resolve_vector_dimension
)
from loki.transformations.sanitise import do_resolve_associates
from loki.transformations.utilities import (
    check_routine_sequential, rename_variables
)


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

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        do_resolve_associates(routine)

        # Resolve WHERE clauses
        resolve_masked_statements(routine, dimension=self.horizontal)

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
