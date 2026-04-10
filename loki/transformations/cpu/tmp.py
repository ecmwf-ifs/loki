# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.ir import nodes as ir, FindNodes, Transformer

from loki.transformations.array_indexing import resolve_vector_dimension, resolve_vector_notation
from loki.transformations.sanitise import do_resolve_associates
from loki.transformations.utilities import (
    check_routine_sequential, rename_variables
)


__all__ = ['CPUBaseTransformation']


class AnnotateLoopTransformer(Transformer):
    """
    A :any:`Transformer` that annotates horizontal vector loops with
    SIMD pragmas (``!$OMP SIMD``, ``!DIR$ IVDEP``).

    Parameters
    ----------
    dimension : :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """

    def __init__(self, dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimension = dimension

    def visit_Loop(self, loop, **kwargs):
        if loop.variable == self.dimension.index:
            pragma = (ir.Pragma(keyword='omp', content='simd'),)
            pragma_post = (ir.Pragma(keyword='omp', content='end simd'),)
            return (
                ir.Comment(text='!DIR$ IVDEP'),
                self._rebuild(
                    loop, self.visit(loop.children, **kwargs),
                    pragma=pragma, pragma_post=pragma_post
                ),
            )

        # Rebuild loop after recursing to children
        return self._rebuild(loop, self.visit(loop.children, **kwargs))


class CPUBaseTransformation(Transformation):
    """
    A basic set of utilities for CPU vectorisation. Resolves
    associations, vector notation, and annotates horizontal loops
    with SIMD pragmas.

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
        dimension : :any:`Dimension`
            :any:`Dimension` object to rename the index aliases
            to the first/former index.
        """
        if len(dimension.indices) > 1:
            symbol_map = {index: dimension.index for index in dimension.indices[1:]}
            rename_variables(routine, symbol_map)

    @staticmethod
    def is_elemental(routine):
        """
        Check whether :any:`Subroutine` ``routine`` is an elemental routine.

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
        Apply CPUBase utilities to a :any:`Subroutine`.

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

    def process_kernel(self, routine, rename_indices=False):
        """
        Applies the CPUBase utilities to a "kernel". This consists of
        resolving associations, masked statements and vector notation,
        and annotating horizontal loops with SIMD pragmas.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Bail if routine is marked as sequential
        if check_routine_sequential(routine):
            return

        # Bail if routine is elemental
        if self.is_elemental(routine):
            return

        if rename_indices:
            self.rename_index_aliases(routine, dimension=self.horizontal)

        # Resolve associates
        do_resolve_associates(routine)

        # Resolve vector notation, eg. VARIABLE(KIDIA:KFDIA)
        resolve_vector_dimension(routine, dimension=self.horizontal)
        resolve_vector_notation(routine)

        routine.body = AnnotateLoopTransformer(dimension=self.horizontal).visit(routine.body)

    def process_driver(self, routine):
        """
        Applies the CPUBase utilities to a "driver". This consists of
        resolving associations and vector notation.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """
        do_resolve_associates(routine)
        resolve_vector_dimension(routine, dimension=self.horizontal)
        resolve_vector_notation(routine)

        routine.body = AnnotateLoopTransformer(dimension=self.horizontal).visit(routine.body)

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
        resolve_vector_dimension(routine, dimension=self.horizontal)
