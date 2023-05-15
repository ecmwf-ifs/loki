# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki import Transformation, FindNodes, ir, NestedTransformer, Transformer
from transformations.scc_base import SCCBaseTransformation

__all__ = ['SCCRevectorTransformation']

class SCCRevectorTransformation(Transformation):
    """
    A transformation to wrap thread-parallel IR sections within a horizontal loop.
    This transformation relies on markers placed by :any:`SCCDevectorTransformation`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    directive : string or None
        Directives flavour to use for parallelism annotations; either
        ``'openacc'`` or ``None``.
    """

    def __init__(self, horizontal, directive):
        self.horizontal = horizontal

        assert directive in [None, 'openacc']
        self.directive = directive

        self._processed = {}

    @classmethod
    def wrap_vector_section(cls, section, routine, horizontal):
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
        index = SCCBaseTransformation.get_integer_variable(routine, horizontal.index)
        bounds = sym.LoopRange((v_start, v_end))

        # Ensure we clone all body nodes, to avoid recursion issues
        vector_loop = ir.Loop(variable=index, bounds=bounds, body=Transformer().visit(section))

        # Add a comment before the pragma-annotated loop to ensure
        # we do not overlap with neighbouring pragmas
        return (ir.Comment(''), vector_loop)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCRevector utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """

        # TODO: we only need this here until the scheduler can combine multiple transformations into single pass
        # Bail if routine has already been processed
        if self._processed.get(routine, None):
            return

        role = kwargs['role']
        item = kwargs.get('item', None)

        if role == 'kernel':
            self.process_kernel(routine)

        # Mark routine as processed
        self._processed[routine] = True

    def process_kernel(self, routine):
        """
        Applies the SCCRevector utilities to a "kernel" and wraps all thread-parallel sections within
        a horizontal loop. The markers placed by :any:`SCCDevectorTransformation` are removed.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Promote vector loops to be the outermost loop dimension in the kernel
        mapper = {s.body: self.wrap_vector_section(s.body, routine, self.horizontal)
                          for s in FindNodes(ir.Section).visit(routine.body)
                          if s.label == 'vector_section'}
        routine.body = NestedTransformer(mapper).visit(routine.body)

        # Remove section wrappers
        section_mapper = {s: s.body for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section'}
        if section_mapper:
            routine.body = Transformer(section_mapper).visit(routine.body)
