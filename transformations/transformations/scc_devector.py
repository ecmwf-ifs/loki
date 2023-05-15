# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from more_itertools import split_at

from loki import (
     Transformation, FindNodes, ir, FindScopes, as_tuple, flatten, Transformer,
     NestedTransformer, FindVariables
)

__all__ = ['SCCDevectorTransformation']

class SCCDevectorTransformation(Transformation):
    """
    A set of utilities that can be used to strip vector loops from a :any:`Subroutine`
    and determine the regions of the IR to be placed within thread-parallel loop directives.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    directive: string or None
        Directives flavour to use for parallelism annotations; either
        ``'openacc'`` or ``None``.
    """

    def __init__(self, horizontal, directive):
        self.horizontal = horizontal

        assert directive in [None, 'openacc']
        self.directive = directive

        self._processed = {}

    @classmethod
    def kernel_remove_vector_loops(cls, routine, horizontal):
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

    @classmethod
    def extract_vector_sections(cls, section, horizontal):
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

        _scope_note_types = (ir.Loop, ir.Conditional, ir.MultiConditional)

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
                ancestor_scopes = [a for a in ancestors if isinstance(a, _scope_note_types)]
                if len(ancestor_scopes) > 0 and ancestor_scopes[0] not in separator_nodes:
                    separator_nodes.append(ancestor_scopes[0])

        # Extract contiguous node sections between separator nodes
        assert all(n in section for n in separator_nodes)
        subsections = [as_tuple(s) for s in split_at(section, lambda n: n in separator_nodes)]

        # Filter sub-sections that do not use the horizontal loop index variable
        subsections = [s for s in subsections if horizontal.index in list(FindVariables().visit(s))]

        # Recurse on all separator nodes that might contain further vector sections
        for separator in separator_nodes:

            if isinstance(separator, ir.Loop):
                subsec_body = cls.extract_vector_sections(separator.body, horizontal)
                if subsec_body:
                    subsections += subsec_body

            if isinstance(separator, ir.Conditional):
                subsec_body = cls.extract_vector_sections(separator.body, horizontal)
                if subsec_body:
                    subsections += subsec_body
                subsec_else = cls.extract_vector_sections(separator.else_body, horizontal)
                if subsec_else:
                    subsections += subsec_else

            if isinstance(separator, ir.MultiConditional):
                for body in separator.bodies:
                    subsec_body = cls.extract_vector_sections(body, horizontal)
                    if subsec_body:
                        subsections += subsec_body
                subsec_else = cls.extract_vector_sections(separator.else_body, horizontal)
                if subsec_else:
                    subsections += subsec_else

        return subsections

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCDevector utilities to a :any:`Subroutine`.

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
        Applies the SCCDevector utilities to a "kernel". This consists simply
        of stripping vector loops and determing which sections of the IR can be
        placed within thread-parallel loops.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Remove all vector loops over the specified dimension
        self.kernel_remove_vector_loops(routine, self.horizontal)

        # Replace sections with marked Section node
        section_mapper = {s: ir.Section(body=s, label='vector_section')
                          for s in self.extract_vector_sections(routine.body.body, self.horizontal)}
        routine.body = NestedTransformer(section_mapper).visit(routine.body)
