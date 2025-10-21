# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from more_itertools import split_at

from loki.analyse import dataflow_analysis_attached
from loki.batch import Transformation
from loki.ir import (
    nodes as ir, FindNodes, FindScopes, FindVariables, Transformer,
    NestedTransformer, is_loki_pragma, pragmas_attached,
)
from loki.tools import as_tuple, flatten
from loki.types import BasicType
from loki.expression import symbols as sym

from loki.transformations.utilities import (
    find_driver_loops, check_routine_sequential
)


__all__ = [
    'RemoveLoopTransformer', 'SCCDevectorTransformation',
]


class RemoveLoopTransformer(Transformer):
    """
    A :any:`Transformer` that removes all loops over the specified
    dimension.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """
    # pylint: disable=unused-argument

    def __init__(self, dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimension = dimension

    def visit_Loop(self, loop, **kwargs):
        if loop.variable == self.dimension.index:
            # Recurse and return body as replacement
            return self.visit(loop.body, **kwargs)

        # Rebuild loop after recursing to children
        return self._rebuild(loop, self.visit(loop.children, **kwargs))


class SCCDevectorTransformation(Transformation):
    """
    A set of utilities that can be used to strip vector loops from a :any:`Subroutine`
    and determine the regions of the IR to be placed within thread-parallel loop directives.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    trim_vector_sections : bool
        Flag to trigger trimming of extracted vector sections to remove
        nodes that are not assignments involving vector parallel arrays.
    """

    _separator_node_types = (ir.Loop, ir.Conditional, ir.MultiConditional)

    def __init__(self, horizontal, trim_vector_sections=False):
        self.horizontal = horizontal
        self.trim_vector_sections = trim_vector_sections

    @classmethod
    def _add_separator(cls, node, section, separator_nodes):
        """
        Add either the current node or its outermost parent node from the list of types
        defining a vector region separator (:attr:`separator_node_types`) to the list of
        separator nodes.
        """

        if node in section:
            # If the node is at the current section's level, it's a separator
            separator_nodes.append(node)

        else:
            # If the node is deeper in the IR tree, it's highest ancestor is used
            ancestors = flatten(FindScopes(node).visit(section))
            ancestor_scopes = [a for a in ancestors if isinstance(a, cls._separator_node_types)]
            if len(ancestor_scopes) > 0 and ancestor_scopes[0] not in separator_nodes:
                separator_nodes.append(ancestor_scopes[0])

        return separator_nodes

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

        # Identify outer "scopes" (loops/conditionals) constrained by recursive routine calls
        calls = FindNodes(ir.CallStatement).visit(section)
        separator_nodes = []

        for call in calls:

            # check if calls have been enriched
            if not call.routine is BasicType.DEFERRED:
                # check if called routine is marked as sequential
                if check_routine_sequential(routine=call.routine):
                    continue

            separator_nodes = cls._add_separator(call, section, separator_nodes)

        for pragma in FindNodes(ir.Pragma).visit(section):
            # Reductions over thread-parallel regions should be marked as a separator node
            if (is_loki_pragma(pragma, starts_with='vector-reduction') or
                is_loki_pragma(pragma, starts_with='end vector-reduction') or
                is_loki_pragma(pragma, starts_with='separator')):

                separator_nodes = cls._add_separator(pragma, section, separator_nodes)

        for assign in FindNodes(ir.Assignment).visit(section):
            if assign.ptr and isinstance(assign.rhs, sym.Array):
                if assign.rhs.shape is not None and any(s in assign.rhs.shape for s in horizontal.size_expressions):
                    separator_nodes = cls._add_separator(assign, section, separator_nodes)

            if isinstance(assign.rhs, sym.InlineCall):
                # filter out array arguments
                # we can't use arg_map here because intrinsic functions are not enriched
                _params = assign.rhs.parameters + as_tuple(assign.rhs.kw_parameters.values())
                _params = [p for p in _params if isinstance(p, sym.Array)]

                # check if a horizontal array is passed as an argument, meaning we have a vector
                # InlineCall, e.g. an array reduction intrinsic
                for p in _params:
                    if any(s in (p.dimensions or p.shape) for s in horizontal.size_expressions):
                        separator_nodes = cls._add_separator(assign, section, separator_nodes)

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
                # we need to prevent that all (possibly nested) 'else_bodies' are completely wrapped as a section,
                # as 'Conditional's rely on the fact that the first element of each 'else_body'
                # (if 'has_elseif') is a Conditional itself
                for ebody in separator.else_bodies:
                    subsections += cls.extract_vector_sections(ebody, horizontal)

            if isinstance(separator, (ir.MultiConditional, ir.TypeConditional)):
                for body in separator.bodies:
                    subsec_body = cls.extract_vector_sections(body, horizontal)
                    if subsec_body:
                        subsections += subsec_body
                subsec_else = cls.extract_vector_sections(separator.else_body, horizontal)
                if subsec_else:
                    subsections += subsec_else

        return subsections

    @classmethod
    def get_trimmed_sections(cls, routine, horizontal, sections):
        """
        Trim extracted vector sections to remove nodes that are not assignments
        involving vector parallel arrays.
        """

        trimmed_sections = ()
        with dataflow_analysis_attached(routine):
            for sec in sections:
                vec_nodes = [node for node in sec if horizontal.index.lower() in node.uses_symbols]
                start = sec.index(vec_nodes[0])
                end = sec.index(vec_nodes[-1])

                trimmed_sections += (sec[start:end+1],)

        return trimmed_sections

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
        role = kwargs['role']
        targets = kwargs.get('targets', ())

        if role == 'kernel':
            self.process_kernel(routine)
        if role == "driver":
            self.process_driver(routine, targets=targets)

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
        routine.body = RemoveLoopTransformer(dimension=self.horizontal).visit(routine.body)

        # Extract vector-level compute sections from the kernel
        sections = self.extract_vector_sections(routine.body.body, self.horizontal)

        if self.trim_vector_sections:
            sections = self.get_trimmed_sections(routine, self.horizontal, sections)

        # Replace sections with marked Section node
        section_mapper = {s: ir.Section(body=s, label='vector_section') for s in sections}
        routine.body = NestedTransformer(section_mapper).visit(routine.body)

    def process_driver(self, routine, targets=()):
        """
        Applies the SCCDevector utilities to a "driver". This consists simply
        of stripping vector loops and determining which sections of the IR can be
        placed within thread-parallel loops.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """

        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            driver_loops = find_driver_loops(section=routine.body, targets=targets)

        # remove vector loops
        driver_loop_map = {}
        for loop in driver_loops:
            new_driver_loop = RemoveLoopTransformer(dimension=self.horizontal).visit(loop.body)
            new_driver_loop = loop.clone(body=new_driver_loop)
            sections = self.extract_vector_sections(new_driver_loop.body, self.horizontal)
            if self.trim_vector_sections:
                sections = self.get_trimmed_sections(new_driver_loop, self.horizontal, sections)
            section_mapper = {s: ir.Section(body=s, label='vector_section') for s in sections}
            new_driver_loop = NestedTransformer(section_mapper).visit(new_driver_loop)
            driver_loop_map[loop] = new_driver_loop
        routine.body = Transformer(driver_loop_map).visit(routine.body)
