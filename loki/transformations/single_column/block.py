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
    NestedTransformer, is_loki_pragma, pragmas_attached, SubstituteExpressions
)
from loki.tools import as_tuple, flatten, CaseInsensitiveDict
from loki.types import BasicType

from loki.transformations.utilities import (
    find_driver_loops, check_routine_sequential
)


__all__ = [
    'SCCBlockSectionTransformation', 'SCCBlockSectionToLoopTransformation',
]


class ReblockSectionTransformer(Transformer):
    """
    :any:`Transformer` that replaces :any:`Section` objects labelled
    with ``"vector_section"`` with vector-level loops across the
    horizontal.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    insert_pragma: bool, optional
        Adds a ``!$loki vector`` pragma around the created loop
    """
    # pylint: disable=unused-argument

    def __init__(self, routine, item, horizontal, *args, insert_pragma=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.routine = routine
        self.horizontal = horizontal

        self.insert_pragma = insert_pragma
        self.item = item

        if 'LowerBlockIndex' in item.trafo_data:
            self.driver_loop = item.trafo_data['LowerBlockIndex']['driver_loop']
        else:
            self.driver_loop = None

    def visit_Section(self, s, **kwargs):
        if s.label == 'block_section':
            # Derive the loop bounds wrap section in loop
            # bounds = get_loop_bounds(self.routine, dimension=self.horizontal)
            symbol_map = self.routine.symbol_map
            sizes = tuple(
                self.routine.resolve_typebound_var(size, symbol_map) for size in self.horizontal.size_expressions
                if size.split('%')[0] in symbol_map
            )
            vector_length = f' vector_length({sizes[0]})' if sizes else ''
            if self.driver_loop is None:
                # TODO: raise proper exception
                assert False
            else:
                return (ir.Comment(text='! START OF BLOCK LOOP'), ir.Comment(text=''),
                        self.driver_loop.clone(body=self.driver_loop.body+s.body, pragma=(ir.Pragma(keyword='loki', content=f'loop driver{vector_length}'),)),
                        ir.Comment(text=''), 
                        ir.Comment(text='! END OF BLOCK LOOP'))
        
        # Rebuild loop after recursing to children
        return self._rebuild(s, self.visit(s.children))


class SCCBlockSectionToLoopTransformation(Transformation):

    def __init__(self, block_dim, horizontal):
        self.block_dim = block_dim
        self.horizontal = horizontal

    def activate_pragmas(self, routine):
        # !$loki inactive-small-kernels routine seq
        pragmas = FindNodes(ir.Pragma).visit(routine.body)
        for pragma in pragmas:
            if is_loki_pragma(pragma, starts_with='inactive-small-kernels'):
                pragma._update(content=pragma.content.replace('inactive-small-kernels', ''))

    def get_block_index(self, routine, variable_map, index):
        """
        Utility to retrieve the block-index loop induction variable.
        """
        if (block_index := variable_map.get(index, None)):
            return block_index
        if (index.split('%', maxsplit=1)[0] in variable_map):
            block_index = index.split('%', maxsplit=1)
            return routine.resolve_typebound_var(block_index[0], variable_map)
        return None

    def _create_local_copies(self, routine):
        routine_variable_map = routine.variable_map
        create_local_copy = []
        for _index in self.block_dim.indices:
            if not "%" in _index:
                continue
            if (block_index := self.get_block_index(routine, routine_variable_map, _index)):
                create_local_copy.append(block_index)
        local_copy_map = {var: var.clone(name=f'local_{var.name}', type=var.type.clone(intent=None)) for var in create_local_copy}
        routine.body = SubstituteExpressions(local_copy_map).visit(routine.body)
        routine.variables += as_tuple(local_copy_map.values())

        new_assignments = ()
        for key, val in local_copy_map.items():
            new_assignments += (ir.Assignment(lhs=val, rhs=key),)
        if new_assignments:
            routine.body.prepend(new_assignments)


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
        item = kwargs.get('item', None)

        if role == 'kernel':
            routine.body = ReblockSectionTransformer(routine, item, self.horizontal).visit(routine.body)
            self.activate_pragmas(routine)
            if 'LowerBlockIndex' in item.trafo_data:
                self._create_local_copies(routine)


class SCCBlockSectionTransformation(Transformation):
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

    def __init__(self, block_dim, trim_block_sections=True):
        self.block_dim = block_dim
        self.trim_block_sections = trim_block_sections

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
    def extract_block_sections(cls, section, block_dim, successor_map):
        """
        Extract a contiguous sections of nodes that contains vector-level
        computations and are not interrupted by recursive subroutine calls
        or nested control-flow structures.

        Parameters
        ----------
        section : tuple of :any:`Node`
            A section of nodes from which to extract vector-level sub-sections
        block_dim: :any:`Dimension`
            The dimension specifying the block dimension
        """

        # Identify outer "scopes" (loops/conditionals) constrained by recursive routine calls
        # with pragmas_attached(section, ir.CallStatement):
        calls = FindNodes(ir.CallStatement).visit(section)
        separator_nodes = []

        for call in calls:

            # check if calls have been enriched
            if not call.routine is BasicType.DEFERRED:
                # check if called routine is marked as sequential
                if check_routine_sequential(routine=call.routine):
                    continue
            call_pragmas = call.pragma
            if not call_pragmas:
                continue
            early_exit = True
            for pragma in call_pragmas:
                if pragma.keyword.lower() == 'loki' and pragma.content.lower() == "small-kernels":
                    successor_map[str(call.name)].trafo_data['BlockSectionTrafo'] = True
                    early_exit = False
                    break
            if early_exit:
                continue
        
            separator_nodes = cls._add_separator(call, section, separator_nodes)

        subsections = [as_tuple(s) for s in split_at(section, lambda n: n in separator_nodes)]

        subsections = [s for s in subsections if any([index in list(FindVariables().visit(s)) for index in block_dim.indices])]

        # Recurse on all separator nodes that might contain further vector sections
        for separator in separator_nodes:

            if isinstance(separator, ir.Conditional):
                subsec_body = cls.extract_block_sections(separator.body, block_dim, successor_map)
                if subsec_body:
                    subsections += subsec_body
                # we need to prevent that all (possibly nested) 'else_bodies' are completely wrapped as a section,
                # as 'Conditional's rely on the fact that the first element of each 'else_body'
                # (if 'has_elseif') is a Conditional itself
                for ebody in separator.else_bodies:
                    subsections += cls.extract_block_sections(ebody, block_dim, successor_map)

            if isinstance(separator, (ir.MultiConditional, ir.TypeConditional)):
                for body in separator.bodies:
                    subsec_body = cls.extract_block_sections(body, block_dim, successor_map)
                    if subsec_body:
                        subsections += subsec_body
                subsec_else = cls.extract_block_sections(separator.else_body, block_dim, successor_map)
                if subsec_else:
                    subsections += subsec_else

        return subsections

    @classmethod
    def get_trimmed_sections(cls, routine, block_dim, sections):
        """
        Trim extracted vector sections to remove nodes that are not assignments
        involving vector parallel arrays.
        """

        trimmed_sections = ()
        with dataflow_analysis_attached(routine):
            for sec in sections:
                block_nodes = [node for node in sec if any([index.lower() in node.uses_symbols for index in block_dim.indices])]
                start = sec.index(block_nodes[0])
                # don't loose e.g. loop vector pragmas ...
                if isinstance(sec[start-1], ir.Pragma):
                    start -= 1
                end = sec.index(block_nodes[-1])

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
        targets = kwargs.get('targets', ())
        item = kwargs.get('item', None)
        role = kwargs.get('role', None)
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = as_tuple(sub_sgraph.successors(item)) if sub_sgraph is not None else ()

        successor_map = CaseInsensitiveDict(
            (successor.local_name, successor)
            for successor in successors
        )

        if role == 'kernel':
            self.process_kernel(routine, item, successor_map)
        if role == "driver":
            self.process_driver(routine, item, successor_map, targets=targets)

    def process_driver(self, routine, item, successor_map, targets):
        with pragmas_attached(routine, ir.CallStatement):
            calls = FindNodes(ir.CallStatement).visit(routine.body)

            for call in calls:
                call_pragmas = call.pragma
                if not call_pragmas:
                    continue
                for pragma in call_pragmas:
                    if pragma.keyword.lower() == 'loki' and pragma.content.lower() == "small-kernels":
                        successor_map[str(call.name)].trafo_data['BlockSectionTrafo'] = True
        loop_map = {}
        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            driver_loops = find_driver_loops(section=routine.body, targets=targets)
            for driver_loop in driver_loops:
                pragmas = FindNodes(ir.Pragma).visit(driver_loop.body)
                for pragma in pragmas:
                    if pragma.keyword.lower() == 'loki' and pragma.content.lower() == "small-kernels":
                        loop_map[driver_loop] = (ir.Comment(text='! former driver loop ...'), driver_loop.body, ir.Comment(text='! END: former driver loop ...'))
                        break
            if loop_map:
                routine.body = Transformer(loop_map).visit(routine.body)

    def process_kernel(self, routine, item, successor_map):
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
        
        if not item.trafo_data.get('BlockSectionTrafo', False):
            return

        # remove 'loki routine seq/vec' pragmas
        pragmas = [pragma for pragma in FindNodes(ir.Pragma).visit(routine.ir) if is_loki_pragma(pragma, starts_with='routine')]
        pragma_map = {pragma: None for pragma in pragmas}
        routine.spec = Transformer(pragma_map).visit(routine.spec)
        routine.body = Transformer(pragma_map).visit(routine.body)

        # Extract vector-level compute sections from the kernel
        with pragmas_attached(routine, ir.CallStatement):
            sections = self.extract_block_sections(routine.body.body, self.block_dim, successor_map)

        if self.trim_block_sections:
            sections = self.get_trimmed_sections(routine, self.block_dim, sections)

        # Replace sections with marked Section node
        section_mapper = {s: ir.Section(body=s, label='block_section') for s in sections
                if s and [s for s in s if not isinstance(s, (ir.Comment, ir.Pragma, ir.CommentBlock))]}
        routine.body = NestedTransformer(section_mapper).visit(routine.body)
