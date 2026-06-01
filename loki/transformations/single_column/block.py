# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformations for extracting and re-injecting block-level computation
sections in the small-kernels SCC pipeline.

:any:`SCCBlockSectionTransformation` identifies contiguous IR regions
referencing block-dimension variables, wraps them in labelled
:any:`Section` nodes, and unwraps driver loops containing small-kernels
pragma calls.

:any:`SCCBlockSectionToLoopTransformation` re-wraps those labelled
sections in cloned driver loops with ``!$loki loop driver`` pragmas and
creates local copies of block-dimension index variables.

Applied in sequence, the two transformations convert a driver + kernel
pair from block-loop form into small-kernels form:

**Driver before both transformations:**

.. code-block:: fortran

    do ibl = 1, nb
      !$loki small-kernels
      call kernel(nproma, nb, ibl, za)
    end do

**Driver after** :any:`SCCBlockSectionTransformation` **(loop unwrapped):**

.. code-block:: fortran

    ! former driver loop ...
    call kernel(nproma, nb, ibl, za)
    ! END: former driver loop ...

**Kernel before both transformations:**

.. code-block:: fortran

    subroutine kernel(ydcpg_bnds, nproma, nb, za)
      !$loki routine seq
      za(1, ydcpg_bnds%kbl) = 1.0
      !$loki small-kernels
      call child(nproma, nb)
      za(2, ydcpg_bnds%kbl) = 2.0
    end subroutine

**Kernel after** :any:`SCCBlockSectionTransformation` **(sections extracted):**

.. code-block:: fortran

    subroutine kernel(ydcpg_bnds, nproma, nb, za)
      ! [block_section]
      za(1, ydcpg_bnds%kbl) = 1.0
      ! [end block_section]
      call child(nproma, nb)
      ! [block_section]
      za(2, ydcpg_bnds%kbl) = 2.0
      ! [end block_section]
    end subroutine

**Kernel after** :any:`SCCBlockSectionToLoopTransformation` **(block loops re-injected):**

.. code-block:: fortran

    subroutine kernel(ydcpg_bnds, nproma, nb, za)
      local_ydcpg_bnds = ydcpg_bnds
      ! START of Loki inserted block loop

      !$loki loop driver vector_length(nproma)
      do ibl = 1, nb
        za(1, local_ydcpg_bnds%kbl) = 1.0
      end do

      ! END of Loki inserted block loop
      call child(nproma, nb)
      ! START of Loki inserted block loop

      !$loki loop driver vector_length(nproma)
      do ibl = 1, nb
        za(2, local_ydcpg_bnds%kbl) = 2.0
      end do

      ! END of Loki inserted block loop
    end subroutine
"""

from more_itertools import split_at

from loki.analyse import dataflow_analysis_attached
from loki.batch import Transformation
from loki.ir import (
    nodes as ir, FindNodes, FindScopes, FindVariables, Transformer,
    NestedTransformer, is_loki_pragma, pragmas_attached,
    SubstituteExpressions,
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
    :any:`Transformer` that replaces :any:`Section` nodes labelled
    with ``"block_section"`` with block-level loops across the
    block dimension, using a cloned driver loop stored in
    ``item.trafo_data['LowerBlockIndex']['driver_loop']``.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which block sections should be wrapped.
    item : :any:`Item`
        The scheduler item for the routine, carrying ``trafo_data``.
    horizontal : :any:`Dimension`
        The dimension specifying the horizontal vector dimension
        (used for ``vector_length`` pragma annotation).
    """
    # pylint: disable=unused-argument

    def __init__(self, routine, item, horizontal, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.routine = routine
        self.horizontal = horizontal

        if 'LowerBlockIndex' not in item.trafo_data:
            raise RuntimeError(
                f'{self.__class__.__name__}: '
                f"'LowerBlockIndex' missing from item.trafo_data "
                f'for routine {routine.name}'
            )
        self.driver_loop = item.trafo_data['LowerBlockIndex']['driver_loop']

    def visit_Section(self, s, **kwargs):
        """
        Replace ``block_section``-labelled :any:`Section` nodes with a
        cloned driver loop containing the section body and a
        ``!$loki loop driver`` pragma.
        """
        if s.label == 'block_section':
            symbol_map = self.routine.symbol_map
            sizes = tuple(
                self.routine.resolve_typebound_var(size, symbol_map)
                for size in self.horizontal.size_expressions
                if size.split('%')[0] in symbol_map
            )
            vector_length = f' vector_length({sizes[0]})' if sizes else ''

            return (
                ir.Comment(text='! START of Loki inserted block loop'),
                ir.Comment(text=''),
                self.driver_loop.clone(
                    body=self.driver_loop.body + s.body,
                    pragma=(ir.Pragma(
                        keyword='loki',
                        content=f'loop driver{vector_length}'
                    ),)
                ),
                ir.Comment(text=''),
                ir.Comment(text='! END of Loki inserted block loop'),
            )

        # Rebuild section after recursing to children
        return self._rebuild(s, self.visit(s.children))


class SCCBlockSectionToLoopTransformation(Transformation):
    """
    Wrap identified block-level sections in driver block loops and
    create local copies of block-dimension index variables.

    This transformation operates on kernel routines that have been
    processed by :any:`SCCBlockSectionTransformation` (which wraps
    block-dimension computation regions in ``Section(label='block_section')``
    nodes).  It:

    1. Calls :any:`ReblockSectionTransformer` to replace those labelled
       sections with cloned driver loops carrying ``!$loki loop driver``
       pragmas.
    2. Activates ``!$loki inactive-small-kernels`` pragmas by stripping
       the ``inactive-small-kernels`` prefix.
    3. Creates local copies of derived-type block-index variables
       (e.g. ``YDCPG_BNDS%KBL`` → ``local_KBL``).

    **Before** (after :any:`SCCBlockSectionTransformation`):

    .. code-block:: fortran

        subroutine kernel(ydcpg_bnds, nproma, nb)
          ! [block_section]
          za(1, ydcpg_bnds%kbl) = 1.0
          ! [end block_section]
          !$loki inactive-small-kernels data present(za)
        end subroutine

    **After:**

    .. code-block:: fortran

        subroutine kernel(ydcpg_bnds, nproma, nb)
          local_ydcpg_bnds = ydcpg_bnds
          ! START of Loki inserted block loop

          !$loki loop driver vector_length(nproma)
          do ibl = 1, nb
            za(1, local_ydcpg_bnds%kbl) = 1.0
          end do

          ! END of Loki inserted block loop
          !$loki data present(za)
        end subroutine

    Parameters
    ----------
    block_dim : :any:`Dimension`
        Dimension object describing the blocking data dimension.
    horizontal : :any:`Dimension`
        Dimension object describing the horizontal (column) dimension.
    """

    def __init__(self, block_dim, horizontal):
        self.block_dim = block_dim
        self.horizontal = horizontal

    @staticmethod
    def activate_pragmas(routine):
        """
        Replace ``!$loki inactive-small-kernels`` pragmas with active
        ``!$loki`` pragmas by stripping the ``inactive-small-kernels``
        prefix from the pragma content.
        """
        pragmas = FindNodes(ir.Pragma).visit(routine.body)
        for pragma in pragmas:
            if is_loki_pragma(pragma, starts_with='inactive-small-kernels'):
                pragma._update(
                    content=pragma.content.replace('inactive-small-kernels', '')
                )

    @staticmethod
    def get_block_index(routine, variable_map, index):
        """
        Resolve a block-index name to its corresponding variable from
        the routine's variable map.

        Handles both simple names and derived-type member access names
        (e.g. ``YDCPG_BNDS%KBL``).

        Parameters
        ----------
        routine : :any:`Subroutine`
            The routine whose variables are searched.
        variable_map : dict
            Mapping of variable names to :any:`Variable` objects.
        index : str
            The block-index name to resolve (may contain ``%``).

        Returns
        -------
        :any:`Variable` or None
        """
        if (block_index := variable_map.get(index, None)) is not None:
            return block_index
        if index.split('%', maxsplit=1)[0] in variable_map:
            return routine.resolve_typebound_var(
                index.split('%', maxsplit=1)[0], variable_map
            )
        return None

    def _create_local_copies(self, routine):
        """
        Create local copies of derived-type block-dimension index
        variables and prepend ``local_X = X`` assignments to the
        routine body.
        """
        routine_variable_map = routine.variable_map
        create_local_copy = []
        for _index in self.block_dim.indices:
            if '%' not in _index:
                continue
            block_index = self.get_block_index(
                routine, routine_variable_map, _index
            )
            if block_index is not None:
                create_local_copy.append(block_index)

        local_copy_map = {
            var: var.clone(
                name=f'local_{var.name}',
                type=var.type.clone(intent=None)
            )
            for var in create_local_copy
        }
        routine.body = SubstituteExpressions(
            local_copy_map
        ).visit(routine.body)
        routine.variables += as_tuple(local_copy_map.values())

        new_assignments = tuple(
            ir.Assignment(lhs=val, rhs=key)
            for key, val in local_copy_map.items()
        )
        if new_assignments:
            routine.body.prepend(new_assignments)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply block-section-to-loop transformation to a kernel
        :any:`Subroutine`.

        For kernel routines, this replaces ``block_section``-labelled
        :any:`Section` nodes with driver block loops, activates
        inactive small-kernels pragmas, and creates local copies of
        block-dimension index variables.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : str
            Role of the subroutine in the call tree; only ``"kernel"``
            is processed.
        item : :any:`Item`
            Scheduler item carrying ``trafo_data``.
        """
        role = kwargs['role']
        item = kwargs.get('item', None)

        if role == 'kernel':
            routine.body = ReblockSectionTransformer(
                routine, item, self.horizontal
            ).visit(routine.body)
            self.activate_pragmas(routine)
            if 'LowerBlockIndex' in item.trafo_data:
                self._create_local_copies(routine)


class SCCBlockSectionTransformation(Transformation):
    """
    Extract contiguous block-level computation sections from kernel
    routines and wrap them in labelled :any:`Section` nodes for
    downstream processing by :any:`SCCBlockSectionToLoopTransformation`.

    For driver routines, this transformation marks successors with
    ``BlockSectionTrafo`` in ``trafo_data`` and unwraps driver loops
    that contain ``!$loki small-kernels`` pragma calls, replacing them
    with their body content.

    For kernel routines (guarded by ``BlockSectionTrafo`` in
    ``trafo_data``), the transformation:

    1. Removes ``!$loki routine`` pragmas from spec and body.
    2. Extracts contiguous IR sections that reference block-dimension
       variables or contain non-deferred call statements, split at
       call statements annotated with ``!$loki small-kernels`` pragmas.
    3. Optionally trims those sections to only include nodes that
       actually reference block-dimension symbols.
    4. Wraps the resulting sections in ``Section(label='block_section')``
       nodes via :any:`NestedTransformer`.

    For a kernel routine like:

    .. code-block:: fortran

        subroutine kernel(nproma, nb, ibl, za, zb)
          integer, intent(in) :: nproma, nb, ibl
          real, intent(inout) :: za(nproma, nb), zb(nproma, nb)
          !$loki routine seq
          za(1, ibl) = 1.0
          !$loki small-kernels
          call child_kernel(nproma, nb)
          zb(1, ibl) = 2.0
        end subroutine

    the transformation wraps block-dimension sections and removes the
    ``!$loki routine`` pragma:

    .. code-block:: fortran

        subroutine kernel(nproma, nb, ibl, za, zb)
          integer, intent(in) :: nproma, nb, ibl
          real, intent(inout) :: za(nproma, nb), zb(nproma, nb)
          ! [block_section]
          za(1, ibl) = 1.0
          ! [end block_section]
          call child_kernel(nproma, nb)
          ! [block_section]
          zb(1, ibl) = 2.0
          ! [end block_section]
        end subroutine

    For a driver routine, the block loop is unwrapped:

    .. code-block:: fortran

        ! Before:
        do ibl = 1, nb
          !$loki small-kernels
          call kernel(nproma, nb, ibl, za, zb)
        end do

        ! After:
        ! former driver loop ...
        call kernel(nproma, nb, ibl, za, zb)
        ! END: former driver loop ...

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object describing the block data dimension.
    trim_block_sections : bool, optional
        Flag to trigger trimming of extracted block sections to remove
        nodes that are not assignments involving block-dimension arrays.
        Default is ``True``.
    """

    _separator_node_types = (ir.Loop, ir.Conditional, ir.MultiConditional)

    def __init__(self, block_dim, trim_block_sections=True):
        self.block_dim = block_dim
        self.trim_block_sections = trim_block_sections

    @classmethod
    def _add_separator(cls, node, section, separator_nodes):
        """
        Add either the current node or its outermost parent node from
        the list of types defining a block-region separator
        (:attr:`_separator_node_types`) to the list of separator nodes.
        """
        if node in section:
            separator_nodes.append(node)
        else:
            ancestors = flatten(FindScopes(node).visit(section))
            ancestor_scopes = [
                a for a in ancestors
                if isinstance(a, cls._separator_node_types)
            ]
            if ancestor_scopes and ancestor_scopes[0] not in separator_nodes:
                separator_nodes.append(ancestor_scopes[0])

        return separator_nodes

    @staticmethod
    def _resolve_block_indices(routine, block_dim):
        """Resolve block-dimension index strings to routine-scoped expressions."""
        if routine is None:
            return ()

        variable_map = routine.variable_map
        resolved_indices = []
        for index in block_dim.indices:
            if (resolved := variable_map.get(index, None)) is not None:
                resolved_indices.append(resolved)
                continue
            parent = index.split('%', maxsplit=1)[0]
            if parent in variable_map:
                resolved_indices.append(routine.resolve_typebound_var(index, variable_map))
        return as_tuple(resolved_indices)

    @classmethod
    def _section_references_block_index(cls, section, resolved_block_indices):
        """Return True if a section references a resolved block index directly or in dimensions."""
        for var in FindVariables().visit(section):
            if resolved_block_indices:
                if var in resolved_block_indices:
                    return True
                if any(
                        dim in resolved_block_indices
                        for dim in as_tuple(getattr(var, 'dimensions', None) or ())
                ):
                    return True
        return False

    @staticmethod
    def _section_references_block_index_stringwise(section, block_dim):
        """Fallback match for block-index references embedded in rendered variable expressions."""
        block_indices = tuple(index.lower() for index in block_dim.indices)
        for var in FindVariables().visit(section):
            if any(index in str(var).lower() for index in block_indices):
                return True
            if any(
                any(index in str(dim).lower() for index in block_indices)
                for dim in as_tuple(getattr(var, 'dimensions', None) or ())
            ):
                return True
        return False

    @staticmethod
    def _is_explicit_data_pragma(node):
        """Return True for explicit data-management pragmas that must stay outside block loops."""
        if not isinstance(node, ir.Pragma):
            return False

        keyword = node.keyword.lower()
        content = node.content.lower()
        if keyword == 'acc':
            return ('enter' in content or 'exit' in content) and 'data' in content
        if keyword == 'loki':
            return 'unstructured-data' in content
        return False

    @classmethod
    def _node_is_block_anchor(cls, node, resolved_block_indices, block_dim):
        """Return True if the node itself should anchor a trimmed block section."""
        if cls._is_explicit_data_pragma(node):
            return False

        return bool(
            cls._section_references_block_index((node,), resolved_block_indices)
            or cls._section_references_block_index_stringwise((node,), block_dim)
        )

    @classmethod
    def extract_block_sections(cls, section, block_dim, successor_map, routine=None):
        """
        Extract contiguous sections of nodes that contain block-level
        computations, split at call statements annotated with
        ``!$loki small-kernels`` pragmas.

        Recursively processes :any:`Conditional` and
        :any:`MultiConditional` separator nodes to find nested block
        sections.

        Parameters
        ----------
        section : tuple of :any:`Node`
            A section of nodes from which to extract block-level
            sub-sections.
        block_dim : :any:`Dimension`
            The dimension specifying the block dimension.
        successor_map : :any:`CaseInsensitiveDict`
            Mapping of callee names to their scheduler items, used to
            mark successors with ``BlockSectionTrafo``.

        Returns
        -------
        list of tuple
            List of node tuples representing extracted block sections.
        """
        if routine is None and section:
            routine = getattr(section[0], 'scope', None)

        calls = FindNodes(ir.CallStatement).visit(section)
        separator_nodes = []

        for call in calls:
            if call.routine is not BasicType.DEFERRED:
                if check_routine_sequential(routine=call.routine):
                    continue

            call_pragmas = call.pragma
            if not call_pragmas:
                continue

            is_sk_call = False
            for pragma in call_pragmas:
                if (pragma.keyword.lower() == 'loki'
                        and pragma.content.lower() == 'small-kernels'):
                    successor_map[str(call.name)].trafo_data[
                        'BlockSectionTrafo'
                    ] = True
                    is_sk_call = True
                    break

            if not is_sk_call:
                continue

            separator_nodes = cls._add_separator(
                call, section, separator_nodes
            )

        raw_subsections = [
            as_tuple(s)
            for s in split_at(
                section, lambda n: n in separator_nodes
            )
        ]
        resolved_block_indices = cls._resolve_block_indices(routine, block_dim)

        # Keep only subsections that reference block-dim indices or
        # contain resolved call statements
        subsections = [
            s for s in raw_subsections
            if cls._section_references_block_index(s, resolved_block_indices)
            or cls._section_references_block_index_stringwise(s, block_dim)
            or any(
                call
                for call in FindNodes(ir.CallStatement).visit(s)
                if call.routine is not BasicType.DEFERRED
            )
        ]

        # Recurse into separator bodies for nested block sections
        for separator in separator_nodes:
            if isinstance(separator, ir.Conditional):
                subsec_body = cls.extract_block_sections(
                    separator.body, block_dim, successor_map, routine=routine
                )
                if subsec_body:
                    subsections += subsec_body
                for ebody in separator.else_bodies:
                    subsections += cls.extract_block_sections(
                        ebody, block_dim, successor_map, routine=routine
                    )

            if isinstance(
                separator, (ir.MultiConditional, ir.TypeConditional)
            ):
                for body in separator.bodies:
                    subsec_body = cls.extract_block_sections(
                        body, block_dim, successor_map, routine=routine
                    )
                    if subsec_body:
                        subsections += subsec_body
                subsec_else = cls.extract_block_sections(
                    separator.else_body, block_dim, successor_map, routine=routine
                )
                if subsec_else:
                    subsections += subsec_else

        return subsections

    @classmethod
    def get_trimmed_sections(cls, routine, block_dim, sections):
        """
        Trim extracted block sections to remove nodes that are not
        assignments involving block-dimension arrays.

        Uses :func:`dataflow_analysis_attached` to determine which
        nodes reference block-dimension symbols.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The routine containing the sections.
        block_dim : :any:`Dimension`
            The dimension specifying the block dimension.
        sections : tuple of tuple
            The extracted block sections to trim.

        Returns
        -------
        tuple of tuple
            The trimmed block sections.
        """
        trimmed_sections = ()
        resolved_block_indices = cls._resolve_block_indices(routine, block_dim)
        with dataflow_analysis_attached(routine):
            for sec in sections:
                block_nodes = [
                    node for node in sec
                    if any(
                        index.lower() in node.uses_symbols
                        for index in block_dim.indices
                    )
                    and cls._node_is_block_anchor(node, resolved_block_indices, block_dim)
                ]
                if block_nodes:
                    start = sec.index(block_nodes[0])
                    if (start > 0
                            and isinstance(sec[start - 1], ir.Pragma)
                            and 'loop' in sec[start - 1].content.lower()):
                        start -= 1
                    end = sec.index(block_nodes[-1])
                    trimmed_sections += (sec[start:end + 1],)
                else:
                    call_nodes = [
                        node for node in sec
                        if any(
                            call
                            for call in FindNodes(
                                ir.CallStatement
                            ).visit(node)
                            if call.routine is not BasicType.DEFERRED
                        )
                    ]
                    if call_nodes:
                        start = sec.index(call_nodes[0])
                        end = sec.index(call_nodes[-1])
                        trimmed_sections += (sec[start:end + 1],)

        return trimmed_sections

    def process_driver(self, routine, item, successor_map, targets):  # pylint: disable=unused-argument
        """
        Process a driver-level routine by marking call statements that
        have ``!$loki small-kernels`` pragmas for block-section
        treatment, and unwrapping driver loops that contain such calls.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The driver routine to process.
        item : :any:`Item`
            The scheduler item for the routine.
        successor_map : :any:`CaseInsensitiveDict`
            Mapping of callee names to their scheduler items.
        targets : tuple of str
            Target routine names for driver loop detection.
        """
        with pragmas_attached(routine, ir.CallStatement):
            calls = FindNodes(ir.CallStatement).visit(routine.body)
            for call in calls:
                call_pragmas = call.pragma
                if not call_pragmas:
                    continue
                for pragma in call_pragmas:
                    if (pragma.keyword.lower() == 'loki'
                            and pragma.content.lower() == 'small-kernels'):
                        successor_map[str(call.name)].trafo_data[
                            'BlockSectionTrafo'
                        ] = True

        loop_map = {}
        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            driver_loops = find_driver_loops(
                section=routine.body, targets=targets
            )
            for driver_loop in driver_loops:
                pragmas = FindNodes(ir.Pragma).visit(driver_loop.body)
                for pragma in pragmas:
                    if (pragma.keyword.lower() == 'loki'
                            and pragma.content.lower() == 'small-kernels'):
                        loop_map[driver_loop] = (
                            ir.Comment(
                                text='! former driver loop ...'
                            ),
                            driver_loop.body,
                            ir.Comment(
                                text='! END: former driver loop ...'
                            ),
                        )
                        break
            if loop_map:
                routine.body = Transformer(loop_map).visit(routine.body)

    def process_kernel(self, routine, item, successor_map):
        """
        Extract block-level computation sections from a kernel routine
        and wrap them in ``Section(label='block_section')`` nodes.

        Only processes routines that have ``BlockSectionTrafo`` set in
        their ``trafo_data`` (set by :meth:`process_driver` on the
        caller side).

        Parameters
        ----------
        routine : :any:`Subroutine`
            The kernel routine to process.
        item : :any:`Item`
            The scheduler item for the routine.
        successor_map : :any:`CaseInsensitiveDict`
            Mapping of callee names to their scheduler items.
        """
        if not item.trafo_data.get('BlockSectionTrafo', False):
            return

        # Remove !$loki routine pragmas
        pragmas = [
            pragma
            for pragma in FindNodes(ir.Pragma).visit(routine.ir)
            if is_loki_pragma(pragma, starts_with='routine')
        ]
        pragma_map = {pragma: None for pragma in pragmas}
        routine.spec = Transformer(pragma_map).visit(routine.spec)
        routine.body = Transformer(pragma_map).visit(routine.body)

        with pragmas_attached(routine, ir.CallStatement):
            sections = self.extract_block_sections(
                routine.body.body, self.block_dim, successor_map, routine=routine
            )

        if self.trim_block_sections:
            sections = self.get_trimmed_sections(
                routine, self.block_dim, sections
            )

        section_mapper = {
            s: ir.Section(body=s, label='block_section')
            for s in sections
            if s and any(
                not isinstance(n, (ir.Comment, ir.Pragma, ir.CommentBlock))
                for n in s
            )
        }
        routine.body = NestedTransformer(section_mapper).visit(routine.body)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply block-section extraction to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : str
            Role of the subroutine in the call tree
            (``"driver"`` or ``"kernel"``).
        item : :any:`Item`
            Scheduler item carrying ``trafo_data``.
        targets : tuple of str, optional
            Target routine names for driver loop detection.
        sub_sgraph : optional
            Sub-graph of the scheduler providing successor information.
        """
        targets = kwargs.get('targets', ())
        item = kwargs.get('item', None)
        role = kwargs.get('role', None)
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = (
            as_tuple(sub_sgraph.successors(item))
            if sub_sgraph is not None
            else ()
        )

        successor_map = CaseInsensitiveDict(
            (successor.local_name, successor)
            for successor in successors
        )

        if role == 'kernel':
            self.process_kernel(routine, item, successor_map)
        if role == 'driver':
            self.process_driver(
                routine, item, successor_map, targets=targets
            )
