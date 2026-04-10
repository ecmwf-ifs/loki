# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to split monolithic physics subroutines into smaller
subroutines by physical-process section.

This addresses Intel optrpt remark ``#15532`` (compilation-time budget
exceeded) and the structural problem of extreme register pressure
caused by routines with thousands of local temporaries (e.g. CLOUDSC
with ~12,630 temporaries and 3,902 spills).

Two driving modes are supported:

- **Pragma mode** (default): the developer inserts ``!$loki outline``
  / ``!$loki end outline`` pragmas around each section, and Loki's
  built-in :func:`outline_pragma_regions` utility handles the
  extraction.

- **Heuristic mode**: the transformation detects "natural boundaries"
  from comment headers (lines of dashes/equals, ``SECTION N`` markers)
  and outlines the regions between them.

The transformation should run **last** in the CPU vectorisation
pipeline (after T1-T3, T5, T6), because outlining creates subroutine
boundaries that would prevent the other transformations from seeing
the full routine context.

**Numerical impact**: None -- outlining is semantically equivalent.
"""

import re

from loki.batch import Transformation, ProcedureItem
from loki.expression import symbols as sym, Variable
from loki.ir import nodes as ir, FindNodes, FindVariables, CallStatement
from loki.tools import as_tuple
from loki.types import BasicType

from loki.transformations.extract.outline import outline_pragma_regions
from loki.transformations.sanitise import do_resolve_associates
from loki.transformations.utilities import check_routine_sequential


__all__ = ['ExtractOutlinePhysicsSection']


class ExtractOutlinePhysicsSection(Transformation):
    """
    Split large physics routines into smaller subroutines.

    Works in two modes:

    - **pragma** (default): uses existing ``!$loki outline`` markers
      and Loki's built-in :func:`outline_pragma_regions`.
    - **heuristic**: splits at comment-delimited section boundaries.

    The newly created subroutines are placed alongside the original
    routine: in the ``contains`` clause of the enclosing :any:`Module`
    (for module-wrapped routines) or at file scope in the
    :any:`Sourcefile` (for free-standing routines).

    When used with the :any:`Scheduler`, set :attr:`traverse_file_graph`
    to ``True`` so that the transformation receives :any:`Sourcefile`
    objects and can place the new routines at the correct level.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        The dimension object describing the horizontal iteration space.
    mode : str, optional
        Either ``'pragma'`` (default) or ``'heuristic'``.
    min_section_lines : int, optional
        Minimum number of top-level body nodes a detected section must
        have to be outlined.  Default is ``50``.
    """

    # Tell the scheduler to traverse a file graph so we receive
    # Sourcefile objects, allowing us to append new routines at
    # file scope (critical for free-standing subroutines like CLOUDSC
    # where routine.parent is None).
    traverse_file_graph = True

    # Tell the scheduler that this transformation creates new scopes
    # (the outlined subroutines) so that it runs a discovery step
    # afterwards to include them in the dependency graph.
    creates_items = True

    def __init__(self, horizontal, mode='pragma', min_section_lines=50):
        self.horizontal = horizontal
        self.mode = mode
        self.min_section_lines = min_section_lines

    # -----------------------------------------------------------------
    # File-level entry point (used by scheduler with traverse_file_graph)
    # -----------------------------------------------------------------

    def transform_file(self, sourcefile, **kwargs):
        """
        Apply section outlining to all subroutines in a
        :any:`Sourcefile` and append newly created routines at file
        scope.

        This is the primary entry point when processing via the
        :any:`Scheduler` with ``traverse_file_graph = True``.
        """
        item_factory = kwargs.get('item_factory')
        scheduler_config = kwargs.get('scheduler_config')
        items = kwargs.get('items', ())
        file_item = kwargs.get('item')

        for routine in sourcefile.subroutines:
            new_routines = self._outline_routine(routine, **kwargs)
            if new_routines:
                sourcefile.ir.append(new_routines)

                # Register outlined routines as additional dependencies
                # on the parent routine's ProcedureItem so they survive
                # DependencyTransformation.remove_inactive_ir_nodes().
                # This follows the DuplicateKernel pattern in
                # loki/transformations/dependency.py.
                if item_factory and items:
                    self._register_new_items(
                        routine, new_routines, file_item,
                        items, item_factory, scheduler_config
                    )

    # -----------------------------------------------------------------
    # Module-level entry point (for module-wrapped routines)
    # -----------------------------------------------------------------

    def transform_module(self, module, **kwargs):
        """
        Apply section outlining to all subroutines in a :any:`Module`
        and append newly created routines to the module's ``contains``
        clause.
        """
        for routine in module.subroutines:
            new_routines = self._outline_routine(routine, **kwargs)
            if new_routines:
                module.contains.append(new_routines)

    # -----------------------------------------------------------------
    # Subroutine-level entry point (for direct apply in tests)
    # -----------------------------------------------------------------

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply section outlining to a single subroutine.

        When called directly (e.g., ``trafo.apply(routine)``), the
        newly created subroutines are placed in the enclosing
        :any:`Module` (via ``routine.parent``) if available.  For
        free-standing routines with no parent, the outlining still
        modifies the routine in place (replacing pragma regions with
        CALL statements) but the new subroutine definitions can only
        be placed when a parent scope is accessible.

        For full placement support with free-standing routines, apply
        the transformation to the :any:`Sourcefile` instead.
        """
        new_routines = self._outline_routine(routine, **kwargs)
        if new_routines and routine.parent is not None:
            routine.parent.contains.append(new_routines)

    # -----------------------------------------------------------------
    # Core outlining logic
    # -----------------------------------------------------------------

    def _outline_routine(self, routine, **kwargs):
        """
        Run the configured outlining mode on *routine* and return the
        list of newly created :any:`Subroutine` objects.

        Returns ``()`` if the routine is skipped or no regions are
        outlined.
        """
        role = kwargs.get('role', 'kernel')
        if role == 'driver':
            return ()
        if check_routine_sequential(routine):
            return ()

        # Resolve ASSOCIATE blocks before outlining so that the
        # dataflow analysis in outline_pragma_regions sees plain
        # variable references instead of associate-scoped aliases.
        # Without this, derived-type component accesses hidden behind
        # ASSOCIATE aliases cause ordering failures in
        # order_variables_by_type (the type name is not found among
        # the imported symbols).
        do_resolve_associates(routine)

        if self.mode == 'pragma':
            return self._apply_pragma_mode(routine)
        if self.mode == 'heuristic':
            return self._apply_heuristic_mode(routine)
        return ()

    # -----------------------------------------------------------------
    # Scheduler item registration
    # -----------------------------------------------------------------

    @staticmethod
    def _register_new_items(parent_routine, new_routines, file_item,
                            items, item_factory, config):
        """
        Create :any:`ProcedureItem` objects for the newly outlined
        subroutines and register them as ``additional_dependencies``
        on the parent routine's item.

        This ensures the new routines are included in the scheduler's
        SGraph so that later transformations (in particular
        :any:`DependencyTransformation`) do not strip them via
        ``remove_inactive_ir_nodes()``.

        Parameters
        ----------
        parent_routine : :any:`Subroutine`
            The original routine from which sections were outlined.
        new_routines : list of :any:`Subroutine`
            The newly created outlined subroutines.
        file_item : :any:`FileItem`
            The file item passed by the scheduler (``kwargs['item']``).
            Its :attr:`name` is used as the ``scope_name`` when creating
            new :any:`ProcedureItem` objects in the item cache.
        items : tuple of :any:`Item`
            Definition items passed from the scheduler (from
            ``_get_definition_items``).
        item_factory : :any:`ItemFactory`
            The factory used to create/cache items.
        config : :any:`SchedulerConfig`
            The scheduler config.
        """
        # Find the ProcedureItem for the parent routine
        parent_name = parent_routine.name.lower()
        parent_item = None
        for it in as_tuple(items):
            if isinstance(it, ProcedureItem) and it.local_name == parent_name:
                parent_item = it
                break

        if parent_item is None:
            return

        # Use the file item's name as scope_name -- this matches the
        # key under which the FileItem is stored in the item cache
        # (see ItemFactory.get_or_create_file_item_from_path).
        scope_name = file_item.name if file_item else None

        new_items = ()
        for routine in new_routines:
            # Free-standing subroutines have item names like
            # '#routine_name' with no module scope prefix.
            item_name = f'#{routine.name.lower()}'
            new_item = item_factory.get_or_create_item(
                ProcedureItem, item_name, scope_name, config
            )
            if new_item is not None:
                new_items += (new_item,)

        if new_items:
            parent_item.plan_data.setdefault('additional_dependencies', ())
            parent_item.plan_data['additional_dependencies'] += new_items

    # -----------------------------------------------------------------
    # Pragma mode
    # -----------------------------------------------------------------

    def _apply_pragma_mode(self, routine):
        """
        Use Loki's built-in ``outline_pragma_regions`` to extract all
        ``!$loki outline`` marked sections, then apply IFS-specific
        post-processing fixes.

        Returns a list of newly created :any:`Subroutine` objects.
        """
        new_routines = outline_pragma_regions(routine)

        if new_routines:
            self._fix_outlined_routines(routine, new_routines)

        return new_routines

    # -----------------------------------------------------------------
    # Post-processing fixes for outlined routines
    # -----------------------------------------------------------------

    @staticmethod
    def _fix_outlined_routines(parent, new_routines):
        """
        Fix argument mismatches between outlined subroutines and
        their call sites in *parent*.

        Two categories of fixes are applied:

        1. **Remove phantom statement-function arguments**: IFS-style
           ``#include`` files (``fcttre.func.h``, ``fccld.func.h``)
           define statement functions (e.g. ``FOELDCPM``, ``FOEELIQ``,
           ``FOEEICE``, ``FOKOOP``).  Loki's dataflow analysis treats
           these as "used symbols" and adds them to the outlined
           routine's argument list.  However, they are NOT real
           variables — they are statement functions re-declared by the
           ``#include`` inside the outlined routine.  They must be
           removed from both the SUBROUTINE definition and the CALL.

           Detection: a variable has ``BasicType.DEFERRED`` type AND
           is NOT in the parent routine's ``variable_map``.

        2. **Add missing dimension-spec variables**: Array arguments
           may have dimension expressions referencing derived-type
           variables (e.g. ``PGP2DSPP(KLON, YDSPPL%NRFTOTAL)``) that
           are NOT detected by the dataflow analysis because they only
           appear in type specifications, not in executable code.  These
           must be added as ``INTENT(IN)`` arguments.

        Parameters
        ----------
        parent : :any:`Subroutine`
            The parent routine containing the CALL statements.
        new_routines : list of :any:`Subroutine`
            The outlined subroutines to fix.
        """
        parent_vmap = parent.variable_map

        # Build a map from outlined routine name -> CallStatement
        call_map = {}
        for call in FindNodes(CallStatement).visit(parent.body):
            call_map[call.name.name.upper()] = call

        for routine in new_routines:
            call = call_map.get(routine.name.upper())
            if call is None:
                continue

            # --- Fix 1: Remove phantom statement-function arguments ---
            phantom_names = set()
            for arg in routine.arguments:
                if (hasattr(arg, 'type') and
                        isinstance(arg.type.dtype, BasicType) and
                        arg.type.dtype == BasicType.DEFERRED and
                        arg.name not in parent_vmap):
                    phantom_names.add(arg.name)

            if phantom_names:
                # Remove from routine arguments
                new_args = tuple(
                    a for a in routine.arguments
                    if a.name not in phantom_names
                )
                # Remove from routine variables (removes declarations)
                new_vars = tuple(
                    v for v in routine.variables
                    if v.name not in phantom_names
                )
                routine.arguments = new_args
                routine.variables = new_vars

                # Remove from call arguments — match by position since
                # the call args correspond 1:1 with the old routine args
                # BEFORE we modified them.  However, the call was already
                # built by outline_region() using call_arg_map which
                # may have FEWER args (the phantom was in the sub but
                # not the call).  So we just filter by name.
                new_call_args = tuple(
                    a for a in call.arguments
                    if a.name not in phantom_names
                )
                call._update(arguments=new_call_args)

            # --- Fix 2: Add missing dimension-spec variables ---
            # Scan all array arguments for dimension expressions that
            # reference variables not already in the argument list.
            existing_arg_names = {a.name.upper() for a in routine.arguments}
            missing_vars = {}  # name -> Variable from parent

            for arg in routine.arguments:
                if isinstance(arg, sym.Array) and arg.type.shape:
                    for dim_expr in arg.type.shape:
                        # Find all variables used in this dimension
                        for v in FindVariables().visit(dim_expr):
                            # Get the root parent for derived-type access
                            root = v.parents[0] if v.parent else v
                            root_name = root.name.upper()
                            if (root_name not in existing_arg_names and
                                    root_name not in missing_vars and
                                    root.name in parent_vmap):
                                missing_vars[root_name] = parent_vmap[root.name]

            if missing_vars:
                for name, parent_var in missing_vars.items():
                    # Clone the variable into the outlined routine's scope
                    # with INTENT(IN)
                    new_var = parent_var.clone(
                        type=parent_var.type.clone(
                            intent='in', allocatable=None, target=None
                        ),
                        scope=routine
                    )

                    # Add to routine arguments and variables
                    # Insert derived types at the front (before arrays)
                    # routine.arguments = routine.arguments + (new_var,)
                    print(f"adding new var {new_var}")
                    routine.arguments = (new_var,) + routine.arguments
                    # routine.variables = routine.variables + (new_var,)

                    # Add to call arguments
                    call._update(
                        arguments=(parent_var,) + call.arguments#  + (parent_var,)
                    )

    # -----------------------------------------------------------------
    # Heuristic mode
    # -----------------------------------------------------------------

    _SECTION_HEADER_RE = re.compile(
        r'!\s*[-=]{5,}'       # line of dashes or equals (5+)
        r'|!\s*SECTION\s+\d'  # "SECTION N"
        r'|!\s*\d+\.\s+[A-Z]'  # "N. Title"
        , re.IGNORECASE
    )

    def _apply_heuristic_mode(self, routine):
        """
        Detect section boundaries from comment headers, inject
        ``!$loki outline`` pragma pairs around each qualifying section,
        then delegate to :func:`outline_pragma_regions` for the actual
        extraction.

        Returns a list of newly created :any:`Subroutine` objects.
        """
        # Find section-header comments in the routine body
        all_comments = FindNodes(ir.Comment).visit(routine.body)
        headers = [
            c for c in all_comments
            if c.text and self._SECTION_HEADER_RE.search(c.text)
        ]

        if len(headers) < 2:
            # Need at least two headers to define one section
            return ()

        # Build a flat list of top-level body nodes for slicing
        body_nodes = list(routine.body.body)

        # Find the index of each header in the body
        header_indices = []
        for h in headers:
            for i, node in enumerate(body_nodes):
                if node is h:
                    header_indices.append(i)
                    break

        if len(header_indices) < 2:
            return ()

        # Determine which sections qualify (meet min_section_lines)
        # and inject !$loki outline pragmas around them.
        # We build the new body with pragma pairs inserted.
        sections_to_outline = []
        counter = 0

        for idx in range(len(header_indices) - 1):
            start = header_indices[idx]
            end = header_indices[idx + 1]
            section_nodes = body_nodes[start:end]

            if len(section_nodes) < self.min_section_lines:
                continue

            counter += 1
            section_name = self._extract_section_name(
                headers[idx].text, counter
            )
            new_name = f'{routine.name}_{section_name}'.upper()
            sections_to_outline.append((start, end, new_name))

        if not sections_to_outline:
            return ()

        # Build a new body list with pragma pairs injected
        new_body = []
        covered = set()
        for start, end, name in sections_to_outline:
            for i in range(start, end):
                covered.add(i)

        pos = 0
        for start, end, name in sections_to_outline:
            # Add any nodes before this section that aren't covered
            while pos < start:
                new_body.append(body_nodes[pos])
                pos += 1

            # Insert !$loki outline pragma before the section
            pragma_start = ir.Pragma(
                keyword='loki', content=f'outline name({name})'
            )
            new_body.append(pragma_start)

            # Add section body nodes
            for i in range(start, end):
                new_body.append(body_nodes[i])
                pos = i + 1

            # Insert !$loki end outline pragma after the section
            pragma_end = ir.Pragma(keyword='loki', content='end outline')
            new_body.append(pragma_end)

        # Add any remaining nodes after the last section
        while pos < len(body_nodes):
            new_body.append(body_nodes[pos])
            pos += 1

        # Update the routine body with the injected pragmas
        routine.body = routine.body.clone(body=as_tuple(new_body))

        # Now delegate to outline_pragma_regions which handles
        # dataflow analysis and region extraction properly
        return outline_pragma_regions(routine)

    @staticmethod
    def _extract_section_name(comment_text, fallback_idx):
        """
        Try to extract a descriptive section name from a comment header.
        Falls back to ``SECTION_<idx>``.
        """
        # Try "SECTION N: title" pattern
        m = re.search(r'SECTION\s+(\d+)\s*[:\-]?\s*(\w+)?',
                       comment_text, re.IGNORECASE)
        if m:
            name = m.group(2) or f'SECTION_{m.group(1)}'
            return re.sub(r'[^A-Za-z0-9_]', '_', name)

        # Try "N. Title" pattern
        m = re.search(r'(\d+)\.\s+([A-Za-z]\w*)', comment_text)
        if m:
            return re.sub(r'[^A-Za-z0-9_]', '_', m.group(2))

        return f'SECTION_{fallback_idx}'
