# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to create local scalar copies of block-dimension and
horizontal-dimension index variables inside kernel routines.

This prevents accidental aliasing when the caller-side variable is a
derived-type component (e.g. ``YDCPG_BNDS%KIDIA``).
"""

from loki.batch import Transformation
from loki.ir import (
    nodes as ir,
    FindNodes, Pragma, Transformer, SubstituteExpressions,
    is_loki_pragma, get_pragma_parameters
)
from loki.tools import as_tuple

__all__ = ['CreateLocalCopiesTransformation']


class CreateLocalCopiesTransformation(Transformation):
    """
    Create local scalar copies of block-dimension and horizontal-dimension
    index variables inside kernel routines.

    For each variable in ``block_dim.indices``, ``horizontal.upper``, and
    ``horizontal.lower`` that appears in the routine's variable map, a new
    ``local_<name>`` variable is created and all body references are
    rewritten to use the local copy.  This prevents accidental aliasing
    when the caller-side variable is a derived-type component (e.g.
    ``YDCPG_BNDS%KIDIA``).

    Variables matching ``horizontal.sizes`` are excluded because those are
    array dimension sizes (e.g. ``KLON``, ``NPROMA``) that must retain
    their original name in pool-allocator expressions and other code
    outside the block loop.

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

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply local-copy creation to a kernel :any:`Subroutine`.

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
            if 'LowerBlockIndex' in item.trafo_data:
                self._create_local_copies(routine)
            # Always clean up device-present pragmas for bounds-parent
            # variables, even when LowerBlockIndex did not propagate to
            # this item (e.g. sub-sub-kernels not directly marked with
            # ``!$loki small-kernels``).
            self._remove_bounds_parents_from_device_present(routine)

    @staticmethod
    def get_block_index(routine, variable_map, index):
        """
        Resolve *index* (a string from ``block_dim.indices`` or
        ``horizontal._upper/_lower``) to a Loki variable in *routine*.

        Handles both plain names and ``%``-separated derived-type paths.
        For derived-type components, this returns the parent object so local
        copies can be created as valid Fortran symbols (e.g. ``local_bnds``),
        rather than illegal component declarations like ``local_bnds%kbl``.
        """
        if (block_index := variable_map.get(index, None)):
            return block_index
        parent = index.split('%', maxsplit=1)[0]
        if parent in variable_map:
            return routine.resolve_typebound_var(parent, variable_map)
        return None

    def _create_local_copies(self, routine):
        """
        Create ``local_<name>`` copies of block-dimension indices and
        horizontal bounds (excluding ``horizontal.sizes``), substitute
        body references, and clean up ``!$loki device-present`` pragmas.
        """
        routine_variable_map = routine.variable_map

        # Exclude horizontal.sizes — these are array dimension sizes
        # (e.g. KLON, NPROMA) not loop bounds.
        size_names = {
            s.split('%')[-1].lower()
            for s in (self.horizontal.sizes or ())
        }

        create_local_copy = []
        for _index in self.block_dim.indices + self.horizontal._upper + self.horizontal._lower:
            if _index.split('%')[-1].lower() in size_names:
                continue
            block_index = self.get_block_index(routine, routine_variable_map, _index)
            if block_index is not None:
                create_local_copy.append(block_index)

        local_copy_map = {
            var: var.clone(
                name=f'local_{var.name}',
                type=var.type.clone(intent=None)
            )
            for var in create_local_copy
            if f'local_{var.name}' not in routine_variable_map
        }
        routine.body = SubstituteExpressions(local_copy_map).visit(routine.body)
        routine.variables += as_tuple(local_copy_map.values())

        new_assignments = tuple(
            ir.Assignment(lhs=val, rhs=key)
            for key, val in local_copy_map.items()
        )
        if new_assignments:
            routine.body.prepend(new_assignments)

        # Remove replaced variable names from !$loki device-present pragmas.
        # We use create_local_copy (not local_copy_map) because an earlier
        # pipeline step may have already created the local copy, causing
        # local_copy_map to be empty here.
        replaced_names = {str(var.name).lower() for var in create_local_copy}
        if replaced_names:
            self._filter_device_present_pragma(routine, replaced_names)

    @staticmethod
    def _filter_device_present_pragma(routine, names_to_remove):
        """
        Remove variable names in *names_to_remove* from any
        ``!$loki device-present vars(...)`` pragmas in *routine*.
        """
        pragma_map = {}
        for pragma in FindNodes(Pragma).visit(routine.body):
            if not is_loki_pragma(pragma, starts_with='device-present'):
                continue
            params = get_pragma_parameters(
                pragma, starts_with='device-present',
                only_loki_pragmas=False
            )
            if params is None or 'vars' not in params:
                continue
            var_list = [v.strip() for v in params['vars'].split(',')]
            filtered = [v for v in var_list if v.lower() not in names_to_remove]
            if len(filtered) < len(var_list):
                if filtered:
                    new_content = f'device-present vars({", ".join(filtered)})'
                else:
                    new_content = 'device-present'
                pragma_map[pragma] = pragma.clone(content=new_content)
        if pragma_map:
            routine.body = Transformer(pragma_map).visit(routine.body)

    def _get_bounds_parent_names(self):
        """
        Return the set of lower-cased parent variable names from all
        ``block_dim.indices``, ``horizontal.upper``, and ``horizontal.lower``
        entries that contain a ``%`` (i.e. are derived-type components).

        For example, ``YDCPG_BNDS%KIDIA`` yields ``ydcpg_bnds``.
        """
        parents = set()
        for _index in self.block_dim.indices + self.horizontal._upper + self.horizontal._lower:
            if '%' in _index:
                parents.add(_index.split('%')[0].lower())
        return parents

    def _remove_bounds_parents_from_device_present(self, routine):
        """
        Remove bounds-parent variable names (e.g. ``YDCPG_BNDS``) from
        ``!$loki device-present vars(...)`` pragmas.

        These variables are derived-type containers for loop bounds and
        are never accessed as device data.  The annotation step adds them
        because they appear as derived-type arguments, but they should be
        excluded from ``present()`` clauses.
        """
        bounds_parents = self._get_bounds_parent_names()
        if bounds_parents:
            self._filter_device_present_pragma(routine, bounds_parents)
