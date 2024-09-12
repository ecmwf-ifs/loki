# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utility routines that provide transformations for code regions.

"""
from collections import defaultdict

from loki.ir import (
    Comment, Loop, Pragma, PragmaRegion, FindNodes, FindScopes,
    MaskedTransformer, NestedMaskedTransformer, is_loki_pragma,
    get_pragma_parameters, pragma_regions_attached
)
from loki.logging import info
from loki.tools import as_tuple, flatten

from loki.transformations.array_indexing import (
    promotion_dimensions_from_loop_nest, promote_nonmatching_variables
)


__all__ = ['region_hoist']


def region_hoist(routine):
    """
    Hoist one or multiple code regions annotated by pragma ranges and insert
    them at a specified target location.

    The pragma syntax for annotating the regions to hoist is
    ``!$loki region-hoist [group(group-name)] [collapse(n) [promote(var-name, var-name, ...)]]``
    and ``!$loki end region-hoist``.
    The optional ``group(group-name)`` can be provided when multiple regions
    are to be hoisted and inserted at different positions. Multiple pragma
    ranges can be specified for the same group, all of which are then moved to
    the target location in the same order as the pragma ranges appear.
    The optional ``collapse(n)`` parameter specifies that ``n`` enclosing scopes
    (such as loops, conditionals, etc.) should be re-created at the target location.
    Optionally, this can be combined with variable promotion using ``promote(...)``.
    """
    hoist_targets = defaultdict(list)
    hoist_regions = defaultdict(list)

    # Find all region-hoist pragma regions
    with pragma_regions_attached(routine):
        for region in FindNodes(PragmaRegion).visit(routine.body):
            if is_loki_pragma(region.pragma, starts_with='region-hoist'):
                parameters = get_pragma_parameters(region.pragma, starts_with='region-hoist')
                group = parameters.get('group', 'default')
                hoist_regions[group] += [(region.pragma, region.pragma_post)]

    # Find all region-hoist targets
    for pragma in FindNodes(Pragma).visit(routine.body):
        if is_loki_pragma(pragma, starts_with='region-hoist'):
            parameters = get_pragma_parameters(pragma, starts_with='region-hoist')
            if 'target' in parameters:
                group = parameters.get('group', 'default')
                hoist_targets[group] += [pragma]

    if not hoist_regions:
        return

    # Group-by-group extract the regions and build the node replacement map
    hoist_map = {}
    promotion_vars_dims = {}  # Variables to promote with new dimension
    promotion_vars_index = {}  # Variable subscripts to promote with new indices
    starts, stops = [], []
    for group, regions in hoist_regions.items():
        if not group in hoist_targets or not hoist_targets[group]:
            raise RuntimeError(f'No region-hoist target for group {group} defined.')
        if len(hoist_targets[group]) > 1:
            raise RuntimeError(f'Multiple region-hoist targets given for group {group}')

        hoist_body = ()
        for start, stop in regions:
            parameters = get_pragma_parameters(start, starts_with='region-hoist')

            # Extract the region to hoist
            collapse = int(parameters.get('collapse', 0))
            if collapse > 0:
                scopes = FindScopes(start).visit(routine.body)[0]
                if len(scopes) <= collapse:
                    raise RuntimeError(f'Not enough enclosing scopes for collapse({collapse})')
                scopes = scopes[-(collapse+1):]
                region = NestedMaskedTransformer(start=start, stop=stop, mapper={start: None}).visit(scopes[0])

                # Promote variables given in promotion list
                loops = [scope for scope in scopes if isinstance(scope, Loop)]
                promote_vars = [var.strip().lower()
                                for var in get_pragma_parameters(start).get('promote', '').split(',') if var]
                promotion_vars_dims, promotion_vars_index = promotion_dimensions_from_loop_nest(
                    promote_vars, loops, promotion_vars_dims, promotion_vars_index)
            else:
                region = MaskedTransformer(start=start, stop=stop, mapper={start: None}).visit(routine.body)

            # Append it to the group's body, wrapped in comments
            begin_comment = Comment(f'! Loki {start.content}')
            end_comment = Comment(f'! Loki {stop.content}')
            hoist_body += as_tuple(flatten([begin_comment, region, end_comment]))

            # Register start and end nodes for transformer mask
            starts += [stop]
            stops += [start]

            # Replace end pragma by comment
            comment = Comment(f'! Loki {start.content} - region hoisted')
            hoist_map[stop] = comment

        # Insert target <-> hoisted regions into map
        hoist_map[hoist_targets[group][0]] = hoist_body

    routine.body = MaskedTransformer(active=True, start=starts, stop=stops, mapper=hoist_map).visit(routine.body)
    num_targets = sum(1 for pragma in hoist_map if 'target' in get_pragma_parameters(pragma))
    info('%s: hoisted %d region(s) in %d group(s)', routine.name, len(hoist_map) - num_targets, num_targets)
    promote_nonmatching_variables(routine, promotion_vars_dims, promotion_vars_index)
