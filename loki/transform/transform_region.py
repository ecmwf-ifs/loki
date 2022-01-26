"""
Collection of utility routines that provide transformations for code regions.

"""
from collections import defaultdict

from loki import Subroutine, info
from loki.analyse import dataflow_analysis_attached
from loki.expression import FindVariables, Variable
from loki.ir import CallStatement, Comment, Import, Loop, Pragma, PragmaRegion, Section
from loki.pragma_utils import is_loki_pragma, get_pragma_parameters, pragma_regions_attached
from loki.tools import as_tuple, flatten, CaseInsensitiveDict
from loki.transform.transform_array_indexing import (
    promotion_dimensions_from_loop_nest, promote_nonmatching_variables
)
from loki.visitors import (
    FindNodes, FindScopes, MaskedTransformer, NestedMaskedTransformer, Transformer
)

__all__ = ['region_hoist', 'region_to_call']


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
                    RuntimeError(f'Not enough enclosing scopes for collapse({collapse})')
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


def region_to_call(routine):
    """
    Convert regions annotated with ``!$loki region-to-call`` pragmas to subroutine calls.


    The pragma syntax for regions to convert to subroutines is
    ``!$loki region-to-call [name(...)] [in(...)] [out(...)] [inout(...)]``
    and ``!$loki end region-to-call``.

    A new subroutine is created with the provided name (or an auto-generated default name
    derived from the current subroutine name) and the content of the pragma region as body.

    Variables provided with the ``in``, ``out`` and ``inout`` options are used as
    arguments in the routine with the corresponding intent, all other variables used in this
    region are assumed to be local variables.

    The pragma region in the original routine is replaced by a call to the new subroutine.

    :param :class:``Subroutine`` routine:
        the routine to modify.

    :return: the list of newly created subroutines.

    """
    counter = 0
    routines, starts, stops = [], [], []
    imports = {var for imprt in FindNodes(Import).visit(routine.spec) for var in imprt.symbols}
    mask_map = {}
    with pragma_regions_attached(routine):
        with dataflow_analysis_attached(routine):
            for region in FindNodes(PragmaRegion).visit(routine.body):
                if not is_loki_pragma(region.pragma, starts_with='region-to-call'):
                    continue

                # Name the external routine
                parameters = get_pragma_parameters(region.pragma, starts_with='region-to-call')
                name = parameters.get('name', f'{routine.name}_region_to_call_{counter}')
                counter += 1

                # Create the external subroutine containing the routine's imports and the region's body
                spec = Section(body=Transformer().visit(FindNodes(Import).visit(routine.spec)))
                body = Section(body=Transformer().visit(region.body))
                region_routine = Subroutine(name, spec=spec, body=body)

                # Use dataflow analysis to find in, out and inout variables to that region
                # (ignoring any symbols that are external imports)
                region_in_args = region.uses_symbols - region.defines_symbols - imports
                region_inout_args = region.uses_symbols & region.defines_symbols - imports
                region_out_args = region.defines_symbols - region.uses_symbols - imports

                # Remove any parameters from in args
                region_in_args = {arg for arg in region_in_args if not arg.type.parameter}

                # Extract arguments given in pragma annotations
                region_var_map = CaseInsensitiveDict(
                    (v.name, v.clone(dimensions=None))
                    for v in FindVariables().visit(region.body) if v.clone(dimensions=None) not in imports
                )
                pragma_in_args = {region_var_map[v.lower()] for v in parameters.get('in', '').split(',') if v}
                pragma_inout_args = {region_var_map[v.lower()] for v in parameters.get('inout', '').split(',') if v}
                pragma_out_args = {region_var_map[v.lower()] for v in parameters.get('out', '').split(',') if v}

                # Override arguments according to pragma annotations
                region_in_args = (region_in_args - (pragma_inout_args | pragma_out_args)) | pragma_in_args
                region_inout_args = (region_inout_args - (pragma_in_args | pragma_out_args)) | pragma_inout_args
                region_out_args = (region_out_args - (pragma_in_args | pragma_inout_args)) | pragma_out_args

                # Now fix the order
                region_inout_args = as_tuple(region_inout_args)
                region_in_args = as_tuple(region_in_args)
                region_out_args = as_tuple(region_out_args)

                # Set the list of variables used in region routine (to create declarations)
                # and put all in the new scope
                region_routine_variables = {v.clone(dimensions=v.type.shape or None)
                                            for v in FindVariables().visit(region_routine.body)
                                            if v.name in region_var_map}
                region_routine.variables = as_tuple(region_routine_variables)
                region_routine.rescope_symbols()

                # Build the call signature
                region_routine_var_map = region_routine.variable_map
                region_routine_arguments = []
                for intent, args in zip(('in', 'inout', 'out'), (region_in_args, region_inout_args, region_out_args)):
                    for arg in args:
                        local_var = region_routine_var_map[arg.name]
                        local_var = local_var.clone(type=local_var.type.clone(intent=intent))
                        region_routine_var_map[arg.name] = local_var
                        region_routine_arguments += [local_var]

                # We need to update the list of variables again to avoid duplicate declarations
                region_routine.variables = as_tuple(region_routine_var_map.values())
                region_routine.arguments = as_tuple(region_routine_arguments)

                # insert into list of new routines
                routines.append(region_routine)

                # Register start and end nodes in transformer mask for original routine
                starts += [region.pragma_post]
                stops += [region.pragma]

                # Replace end pragma by call in original routine
                call_arguments = region_in_args + region_inout_args + region_out_args
                call = CallStatement(name=Variable(name=name), arguments=call_arguments)
                mask_map[region.pragma_post] = call

    routine.body = MaskedTransformer(active=True, start=starts, stop=stops, mapper=mask_map).visit(routine.body)
    info('%s: converted %d region(s) to calls', routine.name, counter)

    return routines
