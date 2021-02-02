"""
Collection of utility routines that provide transformations for code regions.

"""
from collections import defaultdict

from loki import Scope, Subroutine, info
from loki.analyse import dataflow_analysis_attached
from loki.expression import (
    symbols as sym, FindTypedSymbols, SubstituteExpressions
)
from loki.ir import CallStatement, Comment, Import, Loop, Pragma, PragmaRegion, Section
from loki.pragma_utils import is_loki_pragma, get_pragma_parameters, pragma_regions_attached
from loki.tools import as_tuple, flatten
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
            raise RuntimeError('No region-hoist target for group {} defined.'.format(group))
        if len(hoist_targets[group]) > 1:
            raise RuntimeError('Multiple region-hoist targets given for group {}'.format(group))

        hoist_body = ()
        for start, stop in regions:
            parameters = get_pragma_parameters(start, starts_with='region-hoist')

            # Extract the region to hoist
            collapse = int(parameters.get('collapse', 0))
            if collapse > 0:
                scopes = FindScopes(start).visit(routine.body)[0]
                if len(scopes) <= collapse:
                    RuntimeError('Not enough enclosing scopes for collapse({})'.format(collapse))
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
            begin_comment = Comment('! Loki {}'.format(start.content))
            end_comment = Comment('! Loki {}'.format(stop.content))
            hoist_body += as_tuple(flatten([begin_comment, region, end_comment]))

            # Register start and end nodes for transformer mask
            starts += [stop]
            stops += [start]

            # Replace end pragma by comment
            comment = Comment('! Loki {} - region hoisted'.format(start.content))
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
    def _build_arg_list(region_args, intent, scope):
        """Helper routine to re-scope and clone arguments for call statement and new routine."""
        arguments, call_arguments = [], []
        for arg in region_args:
            call_arguments += [arg.clone(dimensions=None)]
            if isinstance(arg, sym.Array):
                amap = {v: v.clone(scope=scope) for v in FindTypedSymbols(unique=False).visit(arg.shape)}
                dim = sym.ArraySubscript(SubstituteExpressions(amap).visit(arg.shape))
                arguments += [arg.clone(scope=scope, type=arg.type.clone(intent=intent), dimensions=dim)]
            else:
                arguments += [arg.clone(scope=scope, type=arg.type.clone(intent=intent))]
        return arguments, call_arguments

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
                name = parameters.get('name', '{}_region_to_call_{}'.format(routine.name, counter))
                counter += 1

                # Use dataflow analysis to find in, out and inout variables to that region
                # (ignoring any symbols that are external imports)
                region_inout_args = region.uses_symbols & region.defines_symbols - imports
                region_in_args = region.uses_symbols - region.defines_symbols - imports
                region_out_args = region.defines_symbols - region.uses_symbols - imports

                region_var_map = {v.name.lower(): v for v in FindTypedSymbols().visit(region.body) if v not in imports}
                region_local_vars = set(region_var_map.values()) - region_in_args - region_out_args - region_inout_args

                # Extract arguments given in pragma annotations
                pragma_inout_args = {region_var_map[v.lower()] for v in parameters.get('inout', '').split(',') if v}
                pragma_in_args = {region_var_map[v.lower()] for v in parameters.get('in', '').split(',') if v}
                pragma_out_args = {region_var_map[v.lower()] for v in parameters.get('out', '').split(',') if v}

                # Override arguments according to pragma annotations
                region_inout_args = (region_inout_args - (pragma_in_args | pragma_out_args)) | pragma_inout_args
                region_in_args = (region_in_args - (pragma_inout_args | pragma_out_args)) | pragma_in_args
                region_out_args = (region_out_args - (pragma_in_args | pragma_inout_args)) | pragma_out_args

                # Copy body for external routine
                scope = Scope(parent=routine.scope.parent)
                body_map = {v: v.clone(scope=scope) for v in FindTypedSymbols(unique=False).visit(region.body)}
                body = Section(body=SubstituteExpressions(body_map).visit(region.body))

                # Copy imports from spec for external routine
                spec = Transformer().visit(FindNodes(Import).visit(routine.spec))
                spec_map = {v: v.clone(scope=scope) for v in FindTypedSymbols(unique=False).visit(spec)}
                spec = Section(body=SubstituteExpressions(spec_map).visit(spec))

                # Build the call signature
                in_args, call_in_args = _build_arg_list(region_in_args, 'in', scope)
                inout_args, call_inout_args = _build_arg_list(region_inout_args, 'inout', scope)
                out_args, call_out_args = _build_arg_list(region_out_args, 'out', scope)
                variables, _ = _build_arg_list(region_local_vars, None, scope)

                # Create the external routine and insert it into the list
                region_routine = Subroutine(name, spec=spec, body=body, scope=scope)
                region_routine.variables = variables
                region_routine.arguments = in_args + inout_args + out_args
                routines.append(region_routine)

                # Register start and end nodes in transformer mask for original routine
                starts += [region.pragma_post]
                stops += [region.pragma]

                # Replace end pragma by call in original routine
                call_arguments = call_in_args + call_inout_args + call_out_args
                call = CallStatement(name=name, arguments=call_arguments)
                mask_map[region.pragma_post] = call

    routine.body = MaskedTransformer(active=True, start=starts, stop=stops, mapper=mask_map).visit(routine.body)
    info('%s: converted %d region(s) to calls', routine.name, counter)

    return routines
