# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.analyse import dataflow_analysis_attached
from loki.expression import Variable
from loki.ir import (
    CallStatement, Import, PragmaRegion, Section, FindNodes,
    FindVariables, MaskedTransformer, Transformer, is_loki_pragma,
    get_pragma_parameters, pragma_regions_attached
)
from loki.logging import info
from loki.subroutine import Subroutine
from loki.tools import as_tuple, CaseInsensitiveDict


__all__ = ['extract_marked_subroutines']


def extract_marked_subroutines(routine):
    """
    Convert regions annotated with ``!$loki extract`` pragmas to subroutine calls.


    The pragma syntax for regions to convert to subroutines is
    ``!$loki extract [name(...)] [in(...)] [out(...)] [inout(...)]``
    and ``!$loki end extract``.

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
                if not is_loki_pragma(region.pragma, starts_with='extract'):
                    continue

                # Name the external routine
                parameters = get_pragma_parameters(region.pragma, starts_with='extract')
                name = parameters.get('name', f'{routine.name}_extracted_{counter}')
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
