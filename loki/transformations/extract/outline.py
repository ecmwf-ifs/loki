# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.analyse import dataflow_analysis_attached
from loki.expression import symbols as sym, Variable
from loki.ir import (
    CallStatement, Import, PragmaRegion, Section, FindNodes,
    FindVariables, Transformer, is_loki_pragma,
    get_pragma_parameters, pragma_regions_attached
)
from loki.logging import info
from loki.subroutine import Subroutine
from loki.tools import as_tuple
from loki.types import BasicType, DerivedType



__all__ = ['outline_region', 'outline_pragma_regions']


def order_variables_by_type(variables, imports=None):
    """
    Apply a default ordering to variables based on their type, so that
    their use in declaration lists is unified.
    """
    variables = sorted(variables, key=str)  # Lexicographical base order

    derived = tuple(
        v for v in variables
        if isinstance(v.type.dtype, DerivedType) or v.type.dtype == BasicType.DEFERRED
    )

    if imports:
        # Order derived types by the order of their type in imports
        imported_symbols = tuple(s for i in imports for s in i.symbols if not i.c_import)
        derived = tuple(sorted(derived, key=lambda x: imported_symbols.index(x.type.dtype.name)))

    # Order declarations by type and put arrays before scalars
    non_derived = tuple(v for v in variables if v not in derived)
    arrays = tuple(v for v in non_derived if isinstance(v, sym.Array))
    scalars = tuple(v for v in non_derived if isinstance(v, sym.Scalar))
    assert len(derived) + len(arrays) + len(scalars) == len(variables)

    return derived + arrays + scalars


def outline_region(region, name, imports, intent_map=None):
    """
    Creates a new :any:`Subroutine` object from a given :any:`PragmaRegion`.

    Parameters
    ----------
    region : :any:`PragmaRegion`
        The region that holds the body for which to create a subroutine.
    name : str
        Name of the new subroutine
    imports : tuple of :any:`Import`, optional
        List of imports to replicate in the new subroutine
    intent_map : dict, optional
        Mapping of instent strings to list of variables to override intents

    Returns
    -------
    tuple of :any:`CallStatement` and :any:`Subroutine`
        The newly created call and respectice subroutine.
    """
    intent_map = intent_map or {}
    imports = as_tuple(imports)
    imported_symbols = {var for imp in imports for var in imp.symbols}
    # Special-case for IFS-style C-imports
    imported_symbols |= {
        str(imp.module).split('.', maxsplit=1)[0] for imp in imports if imp.c_import
    }

    # Create the external subroutine containing the routine's imports and the region's body
    spec = Section(body=imports)
    body = Section(body=Transformer().visit(region.body))
    region_routine = Subroutine(name, spec=spec, body=body)

    # Filter derived-type component accesses and only use the root parent
    region_uses_symbols = {s.parents[0] if s.parent else s for s in region.uses_symbols}
    region_defines_symbols = {s.parents[0] if s.parent else s for s in region.defines_symbols}

    # Use dataflow analysis to find in, out and inout variables to that region
    # (ignoring any symbols that are external imports)
    region_in_args = region_uses_symbols - region_defines_symbols - imported_symbols
    region_inout_args = region_uses_symbols & region_defines_symbols - imported_symbols
    region_out_args = region_defines_symbols - region_uses_symbols - imported_symbols

    # Remove any parameters from in args
    region_in_args = {arg for arg in region_in_args if not arg.type.parameter}

    # Extract arguments given in pragma annotations
    pragma_in_args = {v.clone(scope=region_routine) for v in intent_map['in']}
    pragma_inout_args = {v.clone(scope=region_routine) for v in intent_map['inout']}
    pragma_out_args = {v.clone(scope=region_routine) for v in intent_map['out']}

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
    region_routine_variables = tuple(
        v.clone(dimensions=v.type.shape or None, scope=region_routine)
        for v in FindVariables().visit(region.body)
        if v.clone(dimensions=None) not in imported_symbols
    )
    # Filter out derived-type component variables from declarations
    region_routine_variables = tuple(
        v.parents[0] if v.parent else v for v in region_routine_variables
    )

    # Build the call signature
    region_routine_var_map = {v.name: v for v in region_routine_variables}
    region_routine_arguments = []
    for intent, args in zip(('in', 'inout', 'out'), (region_in_args, region_inout_args, region_out_args)):
        for arg in args:
            local_var = region_routine_var_map.get(arg.name, arg)
            # Sanitise argument types
            local_var = local_var.clone(
                type=local_var.type.clone(intent=intent, allocatable=None, target=None),
                scope=region_routine
            )

            region_routine_var_map[arg.name] = local_var
            region_routine_arguments += [local_var]

    # Order the arguments and local declaration lists and put arguments first
    region_routine_locals = tuple(
        v for v in region_routine_variables if not v in region_routine_arguments
    )
    region_routine_arguments = order_variables_by_type(region_routine_arguments, imports=imports)
    region_routine_locals = order_variables_by_type(region_routine_locals, imports=imports)

    region_routine.variables = region_routine_arguments + region_routine_locals
    region_routine.arguments = region_routine_arguments

    # Ensure everything has been rescoped
    region_routine.rescope_symbols()

    # Create the call according to the wrapped code region
    call_arg_map = {v.name: v for v in region_in_args + region_inout_args + region_out_args}
    call_arguments = tuple(call_arg_map[a.name] for a in region_routine_arguments)
    call = CallStatement(name=Variable(name=name), arguments=call_arguments, kwarguments=())

    return call, region_routine


def outline_pragma_regions(routine):
    """
    Convert regions annotated with ``!$loki outline`` pragmas to subroutine calls.

    The pragma syntax for regions to convert to subroutines is
    ``!$loki outline [name(...)] [in(...)] [out(...)] [inout(...)]``
    and ``!$loki end outline``.

    A new subroutine is created with the provided name (or an auto-generated default name
    derived from the current subroutine name) and the content of the pragma region as body.

    Variables provided with the ``in``, ``out`` and ``inout`` options are used as
    arguments in the routine with the corresponding intent, all other variables used in this
    region are assumed to be local variables.

    The pragma region in the original routine is replaced by a call to the new subroutine.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The routine from which to extract marked pragma regions.

    Returns
    -------
    list of :any:`Subroutine`
        the list of newly created subroutines.
    """
    counter = 0
    routines = []
    imports = routine.imports
    parent_vmap = routine.variable_map
    mapper = {}
    with pragma_regions_attached(routine):
        with dataflow_analysis_attached(routine):
            for region in FindNodes(PragmaRegion).visit(routine.body):
                if not is_loki_pragma(region.pragma, starts_with='outline'):
                    continue

                # Name the external routine
                parameters = get_pragma_parameters(region.pragma, starts_with='outline')
                name = parameters.get('name', f'{routine.name}_outlined_{counter}')
                counter += 1

                # Extract explicitly requested symbols from context
                intent_map = {}
                intent_map['in'] = tuple(parent_vmap[v] for v in parameters.get('in', '').split(',') if v)
                intent_map['inout'] = tuple(parent_vmap[v] for v in parameters.get('inout', '').split(',') if v)
                intent_map['out'] = tuple(parent_vmap[v] for v in parameters.get('out', '').split(',') if v)

                call, region_routine = outline_region(region, name, imports, intent_map=intent_map)

                # insert into list of new routines
                routines.append(region_routine)

                # Replace region by call in original routine
                mapper[region] = call

            routine.body = Transformer(mapper=mapper).visit(routine.body)
    info('%s: converted %d region(s) to calls', routine.name, counter)

    return routines
