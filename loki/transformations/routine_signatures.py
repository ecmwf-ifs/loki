# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utilities and transformations altering routine signatures.
"""

import os
import itertools as it
from loki.batch import Transformation, ProcedureItem
from loki.ir import (
    VariableDeclaration, FindVariables,
    Transformer, FindNodes, CallStatement,
    SubstituteExpressions
)
from loki.tools import as_tuple, flatten
from loki.types import BasicType

__all__ = ['RemoveDuplicateArgs', 'remove_duplicate_args_from_calls',
           'modify_variable_declarations']


class RemoveDuplicateArgs(Transformation):
    """
    Transformation to remove duplicate arguments for both caller
    and callee. 

    .. warning::
        this won't work properly for multiple calls to the same routine
        with differing duplicate arguments

    Parameters
    ----------
    recurse_to_kernels : bool, optional
        Remove duplicate arguments only at the driver level or recurse to
        (nested) kernels (Default: `True`).
    rename_common : bool, optional
        Try to rename dummy arguments in called routines that received the same argument
        on the caller side, by finding a common name pattern in those names (Default: `False`).
    """

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, recurse_to_kernels=True, rename_common=False):
        self.recurse_to_kernels = recurse_to_kernels
        self.rename_common = rename_common

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs['role']
        if role == 'driver' or self.recurse_to_kernels:
            remove_duplicate_args_from_calls(routine, rename_common=self.rename_common)

def remove_duplicate_args_from_calls(routine, rename_common=False):
    """
    Utility to remove duplicate arguments from calls in :data:`routine`
    
    This updates the calls as well as the called routines. It requires calls
    to be enriched with interprocedural information.

    .. warning::
        this won't work properly for multiple calls to the same routine
        with differing duplicate arguments

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine where calls should be transformed.
    rename_common : bool, optional
        Try to rename dummy arguments in called routines that received the same argument
        on the caller side, by finding a common name pattern in those names (Default: `False`).
    """

    def remove_duplicate_args_call(call):
        arg_map = {}
        for routine_arg, call_arg in call.arg_iter():
            arg_map.setdefault(call_arg, []).append(routine_arg)
        # filter duplicate kwargs (comparing to the other kwarguments)
        _new_kwargs = as_tuple(list(kw_vals)[0] for g, kw_vals in it.groupby(call.kwarguments, key=lambda x: x[1]))
        # filter duplicate kwargs (comparing to the arguments)
        new_kwargs = tuple(kwarg for kwarg in _new_kwargs if kwarg[1] not in call.arguments)
        # (filter duplicate arguments and) update call
        call._update(arguments=as_tuple(dict.fromkeys(call.arguments)), kwarguments=new_kwargs)
        return arg_map

    def modify_callee(callee, callee_arg_map):

        def allowed_rename(routine, rename):
            # check whether rename is already "used" in routine
            if rename in routine.arguments or rename in routine.variables:
                return False
            return True

        combine = [routine_args for call_arg, routine_args in callee_arg_map.items() if len(routine_args) > 1]
        if rename_common:
            matches = [
                os.path.commonprefix([str(elem.name) for elem in args]).rstrip('_') or
                os.path.commonprefix([str(elem.name)[::-1] for elem in args]).rstrip('_')[::-1]
                for args in combine
            ]
            rename_common_map = {c[0].name: m for c, m in zip(combine, matches) if m}
            # check whether found rename is already "used" in routine
            unallowed_renames = ()
            for name, rename in rename_common_map.items():
                if not allowed_rename(callee, rename):
                    unallowed_renames += (name,)
            # and if already "used", remove and use instead default
            for key in unallowed_renames:
                del rename_common_map[key]
        else:
            rename_common_map = {}
        redundant = flatten([routine_args[1:] for routine_args in combine])
        combine_map = {routine_args[0]: as_tuple(routine_args[1:]) for routine_args in combine}
        arg_map = {arg.name: rename_common_map.get(common_arg.name, common_arg.name)
                   for common_arg, redundant_args in combine_map.items() for arg in redundant_args}
        # remove duplicates from callee.arguments
        new_routine_args = tuple(arg for arg in callee.arguments if arg not in redundant)
        # rename if common name is possible
        new_routine_args = as_tuple(arg.clone(name=rename_common_map[arg.name])
                if arg.name in rename_common_map else arg for arg in new_routine_args)
        callee.arguments = new_routine_args

        # rename usage/occurences in callee.body
        var_map = {}
        variables = FindVariables(unique=False).visit(callee.body)
        var_map = {var: var.clone(name=arg_map[var.name]) for var in variables if var.name in arg_map}
        var_map.update({var: var.clone(name=rename_common_map[var.name]) for var in variables
            if var.name in rename_common_map})
        callee.body = SubstituteExpressions(var_map).visit(callee.body)
        # modify the variable declarations, thus remove redundant variable declarations and possibly rename
        modify_variable_declarations(callee, remove_symbols=redundant, rename_symbols=rename_common_map)
        # store the information for possibly later renaming kwarguments on caller side
        return rename_common_map

    def rename_kwarguments(relevant_calls, rename_common_map_routine):
        for call in relevant_calls:
            kwarguments = call.kwarguments
            if kwarguments:
                call_name = str(call.routine.name).lower()
                new_kwargs = as_tuple((rename_common_map_routine[call_name][kw[0]], kw[1])
                        if kw[0] in rename_common_map_routine[call_name] else kw for kw in kwarguments)
                call._update(kwarguments=new_kwargs)

    calls = FindNodes(CallStatement).visit(routine.body)
    call_arg_map = {}
    relevant_calls = []
    # adapt call statements (and remove duplicate args/kwargs)
    for call in calls:
        if call.routine is BasicType.DEFERRED:
            continue
        call_arg_map[call.routine] = remove_duplicate_args_call(call)
        relevant_calls.append(call)
    rename_common_map_routine = {}
    # modify/adapt callees
    for callee, callee_arg_map in call_arg_map.items():
        rename_common_map_routine[str(callee.name).lower()] = modify_callee(callee, callee_arg_map)
    # handle possibly renamed kwarguments on caller side
    if rename_common:
        rename_kwarguments(relevant_calls, rename_common_map_routine)


def modify_variable_declarations(routine, remove_symbols=(), rename_symbols=None):
    """
    Utility to modify variable declarations by either removing symbols or renaming
    symbols.

    .. note::
        This utility only works on the variable declarations itself and
        won't modify variable/symbol usages elsewhere!

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine to be transformed.
    remove_symbols : list, tuple
        List of symbols for which their declaration should be removed.
    rename_symbols : dict
        Dict/Map of symbols for which their declaration should be renamed.
    """
    rename_symbols = rename_symbols if rename_symbols is not None else {}
    var_decls = FindNodes(VariableDeclaration).visit(routine.spec)
    remove_symbol_names = [var.name.lower() for var in remove_symbols]
    decl_map = {}
    already_declared = ()
    for decl in var_decls:
        symbols = [symbol for symbol in decl.symbols if symbol.name.lower() not in remove_symbol_names]
        symbols = [symbol.clone(name=rename_symbols[symbol.name])
                if symbol.name in rename_symbols else symbol for symbol in symbols]
        symbols = [symbol for symbol in symbols if not symbol.name.lower() in already_declared]
        already_declared += tuple(symbol.name.lower() for symbol in symbols)
        if symbols and symbols != decl.symbols:
            decl_map[decl] = decl.clone(symbols=as_tuple(symbols))
        else:
            if not symbols:
                decl_map[decl] = None
    routine.spec = Transformer(decl_map).visit(routine.spec)
