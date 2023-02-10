# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformations dealing with derived types in subroutines and
derived-type arguments in complex calling structures.

 * DerivedTypeArgumentsExpansionTransformation:
        Transformation to resolve array-of-structure (AOS) uses of derived-type
        variables to explicitly expose arrays from which to hoist dimensions.
"""

from collections import defaultdict
from loki import (
    Transformation, FindVariables, FindNodes, Transformer,
    SubstituteExpressions, CallStatement, Variable,
    RangeIndex, as_tuple, BasicType, DerivedType, CaseInsensitiveDict,
    warning, debug
)


__all__ = ['DerivedTypeArgumentsExpansionAnalysis', 'DerivedTypeArgumentsExpansionTransformation']


class DerivedTypeArgumentsExpansionAnalysis(Transformation):
    """
    Analysis step for :any:`DerivedTypeArgumentsExpansionTransformation`

    This has to be applied over the call tree in reverse order before calling
    :any:`DerivedtypeArgumentsExpansionTransformation`, i.e. callees have to be visited
    before their callerss, to ensure derived type arguments can be expanded
    across nested call trees.

    This analysis step collects for all derived type arguments the member
    variables used in routine, and annotates the argument's ``type`` with an
    ``expansion_names`` property.
    """

    _key = 'DerivedTypeArgumentsExpansionTransformation'

    def __init__(self, key=None, **kwargs):
        if key is not None:
            self._key = key
        super().__init__(**kwargs)

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs.get('role')
        item = kwargs.get('item')

        # Bail out if the current subroutine is not part of the call tree
        if item and item.local_name != routine.name.lower():
            return

        # Initialize the transformation data dictionary
        item.trafo_data[self._key] = {}

        if role == 'kernel':
            successors = [child for child in kwargs.get('successors', []) if self._key in child.trafo_data]
            child_expansion_maps = CaseInsensitiveDict(
                (child.local_name, child.trafo_data[self._key]['expansion_map'])
                for child in successors
            )
            item.trafo_data[self._key]['expansion_map'] = self.routine_expansion_map(item.routine, child_expansion_maps)

    @classmethod
    def routine_expansion_map(cls, routine, child_expansion_maps):
        expansion_candidates = defaultdict(set)

        # Add all variables used in the routine that have parents
        # NB: We do this IR-node by IR-node, because we need to handle call statements
        # separately to avoid adding here members that are flattened later on
        candidates = {
            var
            for node, var_list in FindVariables(recurse_to_parent=False, with_ir_node=True).visit(routine.ir)
            if not isinstance(node, CallStatement)
            for var in var_list if var.parent is not None
        }
        for var in candidates:
            parent, expansion = cls.expand_derived_type_member(var)
            if expansion:
                expansion_candidates[parent].add(expansion)

        # Add all expansion names from derived type unrolling in calls
        for call in FindNodes(CallStatement).visit(routine.body):
            child_candidates = cls.call_expansion_candidates(call, child_expansion_maps.get(str(call.name), {}))
            for parent, expansion in child_candidates.items():
                expansion_candidates[parent] |= expansion

        # Build the final expansion map for derived type arguments only using alphabetic order
        expansion_map = CaseInsensitiveDict()
        for arg in routine.arguments:
            if arg in expansion_candidates:
                arg_type = arg.type
                if not isinstance(arg_type.dtype, DerivedType) or arg_type.dtype.typedef is BasicType.DEFERRED:
                    warning(f'Type definition not attached for {arg.name}. Cannot expand arguments.')
                    continue

                # To accomodate nested derived types, we build a list of expansions for each
                # member of the derived type
                members_to_expand = defaultdict(set)
                for candidate in expansion_candidates[arg]:
                    members_to_expand[candidate.split('%')[0].lower()] |= {candidate}

                expansion_map[arg.name] = as_tuple(sorted([
                    expansion_name
                    for var in arg_type.dtype.typedef.variables
                    for expansion_name in members_to_expand.get(var.name.lower(), [])
                ]))

        return expansion_map

    @staticmethod
    def expand_derived_type_member(var):
        parents = var.parents
        if not parents:
            return var.name.lower(), None

        if any(hasattr(p, 'dimensions') for p in parents):
            # We unroll the derived type member as far as possible, stopping at
            # the occurence of an intermediate derived type array
            unrolled_parents = []
            for parent in parents[1:]:
                unrolled_parents += [parent]
                if hasattr(parent, 'dimensions'):
                    break

            if not unrolled_parents:
                debug(f'Array of derived types {var!s}. Cannot expand argument.')
                return parents[0].name.lower(), None

            debug(f'Array of derived types {var!s}. Can only partially expand argument.')
            return parents[0].name.lower(), ('%'.join(p.name for p in unrolled_parents))

        return parents[0].name.lower(), '%'.join([p.name for p in parents[1:]] + [var.basename]).lower()

    @classmethod
    def call_expansion_candidates(cls, call, child_map):
        """
        Create the call's argument list with derived type arguments expanded

        Parameters
        ----------
        call : :any:`CallStatement`
            The call statement to process
        expansion_map : :any:`CaseInsensitiveDict`
            Map of maps, specifying for each child routine the derived type arguments
            and the names of members of that derived type argument that need to be expanded.

        Returns
        -------
        tuple :
            The list of derived type arguments that need to be expanded
        """
        expansion_candidates = defaultdict(set)

        # Can only expand on caller side
        if call.not_active or call.routine is BasicType.DEFERRED:
            for var in FindVariables().visit(call):
                parent, expansion = cls.expand_derived_type_member(var)
                if expansion:
                    expansion_candidates[parent].add(expansion)
            return expansion_candidates

        for kernel_arg, caller_arg in call.arg_iter():
            if kernel_arg.name in child_map or getattr(caller_arg, 'parent'):
                parent, expansion = cls.expand_derived_type_member(caller_arg)

                if expansion:
                    # Check if this argument has been expanded further on the kernel side,
                    # otherwise add to expansion candidates as is
                    if kernel_arg.name in child_map and not hasattr(caller_arg, 'dimensions'):
                        expansion_candidates[parent] |= {
                            f'{expansion}%{v}' for v in child_map[kernel_arg.name]
                        }
                    else:
                        expansion_candidates[parent].add(expansion)

        return expansion_candidates


class DerivedTypeArgumentsExpansionTransformation(Transformation):
    """
    Transformation to remove derived types from subroutine signatures
    by replacing the relevant arguments with all used members of the type.

    The convention used is: ``derived%var => derived_var``.

    This requires a previous analysis step using
    :any;`DerivedTypeArgumentsAnalysis`.

    The transformation has to be applied in forward order, i.e. caller
    has to be processed before callee.
    """

    _key = 'DerivedTypeArgumentsExpansionTransformation'

    def __init__(self, key=None, **kwargs):
        if key is not None:
            self._key = key
        super().__init__(**kwargs)

    def transform_subroutine(self, routine, **kwargs):
        # Determine role in bulk-processing use case
        role = kwargs.get('role')
        item = kwargs.get('item')

        # Bail out if the current subroutine is not part of the call tree
        if item and item.local_name != routine.name.lower():
            return

        successors = [child for child in kwargs.get('successors', []) if self._key in child.trafo_data]
        child_expansion_maps = CaseInsensitiveDict(
            (child.local_name, child.trafo_data[self._key]['expansion_map'])
            for child in successors
        )

        # Apply caller transformation first to update call
        # signatures...
        self.expand_derived_args_caller(routine, child_expansion_maps)

        # ...before updating all other uses in the routine
        # and the routine's signature
        if role == 'kernel':
            self.expand_derived_args_routine(routine, item.trafo_data[self._key]['expansion_map'])

    def expand_derived_args_caller(self, caller, child_expansion_maps):
        """
        Flatten all derived-type call arguments used in the target
        :any:`Subroutine` for all active :any:`CallStatement` nodes.

        The convention used is: ``derived%var => derived_var``.

        Parameters
        ----------
        caller : :any:`Subroutine`
            The routine in which to transform call statements
        """
        call_mapper = {}
        for call in FindNodes(CallStatement).visit(caller.body):
            if not call.not_active and call.routine is not BasicType.DEFERRED:
                if str(call.name) not in child_expansion_maps:
                    continue
                # Set the new call signature on the IR node
                expanded_arguments = self.expand_call_arguments(call, child_expansion_maps[str(call.name)])
                call_mapper[call] = call.clone(arguments=expanded_arguments)

        # Rebuild the caller's IR tree
        if call_mapper:
            caller.body = Transformer(call_mapper).visit(caller.body)

    @staticmethod
    def expand_call_arguments(call, expansion_map):
        """
        Create the call's argument list with derived type arguments expanded

        Parameters
        ----------
        call : :any:`CallStatement`
            The call statement to process
        expansion_map : :any:`CaseInsensitiveDict`
            Map of derived type names to names of members that need to be expanded.

        Returns
        -------
        tuple :
            The argument list with derived type arguments expanded
        """
        arguments = []
        for kernel_arg, caller_arg in call.arg_iter():
            if kernel_arg.name in expansion_map:
                arg_type = kernel_arg.type
                if not isinstance(arg_type.dtype, DerivedType) or arg_type.dtype.typedef is BasicType.DEFERRED:
                    warning(f'Type definition wrong or not attached for {caller_arg.name}. Cannot expand arguments.')
                    continue

                # Found derived-type argument, unroll according to candidate map
                for member in expansion_map[kernel_arg.name]:
                    # We build the name bit-by-bit for nested derived type arguments
                    arg_member = caller_arg
                    for n in member.split('%'):
                        arg_member = Variable(name=f'{arg_member.name}%{n}', parent=arg_member, scope=arg_member.scope)
                    arguments += [arg_member]
            else:
                arguments += [caller_arg]

        return as_tuple(arguments)

    def expand_derived_args_routine(self, routine, expansion_map):
        """
        Unroll all derived-type arguments used in the subroutine
        signature, declarations and body.

        The convention used is: ``derived%var => derived_var``
        """
        # Build a map from derived type arguments to expanded arguments
        argument_map = {}
        for arg in routine.arguments:
            if arg.name in expansion_map:
                new_args = []
                for member in expansion_map[arg.name]:
                    # Instantiate the expanded argument's non-expanded counterpart
                    arg_member = arg
                    for n in member.split('%'):
                        arg_member = Variable(name=f'{arg_member.name}%{n}', parent=arg_member, scope=routine)

                    # Use same argument intent and insert `:` range dimensions
                    new_type = arg_member.type.clone(intent=arg.type.intent)
                    new_dims = tuple(RangeIndex((None, None)) for _ in new_type.shape or [])

                    # Create the expanded argument
                    new_name = arg_member.name.replace('%', '_')
                    new_args += [arg_member.clone(name=new_name, parent=None, type=new_type, dimensions=new_dims)]

                argument_map[arg] = as_tuple(new_args)

        # Update arguments list
        routine.arguments = [a for arg in routine.arguments for a in argument_map.get(arg, [arg])]

        # Update variable list, too, as this triggers declaration generation
        routine.variables = [v for var in routine.variables for v in argument_map.get(var, [var])]

        # Substitue all use in the routine spec and body
        argnames = [arg.name.lower() for arg in argument_map]

        vmap = {}
        for var in FindVariables(unique=False).visit(routine.ir):
            parents = var.parents
            if parents and parents[0] in argnames and not any(hasattr(v, 'dimensions') for v in parents):
                # Note: The ``type=None`` prevents this clone from overwriting the type
                # we just derived above, as it would otherwise use whatever type we
                # had derived previously (ie. the type from the struct definition.)
                vmap[var] = var.clone(name=var.name.replace('%', '_'), parent=None, type=None)


        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
        routine.body = SubstituteExpressions(vmap).visit(routine.body)
