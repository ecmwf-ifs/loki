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
import re
from loki import (
    Transformation, FindVariables, FindNodes, Transformer,
    SubstituteExpressions, CallStatement, RangeIndex, as_tuple,
    BasicType, DerivedType, CaseInsensitiveDict, warning, debug,
    ProcedureDeclaration, recursive_expression_map_update
)
from transformations.scc_cuf import is_elemental


__all__ = ['DerivedTypeArgumentsExpansionAnalysis', 'DerivedTypeArgumentsExpansionTransformation']


class DerivedTypeArgumentsExpansionAnalysis(Transformation):
    """
    Analysis step for :any:`DerivedTypeArgumentsExpansionTransformation`

    This has to be applied over the call tree in reverse order before calling
    :any:`DerivedtypeArgumentsExpansionTransformation`, i.e. callees have to be visited
    before their callers, to ensure derived type arguments can be expanded
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
        """
        Build the :data:`expansion_map` for the given :data:`routine`

        This inspects the use of derived type variables inside the routine,
        building a map that contains for every derived type argument
        the list of members that are used from that derived type.

        The use of derived type members in called routines is taken into
        account via their respective expansion maps, provided in
        :data:`child_expansion_maps`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The routine for which to build the map
        child_expansion_maps : :any:`CaseInsensitiveDict` of (str, :any:`CaseInsensitiveDict`)
            Dictionary containing the expansion maps of every child routine

        Returns
        -------
        :any:`CaseInsensitiveDict` of (str, tuple)
            The expansion map, mapping derived type arguments to the members that
            need to be expanded
        """
        is_elemental_routine = is_elemental(routine)
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
            parent, expansion = cls.expand_derived_type_member(var, is_elemental_routine)
            if expansion:
                expansion_candidates[parent].add(expansion)

        # Add all expansion names from derived type unrolling in calls
        for call in FindNodes(CallStatement).visit(routine.body):
            # Determine correct call name, in case it has been renamed via import
            call_name = call.name.type.use_name or str(call.name)
            child_candidates = cls.call_expansion_candidates(
                call, child_expansion_maps.get(call_name, {}), is_elemental_routine
            )
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
                    base_member = [*candidate.parents, candidate][0]
                    members_to_expand[base_member.name.lower()] |= {candidate}

                expansion_map[arg.name] = as_tuple(sorted([
                    expansion_name
                    for var in arg_type.dtype.typedef.variables
                    for expansion_name in members_to_expand.get(var.name.lower(), [])
                ], key=lambda v: str(v).lower()))

        return expansion_map

    @staticmethod
    def expand_derived_type_member(var, is_elemental_routine):
        """
        Determine the member expansion for a derived type member variable

        For a given derived type member use :data:`var`, this determines
        the name of the root parent and the member expansion.

        A few examples to illustrate the behaviour, with Fortran variable use
        in the left column and return value of this routine on the right:

        .. code-block::
            var name            | return value (parent_name, expansion)  | remarks
           ---------------------+----------------------------------------+---------------
            SOME_VAR            | ('some_var', None)                     | No expansion
            SOME%VAR            | ('some', 'var')                        |
            ARRAY(5)%VAR        | ('array', None)                        | Cannot expand array of derived types
            SOME%NESTED%VAR     | ('some', 'nested%var')                 |
            NESTED%ARRAY(I)%VAR | ('nested', 'array')                    | Partial expansion

        For elemental routines, we may need to hoist subscript expressions to the caller side,
        in which case the corresponding index expression is included in the expansion field,
        for example:

        .. code-block::
            var name            | return value (parent_name, expansion)
           ---------------------+---------------------------------------
            SOME%VAR(IDX)       | ('some', 'var(idx)')

        Note, that this currently supports only index expressions where another argument
        is directly used as the index variable.

        Parameters
        ----------
        var : :any:`MetaSymbol`
            The use of a derived type member
        is_elemental_routine : bool
            Flag to indicate that the current routine is an ``ELEMENTAL`` routine

        Returns
        -------
        (str, str or NoneType)
        """
        parents = var.parents
        if not parents:
            return var.name.lower(), None

        if any(hasattr(p, 'dimensions') for p in parents):
            # We unroll the derived type member as far as possible, stopping at
            # the occurence of an intermediate derived type array
            # Note that we set scope=None, which detaches the symbol from the current
            # scope and stores the type information locally on the symbol. This makes
            # them available later on without risking losing this information due to
            # intermediate rescoping operations
            unrolled_parents = None
            for parent in parents[1:]:
                unrolled_parents = parent.clone(parent=unrolled_parents, scope=None, dimensions=None)
                if hasattr(parent, 'dimensions'):
                    break

            if not unrolled_parents:
                debug(f'Array of derived types {var!s}. Cannot expand argument.')
                return parents[0].name.lower(), None

            debug(f'Array of derived types {var!s}. Can only partially expand argument.')
            return parents[0].name.lower(), unrolled_parents

        unrolled_parents = None
        for parent in parents[1:]:
            unrolled_parents = parent.clone(parent=unrolled_parents, scope=None)

        if is_elemental_routine:
            # In elemental routines, we can have scalar derived type arguments with
            # array members, where the subscript operation is performed inside the elemental
            # routine. Because Fortran mandates (for good reasons!) scalar arguments in
            # elemental routines, we have to include the dimensions in the
            # argument expansion and hoist them to the caller side.
            unrolled_parents = var.clone(parent=unrolled_parents, scope=None)
        else:
            unrolled_parents = var.clone(parent=unrolled_parents, scope=None, dimensions=None)
        return parents[0].name.lower(), unrolled_parents

    @classmethod
    def call_expansion_candidates(cls, call, child_map, is_elemental_routine):
        """
        Create the call's argument list with derived type arguments expanded

        Parameters
        ----------
        call : :any:`CallStatement`
            The call statement to process
        expansion_map : :any:`CaseInsensitiveDict`
            Map of maps, specifying for each child routine the derived type arguments
            and the names of members of that derived type argument that need to be expanded.
        is_elemental_routine : bool
            Flag to indicate that the current routine is an ``ELEMENTAL`` routine

        Returns
        -------
        tuple :
            The list of derived type arguments that need to be expanded
        """
        expansion_candidates = defaultdict(set)

        # Can only expand on caller side
        if call.not_active or call.routine is BasicType.DEFERRED:
            for var in FindVariables().visit(call):
                parent, expansion = cls.expand_derived_type_member(var, is_elemental_routine)
                if expansion:
                    expansion_candidates[parent].add(expansion)
            return expansion_candidates

        for kernel_arg, caller_arg in call.arg_iter():
            if kernel_arg.name in child_map or hasattr(caller_arg, 'parent'):
                parent, expansion = cls.expand_derived_type_member(caller_arg, is_elemental_routine)

                # Check if this argument has been expanded further on the kernel side,
                # otherwise add to expansion candidates as is
                if kernel_arg.name in child_map and not hasattr(caller_arg, 'dimensions'):
                    for child in child_map[kernel_arg.name]:
                        unrolled_member = expansion
                        for child_parent in [*child.parents, child]:
                            if unrolled_member:
                                new_name = f'{unrolled_member.name}%{child_parent.name}'
                            else:
                                new_name = child_parent.name
                            unrolled_member = child_parent.clone(name=new_name, parent=unrolled_member, dimensions=None)

                        if unrolled_member:
                            expansion_candidates[parent].add(unrolled_member)
                elif expansion:
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

    _re_non_alphanum = re.compile(r'\W+')

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

    def expand_derived_args_caller(self, routine, child_expansion_maps):
        """
        Flatten all derived-type call arguments used in the provided :data:`routine`
        for all active :any:`CallStatement` nodes.

        The convention used is: ``derived%var => derived_var``.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The routine in which to transform call statements
        child_expansion_maps : :any:`CaseInsensitiveDict` of (str, :any:`CaseInsensitiveDict`)
            Dictionary containing the expansion maps of every child routine
        """
        call_mapper = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            if not call.not_active and call.routine is not BasicType.DEFERRED:
                # Determine correct call name, in case it has been renamed via import
                call_name = call.name.type.use_name or str(call.name)
                if call_name in child_expansion_maps:
                    # Set the new call signature on the IR node
                    expanded_arguments = self.expand_call_arguments(call, child_expansion_maps[call_name])
                    call_mapper[call] = call.clone(arguments=expanded_arguments)

        # Rebuild the routine's IR tree
        if call_mapper:
            routine.body = Transformer(call_mapper).visit(routine.body)

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
                    if getattr(member, 'dimensions', None):
                        # Extract all variables used in the index expression, and try
                        # to match them to arguments in the call signature
                        vmap = {}
                        try:
                            for var in FindVariables().visit(member.dimensions):
                                arg_index = call.routine.arguments.index(var)
                                vmap[var] = call.arguments[arg_index]
                        except ValueError as exc:
                            raise NotImplementedError(
                                'Transformation supports only kernel arguments as variables in index expressions'
                            ) from exc
                        dimensions = SubstituteExpressions(vmap).visit(member.dimensions)
                    else:
                        dimensions = None

                    # We build the name bit-by-bit to obtain nested derived type arguments
                    # relative to the local derived type variable
                    arg_member = caller_arg
                    for child in member.parents:
                        arg_member = child.clone(
                            name=f'{arg_member.name}%{child.name}', parent=arg_member,
                            scope=arg_member.scope
                        )
                    arg_member = member.clone(
                        name=f'{arg_member.name}%{member.name}', parent=arg_member,
                        scope=arg_member.scope, dimensions=dimensions
                    )

                    arguments += [arg_member]
            else:
                arguments += [caller_arg]

        return as_tuple(arguments)

    def expand_derived_args_routine(self, routine, expansion_map):
        """
        Unroll all derived-type arguments used in the subroutine
        signature, declarations and body.

        The convention used is to replace the ``%`` operator by an
        underscore ``_``, for example: ``derived%var => derived_var``.

        In elemental routines, where indexing may be hoisted to the caller,
        a running index is appended to such arguments to distinguish between
        multiple uses of the same member. For example:

        .. code-block::
           derived%var(idx) => derived_var_1
           derived%var(idx+1) => derived_var_2
        """
        is_elemental_routine = is_elemental(routine)
        arg_counter = defaultdict(int)
        arg_map = {}
        def _expanded_name(arg):
            if arg in arg_map:
                return arg_map[arg]

            new_name = arg.name.replace('%', '_')
            if is_elemental_routine and getattr(arg, 'dimensions', None):
                arg_counter[new_name] += 1
                new_name += f'_{arg_counter[new_name]!s}'
                arg_map[arg] = new_name
            return new_name

        # Build a map from derived type arguments to expanded arguments
        argument_map = {}
        typedefs = []
        for arg in routine.arguments:
            if arg.name in expansion_map:
                # Collect the typedefs corresponding to arguments
                if arg.type.dtype.typedef not in typedefs:
                    typedefs += [arg.type.dtype.typedef]

                new_args = []
                for member in expansion_map[arg.name]:
                    # Instantiate the expanded argument's non-expanded counterpart
                    arg_member = arg
                    for child in [*member.parents, member]:
                        # Note: We clone the child here to retain the dimensions
                        # from the declaration in the typedef. To not overwrite the stored type
                        # information from the routine's symbol table, we use type=None
                        arg_member = child.clone(name=f'{arg_member.name}%{child.name}', parent=arg_member, scope=routine, type=None)

                    if is_elemental_routine:
                        # Use same argument intent, dismiss all initializer and other array attributes
                        # and discard all dimensions
                        new_type = arg_member.type.clone(
                            intent=arg.type.intent, initial=None, allocatable=None, target=None, pointer=None,
                            shape=None
                        )
                        new_dims = None
                    else:
                        # Use same argument intent, dismiss any initializer, and insert `:` range dimensions
                        new_type = arg_member.type.clone(intent=arg.type.intent, initial=None)
                        new_dims = tuple(RangeIndex((None, None)) for _ in new_type.shape or [])

                    # Create the expanded argument
                    new_name = _expanded_name(arg_member)
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
                if is_elemental_routine:
                    vmap[var] = var.clone(name=_expanded_name(var), parent=None, type=None, dimensions=None)
                else:
                    vmap[var] = var.clone(name=_expanded_name(var), parent=None, type=None)

        vmap = recursive_expression_map_update(vmap)

        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
        routine.body = SubstituteExpressions(vmap).visit(routine.body)

        # Update procedure bindings by specifying NOPASS attribute
        for tdef in typedefs:
            for decl in tdef.declarations:
                if isinstance(decl, ProcedureDeclaration) and not decl.generic:
                    for proc in decl.symbols:
                        if routine.name == proc or routine.name in as_tuple(proc.type.bind_names):
                            proc.type = proc.type.clone(pass_attr=False)
