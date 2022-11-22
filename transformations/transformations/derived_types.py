# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformations dealing with derived types in subroutines and
derived-type arguments in complex calling structures.

 * DerivedTypeArgumentsTransformation:
        Transformation to resolve array-of-structure (AOS) uses of derived-type
        variables to explicitly expose arrays from which to hoist dimensions.
"""

from collections import defaultdict
from loki import (
    Transformation, FindVariables, FindNodes, Transformer,
    SubstituteExpressions, CallStatement, Variable, SymbolAttributes,
    RangeIndex, as_tuple, BasicType, DerivedType, CaseInsensitiveDict
)


__all__ = ['DerivedTypeArgumentsAnalysis', 'DerivedTypeArgumentsTransformation']


class DerivedTypeArgumentsAnalysis(Transformation):
    """
    Analysis step for :any:`DerivedTypeArgumentsTransformation`

    This has to be applied over the call tree in reverse order before calling
    :any:`DerivedtypeArgumentsTransformation`, i.e. callees have to be visited
    before their callerss, to ensure derived type arguments can be expanded
    across nested call trees.

    This analysis step collects for all derived type arguments the member
    variables used in routine, and annotates the argument's ``type`` with an
    ``expansion_names`` property.
    """

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs.get('role')

        if role == 'kernel':
            candidates = DerivedTypeArgumentsTransformation.used_derived_type_arguments(routine)

            for arg in routine.arguments:
                if arg in candidates:
                    expansion_names = [v.basename.lower() for v in candidates[arg]]
                    arg.type = arg.type.clone(expansion_names=as_tuple(expansion_names))


class DerivedTypeArgumentsTransformation(Transformation):
    """
    Transformation to remove derived types from subroutine signatures
    by replacing the relevant arguments with all used members of the type.

    The convention used is: ``derived%var => derived_var``.

    This requires a previous analysis step using
    :any;`DerivedTypeArgumentsAnalysis`.

    The transformation has to be applied in forward order, i.e. caller
    has to be processed before callee.
    """

    def transform_subroutine(self, routine, **kwargs):
        # Determine role in bulk-processing use case
        role = kwargs.get('role')

        # Apply caller transformation first to update call
        # signatures...
        self.flatten_derived_args_caller(routine)

        # ...before updating all other uses in the routine
        # and the routine's signature
        if role == 'kernel':
            self.flatten_derived_args_routine(routine)

    @staticmethod
    def expand_call_arguments(call):
        """
        Create the call's argument list with derived type arguments expanded

        This requires previous callee-side annotation via
        :any:`DerivedTypeArgumentsAnalysis`. All derived type arguments on
        callee-side that have an ``expand_names`` attribute on their type
        are replaced by the corresponding derived type members.

        Returns
        -------
        tuple :
            The argument list with derived type arguments expanded
        """
        arguments = []
        for kernel_arg, caller_arg in call.arg_iter():
            arg_type = kernel_arg.type
            if isinstance(arg_type.dtype, DerivedType) and arg_type.expansion_names:
                # Map of derived type members
                var_map = CaseInsensitiveDict(
                    (var.name, var) for var in arg_type.dtype.typedef.variables
                )

                # Found derived-type argument, unroll according to candidate map
                for var_name in arg_type.expansion_names:
                    type_var = var_map[var_name]
                    # Insert `:` range dimensions into newly generated args
                    new_dims = tuple(RangeIndex((None, None)) for _ in type_var.type.shape or [])
                    arguments += [Variable(
                        name=f'{caller_arg.name}%{var_name}', parent=caller_arg, dimensions=new_dims,
                        scope=caller_arg.scope
                    )]
            else:
                arguments += [caller_arg]

        return as_tuple(arguments)

    @classmethod
    def used_derived_type_arguments(cls, routine):
        """
        Find all derived-type arguments used in a given routine.

        Returns
        -------
        dict
            A map of ``arg => [type_vars]``, where ``type_var``
            is a :any:`Variable` for each derived sub-variable
            defined in the original compound type.
        """
        # Get all variables used in the kernel that have parents
        variables = FindVariables(unique=True).visit(routine.ir)
        variables = [v for v in variables if hasattr(v, 'parent') and v.parent is not None]

        # Get all expansion names from derived type unrolling in calls
        for call in FindNodes(CallStatement).visit(routine.body):
            if not call.not_active and call.routine is not BasicType.DEFERRED:
                variables += cls.expand_call_arguments(call)

        candidates = defaultdict(list)
        for arg in routine.arguments:
            # Get the list of variables declared inside the derived type
            # (This property is None for non-derived type variables and empty
            # if we don't have the derived type definition available)
            arg_variables = as_tuple(arg.variables)
            if not arg_variables or all(not v.type.pointer and not v.type.allocatable for v in arg_variables):
                # Skip non-derived types or with no array members
                continue

            # Add candidate type variables, preserving order from the typedef
            arg_member_vars = set(v.basename.lower() for v in variables if v.parent == arg.name)
            candidates[arg] += [v for v in arg_variables if v.basename.lower() in arg_member_vars]
        return candidates

    def flatten_derived_args_caller(self, caller):
        """
        Flatten all derived-type call arguments used in the target
        :any:`Subroutine` for all active :any:`CallStatement` nodes.

        The convention used is: ``derived%var => derived_var``.

        This requires the callee to have been transformed first.

        Parameters
        ----------
        caller : :any:`Subroutine`
            The routine in which to transform call statements
        """
        call_mapper = {}
        for call in FindNodes(CallStatement).visit(caller.body):
            if not call.not_active and call.routine is not BasicType.DEFERRED:
                # Set the new call signature on the IR node
                call_mapper[call] = call.clone(arguments=self.expand_call_arguments(call))

        # Rebuild the caller's IR tree
        caller.body = Transformer(call_mapper).visit(caller.body)

    def flatten_derived_args_routine(self, routine):
        """
        Unroll all derived-type arguments used in the subroutine
        signature, declarations and body.

        The convention used is: ``derived%var => derived_var``
        """
        candidates = self.used_derived_type_arguments(routine)

        # Callee: Establish replacements for declarations and dummy arguments
        new_arguments = list(routine.arguments)
        new_variables = list(routine.variables)
        for arg, type_vars in candidates.items():
            new_vars = []
            for type_var in type_vars:
                # Create a new variable with a new type mimicking the old one
                new_type = SymbolAttributes(type_var.type.dtype, kind=type_var.type.kind,
                                            intent=arg.type.intent or 'inout', shape=type_var.type.shape)
                new_name = f'{arg.name}_{type_var.basename}'
                new_dimensions = new_type.shape if new_type.shape else None
                new_var = Variable(name=new_name, type=new_type, dimensions=new_dimensions, scope=routine)
                new_vars += [new_var]

            # Replace variable in subroutine argument list
            i = new_arguments.index(arg)
            new_arguments[i:i+1] = new_vars

            # Also replace the variable in the variable list to
            # trigger the re-generation of the according declaration.
            i = new_variables.index(arg)
            new_variables[i:i+1] = new_vars

        # Apply replacements to routine by setting the properties
        routine.arguments = new_arguments
        routine.variables = new_variables

        # Create a variable substitution mapper and apply to body
        argnames = [arg.name.lower() for arg in candidates.keys()]
        variables = FindVariables(unique=False).visit(routine.body)
        variables = [v for v in variables
                     if hasattr(v, 'parent') and str(v.parent).lower() in argnames]
        # Note: The ``type=None`` prevents this clone from overwriting the type
        # we just derived above, as it would otherwise use whaterever type we
        # had derived previously (ie. the pointer type from the struct definition.)
        vmap = {v: v.clone(name=v.name.replace('%', '_'), parent=None, type=None)
                for v in variables}

        routine.body = SubstituteExpressions(vmap).visit(routine.body)
