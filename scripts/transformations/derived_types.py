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
    RangeIndex, as_tuple
)


__all__ = ['DerivedTypeArgumentsTransformation']


class DerivedTypeArgumentsTransformation(Transformation):
    """
    Transformation to remove derived types from subroutine signatures
    by replacing the relevant derived arguments with the sub-variables
    used in the called routine. The equivalent change is also applied
    to all callers of the transformed subroutines.

    Note, due to the dependency between caller and callee, this
    transformation should be applied atomically to sets of subroutine,
    if further transformations depend on the accurate signatures and
    call arguments.
    """

    def transform_subroutine(self, routine, **kwargs):
        # Determine role in bulk-processing use case
        task = kwargs.get('task', None)
        role = kwargs.get('role') if task is None else task.config['role']

        # Apply argument transformation, caller first!
        self.flatten_derived_args_caller(routine)
        if role == 'kernel':
            self.flatten_derived_args_routine(routine)

    @staticmethod
    def _derived_type_arguments(routine):
        """
        Find all derived-type arguments used in a given routine.

        :return: A map of ``arg => [type_vars]``, where ``type_var``
                 is a :class:`Variable` for each derived sub-variable
                 defined in the original compound type.
        """
        # Get all variables used in the kernel that have parents
        variables = FindVariables(unique=True).visit(routine.ir)
        variables = [v for v in variables if hasattr(v, 'parent') and v.parent is not None]
        candidates = defaultdict(list)

        for arg in routine.arguments:
            # Get the list of variables declared inside the derived type
            # (This property is None for non-derived type variables and empty
            # if we don't have the derived type definition available)
            arg_variables = as_tuple(arg.variables)
            if not arg_variables or all(not v.type.pointer and not v.type.allocatable for v in arg.variables):
                # Skip non-derived types or with no array members
                continue

            # Add candidate type variables, preserving order from the typedef
            arg_member_vars = set(v.basename.lower() for v in variables
                                  if v.parent.name.lower() == arg.name.lower())
            candidates[arg] += [v for v in arg.variables if v.basename.lower() in arg_member_vars]
        return candidates

    def flatten_derived_args_caller(self, caller):
        """
        Flatten all derived-type call arguments used in the target
        :class:`Subroutine` for all active :class:`CallStatement` nodes.

        The convention used is: ``derived%var => derived_var``.

        :param caller: The calling :class:`Subroutine`.
        """
        call_mapper = {}
        for call in FindNodes(CallStatement).visit(caller.body):
            if not call.not_active and call.routine:
                candidates = self._derived_type_arguments(call.routine)

                # Simultaneously walk caller and subroutine arguments
                new_arguments = list(call.arguments)
                for d_arg, k_arg in zip(call.arguments, call.routine.arguments):
                    if k_arg in candidates:
                        # Found derived-type argument, unroll according to candidate map
                        new_args = []
                        for type_var in candidates[k_arg]:
                            # Insert `:` range dimensions into newly generated args
                            new_dims = tuple(RangeIndex((None, None)) for _ in type_var.type.shape or [])
                            new_type = type_var.type.clone(parent=d_arg)
                            new_arg = type_var.clone(dimensions=new_dims, type=new_type,
                                                     parent=d_arg, scope=d_arg.scope)
                            new_args += [new_arg]

                        # Replace variable in dummy signature
                        # TODO: There's no cache anymore, maybe this can be changed?
                        # TODO: This is hacky, but necessary, as the variables
                        # from caller and callee don't cache, so we
                        # need to compare their string representation.
                        new_arg_strs = [str(a) for a in new_arguments]
                        i = new_arg_strs.index(str(d_arg))
                        new_arguments[i:i+1] = new_args

                # Set the new call signature on the IR ndoe
                call_mapper[call] = call.clone(arguments=as_tuple(new_arguments))

        # Rebuild the caller's IR tree
        caller.body = Transformer(call_mapper).visit(caller.body)

    def flatten_derived_args_routine(self, routine):
        """
        Unroll all derived-type arguments used in the subroutine
        signature, declarations and body.

        The convention used is: ``derived%var => derived_var``
        """
        candidates = self._derived_type_arguments(routine)

        # Callee: Establish replacements for declarations and dummy arguments
        new_arguments = list(routine.arguments)
        new_variables = list(routine.variables)
        for arg, type_vars in candidates.items():
            new_vars = []
            for type_var in type_vars:
                # Create a new variable with a new type mimicking the old one
                new_type = SymbolAttributes(type_var.type.dtype, kind=type_var.type.kind,
                                            intent=arg.type.intent, shape=type_var.type.shape)
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
