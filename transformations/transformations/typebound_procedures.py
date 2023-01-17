from collections import defaultdict
from loki import (
    Transformation, FindNodes, CallStatement, Transformer, BasicType, Variable,
    Module, warning, as_tuple, Import
)

__all__ = ['TypeboundProcedureCallTransformation']


class TypeboundProcedureCallTransformation(Transformation):
    """
    Replace calls to type-bound procedures with direct calls to the
    corresponding subroutines/functions

    Instead of calling a type-bound procedure, e.g. ``CALL my_type%proc``,
    it is possible to import the bound procedure and call it directly, with
    the derived type as first argument, i.e. ``CALL proc(my_type)``.
    This transformation replaces all calls to type-bound procedures accordingly
    and inserts necessary imports.

    Also, for some compilers these direct calls seem to require an explicit
    ``INTENT`` specification on the polymorphic derived type dummy argument,
    which is set to `INOUT` by default, if missing.
    """

    def apply_default_polymorphic_intent(self, routine):
        """
        Utility routine to set a default ``INTENT(INOUT)`` on polymorphic dummy
        arguments (i.e. declared via ``CLASS``) that don't have an explicit intent
        """
        for arg in routine.arguments:
            type_ = arg.type
            if type_.polymorphic and not type_.intent:
                arg.type = type_.clone(intent='inout')

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply the transformation of calls to the given :data:`routine`
        """
        self.apply_default_polymorphic_intent(routine)

        new_subroutine_imports = defaultdict(set)
        call_mapper = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            if not call.name.parent:
                continue

            # Get the parent variable corresponding to the declared variable (i.e. outermost parent)
            parent = call.name.parent
            while parent.parent is not None:
                parent = parent.parent

            if call.name.type.bind_names is not None:
                new_name = call.name.type.bind_names[0]
            elif parent.type.dtype.typedef is not BasicType.DEFERRED:
                # Fiddle our way through derived type nesting until we obtain the symbol corresponding
                # to the procedure-binding in the TypeDef
                local_parent = None
                local_var = parent
                try:
                    for local_name in call.name.name_parts[1:]:
                        local_parent = local_var
                        local_var = local_var.type.dtype.typedef.variable_map[local_name]
                except AttributeError:
                    warning('Type definitions incomplete for %s in %s', call.name, routine.name)
                    continue

                if local_var.type.dtype.is_generic:
                    warning('Cannot resolve generic binding %s (not implemented) in %s', call.name, routine.name)
                    continue

                if local_var.type.bind_names is not None:
                    # Although this should have ben taken care of by the first if branch,
                    # this may trigger here when the bind_names property hasn't been imported
                    # into the local symbol table
                    new_name = local_var.type.bind_names[0]
                else:
                    # If the binding doesn't have any specific bind_names, this means the
                    # corresponding subroutine has the same name and should be declared
                    # in the same module as the typedef
                    new_name = Variable(name=local_var.name, scope=local_parent.type.dtype.typedef.parent)
            else:
                # We don't have any binding information available
                continue

            # Mark the call for replacement
            new_arguments = (call.name.parent,) + call.arguments
            call_mapper[call] = call.clone(name=new_name.rescope(scope=routine), arguments=new_arguments)

            # Add the subroutine to the list of symbols that need to be imported
            if isinstance(new_name.scope, Module):
                module_name = new_name.scope.name.lower()
            else:
                module_name = new_name.type.dtype.procedure.procedure_symbol.scope.name.lower()
            new_subroutine_imports[module_name.lower()] |= {new_name.name.lower()}

        # Replace the calls
        routine.body = Transformer(call_mapper).visit(routine.body)

        # Add missing imports
        if routine.parent:
            current_module = routine.parent.name.lower()
        else:
            current_module = None
        imported_symbols = routine.imported_symbols
        new_imports = []
        for modname, routines in new_subroutine_imports.items():
            if modname != current_module:
                new_symbols = tuple(Variable(name=s, scope=routine) for s in routines if s not in imported_symbols)
                if new_symbols:
                    new_imports += [Import(module=modname, symbols=new_symbols)]
        routine.spec.prepend(as_tuple(new_imports))
