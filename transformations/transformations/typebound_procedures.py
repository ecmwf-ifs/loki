from collections import defaultdict
from loki import (
    Transformation, FindNodes, CallStatement, Transformer, BasicType, Variable,
    Module, warning, as_tuple, Import, FindInlineCalls, SubstituteExpressions
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
    which is set to `INOUT` by default, if missing. This behaviour can be switched
    off by setting :data:`fix_intent` to `False`.

    Parameters
    ----------
    fix_intent: bool
        Supply ``INOUT`` as intent on polymorphic dummy arguments  without
    """

    def __init__(self, fix_intent=True, **kwargs):
        super().__init__(**kwargs)
        self.fix_intent = fix_intent
        self.inline_call_dependencies = defaultdict(set)

    def apply_default_polymorphic_intent(self, routine):
        """
        Utility routine to set a default ``INTENT(INOUT)`` on polymorphic dummy
        arguments (i.e. declared via ``CLASS``) that don't have an explicit intent
        """
        for arg in routine.arguments:
            type_ = arg.type
            if type_.polymorphic and not type_.intent:
                arg.type = type_.clone(intent='inout')

    def get_procedure_name(self, routine, proc_symbol):
        if proc_symbol.type.bind_names is not None:
            return proc_symbol.type.bind_names[0]

        parent = proc_symbol.parents[0]
        if parent.type.dtype.typedef is not BasicType.DEFERRED:
            # Fiddle our way through derived type nesting until we obtain the symbol corresponding
            # to the procedure-binding in the TypeDef
            local_parent = None
            local_var = parent
            try:
                for local_name in proc_symbol.name_parts[1:]:
                    local_parent = local_var
                    local_var = local_var.type.dtype.typedef.variable_map[local_name]
            except AttributeError:
                warning('Type definitions incomplete for %s in %s', proc_symbol, routine.name)
                return None

            if local_var.type.dtype.is_generic:
                warning('Cannot resolve generic binding %s (not implemented) in %s', proc_symbol, routine.name)
                return None

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
            return new_name

        # We don't have any binding information available
        return None

    def add_inline_call_dependency(self, caller, callee):
        caller_module = getattr(caller.parent, 'name', '')
        callee_module = getattr(callee.parent, 'name', '')
        self.inline_call_dependencies[f'{caller_module}#{caller.name}'.lower()] |= {
            f'{callee_module}#{callee.name}'.lower()
        }

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply the transformation of calls to the given :data:`routine`
        """
        item = kwargs.get('item')

        # Bail out if the current subroutine is not part of the call tree
        if item and item.local_name != routine.name.lower():
            return

        # Fix any wrong intents on polymorphic arguments
        # (sadly, it's not uncommon to omit the intent specification on the CLASS declaration,
        # so we set them to `inout` here for any missing intents)
        if self.fix_intent:
            self.apply_default_polymorphic_intent(routine)

        if routine.parent:
            current_module = routine.parent.name.lower()
        else:
            current_module = None

        # Collect names of new procedure symbols that need to be imported
        new_procedure_imports = defaultdict(set)

        # Search for typebound procedure calls to subroutines
        call_mapper = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name.parent:
                new_proc_symbol = self.get_procedure_name(routine, call.name)
                if not new_proc_symbol:
                    continue

                # Mark the call for replacement
                new_arguments = (call.name.parent,) + call.arguments
                call_mapper[call] = call.clone(name=new_proc_symbol.rescope(scope=routine), arguments=new_arguments)

                # Add the subroutine to the list of symbols that need to be imported
                if isinstance(new_proc_symbol.scope, Module):
                    module_name = new_proc_symbol.scope.name.lower()
                else:
                    module_name = new_proc_symbol.type.dtype.procedure.procedure_symbol.scope.name.lower()

                if module_name != current_module:
                    new_procedure_imports[module_name] |= {new_proc_symbol.name.lower()}

        # Replace the calls
        if call_mapper:
            routine.body = Transformer(call_mapper).visit(routine.body)

        # Search for typebound procedure calls to functions
        expr_mapper = {}
        for call in FindInlineCalls().visit(routine.body):
            if call.function.parent:
                new_proc_symbol = self.get_procedure_name(routine, call.function)
                if not new_proc_symbol:
                    continue

                new_arguments = (call.function.parent,) + call.parameters
                expr_mapper[call] = call.clone(
                    function=new_proc_symbol.rescope(scope=routine), parameters=new_arguments
                )

                # Add the function to the list of symbols that need to be imported
                if isinstance(new_proc_symbol.scope, Module):
                    module_name = new_proc_symbol.scope.name.lower()
                else:
                    module_name = new_proc_symbol.type.dtype.procedure.procedure_symbol.scope.name.lower()

                if module_name != current_module:
                    new_procedure_imports[module_name] |= {new_proc_symbol.name.lower()}
                    self.add_inline_call_dependency(routine, call.function.type.dtype.procedure)

        # Replace the inline calls
        if expr_mapper:
            routine.body = SubstituteExpressions(expr_mapper).visit(routine.body)

        # Add missing imports
        imported_symbols = routine.imported_symbols
        new_imports = []
        for module_name, proc_symbols in new_procedure_imports.items():
            new_symbols = tuple(Variable(name=s, scope=routine) for s in proc_symbols if s not in imported_symbols)
            if new_symbols:
                new_imports += [Import(module=module_name, symbols=new_symbols)]

        if new_imports:
            routine.spec.prepend(as_tuple(new_imports))
