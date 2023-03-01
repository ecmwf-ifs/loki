from collections import defaultdict
from loki import (
    Transformation, Transformer, BasicType, Variable, Module,
    ExpressionRetriever, InlineCall, recursive_expression_map_update,
    SubstituteExpressionsMapper, warning, as_tuple, Import
)

__all__ = ['TypeboundProcedureCallTransformation']


def get_procedure_name(proc_symbol, routine_name):
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
            warning('Type definitions incomplete for %s in %s', proc_symbol, routine_name)
            return None

        if local_var.type.dtype.is_generic:
            warning('Cannot resolve generic binding %s (not implemented) in %s', proc_symbol, routine_name)
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


class TypeboundProcedureCallTransformer(Transformer):

    def __init__(self, routine_name, current_module, **kwargs):
        super().__init__(inplace=True, **kwargs)
        self.routine_name = routine_name
        self.current_module = current_module
        self.new_procedure_imports = defaultdict(set)
        self.new_dependencies = set()
        self._retriever = ExpressionRetriever(
            lambda e: isinstance(e, InlineCall) and e.function.parent,
            recurse_to_parent=False
        )

    def retrieve(self, o):
        return self._retriever.retrieve(o)

    def visit_CallStatement(self, o, **kwargs):
        rebuilt = {k: self.visit(c, **kwargs) for k, c in zip(o._traversable, o.children)}
        if rebuilt['name'].parent:
            new_proc_symbol = get_procedure_name(rebuilt['name'], self.routine_name)

            if new_proc_symbol:
                # Add the derived type as first argument to the call
                rebuilt['arguments'] = (rebuilt['name'].parent, ) + rebuilt['arguments']

                # Add the subroutine to the list of symbols that need to be imported
                if isinstance(new_proc_symbol.scope, Module):
                    module_name = new_proc_symbol.scope.name.lower()
                else:
                    module_name = new_proc_symbol.type.dtype.procedure.procedure_symbol.scope.name.lower()

                if module_name != self.current_module:
                    self.new_procedure_imports[module_name].add(new_proc_symbol.name.lower())

                rebuilt['name'] = new_proc_symbol
        children = [rebuilt[k] for k in o._traversable]
        return self._rebuild(o, children)

    def visit_Expression(self, o, **kwargs):
        inline_calls = self.retrieve(o)
        if not inline_calls:
            return o

        expr_map = {}
        for call in inline_calls:
            new_proc_symbol = get_procedure_name(call.function, self.routine_name)

            if new_proc_symbol:
                new_arguments = (call.function.parent,) + call.parameters
                expr_map[call] = call.clone(
                    function=new_proc_symbol.rescope(scope=kwargs['scope']),
                    parameters=new_arguments
                )
                # Add the function to the list of symbols that need to be imported
                if isinstance(new_proc_symbol.scope, Module):
                    module_name = new_proc_symbol.scope.name.lower()
                else:
                    module_name = new_proc_symbol.type.dtype.procedure.procedure_symbol.scope.name.lower()

                if module_name != self.current_module:
                    self.new_procedure_imports[module_name].add(new_proc_symbol.name.lower())
                    self.new_dependencies.add(call.function.type.dtype.procedure)

        if not expr_map:
            return o

        expr_map = recursive_expression_map_update(expr_map)
        return SubstituteExpressionsMapper(expr_map)(o)



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

        transformer = TypeboundProcedureCallTransformer(routine.name, current_module)
        routine.body = transformer.visit(routine.body, scope=routine)
        new_procedure_imports = transformer.new_procedure_imports

        # Add new dependencies
        for callee in transformer.new_dependencies:
            self.add_inline_call_dependency(routine, callee)

        # Add missing imports
        imported_symbols = routine.imported_symbols
        new_imports = []
        for module_name, proc_symbols in new_procedure_imports.items():
            new_symbols = tuple(Variable(name=s, scope=routine) for s in proc_symbols if s not in imported_symbols)
            if new_symbols:
                new_imports += [Import(module=module_name, symbols=new_symbols)]

        if new_imports:
            routine.spec.prepend(as_tuple(new_imports))
            if item:
                item.clear_cached_property('imports')
