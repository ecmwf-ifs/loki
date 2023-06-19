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
    Transformation, FindVariables, FindNodes, Transformer, SubstituteExpressions,
    ExpressionRetriever, SubstituteExpressionsMapper, recursive_expression_map_update,
    Module, Import, CallStatement, InlineCall, Variable, SymbolAttributes,  RangeIndex,
    as_tuple, BasicType, warning
)


__all__ = ['DerivedTypeArgumentsTransformation', 'TypeboundProcedureCallTransformation']


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
            if not call.not_active and call.routine is not BasicType.DEFERRED:
                candidates = self._derived_type_arguments(call.routine)

                # Simultaneously walk caller and subroutine arguments
                new_arguments = list(call.arguments)
                for k_arg, d_arg in call.arg_iter():
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


def get_procedure_symbol_from_typebound_procedure_symbol(proc_symbol, routine_name):
    """
    Utility routine that returns the :any:`ProcedureSymbol` of the :any:`Subroutine`
    that a typebound procedure corresponds to.

    .. warning::
       Resolving generic bindings is currently not implemented

    This uses binding information (such as ``proc_symbol.type.bind_names``) or the
    :any:`TypeDef` to resolve the procedure binding. If the type information is
    incomplete or the resolution fails for other reasons, ``None`` is returned.

    Parameters
    ----------
    proc_symbol : :any:`ProcedureSymbol`
        The typebound procedure symbol that is to be resolved
    routine_name : str
        The name of the routine :data:`proc_symbol` appears in. This is used for
        logging purposes only

    Returns
    -------
    :any:`ProcedureSymbol` or None
        The procedure symbol of the :any:`Subroutine` or ``None`` if it fails to resolve
    """
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
    """
    Transformer to carry out the replacement of subroutine and inline function
    calls to typebound procedures by direct calls to the respective procedures

    During the transformer pass, this identifies also new dependencies due to
    inline function calls, which the :any:`Scheduler` may not be able to
    discover otherwise at the moment.

    Parameters
    ----------
    routine_name : str
        The name of the :any:`Subroutine` the replacement takes place. This is used
        for logging purposes only.
    current_module : str
        The name of the enclosing module. This is used to determine whether the
        resolved procedure needs to be added as an import.

    Attributes
    ----------
    new_procedure_imports : dict
        After a transformer pass, this will contain the mapping
        ``{module: {proc_name, proc_name, ...}}`` for new imports that are required
        as a consequence of the replacement.
    new_dependencies : set
        New dependencies due to inline function calls that are identified during the
        transformer pass.
    """

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
        """
        Rebuild a :any:`CallStatement`

        If this is a call to a typebound procedure, resolve the procedure binding and
        insert the derived type as the first argument in the call statement.
        """
        rebuilt = {k: self.visit(c, **kwargs) for k, c in zip(o._traversable, o.children)}
        if rebuilt['name'].parent:
            new_proc_symbol = get_procedure_symbol_from_typebound_procedure_symbol(rebuilt['name'], self.routine_name)

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
        """
        Return the expression unchanged unless there are :any:`InlineCall` nodes in the expression
        that are calls to typebound procedures, which are replaced by direct calls to the function
        with the derived type added as the first argument.
        """
        inline_calls = self.retrieve(o)
        if not inline_calls:
            return o

        expr_map = {}
        for call in inline_calls:
            new_proc_symbol = get_procedure_symbol_from_typebound_procedure_symbol(call.function, self.routine_name)

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
    fix_intent : bool
        Supply ``INOUT`` as intent on polymorphic dummy arguments missing an intent

    Attributes
    ----------
    inline_call_dependencies : dict
        Additional call dependencies identified during the transformer pass that can
        be registered in the :any:`Scheduler` via :any:`Scheduler.add_dependencies`.
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

    def add_inline_call_dependency(self, caller, callee):
        """
        Register a new dependency due to an inline call from :data:`caller` to :data:`callee`

        These dependencies are later on available to query via :attr:`inline_call_dependencies`.
        """
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
