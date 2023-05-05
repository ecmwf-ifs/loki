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
try:
    from fparser.two.Fortran2003 import Intrinsic_Name
    _intrinsic_fortran_names = Intrinsic_Name.function_names
except ImportError:
    _intrinsic_fortran_names = ()
from loki import (
    Transformation, FindVariables, FindNodes, FindInlineCalls, Transformer,
    SubstituteExpressions, SubstituteExpressionsMapper, recursive_expression_map_update,
    Module, Import, CallStatement, ProcedureDeclaration, InlineCall, Variable, RangeIndex,
    BasicType, DerivedType, as_tuple, warning, debug, CaseInsensitiveDict
)
from transformations.scc_cuf import is_elemental


__all__ = ['DerivedTypeArgumentsTransformation', 'TypeboundProcedureCallTransformation']


class DerivedTypeArgumentsTransformation(Transformation):
    """
    Remove derived types from procedure signatures by replacing the
    relevant derived type arguments by its member variables

    .. note::
       This transformation requires a Scheduler traversal that
       processes callees before callers.

    On the caller side, this updates calls to transformed subroutines
    and functions by passing the relevant derived type member variables
    instead of the original derived type argument. This uses information
    from previous application of this transformation to the called
    procedure.
    For calls to elemental routines, this may include the hoisting of
    a dimension expression to the caller. Where this dimension
    expression requires additional module variable imports, these
    imports are added on the caller side and the ``imports``
    cached property on the callers :any:`Item` is invalidated.

    On the callee side, this identifies derived type member variable
    usage, builds an expansion mapping, updates the procedure's
    signature accordingly, and substitutes the variable's use inside
    the routine. The information about the expansion map is stored
    in the :any:`Item`'s ``trafo_data``.
    See :meth:`expand_derived_args_kernel` for more information.
    """

    _key = 'DerivedTypeArgumentsTransformation'

    class _VariableCache:

        def __init__(self):
            self._var_cache = {}
            self._var_counter = defaultdict(int)

        def __getitem__(self, var):
            return self._var_cache[var]

        def __setitem__(self, var, value):
            self._var_cache[var] = value

        def __contains__(self, var):
            return var in self._var_cache

        def incr(self, var):
            self._var_counter[var.name.lower()] += 1

        def count(self, var):
            return self._var_counter[var.name.lower()]


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
        if item:
            item.trafo_data[self._key] = {}

        # Extract expansion maps and argument re-mapping for successors
        successors = [child for child in kwargs.get('successors', []) if self._key in child.trafo_data]
        renamed_import_map = {
            import_.module.lower(): {
                s.type.use_name.lower(): s.name.lower()
                for s in import_.symbols if s.type.use_name
            }
            for import_ in routine.imports + getattr(routine.parent, 'imports', [])
        }
        successors_data = CaseInsensitiveDict(
            (
                renamed_import_map.get(child.scope_name, {}).get(child.local_name, child.local_name),
                child.trafo_data[self._key]
            )
            for child in successors
        )

        # Apply caller transformation first to update calls to successors...
        dependencies_updated = self.expand_derived_args_caller(routine, successors_data)

        # ...and invalidate cached properties if dependencies have changed...
        if dependencies_updated and item:
            item.clear_cached_property('imports')

        # ...before updating the routine's signature and replacing
        # use of members in the body
        if role == 'kernel':
            # Expand derived type arguments in kernel
            orig_argnames = tuple(arg.lower() for arg in routine.argnames)
            expansion_map = self.expand_derived_args_kernel(routine)
            if item:
                item.trafo_data[self._key] = {
                    'orig_argnames': orig_argnames,
                    'expansion_map': expansion_map,
                }

    def expand_derived_args_caller(self, routine, successors_data):
        """
        For all active :any:`CallStatement` nodes, apply the derived type argument
        expansion on the caller side.

        The convention used is: ``derived%var => derived_var``.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The routine in which to transform call statements
        successors_data : :any:`CaseInsensitiveDict` of (str, dict)
            Dictionary containing the expansion maps (key ``'expansion_map'``) and
            original argnames (key ``'orig_argnames'``) of every child routine

        Returns
        -------
        bool
            Flag to indicate that dependencies have been changed (e.g. via new imports)
        """
        other_symbols = set()
        call_mapper = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            if not call.not_active:
                call_name = str(call.name)
                if call_name in successors_data:
                    # Set the new call signature on the IR node
                    arguments, kwarguments, others = self.expand_call_arguments(call, successors_data[call_name])
                    call_mapper[call] = call.clone(arguments=arguments, kwarguments=kwarguments)
                    other_symbols.update(others)

        # Rebuild the routine's IR tree
        if call_mapper:
            routine.body = Transformer(call_mapper).visit(routine.body)

        call_mapper = {}
        for call in FindInlineCalls().visit(routine.body):
            if (call_name := str(call.name)) in successors_data:
                # Set the new call signature on the expression node
                arguments, kwarguments, others = self.expand_call_arguments(call, successors_data[call_name])
                call_mapper[call] = call.clone(parameters=arguments, kw_parameters=kwarguments)
                other_symbols.update(others)

        if call_mapper:
            routine.body = SubstituteExpressions(call_mapper).visit(routine.body)

        # Add parameter declarations or imports
        if other_symbols:
            new_symbols = []
            new_imports = []
            for s in other_symbols:
                if s.type.imported:
                    if not s.type.module:
                        raise RuntimeError(
                            f'Incomplete type information available for {s!s}'
                        )
                    new_imports += [Import(module=s.type.module.name, symbols=(s.rescope(routine),))]
                else:
                    new_symbols += [s.rescope(routine)]
            if new_symbols:
                routine.variables += as_tuple(new_symbols)
            if new_imports:
                routine.spec.prepend(as_tuple(new_imports))
                return True

        return False

    @staticmethod
    def _expand_relative_to_local_var(local_var, expansion_components):
        """
        Utility routine that returns an expanded (nested) derived type argument
        relative to the local derived type variable declared on caller side

        Example: A subroutine that previously accepted a derived type argument
        ``some_arg`` uses only a member variable ``some_arg%nested_thing%var``,
        which is now replaced in the procedure interface by
        ``some_arg_nested_thing_var``. On the caller side, ``local_var`` is the
        instance of the derived type argument that used to be passed to the
        subroutine call. This utility routine determines and returns
        the new call argument ``local_var%nested_thing%var`` that has to be
        passed instead.
        """
        # We build the name bit-by-bit to obtain nested derived type arguments
        # relative to the local derived type variable
        for child in expansion_components:
            local_var = child.clone(
                name=f'{local_var.name}%{child.name}',
                parent=local_var,
                scope=local_var.scope
            )
        return local_var

    @classmethod
    def _expand_call_argument_dimensions(cls, call, expanded_arg, orig_argnames):
        """
        Utility routine that expands the dimension expression in a derived type
        argument expansion by matching symbols in the expression against other
        procedure arguments. Imported module level symbols or parameters that
        are used in the expression are identified and returned.

        The return value is a 2-tuple consisting of the dimension expression,
        where arguments are substituted with their caller-side expressions, and
        a set that contains parameters and module variables.
        """
        if not getattr(expanded_arg, 'dimensions', None):
            return None, set()

        # Extract all variables used in the index expression, and try
        # to match them to arguments in the call signature
        vmap = {}
        other_symbols = set()
        for var in FindVariables().visit(expanded_arg.dimensions):
            components = [*var.parents, var]

            try:
                arg_index = orig_argnames.index(components[0])
            except ValueError as exc:
                if var.type.imported or var.type.parameter:
                    other_symbols.add(var)
                elif var not in _intrinsic_fortran_names:
                    raise NotImplementedError(
                        f'Unsupported variable {var} in index expression of {expanded_arg}'
                    ) from exc
                continue

            vmap[var] = cls._expand_relative_to_local_var(call.arguments[arg_index], components[1:])

        vmap = recursive_expression_map_update(vmap)
        return SubstituteExpressions(vmap).visit(expanded_arg.dimensions), other_symbols

    @classmethod
    def _expand_call_argument(cls, call, caller_arg, expansion_list, orig_argnames):
        """
        Utility routine to expand :data:`caller_arg` in a subroutine call :data:`call`
        according to the provided :data:`expansion_list` and original arguments of the
        call target, as given in :data:`orig_argnames

        It returns a 2-tuple consisting of a list of new arguments and other symbols
        that become relevant on the caller side as a consequence of expanding a
        dimension expression, needing either a matching parameter declaration or module
        import.
        """
        other_symbols = set()

        # Found derived-type argument, unroll according to candidate map
        arguments = []
        for member in expansion_list:
            arg_member = cls._expand_relative_to_local_var(caller_arg, [*member.parents[1:], member])
            dimensions, others = cls._expand_call_argument_dimensions(call, member, orig_argnames)
            other_symbols.update(others)

            arguments += [arg_member.clone(dimensions=dimensions)]

        return arguments, other_symbols

    @classmethod
    def expand_call_arguments(cls, call, successor_data):
        """
        Create the call's argument list with derived type arguments expanded

        Parameters
        ----------
        call : :any:`CallStatement`
            The call statement to process
        successor_data : dict
            Dictionary containing the expansion map (key ``'expansion_map'``) and
            original argnames (key ``'orig_argnames'``) of the called routine

        Returns
        -------
        (tuple, tuple, set) :
            The argument and keyword argument list with derived type arguments expanded,
            and a set of additional symbols to cater for (either import them or replicate
            the parameter definition)
        """
        expansion_map = successor_data['expansion_map']
        orig_argnames = successor_data['orig_argnames']

        other_symbols = set()

        arguments = []
        for kernel_argname, caller_arg in zip(orig_argnames, call.arguments):
            if kernel_argname in expansion_map:
                new_arguments, others = cls._expand_call_argument(
                    call, caller_arg,
                    expansion_map[kernel_argname], orig_argnames
                )
                arguments += new_arguments
                other_symbols.update(others)
            else:
                arguments += [caller_arg]

        kwarguments = []
        for kernel_argname, caller_arg in call.kwarguments:
            if kernel_argname in expansion_map:
                new_arguments, others = cls._expand_call_argument(
                    call, caller_arg,
                    expansion_map[kernel_argname], orig_argnames
                )
                kwarguments += list(zip(expansion_map[kernel_argname], new_arguments))
                other_symbols.update(others)
            else:
                kwarguments += [(kernel_argname, caller_arg)]

        return as_tuple(arguments), as_tuple(kwarguments), other_symbols

    @staticmethod
    def _expand_kernel_variable(var, var_cache=None, **kwargs):
        """
        Utility routine that yields the expanded variable in the
        kernel for a given derived type variable member use :data:`var`
        """
        if var_cache is not None and var in var_cache:
            # If we have a variable cache with var already in there,
            # we can directly return that
            if kwargs:
                return var_cache[var].clone(**kwargs)
            return var_cache[var]

        new_name = var.name.replace('%', '_')

        if var_cache is not None and getattr(var, 'dimensions', None):
            # For elemental routines we use a variable cache, since we
            # have to expand every use of array values as separate
            # arguments
            var_cache.incr(var)
            new_name += f'_{var_cache.count(var)!s}'
            new_var = var.clone(name=new_name, parent=None, **kwargs)
            var_cache[var] = new_var
            return new_var

        return var.clone(name=new_name, parent=None, **kwargs)

    def expand_derived_args_kernel(self, routine):
        """
        Find the use of member variables for derived type arguments of
        :data:`routine`, update the call signature to directly pass the
        variable and substitute its use in the routine's body.

        Note that this will only carry out replacements for derived types
        that contain an allocatable, pointer, or nested derived type member.

        See :meth:`expand_derived_type_member` for more details on how
        the expansion is performed.
        """
        is_elemental_routine = is_elemental(routine)
        if is_elemental_routine:
            var_cache = self._VariableCache()
        else:
            var_cache = None

        # All derived type arguments are candidates for expansion
        candidates = []
        for arg in routine.arguments:
            if isinstance(arg.type.dtype, DerivedType):
                arg_variables = as_tuple(arg.variables)
                if any(v.type.pointer or v.type.allocatable or
                       isinstance(v.type.dtype, DerivedType) for v in arg_variables):
                    # Only include derived types with array members or nested derived types
                    candidates += [arg]

        # Inspect all derived type member use and determine their expansion
        expansion_map = defaultdict(set)
        non_expansion_map = defaultdict(set)
        vmap = {}
        for var in FindVariables(recurse_to_parent=False).visit(routine.ir):
            if var.parent:
                declared_var, expansion, local_use = self.expand_derived_type_member(
                    var, is_elemental_routine, var_cache
                )
                if expansion and declared_var in candidates:
                    # Mark this derived type member for expansion
                    expansion_map[declared_var].add(expansion)
                    vmap[var] = local_use
                elif declared_var in candidates:
                    non_expansion_map[declared_var].add(var)

        # Update the expansion map by re-adding the derived type argument when
        # there are non-expanded members left
        # Here, we determine the ordering in the updated call signature
        #   Note: the _local_ name for expanded arguments in elemental routines is not
        #   always the same, because the order in which variable names are created in the
        #   var cache is random (hash sorting of set), however, the order in which they
        #   appear in the updated routine signature is always the same
        expansion_map = dict(expansion_map)
        for arg in candidates:
            if arg in expansion_map:
                sorted_expansion = sorted(expansion_map[arg], key=lambda v: str(v).lower())
                if arg in non_expansion_map:
                    expansion_map[arg] = (arg, *sorted_expansion)
                else:
                    expansion_map[arg] = tuple(sorted_expansion)

        def shape_to_assumed_dim(shape):
            if is_elemental_routine or not shape:
                return None
            return tuple(RangeIndex((None, None)) for _ in shape)

        def update_var_type(arg, var):
            if is_elemental_routine:
                return var.type.clone(
                    intent=arg.type.intent, initial=None,
                    allocatable=None, target=None, pointer=None, shape=None
                )
            return var.type.clone(intent=arg.type.intent, initial=None)

        # Build the arguments map to update the call signature
        arguments_map = {}
        for arg in routine.arguments:
            if arg in expansion_map:
                arguments_map[arg] = [
                    self._expand_kernel_variable(
                        var, type=update_var_type(arg, var), dimensions=shape_to_assumed_dim(var.type.shape),
                        scope=routine, var_cache=var_cache
                    )
                    for var in expansion_map[arg]
                ]

        # Update arguments list
        routine.arguments = [a for arg in routine.arguments for a in arguments_map.get(arg, [arg])]

        # Update variable list, too, as this triggers declaration generation
        routine.variables = [v for var in routine.variables for v in arguments_map.get(var, [var])]

        # Substitue derived type member use in the spec and body
        vmap = recursive_expression_map_update(vmap)
        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
        routine.body = SubstituteExpressions(vmap).visit(routine.body)

        # Update procedure bindings by specifying NOPASS attribute
        for arg in arguments_map:
            for decl in arg.type.dtype.typedef.declarations:
                if isinstance(decl, ProcedureDeclaration) and not decl.generic:
                    for proc in decl.symbols:
                        if routine.name == proc or routine.name in as_tuple(proc.type.bind_names):
                            proc.type = proc.type.clone(pass_attr=False)

        return expansion_map

    @classmethod
    def expand_derived_type_member(cls, var, is_elemental_routine, var_cache=None):
        """
        Determine the member expansion for a derived type member variable

        For a derived type member variable, provided as :data:`var`, this determines
        the name of the root parent and the member expansion.

        A few examples to illustrate the behaviour, with the Fortran variable use
        that :data:`var` represents in the left column and corresponding return value
        of this routine on the right:

        .. code-block::
            var name            | return value (parent_name, expansion, new use)   | remarks
           ---------------------+--------------------------------------------------+------------------------------------
            SOME_VAR            | ('some_var', None, None)                         | No expansion
            SOME%VAR            | ('some', 'some%var', 'some_var')                 |
            ARRAY(5)%VAR        | ('array', None, None)                            | Can't expand array of derived types
            SOME%NESTED%VAR     | ('some', 'some%nested%var', 'some_nested_var)    |
            NESTED%ARRAY(I)%VAR | ('nested', 'nested%array', 'nested_array(i)%var')| Partial expansion

        For elemental routines, we may need to hoist subscript expressions to the caller side,
        in which case the corresponding index expression is included in the expansion field,
        for example:

        .. code-block::
            var name            | return value (parent_name, expansion, new_use)
           ---------------------+-----------------------------------------------
            SOME%VAR(IDX)       | ('some', 'some%var(idx)', 'some_var')

        Note, that this currently supports only index expressions where another argument,
        a constant (i.e. parameter), or global module variables that can be imported from
        elsewhere, are used.

        Parameters
        ----------
        var : :any:`MetaSymbol`
            The use of a derived type member
        is_elemental_routine : bool
            Flag to indicate that the current routine is an ``ELEMENTAL`` routine

        Returns
        -------
        (:any:`Variable`, :any:`Variable` or None, :any:`Variable` or None)
        """
        parents = var.parents
        if not parents:
            return var, None, None

        if is_elemental_routine:
            # In elemental routines, we can have scalar derived type arguments with
            # array members, where the subscript operation is performed inside the elemental
            # routine. Because Fortran mandates (for good reasons!) scalar arguments in
            # elemental routines, we have to include the dimensions in the
            # argument expansion and hoist them to the caller side.
            expansion = var.clone(scope=None)
            local_use = cls._expand_kernel_variable(
                var, dimensions=None, type=var.type.clone(shape=None), var_cache=var_cache
            )
        else:
            # We unroll the derived type member as far as possible, stopping at
            # the occurence of an intermediate derived type array.
            # Note that we set scope=None, which detaches the symbol from the current
            # scope and stores the type information locally on the symbol. This makes
            # them available later on without risking losing this information due to
            # intermediate rescoping operations
            for idx, parent in enumerate(parents):
                if hasattr(parent, 'dimensions'):
                    expansion = parent.clone(scope=None, dimensions=None)
                    if parent is parents[0]:
                        debug(f'Array of derived types {var!s}. Cannot expand argument.')
                        local_use = var
                    else:
                        debug(f'Array of derived types {var!s}. '
                            f'Can only partially expand argument as {expansion!s}.')
                        local_use = cls._expand_kernel_variable(parent)
                        local_use = cls._expand_relative_to_local_var(local_use, [*parents[idx+1:], var])
                    break
            else:
                # None of the parents had a dimensions attribute, which means we can
                # completely expand
                expansion = var.clone(scope=None, dimensions=None)
                local_use = cls._expand_kernel_variable(var)

        return parents[0], expansion, local_use


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
