# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utility routines to perform code-level force-inlining.


"""
from collections import defaultdict, ChainMap

from loki.batch import Transformation
from loki.ir import (
    Import, Comment, Assignment, VariableDeclaration, CallStatement,
    Transformer, FindNodes, pragmas_attached, is_loki_pragma, Interface,
    StatementFunction, FindVariables, FindInlineCalls, FindLiterals,
    SubstituteExpressions, ExpressionFinder
)
from loki.expression import (
    symbols as sym, LokiIdentityMapper, ExpressionRetriever
)
from loki.types import BasicType
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.logging import error
from loki.subroutine import Subroutine

from loki.transformations.remove_code import do_remove_dead_code
from loki.transformations.sanitise import transform_sequence_association_append_map
from loki.transformations.utilities import (
    single_variable_declaration, recursive_expression_map_update
)


__all__ = [
    'inline_constant_parameters', 'inline_elemental_functions',
    'inline_internal_procedures', 'inline_member_procedures',
    'inline_marked_subroutines', 'InlineTransformation',
    'inline_statement_functions'
]


class InlineTransformation(Transformation):
    """
    :any:`Transformation` class to apply several types of source inlining
    when batch-processing large source trees via the :any:`Scheduler`.

    Parameters
    ----------
    inline_constants : bool
        Replace instances of variables with known constant values by
        :any:`Literal` (see :any:`inline_constant_parameters`); default: False.
    inline_elementals : bool
        Replaces :any:`InlineCall` expression to elemental functions
        with the called function's body (see :any:`inline_elemental_functions`);
        default: True.
    inline_stmt_funcs: bool
        Replaces  :any:`InlineCall` expression to statement functions
        with the corresponding rhs of the statement function if
        the statement function declaration is available; default: False.
    inline_internals : bool
        Inline internal procedure (see :any:`inline_internal_procedures`);
        default: False.
    inline_marked : bool
        Inline :any:`Subroutine` objects marked by pragma annotations
        (see :any:`inline_marked_subroutines`); default: True.
    remove_dead_code : bool
        Perform dead code elimination, where unreachable branches are
        trimmed from the code (see :any:`dead_code_elimination`); default: True
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope during
        internal and pragma-driven inlining.
    adjust_imports : bool
        Adjust imports by removing the symbol of the inlined routine or adding
        imports needed by the imported routine (optional, default: True)
    external_only : bool, optional
        Do not replace variables declared in the local scope when
        inlining constants (default: True)
    resolve_sequence_association: bool
        Resolve sequence association for routines that contain calls to inline (default: False)
    """

    # Ensure correct recursive inlining by traversing from the leaves
    reverse_traversal = True

    # This transformation will potentially change the edges in the callgraph
    creates_items = False

    def __init__(
            self, inline_constants=False, inline_elementals=True,
            inline_stmt_funcs=False, inline_internals=False,
            inline_marked=True, remove_dead_code=True,
            allowed_aliases=None, adjust_imports=True,
            external_only=True, resolve_sequence_association=False
    ):
        self.inline_constants = inline_constants
        self.inline_elementals = inline_elementals
        self.inline_stmt_funcs = inline_stmt_funcs
        self.inline_internals = inline_internals
        self.inline_marked = inline_marked
        self.remove_dead_code = remove_dead_code
        self.allowed_aliases = allowed_aliases
        self.adjust_imports = adjust_imports
        self.external_only = external_only
        self.resolve_sequence_association = resolve_sequence_association
        if self.inline_marked:
            self.creates_items = True

    def transform_subroutine(self, routine, **kwargs):

        # Resolve sequence association in calls that are about to be inlined.
        # This step runs only if all of the following hold:
        # 1) it is requested by the user
        # 2) inlining of "internals" or "marked" routines is activated
        # 3) there is an "internal" or "marked" procedure to inline.
        if self.resolve_sequence_association:
            resolve_sequence_association_for_inlined_calls(
                routine, self.inline_internals, self.inline_marked
            )

        # Replace constant parameter variables with explicit values
        if self.inline_constants:
            inline_constant_parameters(routine, external_only=self.external_only)

        # Inline elemental functions
        if self.inline_elementals:
            inline_elemental_functions(routine)

        # Inline Statement Functions
        if self.inline_stmt_funcs:
            inline_statement_functions(routine)

        # Inline internal (contained) procedures
        if self.inline_internals:
            inline_internal_procedures(routine, allowed_aliases=self.allowed_aliases)

        # Inline explicitly pragma-marked subroutines
        if self.inline_marked:
            inline_marked_subroutines(
                routine, allowed_aliases=self.allowed_aliases,
                adjust_imports=self.adjust_imports
            )

        # After inlining, attempt to trim unreachable code paths
        if self.remove_dead_code:
            do_remove_dead_code(routine)


class InlineSubstitutionMapper(LokiIdentityMapper):
    """
    An expression mapper that defines symbolic substitution for inlining.
    """

    def map_algebraic_leaf(self, expr, *args, **kwargs):
        raise NotImplementedError

    def map_scalar(self, expr, *args, **kwargs):
        parent = self.rec(expr.parent, *args, **kwargs) if expr.parent is not None else None

        scope = kwargs.get('scope') or expr.scope
        # We're re-scoping an imported symbol
        if expr.scope != scope:
            return expr.clone(scope=scope, type=expr.type.clone(), parent=parent)
        return expr.clone(parent=parent)

    map_deferred_type_symbol = map_scalar

    def map_array(self, expr, *args, **kwargs):
        if expr.dimensions:
            dimensions = self.rec(expr.dimensions, *args, **kwargs)
        else:
            dimensions = None
        parent = self.rec(expr.parent, *args, **kwargs) if expr.parent is not None else None

        scope = kwargs.get('scope') or expr.scope
        # We're re-scoping an imported symbol
        if expr.scope != scope:
            return expr.clone(scope=scope, type=expr.type.clone(), parent=parent, dimensions=dimensions)
        return expr.clone(parent=parent, dimensions=dimensions)

    def map_procedure_symbol(self, expr, *args, **kwargs):
        parent = self.rec(expr.parent, *args, **kwargs) if expr.parent is not None else None

        scope = kwargs.get('scope') or expr.scope
        # We're re-scoping an imported symbol
        if expr.scope != scope:
            return expr.clone(scope=scope, type=expr.type.clone(), parent=parent)
        return expr.clone(parent=parent)

    def map_inline_call(self, expr, *args, **kwargs):
        if expr.procedure_type is None or expr.procedure_type is BasicType.DEFERRED:
            # Unkonw inline call, potentially an intrinsic
            # We still need to recurse and ensure re-scoping
            return super().map_inline_call(expr, *args, **kwargs)

        # if it is an inline call to a Statement Function
        if isinstance(expr.routine, StatementFunction):
            function = expr.routine
            # Substitute all arguments through the elemental body
            arg_map = dict(expr.arg_iter())
            fbody = SubstituteExpressions(arg_map).visit(function.rhs)
            return fbody

        function = expr.procedure_type.procedure
        v_result = [v for v in function.variables if v == function.name][0]

        # Substitute all arguments through the elemental body
        arg_map = dict(expr.arg_iter())
        fbody = SubstituteExpressions(arg_map).visit(function.body)

        # Extract the RHS of the final result variable assignment
        stmts = [s for s in FindNodes(Assignment).visit(fbody) if s.lhs == v_result]
        assert len(stmts) == 1
        rhs = self.rec(stmts[0].rhs, *args, **kwargs)
        return rhs

def resolve_sequence_association_for_inlined_calls(routine, inline_internals, inline_marked):
    """
    Resolve sequence association in calls to all member procedures (if ``inline_internals = True``)
    or in calls to procedures that have been marked with an inline pragma (if ``inline_marked = True``).
    If both ``inline_internals`` and ``inline_marked`` are ``False``, no processing is done.
    """
    call_map = {}
    with pragmas_attached(routine, node_type=CallStatement):
        for call in FindNodes(CallStatement).visit(routine.body):
            condition = (
                (inline_marked and is_loki_pragma(call.pragma, starts_with='inline')) or
                (inline_internals and call.routine in routine.routines)
            )
            if condition:
                if call.routine == BasicType.DEFERRED:
                    # NOTE: Throwing error here instead of continuing, because the user has explicitly
                    # asked sequence assoc to happen with inlining, so source for routine should be
                    # found in calls to be inlined.
                    raise ValueError(
                        f"Cannot resolve sequence association for call to ``{call.name}`` " +
                        f"to be inlined in routine ``{routine.name}``, because " +
                        f"the ``CallStatement`` referring to ``{call.name}`` does not contain " +
                        "the source code of the procedure. " +
                        "If running in batch processing mode, please recheck Scheduler configuration."
                    )
                transform_sequence_association_append_map(call_map, call)
        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)

def inline_constant_parameters(routine, external_only=True):
    """
    Replace instances of variables with known constant values by `Literals`.

    Notes
    -----
    The ``.type.initial`` property is used to derive the replacement
    value,a which means for symbols imported from external modules,
    the parent :any:`Module` needs to be supplied in the
    ``definitions`` to the constructor when creating the
    :any:`Subroutine`.

    Variables that are replaced are also removed from their
    corresponding import statements, with empty import statements
    being removed alltogether.

    Parameters
    ----------
    routine : :any:`Subroutine`
         Procedure in which to inline/resolve constant parameters.
    external_only : bool, optional
        Do not replace variables declared in the local scope (default: True)
    """
    # Find all variable instances in spec and body
    variables = FindVariables().visit(routine.ir)

    # Filter out variables declared locally
    if external_only:
        variables = [v for v in variables if v not in routine.variables]

    def is_inline_parameter(v):
        return hasattr(v, 'type') and v.type.parameter and v.type.initial

    # Create mapping for variables and imports
    vmap = {v: v.type.initial for v in variables if is_inline_parameter(v)}

    # Replace kind parameters in variable types
    for variable in routine.variables:
        if is_inline_parameter(variable.type.kind):
            routine.symbol_attrs[variable.name] = variable.type.clone(kind=variable.type.kind.type.initial)
        if variable.type.initial is not None:
            # Substitute kind specifier in literals in initializers (I know...)
            init_map = {literal.kind: literal.kind.type.initial
                        for literal in FindLiterals().visit(variable.type.initial)
                        if is_inline_parameter(literal.kind)}
            if init_map:
                initial = SubstituteExpressions(init_map).visit(variable.type.initial)
                routine.symbol_attrs[variable.name] = variable.type.clone(initial=initial)

    # Update imports
    imprtmap = {}
    substituted_names = {v.name.lower() for v in vmap}
    for imprt in FindNodes(Import).visit(routine.spec):
        if imprt.symbols:
            symbols = tuple(s for s in imprt.symbols if s.name.lower() not in substituted_names)
            if not symbols:
                imprtmap[imprt] = Comment(f'! Loki: parameters from {imprt.module} inlined')
            elif len(symbols) < len(imprt.symbols):
                imprtmap[imprt] = imprt.clone(symbols=symbols)

    # Flush mappings through spec and body
    routine.spec = Transformer(imprtmap).visit(routine.spec)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Clean up declarations that are about to become defunct
    decl_map = {
        decl: None for decl in routine.declarations
        if all(isinstance(s, sym.IntLiteral) for s in decl.symbols)
    }
    routine.spec = Transformer(decl_map).visit(routine.spec)


def inline_elemental_functions(routine):
    """
    Replaces `InlineCall` expression to elemental functions with the
    called functions body.

    Parameters
    ----------
    routine : :any:`Subroutine`
         Procedure in which to inline functions.
    """
    inline_functions(routine, inline_elementals_only=True)


def inline_functions(routine, inline_elementals_only=False, functions=None):
    """
    Replaces `InlineCall` expression to functions with the
    called functions body. Nested calls are handled/inlined through
    an iterative approach calling :any:`_inline_functions`.

    Parameters
    ----------
    routine : :any:`Subroutine`
         Procedure in which to inline functions.
    inline_elementals_only : bool, optional
        Inline elemental routines/functions only (default: False).
    functions : tuple, optional
        Inline only functions that are provided here
        (default: None, thus inline all functions).
    """
    potentially_functions_to_be_inlined = True
    while potentially_functions_to_be_inlined:
        potentially_functions_to_be_inlined = _inline_functions(routine, inline_elementals_only=inline_elementals_only,
                                                           functions=functions)

def _inline_functions(routine, inline_elementals_only=False, functions=None):
    """
    Replaces `InlineCall` expression to functions with the
    called functions body, but doesn't include nested calls!

    Parameters
    ----------
    routine : :any:`Subroutine`
         Procedure in which to inline functions.
    inline_elementals_only : bool, optional
        Inline elemental routines/functions only (default: False).
    functions : tuple, optional
        Inline only functions that are provided here
        (default: None, thus inline all functions).
    
    Returns
    -------
    bool
        Whether inline calls are (potentially) left to be
        inlined in the next call to this function.
    """

    class ExpressionRetrieverSkipInlineCallParameters(ExpressionRetriever):
        """
        Expression retriever skipping parameters of inline calls.
        """
        # pylint: disable=abstract-method

        def __init__(self, query, recurse_query=None, inline_elementals_only=False,
                functions=None, **kwargs):
            self.inline_elementals_only = inline_elementals_only
            self.functions = as_tuple(functions)
            super().__init__(query, recurse_query, **kwargs)

        def map_inline_call(self, expr, *args, **kwargs):
            if not self.visit(expr, *args, **kwargs):
                return
            self.rec(expr.function, *args, **kwargs)
            # SKIP parameters/args/kwargs on purpose
            #  under certain circumstances
            if expr.procedure_type is BasicType.DEFERRED or\
                    (self.inline_elementals_only and\
                    not(expr.procedure_type.is_function and expr.procedure_type.is_elemental)) or\
                    (self.functions and expr.routine not in self.functions):
                for child in expr.parameters:
                    self.rec(child, *args, **kwargs)
                for child in list(expr.kw_parameters.values()):
                    self.rec(child, *args, **kwargs)

            self.post_visit(expr, *args, **kwargs)

    class FindInlineCallsSkipInlineCallParameters(ExpressionFinder):
        """
        Find inline calls but skip/ignore parameters of inline calls.
        """
        retriever = ExpressionRetrieverSkipInlineCallParameters(lambda e: isinstance(e, sym.InlineCall))

    functions = as_tuple(functions)

    # Keep track of removed symbols
    removed_functions = set()

    # Find and filter inline calls and corresponding nodes
    function_calls = {}
    # Find inline calls but skip/ignore inline calls being parameters of other inline calls
    #  to ensure correct ordering of inlining. Those skipped/ignored inline calls will be handled
    #  in the next call to this function.
    retriever = ExpressionRetrieverSkipInlineCallParameters(lambda e: isinstance(e, sym.InlineCall),
            inline_elementals_only=inline_elementals_only, functions=functions)
    # override retriever ...
    FindInlineCallsSkipInlineCallParameters.retriever = retriever
    for node, calls in FindInlineCallsSkipInlineCallParameters(with_ir_node=True).visit(routine.body):
        for call in calls:
            if call.procedure_type is BasicType.DEFERRED or isinstance(call.routine, StatementFunction):
                continue
            if inline_elementals_only:
                if not (call.procedure_type.is_function and call.procedure_type.is_elemental):
                    continue
            if functions:
                if call.routine not in functions:
                    continue
            function_calls.setdefault(str(call.name).lower(),[]).append((call, node))

    if not function_calls:
        return False

    # inline functions
    node_prepend_map = {}
    call_map = {}
    for _, calls_nodes in function_calls.items():
        calls, nodes = list(zip(*calls_nodes))
        for call in calls:
            removed_functions.add(call.procedure_type)
        # collect nodes to be appendes as well as expression replacement for inline call
        inline_node_map, inline_call_map = inline_function_calls(routine, as_tuple(calls),
                                                                 calls[0].routine, as_tuple(nodes))
        for node, nodes_to_prepend in inline_node_map.items():
            node_prepend_map.setdefault(node, []).extend(list(nodes_to_prepend))
        call_map.update(inline_call_map)

    # collect nodes to be prepended for each node that contains (at least one) inline call to a function
    node_map = {}
    for node, prepend_nodes in node_prepend_map.items():
        node_map[node] = as_tuple(prepend_nodes) + (SubstituteExpressions(call_map[node]).visit(node),)
    # inline via prepending the relevant functions
    routine.body = Transformer(node_map).visit(routine.body)
    # We need this to ensure that symbols, as well as nested scopes
    # are correctly attached to each other (eg. nested associates).
    routine.rescope_symbols()

    # Remove all module imports that have become obsolete now
    import_map = {}
    for im in FindNodes(Import).visit(routine.spec):
        if im.symbols and all(s.type.dtype in removed_functions for s in im.symbols):
            import_map[im] = None
    routine.spec = Transformer(import_map).visit(routine.spec)
    return True

def inline_statement_functions(routine):
    """
    Replaces :any:`InlineCall` expression to statement functions with the
    called statement functions rhs.
    """
    # Keep track of removed symbols
    removed_functions = set()

    stmt_func_decls = FindNodes(StatementFunction).visit(routine.spec)
    exprmap = {}
    for call in FindInlineCalls().visit(routine.body):
        proc_type = call.procedure_type
        if proc_type is BasicType.DEFERRED:
            continue
        if proc_type.is_function and isinstance(call.routine, StatementFunction):
            exprmap[call] = InlineSubstitutionMapper()(call, scope=routine)
            removed_functions.add(call.routine)
    # Apply the map to itself to handle nested statement function calls
    exprmap = recursive_expression_map_update(exprmap, max_iterations=10, mapper_cls=InlineSubstitutionMapper)
    # Apply expression-level substitution to routine
    routine.body = SubstituteExpressions(exprmap).visit(routine.body)

    # remove statement function declarations as well as statement function argument(s) declarations
    vars_to_remove = {stmt_func.variable.name.lower() for stmt_func in stmt_func_decls}
    vars_to_remove |= {arg.name.lower() for stmt_func in stmt_func_decls for arg in stmt_func.arguments}
    spec_map = {stmt_func: None for stmt_func in stmt_func_decls}
    for decl in routine.declarations:
        if any(var in vars_to_remove for var in decl.symbols):
            symbols = tuple(var for var in decl.symbols if var not in vars_to_remove)
            if symbols:
                decl._update(symbols=symbols)
            else:
                spec_map[decl] = None
    routine.spec = Transformer(spec_map).visit(routine.spec)

def map_call_to_procedure_body(call, caller, callee=None):
    """
    Resolve arguments of a call and map to the called procedure body.

    Parameters
    ----------
    call : :any:`CallStatment` or :any:`InlineCall`
         Call object that defines the argument mapping
    caller : :any:`Subroutine`
         Procedure (scope) into which the callee's body gets mapped
    callee : :any:`Subroutine`, optional
         Procedure (scope) called. Provide if it differs from
         call.routine.
    """

    def _map_unbound_dims(var, val):
        """
        Maps all unbound dimension ranges in the passed array value
        ``val`` with the indices from the local variable ``var``. It
        returns the re-mapped symbol.

        For example, mapping the passed array ``m(:,j)`` to the local
        expression ``a(i)`` yields ``m(i,j)``.
        """
        new_dimensions = list(val.dimensions)

        indices = [index for index, dim in enumerate(val.dimensions) if isinstance(dim, sym.Range)]

        for index, dim in enumerate(var.dimensions):
            new_dimensions[indices[index]] = dim

        return val.clone(dimensions=tuple(new_dimensions))

    # Get callee from the procedure type
    callee = callee or call.routine
    if callee is BasicType.DEFERRED:
        error(
            '[Loki::TransformInline] Need procedure definition to resolve '
            f'call to {call.name} from {caller}'
        )
        raise RuntimeError('Procedure definition not found! ')

    argmap = {}
    callee_vars = FindVariables().visit(callee.body)

    # Match dimension indexes between the argument and the given value
    # for all occurences of the argument in the body
    for arg, val in call.arg_map.items():
        if isinstance(arg, sym.Array):
            # Resolve implicit dimension ranges of the passed value,
            # eg. when passing a two-dimensional array `a` as `call(arg=a)`
            # Check if val is a DeferredTypeSymbol, as it does not have a `dimensions` attribute
            if not isinstance(val, sym.DeferredTypeSymbol) and val.dimensions:
                qualified_value = val
            else:
                qualified_value = val.clone(
                    dimensions=tuple(sym.Range((None, None)) for _ in arg.shape)
                )

            # If sequence association (scalar-to-array argument passing) is used,
            # we cannot determine the right re-mapped iteration space, so we bail here!
            if not any(isinstance(d, sym.Range) for d in qualified_value.dimensions):
                error(
                    '[Loki::TransformInline] Cannot find free dimension resolving '
                    f' array argument for value "{qualified_value}"'
                )
                raise RuntimeError(
                    f'[Loki::TransformInline] Cannot resolve procedure call to {call.name}'
                )
            arg_vars = tuple(v for v in callee_vars if v.name == arg.name)
            argmap.update((v, _map_unbound_dims(v, qualified_value)) for v in arg_vars)
        else:
            argmap[arg] = val

    # Deal with PRESENT check for optional arguments
    present_checks = tuple(
        check for check in FindInlineCalls().visit(callee.body) if check.function == 'PRESENT'
    )
    present_map = {
        check: sym.Literal('.true.') if check.arguments[0] in [arg.name for arg in call.arg_map]
                                     else sym.Literal('.false.')
        for check in present_checks
    }
    argmap.update(present_map)

    # Recursive update of the map in case of nested variables to map
    argmap = recursive_expression_map_update(argmap, max_iterations=10)

    # Substitute argument calls into a copy of the body
    callee_body = SubstituteExpressions(argmap, rebuild_scopes=True).visit(
        callee.body.body, scope=caller
    )

    # Inline substituted body within a pair of marker comments
    comment = Comment(f'! [Loki] inlined child subroutine: {callee.name}')
    c_line = Comment('! =========================================')
    return (comment, c_line) + as_tuple(callee_body) + (c_line, )


def inline_subroutine_calls(routine, calls, callee, allowed_aliases=None):
    """
    Inline a set of call to an individual :any:`Subroutine` at source level.

    This will replace all :any:`Call` objects to the specified
    subroutine with an adjusted equivalent of the member routines'
    body. For this, argument matching, including partial dimension
    matching for array references is performed, and all
    member-specific declarations are hoisted to the containing
    :any:`Subroutine`.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to inline all calls to the member routine
    calls : tuple or list of :any:`CallStatement`
    callee : :any:`Subroutine`
        The called target subroutine to be inlined in the parent
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope, even
        if they alias with a local declaration.
    """
    allowed_aliases = as_tuple(allowed_aliases)

    # Ensure we process sets of calls to the same callee
    assert all(call.routine == callee for call in calls)
    assert isinstance(callee, Subroutine)

    # Prevent shadowing of callee's variables by renaming them a priori
    parent_variables = routine.variable_map
    duplicates = tuple(
        v for v in callee.variables
        if v.name in parent_variables and v.name.lower() not in callee._dummies
    )
    # Filter out allowed aliases to prevent suffixing
    duplicates = tuple(v for v in duplicates if v.symbol not in allowed_aliases)
    shadow_mapper = SubstituteExpressions(
        {v: v.clone(name=f'{callee.name}_{v.name}') for v in duplicates}
    )
    callee.spec = shadow_mapper.visit(callee.spec)

    var_map = {}
    duplicate_names = {dl.name.lower() for dl in duplicates}
    for v in FindVariables(unique=False).visit(callee.body):
        if v.name.lower() in duplicate_names:
            var_map[v] = v.clone(name=f'{callee.name}_{v.name}')
    var_map = recursive_expression_map_update(var_map)
    callee.body = SubstituteExpressions(var_map).visit(callee.body)

    # Separate allowed aliases from other variables to ensure clean hoisting
    if allowed_aliases:
        single_variable_declaration(callee, variables=allowed_aliases)

    # Get local variable declarations and hoist them
    decls = FindNodes(VariableDeclaration).visit(callee.spec)
    decls = tuple(d for d in decls if all(s.name.lower() not in callee._dummies for s in d.symbols))
    decls = tuple(d for d in decls if all(s not in routine.variables for s in d.symbols))
    # Rescope the declaration symbols
    decls = tuple(d.clone(symbols=tuple(s.clone(scope=routine) for s in d.symbols)) for d in decls)

    # Find and apply symbol remappings for array size expressions
    symbol_map = dict(ChainMap(*[call.arg_map for call in calls]))
    decls = SubstituteExpressions(symbol_map).visit(decls)

    routine.spec.append(decls)

    # Resolve the call by mapping arguments into the called procedure's body
    call_map = {
        call: map_call_to_procedure_body(call, caller=routine) for call in calls
    }

    # Replace calls to child procedure with the child's body
    routine.body = Transformer(call_map).visit(routine.body)

    # We need this to ensure that symbols, as well as nested scopes
    # are correctly attached to each other (eg. nested associates).
    routine.rescope_symbols()

def inline_function_calls(routine, calls, callee, nodes, allowed_aliases=None):
    """
    Inline a set of call to an individual :any:`Subroutine` being functions
    at source level.

    This will replace all :any:`InlineCall` objects to the specified
    subroutine with an adjusted equivalent of the member routines'
    body. For this, argument matching, including partial dimension
    matching for array references is performed, and all
    member-specific declarations are hoisted to the containing
    :any:`Subroutine`.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to inline all calls to the member routine
    calls : tuple or list of :any:`InlineCall`
    callee : :any:`Subroutine`
        The called target function to be inlined in the parent
    nodes : :any:`Node`
        The corresponding nodes the functions are called from.
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope, even
        if they alias with a local declaration.
    """

    def rename_result_name(routine, rename=''):
        callee = routine.clone()
        var_map = {}
        callee_result_var = callee.variable_map[callee.result_name.lower()]
        new_callee_result_var = callee_result_var.clone(name=rename)
        var_map[callee_result_var] = new_callee_result_var
        callee_vars = [var for var in FindVariables().visit(callee.body)
                       if var.name.lower() == callee_result_var.name.lower()]
        var_map.update({var: var.clone(name=rename) for var in callee_vars})
        var_map = recursive_expression_map_update(var_map)
        callee.body = SubstituteExpressions(var_map).visit(callee.body)
        return callee, new_callee_result_var

    allowed_aliases = as_tuple(allowed_aliases)

    # Ensure we process sets of calls to the same callee
    assert all(call.routine == callee for call in calls)
    assert isinstance(callee, Subroutine)

    # Prevent shadowing of callee's variables by renaming them a priori
    parent_variables = routine.variable_map
    duplicates = tuple(
        v for v in callee.variables
        if v.name.lower() != callee.result_name.lower()
        and v.name in parent_variables and v.name.lower() not in callee._dummies
    )
    # Filter out allowed aliases to prevent suffixing
    duplicates = tuple(v for v in duplicates if v.symbol not in allowed_aliases)
    shadow_mapper = SubstituteExpressions(
        {v: v.clone(name=f'{callee.name}_{v.name}') for v in duplicates}
    )
    callee.spec = shadow_mapper.visit(callee.spec)

    var_map = {}
    duplicate_names = {dl.name.lower() for dl in duplicates}
    for v in FindVariables(unique=False).visit(callee.body):
        if v.name.lower() in duplicate_names:
            var_map[v] = v.clone(name=f'{callee.name}_{v.name}')

    var_map = recursive_expression_map_update(var_map)
    callee.body = SubstituteExpressions(var_map).visit(callee.body)

    # Separate allowed aliases from other variables to ensure clean hoisting
    if allowed_aliases:
        single_variable_declaration(callee, variables=allowed_aliases)

    single_variable_declaration(callee, variables=callee.result_name)
    # Get local variable declarations and hoist them
    decls = FindNodes(VariableDeclaration).visit(callee.spec)
    decls = tuple(d for d in decls if all(s.name.lower() != callee.result_name.lower() for s in d.symbols))
    decls = tuple(d for d in decls if all(s.name.lower() not in callee._dummies for s in d.symbols))
    decls = tuple(d for d in decls if all(s not in routine.variables for s in d.symbols))
    # Rescope the declaration symbols
    decls = tuple(d.clone(symbols=tuple(s.clone(scope=routine) for s in d.symbols)) for d in decls)

    # Find and apply symbol remappings for array size expressions
    symbol_map = dict(ChainMap(*[call.arg_map for call in calls]))
    decls = SubstituteExpressions(symbol_map).visit(decls)
    routine.spec.append(decls)

    # Handle result/return var/value
    new_symbols = set()
    result_var_map = {}
    adapted_calls = []
    rename_result_var = not len(nodes) == len(set(nodes))
    for i_call, call in enumerate(calls):
        callee_result_var = callee.variable_map[callee.result_name.lower()]
        prefix = ''
        new_callee_result_var_name = f'{prefix}result_{callee.result_name.lower()}_{i_call}'\
                if rename_result_var else f'{prefix}result_{callee.result_name.lower()}'
        new_callee, new_symbol = rename_result_name(callee, new_callee_result_var_name)
        adapted_calls.append(new_callee)
        new_symbols.add(new_symbol)
        if isinstance(callee_result_var, sym.Array):
            result_var_map[(nodes[i_call], call)] = callee_result_var.clone(name=new_callee_result_var_name,
                    dimensions=None)
        else:
            result_var_map[(nodes[i_call], call)] = callee_result_var.clone(name=new_callee_result_var_name)
    new_symbols = SubstituteExpressions(symbol_map).visit(as_tuple(new_symbols), recurse_to_declaration_attributes=True)
    routine.variables += as_tuple([symbol.clone(scope=routine) for symbol in new_symbols])

    # create node map to map nodes to be prepended (representing the functions) for each node
    node_map = {}
    call_map = {}
    for i_call, call in enumerate(calls):
        node_map.setdefault(nodes[i_call], []).extend(
                list(map_call_to_procedure_body(call, caller=routine, callee=adapted_calls[i_call]))
        )
        call_map.setdefault(nodes[i_call], {}).update({call: result_var_map[(nodes[i_call], call)]})
    return node_map, call_map


def inline_internal_procedures(routine, allowed_aliases=None):
    """
    Inline internal subroutines contained in an individual :any:`Subroutine`.

    Please note that internal functions are not yet supported!

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to inline all member routines
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope, even
        if they alias with a local declaration.
    """

    # Run through all members and invoke individual inlining transforms
    for child in routine.members:
        if child.is_function:
            inline_functions(routine, functions=(child,))
        else:
            calls = tuple(
                call for call in FindNodes(CallStatement).visit(routine.body)
                if call.routine == child
            )
            inline_subroutine_calls(routine, calls, child, allowed_aliases=allowed_aliases)

        # Can't use transformer to replace subroutine/function, so strip it manually
        contains_body = tuple(n for n in routine.contains.body if not n == child)
        routine.contains._update(body=contains_body)


inline_member_procedures = inline_internal_procedures


def inline_marked_subroutines(routine, allowed_aliases=None, adjust_imports=True):
    """
    Inline :any:`Subroutine` objects guided by pragma annotations.

    When encountering :any:`CallStatement` objects that are marked with a
    ``!$loki inline`` pragma, this utility will attempt to replace the call
    with the body of the called procedure and remap all passed arguments
    into the calling procedures scope.

    Please note that this utility requires :any:`CallStatement` objects
    to be "enriched" with external type information.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to look for pragma-marked procedures to inline
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope, even
        if they alias with a local declaration.
    adjust_imports : bool
        Adjust imports by removing the symbol of the inlined routine or adding
        imports needed by the imported routine (optional, default: True)
    """

    with pragmas_attached(routine, node_type=CallStatement):

        # Group the marked calls by callee routine
        call_sets = defaultdict(list)
        no_call_sets = defaultdict(list)
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.routine == BasicType.DEFERRED:
                continue

            if is_loki_pragma(call.pragma, starts_with='inline'):
                call_sets[call.routine].append(call)
            else:
                no_call_sets[call.routine].append(call)

        # Trigger per-call inlining on collected sets
        for callee, calls in call_sets.items():
            if callee:  # Skip the unattached calls (collected under None)
                inline_subroutine_calls(
                    routine, calls, callee, allowed_aliases=allowed_aliases
                )

    # Remove imported symbols that have become obsolete
    if adjust_imports:
        callees = tuple(callee.procedure_symbol for callee in call_sets.keys())
        not_inlined = tuple(callee.procedure_symbol for callee in no_call_sets.keys())

        import_map = {}
        for impt in FindNodes(Import).visit(routine.spec):
            # Remove interface header imports
            if any(f'{c.name.lower()}.intfb.h' == impt.module for c in callees):
                import_map[impt] = None

            if any(s.name in callees for s in impt.symbols):
                new_symbols = tuple(
                    s for s in impt.symbols if s.name not in callees or s.name in not_inlined
                )
                # Remove import if no further symbols used, otherwise clone with new symbols
                import_map[impt] = impt.clone(symbols=new_symbols) if new_symbols else None

        # Remove explicit interfaces of inlined routines
        for intf in routine.interfaces:
            if not intf.spec:
                _body = tuple(
	                    s.type.dtype.procedure for s in intf.symbols
	                    if s.name not in callees or s.name in not_inlined
                )
                if _body:
                    import_map[intf] = intf.clone(body=_body)
                else:
                    import_map[intf] = None

        # Now move any callee imports we might need over to the caller
        new_imports = set()
        imported_module_map = CaseInsensitiveDict((im.module, im) for im in routine.imports)
        for callee in call_sets.keys():
            for impt in callee.imports:

                # Add any callee module we do not yet know
                if impt.module not in imported_module_map:
                    new_imports.add(impt)

                # If we're importing the same module, check for missing symbols
                if m := imported_module_map.get(impt.module):
                    _m = import_map.get(m, m)
                    if not all(s in _m.symbols for s in impt.symbols):
                        new_symbols = tuple(s.rescope(routine) for s in impt.symbols)
                        import_map[m] = m.clone(symbols=tuple(set(_m.symbols + new_symbols)))

        # Finally, apply the import remapping
        routine.spec = Transformer(import_map).visit(routine.spec)

        # Add missing explicit interfaces from inlined subroutines
        new_intfs = []
        intf_symbols = routine.interface_symbols
        for callee in call_sets.keys():
            for intf in callee.interfaces:
                for s in intf.symbols:
                    if not s in intf_symbols:
                        new_intfs += [s.type.dtype.procedure,]

        if new_intfs:
            routine.spec.append(Interface(body=as_tuple(new_intfs)))

        # Add Fortran imports to the top, and C-style interface headers at the bottom
        c_imports = tuple(im for im in new_imports if im.c_import)
        f_imports = tuple(im for im in new_imports if not im.c_import)
        routine.spec.prepend(f_imports)
        routine.spec.append(c_imports)
