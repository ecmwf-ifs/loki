# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import ChainMap

from loki.expression import symbols as sym, ExpressionRetriever
from loki.ir import (
    Transformer, FindNodes, FindVariables, Import, StatementFunction,
    FindInlineCalls, ExpressionFinder, SubstituteExpressions,
    VariableDeclaration
)
from loki.subroutine import Subroutine
from loki.types import BasicType
from loki.tools import as_tuple

from loki.transformations.inline.mapper import InlineSubstitutionMapper
from loki.transformations.inline.procedures import map_call_to_procedure_body
from loki.transformations.utilities import (
    single_variable_declaration, recursive_expression_map_update
)


__all__ = [
    'inline_elemental_functions', 'inline_functions',
    'inline_statement_functions', 'inline_function_calls'
]


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
        potentially_functions_to_be_inlined = _inline_functions(
            routine, inline_elementals_only=inline_elementals_only, functions=functions
        )

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

    # functions are provided, however functions is empty, thus early exit
    if functions is not None and not functions:
        return False
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
    for calls_nodes in function_calls.values():
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
        Set of calls (to the same callee) to be inlined.
    callee : :any:`Subroutine`
        The called target function to be inlined in the parent
    nodes : :any:`Node`
        The corresponding nodes the functions are called from.
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope, even
        if they alias with a local declaration.
    """

    def rename_result_name(routine, rename):
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
