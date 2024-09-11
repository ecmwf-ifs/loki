# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.ir import (
    Transformer, FindNodes, Import, StatementFunction,
    FindInlineCalls, SubstituteExpressions
)
from loki.types import BasicType

from loki.transformations.inline.mapper import InlineSubstitutionMapper
from loki.transformations.utilities import recursive_expression_map_update


__all__ = ['inline_elemental_functions', 'inline_statement_functions']


def inline_elemental_functions(routine):
    """
    Replaces `InlineCall` expression to elemental functions with the
    called functions body. This will attempt to resolve the elemental
    function into a single expression and perform a direct replacement
    at expression level.

    Note, that `InlineCall.function.type` is used to determine if a
    function cal be inlined. For functions imported via module use
    statements. This implies that the module needs to be provided in
    the `definitions` argument to the original ``Subroutine`` constructor.
    """

    # Keep track of removed symbols
    removed_functions = set()

    exprmap = {}
    for call in FindInlineCalls().visit(routine.body):
        if call.procedure_type is BasicType.DEFERRED:
            continue

        if call.procedure_type.is_function and call.procedure_type.is_elemental:
            # Map each call to its substitutions, as defined by the
            # recursive inline substitution mapper
            exprmap[call] = InlineSubstitutionMapper()(call, scope=routine)

            # Mark function as removed for later cleanup
            removed_functions.add(call.procedure_type)

    # Apply expression-level substitution to routine
    routine.body = SubstituteExpressions(exprmap).visit(routine.body)

    # Remove all module imports that have become obsolete now
    import_map = {}
    for im in FindNodes(Import).visit(routine.spec):
        if im.symbols and all(s.type.dtype in removed_functions for s in im.symbols):
            import_map[im] = None
    routine.spec = Transformer(import_map).visit(routine.spec)


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
