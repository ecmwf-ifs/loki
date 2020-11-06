"""
Collection of utility routines to deal general conversion.


"""
from loki.expression import (
    symbols as sym, FindVariables, FindInlineCalls,
    SubstituteExpressions, SubstituteExpressionsMapper
)
from loki.ir import Associate
from loki.visitors import Transformer, FindNodes


__all__ = ['convert_to_lower_case', 'replace_intrinsics', 'resolve_associates']


def convert_to_lower_case(routine):
    """
    Converts all variables and symbols in a subroutine to lower-case.

    Note, this is intended for conversion to case-sensitive languages.

    TODO: Should be extended to `Module` objects.
    """

    # Force all variables in a subroutine body to lower-caps
    vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(routine.body)
            if isinstance(v, (sym.Scalar, sym.Array)) and not v.name.islower()}
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Down-case all subroutine arguments and variables
    mapper = SubstituteExpressionsMapper(vmap)
    routine.arguments = [mapper(arg) for arg in routine.arguments]
    routine.variables = [mapper(var) for var in routine.variables]


def replace_intrinsics(routine, function_map=None, symbol_map=None):
    """
    Replace known numerical intrinsic functions and symbols.

    :param function_map: Map (string: string) for replacing intrinsic
                         functions (`InlineCall` objects).
    :param symbol_map: Map (string: string) for replacing intrinsic
                       symbols (`Variable` objects).
    """
    symbol_map = symbol_map or {}
    function_map = function_map or {}

    callmap = {}
    for c in FindInlineCalls(unique=False).visit(routine.body):
        cname = c.name.lower()

        if cname in symbol_map:
            callmap[c] = sym.Variable(name=symbol_map[cname], scope=routine.scope)

        if cname in function_map:
            fct_symbol = sym.ProcedureSymbol(function_map[cname], scope=routine.scope)
            callmap[c] = sym.InlineCall(fct_symbol, parameters=c.parameters,
                                        kw_parameters=c.kw_parameters)

    # Capture nesting by applying map to itself before applying to the routine
    for _ in range(2):
        mapper = SubstituteExpressionsMapper(callmap)
        callmap = {k: mapper(v) for k, v in callmap.items()}

    routine.body = SubstituteExpressions(callmap).visit(routine.body)


def resolve_associates(routine):
    """
    Resolve implicit struct mappings through "associates"
    """
    assoc_map = {}
    vmap = {}
    for assoc in FindNodes(Associate).visit(routine.body):
        invert_assoc = {v.name: k for k, v in assoc.associations.items()}
        for v in FindVariables(unique=False).visit(routine.body):
            if v.name in invert_assoc:
                vmap[v] = invert_assoc[v.name]
        assoc_map[assoc] = assoc.body
    routine.body = Transformer(assoc_map).visit(routine.body)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)
