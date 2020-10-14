"""
Collection of utility routines to deal general conversion.


"""
from loki.expression import (
    symbols as sym, SubstituteExpressions, FindInlineCalls, SubstituteExpressionsMapper
)


__all__ = ['replace_intrinsics']


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
            callmap[c] = sym.Variable(name=symbol_map[cname], scope=routine.symbols)

        if cname in function_map:
            callmap[c] = sym.InlineCall(function_map[cname], parameters=c.parameters,
                                        kw_parameters=c.kw_parameters)

    # Capture nesting by applying map to itself before applying to the routine
    for _ in range(2):
        mapper = SubstituteExpressionsMapper(callmap)
        callmap = {k: mapper(v) for k, v in callmap.items()}

    routine.body = SubstituteExpressions(callmap).visit(routine.body)
