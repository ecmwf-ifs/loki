"""
Collection of utility routines to deal with general language conversion.


"""
from loki import Subroutine, Module
from loki.expression import (
    symbols as sym, FindVariables, FindInlineCalls,
    SubstituteExpressions, SubstituteExpressionsMapper, FindTypedSymbols
)
from loki.ir import Associate, Import, TypeDef
from loki.visitors import Transformer, FindNodes
from loki.tools import CaseInsensitiveDict
from loki.types import SymbolType, BasicType, DerivedType, ProcedureType


__all__ = ['convert_to_lower_case', 'replace_intrinsics', 'resolve_associates', 'sanitise_imports']


def convert_to_lower_case(routine):
    """
    Converts all variables and symbols in a subroutine to lower-case.

    Note, this is intended for conversion to case-sensitive languages.

    TODO: Should be extended to `Module` objects.
    """

    # Force all variables in a subroutine body to lower-caps
    variables = [v for v in FindVariables().visit(routine.spec)]
    variables += [v for v in FindVariables().visit(routine.body)]
    vmap = {v: v.clone(name=v.name.lower()) for v in variables
            if isinstance(v, (sym.Scalar, sym.Array)) and not v.name.islower()}

    # Capture nesting by applying map to itself before applying to the routine
    for _ in range(2):
        mapper = SubstituteExpressionsMapper(vmap)
        vmap = {k: mapper(v) for k, v in vmap.items()}

    routine.body = SubstituteExpressions(vmap).visit(routine.body)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)

    # Down-case all subroutine arguments and variables
    mapper = SubstituteExpressionsMapper(vmap)

    routine.arguments = [mapper(arg) for arg in routine.arguments]
    routine.variables = [mapper(var) for var in routine.variables]


def replace_intrinsics(routine, function_map=None, symbol_map=None):
    """
    Replace known intrinsic functions and symbols.

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
        invert_assoc = CaseInsensitiveDict({v.name: k for k, v in assoc.associations.items()})
        for v in FindVariables(unique=False).visit(routine.body):
            if v.name in invert_assoc:
                vmap[v] = invert_assoc[v.name]
        assoc_map[assoc] = assoc.body
    routine.body = Transformer(assoc_map).visit(routine.body)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)


def used_names_from_symbol(symbol, modifier=str.lower):
    """
    Helper routine that yields the symbol names for the different types of symbols
    we may encounter.
    """
    if isinstance(symbol, str):
        return {modifier(symbol)}

    if isinstance(symbol, sym.TypedSymbol):
        return {modifier(symbol.name)} | used_names_from_symbol(symbol.type, modifier=modifier)

    if isinstance(symbol, SymbolType):
        if isinstance(symbol.dtype, DerivedType):
            return {modifier(symbol.dtype.name)}
        if isinstance(symbol.dtype, BasicType) and symbol.kind is not None:
            return {modifier(str(symbol.kind))}

    if isinstance(symbol, (DerivedType, ProcedureType)):
        return {modifier(symbol.name)}

    return set()


def eliminate_unused_imports(module_or_routine, used_symbols):
    """
    Eliminate any imported symbols (or imports alltogether) that are not
    in the set of used symbols.
    """
    imports = FindNodes(Import).visit(module_or_routine.spec)
    imported_symbols = [s for im in imports for s in im.symbols or []]

    redundant_symbols = {s for s in imported_symbols if str(s).lower() not in used_symbols}

    if redundant_symbols:
        imprt_map = {}
        for im in imports:
            if im.symbols is not None:
                symbols = [s for s in im.symbols if s not in redundant_symbols]
                if not symbols:
                    # Symbol list is empty: Remove the import
                    imprt_map[im] = None
                elif len(symbols) < len(im.symbols):
                    # Symbol list is shorter than before: We need to replace that import
                    imprt_map[im] = im.clone(symbols=symbols)
        module_or_routine.spec = Transformer(imprt_map).visit(module_or_routine.spec)


def find_and_eliminate_unused_imports(routine):
    """
    Find all unused imported symbols and eliminate them from their import statements
    in the given routine and all contained members.
    Empty import statements are removed.

    The accumulated set of used symbols is returned.
    """
    # We need a custom expression retriever that does not return symbols used in Imports
    class SymbolRetriever(FindTypedSymbols):
        def visit_Import(self, o, **kwargs):  # pylint: disable=unused-argument,no-self-use
            return ()

    # Find all used symbols
    used_symbols = set.union(*[used_names_from_symbol(s)
                               for s in SymbolRetriever().visit([routine.spec, routine.body])])
    used_symbols |= set.union(*[used_names_from_symbol(s) for s in routine.variables])
    for typedef in FindNodes(TypeDef).visit(routine.spec):
        used_symbols |= set.union(*[used_names_from_symbol(s) for s in typedef.variables])

    # Recurse for contained subroutines/functions
    for member in routine.members:
        used_symbols |= find_and_eliminate_unused_imports(member)

    eliminate_unused_imports(routine, used_symbols)
    return used_symbols


def sanitise_imports(module_or_routine):
    """
    Sanitise imports by removing unused symbols and eliminating imports
    with empty symbol lists.

    Note that this is currently limited to imports that are identified to be :class:`Scalar`,
    :class:`Array`, or :class:`ProcedureSymbol`.
    """
    if isinstance(module_or_routine, Subroutine):
        find_and_eliminate_unused_imports(module_or_routine)
    elif isinstance(module_or_routine, Module):
        used_symbols = set()
        for routine in module_or_routine.subroutines:
            used_symbols |= find_and_eliminate_unused_imports(routine)
        eliminate_unused_imports(module_or_routine, used_symbols)
