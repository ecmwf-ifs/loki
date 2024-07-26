# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utility routines to deal with general language conversion.
"""

import platform
from collections import defaultdict
from pymbolic.primitives import Expression
from loki.expression import (
    symbols as sym, FindVariables, FindInlineCalls, FindLiterals,
    SubstituteExpressions, SubstituteExpressionsMapper, ExpressionFinder,
    ExpressionRetriever, TypedSymbol, MetaSymbol
)
from loki.ir import (
    Import, TypeDef, VariableDeclaration, StatementFunction,
    Transformer, FindNodes
)
from loki.module import Module
from loki.subroutine import Subroutine
from loki.tools import CaseInsensitiveDict, as_tuple
from loki.types import SymbolAttributes, BasicType, DerivedType, ProcedureType


__all__ = [
    'convert_to_lower_case', 'replace_intrinsics', 'rename_variables', 'sanitise_imports',
    'replace_selected_kind', 'single_variable_declaration', 'recursive_expression_map_update',
    'get_integer_variable'
]


def single_variable_declaration(routine, variables=None, group_by_shape=False):
    """
    Modify/extend variable declarations to

    * default: only declare one variable each time while preserving the order if ``variables=None`` and
      ``group_by_shape=False``
    * declare variables specified in ``variables``in single/unique declarations if ``variables`` is a tuple
      of variables
    * variable declarations to be grouped according to their shapes if ``group_by_shape=True``

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine in which to modify the variable declarations
    variables: tuple
        Variables to grant unique/single declaration for
    group_by_shape: bool
        Whether to strictly make unique variable declarations or to only disassemble non-arrays and arrays and among
        arrays, arrays with differing shapes.
    """
    decl_map = {}
    for decl in FindNodes(VariableDeclaration).visit(routine.spec):
        if len(decl.symbols) > 1:
            if not group_by_shape:
                unique_symbols = [s for s in decl.symbols if variables is None or s.name in variables]
                if unique_symbols:
                    new_decls = tuple(decl.clone(symbols=(s,)) for s in unique_symbols)
                    retain_symbols = tuple(s for s in decl.symbols if variables is not None and s.name not in variables)
                    if retain_symbols:
                        decl_map[decl] = (decl.clone(symbols=retain_symbols),) + new_decls
                    else:
                        decl_map[decl] = new_decls
            else:
                smbls_by_shape = defaultdict(list)
                for smbl in decl.symbols:
                    smbls_by_shape[getattr(smbl, 'shape', None)] += [smbl]
                decl_map[decl] = tuple(decl.clone(symbols=as_tuple(smbls)) for smbls in smbls_by_shape.values())
    routine.spec = Transformer(decl_map).visit(routine.spec)
    # if variables defined and group_by_shape, first call ignores the variables, thus second call
    if variables and group_by_shape:
        single_variable_declaration(routine=routine, variables=variables, group_by_shape=False)


def convert_to_lower_case(routine):
    """
    Converts all variables and symbols in a subroutine to lower-case.

    Note, this is intended for conversion to case-sensitive languages.

    TODO: Should be extended to `Module` objects.
    """

    # Force all variables in a subroutine body to lower-caps
    variables = FindVariables(unique=False).visit(routine.ir)
    vmap = {
        v: v.clone(name=v.name.lower()) for v in variables
        if isinstance(v, (sym.Scalar, sym.Array, sym.DeferredTypeSymbol)) and not v.name.islower()\
                and not v.case_sensitive
    }

    # Capture nesting by applying map to itself before applying to the routine
    vmap = recursive_expression_map_update(vmap)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)

    # Downcase inline calls to, but only after the above has been propagated,
    # so that we  capture the updates from the variable update in the arguments
    mapper = {
        c: c.clone(function=c.function.clone(name=c.name.lower() if not c.function.case_sensitive else c.name))
        for c in FindInlineCalls().visit(routine.ir) if not c.name.islower()
    }
    mapper.update(
        (stmt.variable, stmt.variable.clone(name=stmt.variable.name.lower()))
        for stmt in FindNodes(StatementFunction).visit(routine.spec)
    )
    routine.spec = SubstituteExpressions(mapper).visit(routine.spec)
    routine.body = SubstituteExpressions(mapper).visit(routine.body)


def replace_intrinsics(routine, function_map=None, symbol_map=None, case_sensitive=False):
    """
    Replace known intrinsic functions and symbols.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine object in which to replace intrinsic calls
    function_map : dict[str, str]
        Mapping from function names (:any:`InlineCall` names) to
        their replacement
    symbol_map : dict[str, str]
        Mapping from intrinsic symbol names to their replacement
    case_sensitive : bool
        Match case for name lookups in :data:`function_map` and :data:`symbol_map`
    """
    symbol_map = symbol_map or {}
    function_map = function_map or {}
    if not case_sensitive:
        symbol_map = CaseInsensitiveDict(symbol_map)
        function_map = CaseInsensitiveDict(function_map)
    # (intrinsic) functions
    callmap = {}
    for call in FindInlineCalls(unique=False).visit(routine.ir):
        if call.name in symbol_map:
            callmap[call] = sym.Variable(name=symbol_map[call.name], scope=routine)

        if call.name in function_map:
            callmap[call.function] = sym.ProcedureSymbol(name=function_map[call.name], scope=routine)

    routine.spec = SubstituteExpressions(callmap).visit(routine.spec)
    routine.body = SubstituteExpressions(callmap).visit(routine.body)

def rename_variables(routine, symbol_map=None):
    """
    Rename symbols/variables including (routine) arguments.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine object in which to rename variables.
    symbol_map : dict[str, str]
        Mapping from symbol/variable names to their replacement.
    """
    symbol_map = CaseInsensitiveDict(symbol_map) or {}
    # rename arguments if necessary
    arguments = ()
    renamed_arguments = ()
    for arg in routine.arguments:
        if arg.name in symbol_map:
            arguments += (arg.clone(name=symbol_map[arg.name]),)
            renamed_arguments += (arg,)
        else:
            arguments += (arg,)
    routine.arguments = arguments
    # remove variable declarations
    var_decls = FindNodes(VariableDeclaration).visit(routine.spec)
    var_decl_map = {}
    for var_decl in var_decls:
        new_symbols = ()
        for symbol in var_decl.symbols:
            if symbol not in renamed_arguments:
                new_symbols += (symbol,)
        if new_symbols:
            var_decl_map[var_decl] = var_decl.clone(symbols=new_symbols)
        else:
            var_decl_map[var_decl] = None
    routine.spec = Transformer(var_decl_map).visit(routine.spec)
    # rename variable declarations and usages
    var_map = {}
    for var in FindVariables(unique=False).visit(routine.ir):
        if var.name in symbol_map:
            new_var = symbol_map[var.name]
            if new_var is not None:
                var_map[var] = var.clone(name=symbol_map[var.name])
    if var_map:
        routine.spec = SubstituteExpressions(var_map).visit(routine.spec)
        routine.body = SubstituteExpressions(var_map).visit(routine.body)
    # update symbol table - remove entries under the previous name
    var_map_names = [key.name.lower() for key in var_map]
    delete = [key for key in routine.symbol_attrs if key.lower() in var_map_names\
            or key.split('%')[0].lower() in var_map_names] # derived types
    for key in delete:
        del routine.symbol_attrs[key]

def used_names_from_symbol(symbol, modifier=str.lower):
    """
    Helper routine that yields the symbol names for the different types of symbols
    we may encounter.
    """
    if isinstance(symbol, str):
        return {modifier(symbol)}

    if isinstance(symbol, (sym.TypedSymbol, sym.MetaSymbol)):
        return {modifier(symbol.name)} | used_names_from_symbol(symbol.type, modifier=modifier)

    if isinstance(symbol, SymbolAttributes):
        if isinstance(symbol.dtype, BasicType) and symbol.kind is not None:
            return {modifier(str(symbol.kind))}
        return used_names_from_symbol(symbol.dtype, modifier=modifier)

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

    redundant_symbols = {s for s in imported_symbols if s.name.lower() not in used_symbols}

    if redundant_symbols:
        imprt_map = {}
        for im in imports:
            if im.symbols is not None:
                symbols = tuple(s for s in im.symbols if s not in redundant_symbols)
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
    class SymbolRetriever(ExpressionFinder):

        retriever = ExpressionRetriever(lambda e: isinstance(e, (TypedSymbol, MetaSymbol)))

        def visit_Import(self, o, **kwargs):  # pylint: disable=unused-argument
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


class IsoFortranEnvMapper:
    """
    Mapper to convert other Fortran kind specifications to their definitions
    from ``iso_fortran_env``.
    """

    selected_kind_calls = ('selected_int_kind', 'selected_real_kind')

    def __init__(self, arch=None):
        if arch is None:
            arch = platform.machine()
        self.arch = arch.lower()
        self.used_names = CaseInsensitiveDict()

    @classmethod
    def is_selected_kind_call(cls, call):
        """
        Return ``True`` if the given call is a transformational function to
        select the kind of an integer or real type.
        """
        return isinstance(call, sym.InlineCall) and call.name.lower() in cls.selected_kind_calls

    @staticmethod
    def _selected_int_kind(r):
        """
        Return number of bytes required by the smallest signed integer type that
        is able to represent all integers n in the range -10**r < n < 10**r.

        This emulates the behaviour of Fortran's ``SELECTED_INT_KIND(R)``.

        Source: numpy.f2py.crackfortran
        https://github.com/numpy/numpy/blob/9e26d1d2be7a961a16f8fa9ff7820c33b25415e2/numpy/f2py/crackfortran.py#L2431-L2444

        :returns int: the number of bytes or -1 if no such type exists.
        """
        m = 10 ** r
        if m <= 2 ** 8:
            return 1
        if m <= 2 ** 16:
            return 2
        if m <= 2 ** 32:
            return 4
        if m <= 2 ** 63:
            return 8
        if m <= 2 ** 128:
            return 16
        return -1

    def map_selected_int_kind(self, scope, r):
        """
        Return the kind of the smallest signed integer type defined in
        ``iso_fortran_env`` that is able to represent all integers n
        in the range -10**r < n < 10**r.
        """
        byte_kind_map = {b: f'INT{8 * b}' for b in [1, 2, 4, 8]}
        kind = self._selected_int_kind(r)
        if kind in byte_kind_map:
            kind_name = byte_kind_map[kind]
            self.used_names[kind_name] = sym.Variable(name=kind_name, scope=scope)
            return self.used_names[kind_name]
        return sym.IntLiteral(-1)

    def _selected_real_kind(self, p, r=0, radix=0):  # pylint: disable=unused-argument
        """
        Return number of bytes required by the smallest real type that fulfils
        the given requirements:

        - decimal precision at least ``p``;
        - decimal exponent range at least ``r``;
        - radix ``r``.

        This resembles the behaviour of Fortran's ``SELECTED_REAL_KIND([P, R, RADIX])``.
        NB: This honors only ``p`` at the moment!

        Source: numpy.f2py.crackfortran
        https://github.com/numpy/numpy/blob/9e26d1d2be7a961a16f8fa9ff7820c33b25415e2/numpy/f2py/crackfortran.py#L2447-L2463

        :returns int: the number of bytes or -1 if no such type exists.
        """
        if p < 7:
            return 4
        if p < 16:
            return 8
        if self.arch.startswith(('aarch64', 'power', 'ppc', 'riscv', 's390x', 'sparc')):
            if p <= 20:
                return 16
        else:
            if p < 19:
                return 10
            if p <= 20:
                return 16
        return -1

    def map_selected_real_kind(self, scope, p, r=0, radix=0):
        """
        Return the kind of the smallest real type defined in
        ``iso_fortran_env`` that is able to fulfil the given requirements
        for decimal precision (``p``), decimal exponent range (``r``) and
        radix (``r``).
        """
        byte_kind_map = {b: f'REAL{8 * b}' for b in [4, 8, 16]}
        kind = self._selected_real_kind(p, r, radix)
        if kind in byte_kind_map:
            kind_name = byte_kind_map[kind]
            self.used_names[kind_name] = sym.Variable(name=kind_name, scope=scope)
            return self.used_names[kind_name]
        return sym.IntLiteral(-1)

    def map_call(self, call, scope):
        if not self.is_selected_kind_call(call):
            return call

        func = getattr(self, f'map_{call.name.lower()}')
        args = [int(arg) for arg in call.parameters]
        kwargs = {key: int(val) for key, val in call.kw_parameters.items()}

        return func(scope, *args, **kwargs)


def replace_selected_kind(routine):
    """
    Find all uses of ``selected_real_kind`` or ``selected_int_kind`` and
    replace them by their ``iso_fortran_env`` counterparts.

    This inserts imports for all used constants from ``iso_fortran_env``.
    """
    mapper = IsoFortranEnvMapper()

    # Find all selected_x_kind calls in spec and body
    calls = [call for call in FindInlineCalls().visit(routine.ir)
             if mapper.is_selected_kind_call(call)]

    # Need to pick out kinds in Literals explicitly
    calls += [literal.kind for literal in FindLiterals().visit(routine.ir)
              if hasattr(literal, 'kind') and mapper.is_selected_kind_call(literal.kind)]

    map_call = {call: mapper.map_call(call, routine) for call in calls}

    # Flush mapping through spec and body
    routine.spec = SubstituteExpressions(map_call).visit(routine.spec)
    routine.body = SubstituteExpressions(map_call).visit(routine.body)

    # Replace calls and literals hidden in variable kinds and inits
    for variable in routine.variables:
        if variable.type.kind is not None and mapper.is_selected_kind_call(variable.type.kind):
            kind = mapper.map_call(variable.type.kind, routine)
            routine.symbol_attrs[variable.name] = variable.type.clone(kind=kind)
        if variable.type.initial is not None:
            if mapper.is_selected_kind_call(variable.type.initial):
                initial = mapper.map_call(variable.type.initial, routine)
                routine.symbol_attrs[variable.name] = variable.type.clone(initial=initial)
            else:
                init_calls = [literal.kind for literal in FindLiterals().visit(variable.type.initial)
                              if hasattr(literal, 'kind') and mapper.is_selected_kind_call(literal.kind)]
                if init_calls:
                    init_map = {call: mapper.map_call(call, routine) for call in init_calls}
                    initial = SubstituteExpressions(init_map).visit(variable.type.initial)
                    routine.symbol_attrs[variable.name] = variable.type.clone(initial=initial)

    # Make sure iso_fortran_env symbols are imported
    if mapper.used_names:
        for imprt in FindNodes(Import).visit(routine.spec):
            if imprt.module.lower() == 'iso_fortran_env':
                # Update the existing iso_fortran_env import
                imprt_symbols = {str(s).lower() for s in imprt.symbols}
                missing_symbols = set(mapper.used_names.keys()) - imprt_symbols
                symbols = as_tuple(imprt.symbols) + tuple(mapper.used_names[s] for s in missing_symbols)

                # Flush the change through the spec
                routine.spec = Transformer({imprt: Import(imprt.module, symbols=symbols)}).visit(routine.spec)
                break
        else:
            # No iso_fortran_env import present, need to insert a new one
            imprt = Import('iso_fortran_env', symbols=as_tuple(mapper.used_names.values()))
            routine.spec.prepend(imprt)


def recursive_expression_map_update(expr_map, max_iterations=10, mapper_cls=SubstituteExpressionsMapper):
    """
    Utility function to apply a substitution map for expressions to itself

    The expression substitution mechanism :any:`SubstituteExpressions` and the
    underlying mapper :any:`SubstituteExpressionsMapper` replace nodes that
    are found in the substitution map by their corresponding replacement.

    However, expression nodes can be nested inside other expression nodes,
    e.g. via the ``parent`` or ``dimensions`` properties of variables.
    In situations, where such expression nodes as well as expression nodes
    appearing inside such properties are marked for substitution, it may
    be necessary to apply the substitution map to itself first. This utility
    routine takes care of that.

    Parameters
    ----------
    expr_map : dict
        The substitution map that should be updated
    max_iterations : int
        Maximum number of iterations, corresponds to the maximum level of
        nesting that can be replaced.
    mapper_cls: :any:`SubstituteExpressionsMapper`
       The underlying mapper to be used (default: :any:`SubstituteExpressionsMapper`).
    """
    def apply_to_init_arg(name, arg, expr, mapper):
        # Helper utility to apply the mapper only to expression arguments and
        # retain the scope while rebuilding the node
        if isinstance(arg, (tuple, Expression)):
            return mapper(arg)
        if name == 'scope':
            return expr.scope
        return arg

    for _ in range(max_iterations):
        # We update the expression map by applying it to the children of each replacement
        # node, thus making sure node replacements are also applied to nested attributes,
        # e.g. call arguments or array subscripts etc.
        mapper = mapper_cls(expr_map)
        prev_map, expr_map = expr_map, {
            expr: type(replacement)(**{
                name: apply_to_init_arg(name, arg, expr, mapper)
                for name, arg in zip(replacement.init_arg_names, replacement.__getinitargs__())
            })
            for expr, replacement in expr_map.items()
        }

        # Check for early termination opportunities
        if prev_map == expr_map:
            break

    return expr_map


def get_integer_variable(routine, name):
    """
    Find a local variable in the routine, or create an integer-typed one.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to find the variable
    name : string
        Name of the variable to find the in the routine.
    """
    if not (v_index := routine.symbol_map.get(name, None)):
        dtype = SymbolAttributes(BasicType.INTEGER)
        v_index = sym.Variable(name=name, type=dtype, scope=routine)
    return v_index
