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
    Transformer, FindNodes, pragmas_attached, is_loki_pragma
)
from loki.expression import (
    symbols as sym, FindVariables, FindInlineCalls, FindLiterals,
    SubstituteExpressions, LokiIdentityMapper
)
from loki.types import BasicType
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.logging import warning, error
from loki.subroutine import Subroutine

from loki.transformations.remove_code import do_remove_dead_code
from loki.transformations.sanitise import transform_sequence_association_append_map
from loki.transformations.utilities import (
    single_variable_declaration, recursive_expression_map_update
)


__all__ = [
    'inline_constant_parameters', 'inline_elemental_functions',
    'inline_internal_procedures', 'inline_member_procedures',
    'inline_marked_subroutines', 'InlineTransformation'
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

    def __init__(
            self, inline_constants=False, inline_elementals=True,
            inline_internals=False, inline_marked=True,
            remove_dead_code=True, allowed_aliases=None,
            adjust_imports=True, external_only=True,
            resolve_sequence_association=False
    ):
        self.inline_constants = inline_constants
        self.inline_elementals = inline_elementals
        self.inline_internals = inline_internals
        self.inline_marked = inline_marked
        self.remove_dead_code = remove_dead_code
        self.allowed_aliases = allowed_aliases
        self.adjust_imports = adjust_imports
        self.external_only = external_only
        self.resolve_sequence_association = resolve_sequence_association

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

        function = expr.procedure_type.procedure
        v_result = [v for v in function.variables if v == function.name][0]

        # Substitute all arguments through the elemental body
        arg_map = dict(zip(function.arguments, expr.parameters))
        fbody = SubstituteExpressions(arg_map).visit(function.body)

        # Extract the RHS of the final result variable assignment
        stmts = [s for s in FindNodes(Assignment).visit(fbody) if s.lhs == v_result]
        assert len(stmts) == 1
        rhs = self.rec(stmts[0].rhs, *args, **kwargs)
        return rhs

def resolve_sequence_association_for_inlined_calls(routine, inline_internals, inline_marked):
    """
    Resolve sequence association in calls to all member procedures (if `inline_internals = True`) 
    or in calls to procedures that have been marked with an inline pragma (if `inline_marked = True`). 
    If both `inline_internals` and `inline_marked` are `False`, no processing is done.
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
                        f"Cannot resolve sequence association for call to `{call.name}` " + 
                        f"to be inlined in routine `{routine.name}`, because " + 
                        f"the `CallStatement` referring to `{call.name}` does not contain " + 
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


def map_call_to_procedure_body(call, caller):
    """
    Resolve arguments of a call and map to the called procedure body.

    Parameters
    ----------
    call : :any:`CallStatment` or :any:`InlineCall`
         Call object that defines the argument mapping
    caller : :any:`Subroutine`
         Procedure (scope) into which the callee's body gets mapped
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
    callee = call.routine
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
        check: sym.Literal('.true.') if check.arguments[0] in call.arg_map else sym.Literal('.false.')
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
            # TODO: Implement for functions!!!
            warning('[Loki::inline] Inlining internal functions is not yet supported, only subroutines!')
        else:
            calls = tuple(
                call for call in FindNodes(CallStatement).visit(routine.body)
                if call.routine == child
            )
            inline_subroutine_calls(routine, calls, child, allowed_aliases=allowed_aliases)

            # Can't use transformer to replace subroutine, so strip it manually
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
                    if not all(s in m.symbols for s in impt.symbols):
                        new_symbols = tuple(s.rescope(routine) for s in impt.symbols)
                        import_map[m] = m.clone(symbols=tuple(set(m.symbols + new_symbols)))

        # Finally, apply the import remapping
        routine.spec = Transformer(import_map).visit(routine.spec)

        # Add Fortran imports to the top, and C-style interface headers at the bottom
        c_imports = tuple(im for im in new_imports if im.c_import)
        f_imports = tuple(im for im in new_imports if not im.c_import)
        routine.spec.prepend(f_imports)
        routine.spec.append(c_imports)
