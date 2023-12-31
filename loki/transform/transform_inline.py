# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utility routines to perform code-level force-inlining.


"""
from collections import defaultdict

from loki.expression import (
    FindVariables, FindInlineCalls, FindLiterals,
    SubstituteExpressions, LokiIdentityMapper
)
from loki.ir import Import, Comment, Assignment, VariableDeclaration, CallStatement
from loki.expression import symbols as sym
from loki.types import BasicType
from loki.visitors import Transformer, FindNodes
from loki.tools import as_tuple
from loki.logging import warning, error
from loki.pragma_utils import pragmas_attached, is_loki_pragma
from loki.subroutine import Subroutine

from loki.transform.transformation import Transformation
from loki.transform.transform_dead_code import dead_code_elimination
from loki.transform.transform_sequence_association import transform_sequence_association
from loki.transform.transform_associates import resolve_associates
from loki.transform.transform_utilities import (
    single_variable_declaration,
    recursive_expression_map_update
)

__all__ = [
    'inline_constant_parameters', 'inline_elemental_functions',
    'inline_internal_procedures', 'inline_member_procedures',
    'inline_marked_subroutines', 'InlineTransformation'
]


class InlineTransformation(Transformation):
    """
    :any:`Transformation` object to apply several types of source inlining
    when batch-processing large source trees via the :any:`Scheduler`.

    Parameters
    ----------
    inline_constants : bool
        Replace instances of variables with known constant values by
        :any:`Literal`; default: False.
    inline_elementals : bool
        Replaces :any:`InlineCall` expression to elemental functions
        with the called function's body; default: True.
    inline_internals : bool
        Perform internal procedure inlining via
        :method:`inline_internal_procedures`; default: False.
    inline_marked : bool
        Inline :any:`Subroutine` objects marked by pragma annotations;
        default: True.
    resolve_associate_mappings : bool
        Resolve ASSOCIATE mappings in body of processed subroutines; default: True.
    resolve_sequence_association : bool
        Replace scalars that are passed to array arguments with array
        ranges; default: False.
    eliminate_dead_code : bool
        Perform dead code elimination, where unreachable branches are
        trimmed from the code; default@ True
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope during
        internal and pragma-driven inlining.
    remove_imports : bool
        Strip unused import symbols after pragma-inlining (optional, default: True)
    external_only : bool, optional
        Do not replace variables declared in the local scope when
        inlining constants (default: True)
    """

    # Ensure correct recursive inlining by traversing from the leaves
    reverse_traversal = True

    def __init__(
            self, inline_constants=False, inline_elementals=True,
            inline_internals=False, inline_marked=True,
            resolve_associate_mappings=True, resolve_sequence_association=False,
            eliminate_dead_code=True, allowed_aliases=None,
            remove_imports=True, external_only=True
    ):
        self.inline_constants = inline_constants
        self.inline_elementals = inline_elementals
        self.inline_internals = inline_internals
        self.inline_marked = inline_marked

        self.resolve_associate_mappings = resolve_associate_mappings
        self.resolve_sequence_association = resolve_sequence_association
        self.eliminate_dead_code = eliminate_dead_code

        self.allowed_aliases = allowed_aliases
        self.remove_imports = remove_imports
        self.external_only = external_only

    def transform_subroutine(self, routine, **kwargs):

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        if self.resolve_associate_mappings:
            resolve_associates(routine)

        # Transform arrays passed with scalar syntax to array syntax
        if self.resolve_sequence_association:
            transform_sequence_association(routine)

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
                remove_imports=self.remove_imports
            )

        # After inlining, attempt to trim unreachable code paths
        if self.eliminate_dead_code:
            dead_code_elimination(routine)


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


def inline_constant_parameters(routine, external_only=True):
    """
    Replace instances of variables with known constant values by `Literals`.

    Parameters
    ----------
    routine : :any:`Subroutine`
         Procedure in which to inline/resolve constant parameters.
    external_only : bool, optional
        Do not replace variables declared in the local scope (default: True)

    Notes
    -----
    The ``.type.initial`` property is used to derive the replacement
    value,a which means for symbols imported from external modules,
    the parent :any:`Module` needs to be supplied in the
    ``definitions`` to the constructor when creating :param:`routine`.

    Variables that are replaced are also removed from their
    corresponding import statements, with empty import statements
    being removed alltogether.
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
        if call.procedure_type is not BasicType.DEFERRED:
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
        if all(hasattr(s, 'type') and s.type.dtype in removed_functions for s in im.symbols):
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
    routine.spec.append(decls)

    # Resolve the call by mapping arguments into the called procedure's body
    call_map = {
        call: map_call_to_procedure_body(call, caller=routine) for call in calls
    }

    # Replace calls to child procedure with the child's body
    routine.body = Transformer(call_map).visit(routine.body)


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


def inline_marked_subroutines(routine, allowed_aliases=None, remove_imports=True):
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
    remove_imports : bool
        Strip unused import symbols after inlining (optional, default: True)
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
    if remove_imports:
        callees = tuple(callee.procedure_symbol for callee in call_sets.keys())
        not_inlined = tuple(callee.procedure_symbol for callee in no_call_sets.keys())

        import_map = {}
        for impt in FindNodes(Import).visit(routine.spec):
            if any(s.name in callees for s in impt.symbols):
                new_symbols = tuple(
                    s for s in impt.symbols if s.name not in callees or s.name in not_inlined
                )
                # Remove import if no further symbols used, otherwise clone with new symbols
                import_map[impt] = impt.clone(symbols=new_symbols) if new_symbols else None
        routine.spec = Transformer(import_map).visit(routine.spec)
