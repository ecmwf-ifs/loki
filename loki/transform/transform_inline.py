"""
Collection of utility routines to perform code-level force-inlining.


"""
from loki.expression import (
    FindVariables, FindInlineCalls, SubstituteExpressions, LokiIdentityMapper
)
from loki.ir import Import, Comment, Assignment
from loki.types import BasicType
from loki.visitors import Transformer, FindNodes


__all__ = ['inline_constant_parameters', 'inline_elemental_functions']


class InlineSubstitutionMapper(LokiIdentityMapper):
    """
    An expression mapper that defines symbolic substitution for inlining.
    """

    def map_scalar(self, expr, *args, **kwargs):
        parent = self.rec(expr.parent, *args, **kwargs) if expr.parent is not None else None

        scope = kwargs.get('scope', None) or expr.scope
        stype = expr.type
        # We're re-scoping an imported symbol
        if expr.scope != scope:
            stype = expr.type.clone()
        return  expr.__class__(expr.name, scope=scope, type=stype, parent=parent, source=expr.source)

    def map_array(self, expr, *args, **kwargs):
        if expr.dimensions:
            dimensions = self.rec(expr.dimensions, *args, **kwargs)
        else:
            dimensions = None
        parent = self.rec(expr.parent, *args, **kwargs) if expr.parent is not None else None

        scope = kwargs.get('scope', None) or expr.scope
        stype = expr.type
        # We're re-scoping an imported symbol
        if expr.scope != scope:
            stype = expr.type.clone()
        return expr.__class__(expr.name, scope=scope, type=stype, parent=parent,
                              dimensions=dimensions, source=expr.source)

    def map_inline_call(self, expr, *args, **kwargs):
        if expr.procedure_type is None or expr.procedure_type is BasicType.DEFERRED:
            # Unkonw inline call, potentially an intrinsic
            # We still need to recurse and ensure re-scoping
            return super().map_inline_call(expr, *args, **kwargs)

        scope = kwargs.get('scope')
        function = expr.procedure_type.procedure.clone(scope=scope)
        v_result = [v for v in function.variables if v == function.name][0]

        # Substitute all arguments through the elemental body
        arg_map = dict(zip(function.arguments, expr.parameters))
        fbody = SubstituteExpressions(arg_map).visit(function.body)

        # Extract the RHS of the final result variable assignment
        stmts = [s for s in FindNodes(Assignment).visit(fbody) if s.lhs == v_result]
        assert len(stmts) == 1
        rhs = stmts[0].rhs

        return self.rec(rhs, *args, **kwargs)


def inline_constant_parameters(routine, external_only=True):
    """
    Replace instances of variables with known constant values by `Literals`.

    :param external_only: Do not replace variables declared in the local scope

    Note, the `.type.initial` property is used to derive the replacement value,
    which means for symbols imported from external modules, the parent `Module`
    needs to be supplied in the `definitions` to the constructor when creating
    :param routine:.

    Variables that are replaced are also removed from their corresponding import
    statements, with empty import statements being removed alltogether.
    """
    # Find all variable instances in spec and body
    variables = FindVariables().visit(routine.ir)

    # Filter out variables declared locally
    if external_only:
        variables = [v for v in variables if v not in routine.variables]

    # Create mapping for variables and imports
    vmap = {v: v.type.initial for v in variables
            if v.type.parameter and v.type.initial}
    imprtmap = {}
    for imprt in FindNodes(Import).visit(routine.spec):
        if imprt.symbols:
            symbols = tuple(s for s in imprt.symbols if s not in vmap)
            if not symbols:
                imprtmap[imprt] = Comment('! Loki: parameters from {} inlined'.format(imprt.module))
            elif len(symbols) < len(imprt.symbols):
                imprtmap[imprt] = imprt.clone(symbols=symbols)

    # Flush mappings through spec and body
    routine.spec = Transformer(imprtmap).visit(routine.spec)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Replace kind parameters in variable types
    for variable in routine.variables:
        if variable.type.kind is not None and variable.type.kind in vmap:
            variable.type = variable.type.clone(kind=vmap[variable.type.kind])


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
    removed_functions = []

    exprmap = {}
    for call in FindInlineCalls().visit(routine.body):
        if call.procedure_type is not BasicType.DEFERRED:
            # Map each call to its substitutions, as defined by the
            # recursive inline stubstitution mapper
            exprmap[call] = InlineSubstitutionMapper()(call, scope=routine.scope)

            # Mark function as removed for later cleanup
            removed_functions.append(call.procedure_type)

    # Apply expression-level substitution to routine
    routine.body = SubstituteExpressions(exprmap).visit(routine.body)

    # Remove all module imports that have become obsolete now
    import_map = {}
    for im in FindNodes(Import).visit(routine.spec):
        if all(s in removed_functions for s in im.symbols):
            import_map[im] = None
    routine.spec = Transformer(import_map).visit(routine.spec)
