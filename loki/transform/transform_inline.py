"""
Collection of utility routines to perform code-level force-inlining.


"""
from loki.expression import (
    symbols as sym, FindVariables, FindInlineCalls, SubstituteExpressions,
    LokiIdentityMapper
)
from loki.ir import Declaration, Import, Comment, Assignment
from loki.types import BasicType
from loki.visitors import Transformer, FindNodes


__all__ = ['inline_constant_parameters']


class InlineSubstitutionMapper(LokiIdentityMapper):
    """
    An expression mapper that defines symbolic substitution for inlining.
    """

    def map_scalar(self, expr, *args, **kwargs):
        # Ensure that re-scope variable symbols
        kwargs['scope'] = kwargs.get('scope', expr.scope)
        return super().map_scalar(expr, *args, **kwargs)

    def map_array(self, expr, *args, **kwargs):
        # Ensure that re-scope variable symbols
        kwargs['scope'] = kwargs.get('scope', expr.scope)
        return super().map_array(expr, *args, **kwargs)

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
    Replace instances of variables with knwon constant values by `Literals`.

    :param external_only: Do not replace variables declared in the local scope

    Note, the `.type.initial` property is used to derive the replacement value,
    which means for symbols imported from external modules, the parent `Module`
    needs to be supplied in the `definitions` to the constructor when creating
    :param routine:.
    """
    # Find all variable instances in spec and body
    variables = [v for v in FindVariables().visit(routine.spec)]
    variables += [v for v in FindVariables().visit(routine.body)]

    # Filter out variables declared locally
    if external_only:
        variables = [v for v in variables if v not in routine.variables]

    # Create mapping and flush through spec and body
    vmap = {v: v.type.initial for v in variables
            if v.type.parameter and v.type.initial}
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)


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
