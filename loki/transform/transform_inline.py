# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utility routines to perform code-level force-inlining.


"""
from loki.expression import (
    FindVariables, FindInlineCalls, FindLiterals,
    SubstituteExpressions, SubstituteExpressionsMapper, LokiIdentityMapper
)
from loki.ir import Import, Comment, Assignment, VariableDeclaration, CallStatement
from loki.expression import symbols as sym
from loki.types import BasicType
from loki.visitors import Transformer, FindNodes
from loki.tools import as_tuple
from loki.logging import warning


__all__ = [
    'inline_constant_parameters', 'inline_elemental_functions',
    'inline_member_procedures'
]


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


def inline_member_routine(routine, member):
    """
    Inline an individual member :any:`Subroutine` at source level.

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
    member : :any:`Subroutine`
        The contained member subroutine to be inlined in the parent
    """

    def _map_unbound_dims(var, val):
        """
        Maps all unbound dimension ranges in the passed array value
        ``val`` with the indices from the local variable ``var``. It
        returns the re-mapped symbol.

        For example, mapping the passed array ``m(:,j)`` to the local
        expression ``a(i)`` yields ``m(i,j)``.
        """
        val_free_dims = tuple(d for d in val.dimensions if isinstance(d, sym.Range))
        var_bound_dims = tuple(d for d in var.dimensions if not isinstance(d, sym.Range))
        mapper = SubstituteExpressionsMapper(dict(zip(val_free_dims, var_bound_dims)))
        return mapper(val)

    # Get local variable declarations and hoist them
    decls = FindNodes(VariableDeclaration).visit(member.spec)
    decls = tuple(d for d in decls if all(s.name not in routine._dummies for s in d.symbols))
    decls = tuple(d for d in decls if all(s not in routine.variables for s in d.symbols))
    routine.spec.append(decls)

    call_map = {}
    for call in FindNodes(CallStatement).visit(routine.body):
        if call.routine == member:
            argmap = {}
            member_vars = FindVariables().visit(member.body)

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
                    arg_vars = tuple(v for v in member_vars if v.name == arg.name)
                    argmap.update((v, _map_unbound_dims(v, qualified_value)) for v in arg_vars)
                else:
                    argmap[arg] = val

            # Substitute argument calls into a copy of the body
            member_body = SubstituteExpressions(argmap).visit(member.body.body)

            # Inline substituted body within a pair of marker comments
            comment = Comment(f'! [Loki] inlined member subroutine: {member.name}')
            c_line = Comment('! =========================================')
            call_map[call] = (comment, c_line) + as_tuple(member_body) + (c_line, )

    # Replace calls to member with the member's body
    routine.body = Transformer(call_map).visit(routine.body)
    # Can't use transformer to replace subroutine, so strip it manually
    contains_body = tuple(n for n in routine.contains.body if not n == member)
    routine.contains._update(body=contains_body)


def inline_member_procedures(routine):
    """
    Inline all member subroutines contained in an individual :any:`Subroutine`.

    Please note that member functions are not yet supported!

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to inline all member routines
    """

    # Run through all members and invoke individual inlining transforms
    for member in routine.members:
        if member.is_function:
            # TODO: Implement for functions!!!
            warning('[Loki::inline] Inlining member functions is not yet supported, only subroutines!')
        else:
            inline_member_routine(routine, member)
