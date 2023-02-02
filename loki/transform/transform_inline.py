# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utility routines to perform code-level force-inlining.


"""
from itertools import zip_longest

from loki.expression import (
    FindVariables, FindInlineCalls, FindLiterals,
    SubstituteExpressions, LokiIdentityMapper
)
from loki.ir import Import, Comment, Assignment, VariableDeclaration, CallStatement
from loki.expression import symbols as sym
from loki.types import BasicType
from loki.visitors import Transformer, FindNodes
from loki.subroutine import Subroutine
from loki.tools import as_tuple


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

    # Get local variable declarations and hoist them
    decls = FindNodes(VariableDeclaration).visit(member.spec)
    decls = tuple(d for d in decls if all(s not in member.arguments for s in d.symbols))
    # decls = tuple(d for d in decls if all(not s.type.intent for s in d.symbols))
    decls = tuple(d for d in decls if all(s not in routine.variables for s in d.symbols))
    # TODO: Take care of aliasing declarations and
    # mutli-declarations, where individual ones need hoisting!

    ############# TODO ####################
    # One of the hoisted declarations loses the type.shape attribute!!!
    expr_map = {}
    for v in FindVariables().visit(member.body):
        expr_map[v] = v.clone(scope=routine, type=v.type.clone())
    decl = SubstituteExpressions(expr_map).visit(decls)

    # TODO: Make sure that if opposing declaration exists, we preface the variable name
    # in fact, we should always to that, maybe?

    routine.spec.append(decls)

    call_map = {}
    for call in FindNodes(CallStatement).visit(routine.body):
        if call.routine == member:
            # Get all array references in the member's body
            member_arrays = FindVariables(unique=True).visit(member.body)

            arg_subs = {}
            arg_subs = dict((k, v) for k, v in call.arg_iter() if isinstance(k, sym.Scalar))
            for k, v in call.arg_iter():
                if isinstance(k, sym.Scalar):
                    arg_subs[k] = v

                if isinstance(k, sym.Array):
                    candidates = tuple(a for a in member_arrays if a.name.lower() == k.name.lower())
                    for c in candidates:

                        # We currently do not support scalar-to-array argument passing here!
                        # TODO: This might need to go!
                        # assert c.shape == v.shape
                        # if not c.shape == v.shape:
                        #     print(f'ml805 shape not matching:: {c.shape} =!= {v.shape}')

                        ### TODO: This still needs work to cover all the corner cases
                        if not v.dimensions:
                            new_dims = c.dimensions
                        else:
                            new_dims = tuple(
                                val if arg == ':' else arg
                                for arg, val in zip_longest(v.dimensions, c.dimensions)
                            )

                        new_type = v.type.clone(shape=c.type.shape)
                        # arg_subs[c] = v.clone(dimensions=c.dimensions, type=new_type)
                        arg_subs[c] = v.clone(dimensions=new_dims, type=new_type)

            # Substitute argument calls into a copy of the body
            member_body = SubstituteExpressions(arg_subs).visit(member.body.body)

            # Inline substituted body within a pair of marker comments
            comment = Comment(f'! [Loki] inlined member subroutine: {member.name}')
            c_line = Comment(f'! =========================================')
            call_map[call] = (comment, c_line) + as_tuple(member_body) + (c_line, )

    # Replace calls to member with the member's body
    routine.body = Transformer(call_map).visit(routine.body)

    argument_map = {a.name: a for a in routine.arguments}
    private_arrays = [v for v in routine.variables if not v.name in argument_map]
    private_arrays = [v for v in private_arrays if isinstance(v, sym.Array)]
    arr_no_shape = [a for a in private_arrays if not a.shape]

    # Ensure that inserted symbols are scoped to the new parent
    routine.rescope_symbols()

    argument_map = {a.name: a for a in routine.arguments}
    private_arrays = [v for v in routine.variables if not v.name in argument_map]
    private_arrays = [v for v in private_arrays if isinstance(v, sym.Array)]
    arr_no_shape = [a for a in private_arrays if not a.shape]

    # Can't use transformer to replace subroutine, so strip it manually
    contains_body = tuple(n for n in routine.contains.body if not n == member)
    routine.contains._update(body=contains_body)


def inline_member_procedures(routine):
    """

    """
    # Run through all members and invoke individual inlining transforms
    for member in routine.members:
        if isinstance(member, Subroutine):
            inline_member_routine(routine, member)
        # TODO: Implement for functions!!!
