# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.ir import (
    Import, Comment, Transformer, FindNodes, FindVariables,
    FindLiterals, SubstituteExpressions
)
from loki.expression import symbols as sym


__all__ = ['inline_constant_parameters']


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
