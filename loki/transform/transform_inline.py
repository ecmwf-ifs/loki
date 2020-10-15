"""
Collection of utility routines to perform code-level force-inlining.


"""
from loki.expression import (
    symbols as sym, FindVariables, SubstituteExpressions
)
from loki.ir import Declaration
from loki.visitors import Transformer, FindNodes


__all__ = ['inline_constant_parameters']


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
