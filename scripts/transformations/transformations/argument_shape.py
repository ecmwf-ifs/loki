"""
Utility transformation that performs inter-procedural analysis to
infer shape symbols for deferred array dimensions. For this it propagates
shape symbols from declaration or dynamic array allocations from a caller
to the called subroutine.
"""


from loki import (
    Transformation, FindNodes, CallStatement, Array, FindVariables, SubstituteExpressions,
    BasicType
)


__all__ = ['ArgumentArrayShapeAnalysis']


class ArgumentArrayShapeAnalysis(Transformation):
    """
    Inter-procedural analysis :any:`Transformation` that infers shape
    symbols for deferred argument array dimensions and sets the
    :param:`shape` attribute accordingly. For this it propagates shape
    symbols from from a caller to the called subroutine.

    Please note that this transformation propagates from the caller
    to the callee, so it needs to be applied in a forward order over
    large sets of interdependent subroutines in a dependency graph via
    the :any:`Scheduler`.

    Please also note that if the call-side shape of an array argument
    is either set, or has already been derived (possibly with
    conflicting information), this transformation will have no effect.
    """

    def transform_subroutine(self, routine, **kwargs):  # pylint: disable=arguments-differ

        for call in FindNodes(CallStatement).visit(routine.body):

            # Skip if call-side info is not available or call is not active
            if call.routine is BasicType.DEFERRED or call.not_active:
                continue

            routine = call.routine

            # Create a variable map with new shape information from source
            vmap = {}
            for arg, val in call.arg_iter():
                if isinstance(arg, Array) and len(arg.shape) > 0:
                    # Only create new shapes for deferred dimension args
                    if all(str(d) == ':' for d in arg.shape):
                        if len(val.shape) == len(arg.shape):
                            # We're passing the full value array, copy shape
                            vmap[arg] = arg.clone(type=arg.type.clone(shape=val.shape))
                        else:
                            # Passing a sub-array of val, find the right index
                            new_shape = [s for s, d in zip(val.shape, val.dimensions)
                                         if str(d) == ':']
                            vmap[arg] = arg.clone(type=arg.type.clone(shape=new_shape))

            # Propagate the updated variables to variable definitions in routine
            routine.variables = [vmap.get(v, v) for v in routine.variables]

            # And finally propagate this to the variable instances
            vname_map = {k.name.lower(): v for k, v in vmap.items()}
            vmap_body = {}
            for v in FindVariables(unique=False).visit(routine.body):
                if v.name.lower() in vname_map:
                    new_shape = vname_map[v.name.lower()].shape
                    vmap_body[v] = v.clone(type=v.type.clone(shape=new_shape))
            routine.body = SubstituteExpressions(vmap_body).visit(routine.body)
