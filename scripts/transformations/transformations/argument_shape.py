"""
Utility transformation that performs inter-procedural analysis to
infer shape symbols for deferred array dimensions. For this it propagates
shape symbols from declaration or dynamic array allocations from a caller
to the called subroutine.
"""


from loki import (
    Transformation, FindNodes, CallStatement, Array, FindVariables,
    SubstituteExpressions, BasicType, as_tuple, Transformer
)


__all__ = ['ArgumentArrayShapeAnalysis', 'ExplicitArgumentArrayShapeTransformation']


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


class ExplicitArgumentArrayShapeTransformation(Transformation):
    """
    Inter-procedural :any:`Transformation` that inserts explicit array
    shape dimensions for :any:`Subroutine` arguments that use deferred
    shape notation, and updates :any:`CallStatement` in any calling
    subroutines in the traversal graph. Critically, this depends on
    the ``.shape`` attribute of the corresponding argument symbol
    being set, which can be derived from a call context via the
    accompanying :any:`ArgumentArrayShapeAnalysis` transformation.
    """

    def transform_subroutine(self, routine, **kwargs):  # pylint: disable=arguments-differ

        # First, replace assumed array shapes with concrete shapes for
        # all arguments if the shape is known.
        arg_map = {}
        for arg in routine.arguments:
            if isinstance(arg, Array):
                assumed = tuple(':' for _ in arg.shape)
                if arg.shape != assumed and arg.dimensions == assumed:
                    arg_map[arg] = arg.clone(dimensions=tuple(arg.shape))
        routine.spec = SubstituteExpressions(arg_map).visit(routine.spec)

        # We also need to ensure that all potential integer dimensions
        # are passed as arguments in deep subroutine call trees.
        call_map = {}
        for call in FindNodes(CallStatement).visit(routine.body):

            # Skip if call-side info is not available or call is not active
            if call.routine is BasicType.DEFERRED or call.not_active:
                continue

            callee = call.routine

            # Collect all potential dimension variables and filter for scalar integers
            dims = set(d for arg in callee.arguments if isinstance(arg, Array) for d in arg.shape)
            dim_vars = tuple(d for d in FindVariables().visit(as_tuple(dims)))

            # Add all new dimension arguments to the callee signature
            new_args = tuple(d for d in dim_vars if d not in callee.arguments)
            new_args = tuple(d for d in new_args if d.type.dtype == BasicType.INTEGER)
            new_args = tuple(d.clone(scope=routine, type=d.type.clone(intent='IN')) for d in new_args)
            callee.arguments += new_args

            # Map all local dimension args to unknown callee dimension args
            if len(callee.arguments) > len(list(call.arg_iter())):
                arg_keys = dict(call.arg_iter()).keys()
                missing = [a for a in callee.arguments if a not in arg_keys
                           and not a.type.optional and a in dim_vars]

                # Add missing dimension variables (scalars
                new_kwargs = tuple((m, m) for m in missing if m.type.dtype == BasicType.INTEGER)
                call_map[call] = call.clone(kwarguments=call.kwarguments + new_kwargs)

        # Replace all adjusted calls on the caller-side
        routine.body = Transformer(call_map).visit(routine.body)
