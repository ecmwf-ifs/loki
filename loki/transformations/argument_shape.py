# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Inter-procedural analysis passes to derive and augment argument array shapes.

A pair of utility :any:`Transformation` classes that allows thea shape
of array arguments with deferred dimensions to be derived from the
calling context via inter-procedural analysis.

To infer the declared dimensions of array arguments
:any:`ArgumentArrayShapeAnalysis` needs to be applied first to set the
``shape`` property on respective :any:`Array` symbols, before
:any:`ExplicitArgumentArrayShapeTransformation` can be applied in a
reverse traversal order to apply the necessary changes to argument
declarations and call signatures.
"""

from loki.batch import Transformation
from loki.expression import symbols as sym, simplify
from loki.ir import (
    nodes as ir, FindNodes, CallStatement, Transformer, FindVariables,
    SubstituteExpressions
)
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import BasicType


__all__ = [
    'ArgumentArrayShapeAnalysis', 'ExplicitArgumentArrayShapeTransformation',
    'infer_array_shape_caller'
]


class ArgumentArrayShapeAnalysis(Transformation):
    """
    Infer shape of array arguments with deferred shape.

    An inter-procedural analysis pass that passively infers the shape
    symbols for argument arrays from calling contexts and sets the
    ``shape`` attribute on :any:`Array` symbols accordingly.

    The shape information is propagated from a caller to the called
    subroutine in a forward traversal of the call-tree. If the
    call-side shape of an array argument is either set, or has already
    been derived (possibly with conflicting information), this
    transformation will have no effect.

    Note: This transformation does not affect the generated source
    code, as it only sets the ``shape`` property, which is ignored
    during the code generation step (:any:`fgen`). To actively change
    the argument array declarations and the resulting source code, the
    :any:`ExplicitArgumentArrayShapeTransformation` needs to be applied
    `after` this transformation.
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
                if isinstance(arg, sym.Array) and len(arg.shape) > 0:
                    # Only create new shapes for deferred dimension args
                    if all(d == ':' for d in arg.shape):
                        if len(val.shape) == len(arg.shape):
                            # We're passing the full value array, copy shape
                            vmap[arg] = arg.clone(type=arg.type.clone(shape=val.shape))
                        else:
                            # Passing a sub-array of val, find the right index
                            new_shape = [s for s, d in zip(val.shape, val.dimensions)
                                         if d == ':']
                            vmap[arg] = arg.clone(type=arg.type.clone(shape=new_shape))

                    elif str(arg.shape[-1])[-1] == '*':
                        expl_shape_len = len(arg.shape) - 1

                        dims = val.shape
                        if val.dimensions:
                            dims = [d for d in val.dimensions if not isinstance(d, sym.Scalar)]
                            # sanitise unbounded ranges in argument
                            dims = [sym.RangeIndex((d.lower or getattr(val.shape, 'lower', sym.IntLiteral(1)),
                                                    d.upper or getattr(val.shape, 'upper', val.shape[i])))
                                                    for i, d in enumerate(dims)]

                        # determine argument dimension sizes
                        sizes = [simplify(sym.Sum((getattr(d, 'upper', d),
                                          sym.Product((sym.IntLiteral(-1), getattr(d, 'lower', sym.IntLiteral(1)))),
                                          sym.IntLiteral(1)))) for d in dims]

                        # determine explicit size corresponding to assumed size
                        expl_size = simplify(sym.Product(tuple(s for s in sizes[expl_shape_len:])))

                        new_shape = list(arg.shape)
                        if isinstance(arg.shape[expl_shape_len], sym.RangeIndex):
                            lower = arg.shape[expl_shape_len].lower
                            upper = simplify(sym.Sum((expl_size, sym.IntLiteral(-1), lower)))
                            new_shape[expl_shape_len] = sym.RangeIndex((lower, upper))
                        else:
                            new_shape[expl_shape_len] = expl_size
                        vmap[arg] = arg.clone(type=arg.type.clone(shape=as_tuple(new_shape)))

            # Propagate the updated variables to variable definitions in routine
            routine.variables = [vmap.get(v, v) for v in routine.variables]

            # And finally propagate this to the variable instances
            vname_map = CaseInsensitiveDict((k.name, v) for k, v in vmap.items())
            vmap_body = {}
            for v in FindVariables(unique=False).visit(routine.body):
                if v.name in vname_map:
                    new_shape = vname_map[v.name].shape
                    vmap_body[v] = v.clone(type=v.type.clone(shape=new_shape))
            routine.body = SubstituteExpressions(vmap_body).visit(routine.body)


class ExplicitArgumentArrayShapeTransformation(Transformation):
    """
    Add dimensions to array arguments and adjust call signatures.

    Adjusts array argument declarations by inserting explicit shape
    variables according to the ``shape`` property of the :any:`Array`
    symbol. This property can be derived from the calling context via
    :any:`ArgumentArrayShapeAnalysis`.

    If the :any:`Scalar` symbol defining an array dimension is not yet
    known in the local :any:`Subroutine`, it gets added to the call
    signature. In the caller routine, the respective :any:`Scalar`
    argument is added to the :any:`CallStatement` via keyword-argument
    notation.

    Note: Since the :any:`CallStatement` needs updating after the called
    :any:`Subroutine` signature, this transformation has to be applied
    in reverse order via ``Scheduler.process(..., reverse=True)``.
    """

    # We need to traverse call tree in reverse to ensure called
    # procedures are updated before callers.
    reverse_traversal = True

    def transform_subroutine(self, routine, **kwargs):  # pylint: disable=arguments-differ

        def assumed(dims):
            return all(d == ':' for d in dims) or str(dims[-1])[-1] == '*'

        # First, replace assumed array shapes with concrete shapes for
        # all arguments if the shape is known.
        arg_map = {}
        for arg in routine.arguments:
            if isinstance(arg, sym.Array):
                if not assumed(arg.shape) and assumed(arg.dimensions):
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
            imported_symbols = callee.imported_symbols
            if callee.parent is not None:
                imported_symbols += callee.parent.imported_symbols

            # Collect all potential dimension variables and filter for scalar integers
            dims = set(d for arg in callee.arguments if isinstance(arg, sym.Array) for d in arg.shape)
            dim_vars = tuple(d for d in FindVariables().visit(as_tuple(dims)))

            # Add all new dimension arguments to the callee signature
            new_args = tuple(d for d in dim_vars if d not in callee.arguments)
            new_args = tuple(d for d in new_args if d.type.dtype == BasicType.INTEGER)
            new_args = tuple(d for d in new_args if d not in imported_symbols)
            new_args = tuple(d.clone(scope=routine, type=d.type.clone(intent='IN')) for d in new_args)
            callee.arguments += new_args

            # Map all local dimension args to unknown callee dimension args
            if len(callee.arguments) > len(list(call.arg_iter())):
                arg_keys = dict(call.arg_iter()).keys()
                missing = [a for a in callee.arguments if a not in arg_keys
                           and not a.type.optional and a in dim_vars]

                # Add missing dimension variables (scalars
                new_kwargs = tuple((str(m), m) for m in missing if m.type.dtype == BasicType.INTEGER)
                call_map[call] = call.clone(kwarguments=call.kwarguments + new_kwargs)

        # Replace all adjusted calls on the caller-side
        routine.body = Transformer(call_map).visit(routine.body)


def infer_array_shape_caller(region):
    """
    Infer shape and type information for argument arrays from callee, if not on argument
    """
    for call in FindNodes(ir.CallStatement).visit(region):
        if call.routine is BasicType.DEFERRED:
            continue

        arg_map = {}
        for param, arg in call.arg_iter():
            if not isinstance(param, sym.Array):
                continue

            if not isinstance(arg, sym.Array) or not arg.dimensions or not arg.shape:
                newdims = tuple(sym.RangeIndex((None, None)) for _ in param.shape)
                newtype = arg.type.clone(dtype=param.type.dtype, kind=param.type.kind, shape=param.shape)
                arg_map[arg] = arg.clone(dimensions=newdims, type=newtype)

        arguments = tuple(arg_map[a] if a in arg_map else a for a in call.arguments)
        kwarguments = tuple((k, arg_map[v] if v in arg_map else v) for k, v in call.kwarguments)
        call._update(arguments=arguments, kwarguments=kwarguments)
