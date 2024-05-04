# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""
A small selection of utility transformations that resolve certain code
constructs to unify code structure and make reasoning about Fortran
code easier.
"""

from loki.batch import Transformation
from loki.expression import FindVariables, SubstituteExpressions, Array, RangeIndex
from loki.ir import CallStatement, FindNodes, Transformer, NestedTransformer
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import BasicType

from loki.transformations.utilities import recursive_expression_map_update


__all__ = [
    'SanitiseTransformation', 'resolve_associates',
    'ResolveAssociatesTransformer', 'transform_sequence_association',
    'transform_sequence_association_append_map'
]


class SanitiseTransformation(Transformation):
    """
    :any:`Transformation` object to apply several code sanitisation
    steps when batch-processing large source trees via the :any:`Scheduler`.

    Parameters
    ----------
    resolve_associate_mappings : bool
        Resolve ASSOCIATE mappings in body of processed subroutines; default: True.
    resolve_sequence_association : bool
        Replace scalars that are passed to array arguments with array
        ranges; default: False.
    """

    def __init__(
            self, resolve_associate_mappings=True, resolve_sequence_association=False
    ):
        self.resolve_associate_mappings = resolve_associate_mappings
        self.resolve_sequence_association = resolve_sequence_association

    def transform_subroutine(self, routine, **kwargs):

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        if self.resolve_associate_mappings:
            resolve_associates(routine)

        # Transform arrays passed with scalar syntax to array syntax
        if self.resolve_sequence_association:
            transform_sequence_association(routine)


def resolve_associates(routine):
    """
    Resolve :any:`Associate` mappings in the body of a given routine.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine for which to resolve all associate blocks.
    """
    routine.body = ResolveAssociatesTransformer().visit(routine.body)

    # Ensure that all symbols have the appropriate scope attached.
    # This is needed, as the parent of a symbol might have changed,
    # which affects the symbol's type-defining scope.
    routine.rescope_symbols()


class ResolveAssociatesTransformer(NestedTransformer):
    """
    :any:`Transformer` class to resolve :any:`Associate` nodes in IR trees

    This will replace each :any:`Associate` node with its own body,
    where all `identifier` symbols have been replaced with the
    corresponding `selector` expression defined in ``associations``.
    """

    def visit_Associate(self, o, **kwargs):
        # First head-recurse, so that all associate blocks beneath are resolved
        body = self.visit(o.body, **kwargs)

        # Create an inverse association map to look up replacements
        invert_assoc = CaseInsensitiveDict({v.name: k for k, v in o.associations})

        # Build the expression substitution map
        vmap = {}
        for v in FindVariables().visit(body):
            if v.name in invert_assoc:
                # Clone the expression to update its parentage and scoping
                inv = invert_assoc[v.name]
                if hasattr(v, 'dimensions'):
                    vmap[v] = inv.clone(dimensions=v.dimensions)
                else:
                    vmap[v] = inv

        # Apply the expression substitution map to itself to handle nested expressions
        vmap = recursive_expression_map_update(vmap)

        # Mark the associate block for replacement with its body, with all expressions replaced
        self.mapper[o] = SubstituteExpressions(vmap).visit(body)

        # Return the original object unchanged and let the tuple injection mechanism take care
        # of replacing it by its body - otherwise we would end up with nested tuples
        return o


def check_if_scalar_syntax(arg, dummy):
    """
    Check if an array argument, arg,
    is passed to an array dummy argument, dummy,
    using scalar syntax. i.e. arg(1,1) -> d(m,n)

    Parameters
    ----------
    arg:   variable
    dummy: variable
    """
    if isinstance(arg, Array) and isinstance(dummy, Array):
        if arg.dimensions:
            if not any(isinstance(d, RangeIndex) for d in arg.dimensions):
                return True
    return False


def transform_sequence_association(routine):
    """
    Housekeeping routine to replace scalar syntax when passing arrays as arguments
    For example, a call like

    .. code-block::

        real :: a(m,n)

        call myroutine(a(i,j))

    where myroutine looks like

    .. code-block::

        subroutine myroutine(a)
            real :: a(5)
        end subroutine myroutine

    should be changed to

    .. code-block::

        call myroutine(a(i:m,j)

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine where calls will be changed

    """

    #List calls in routine, but make sure we have the called routine definition
    calls = (c for c in FindNodes(CallStatement).visit(routine.body) if not c.procedure_type is BasicType.DEFERRED)
    call_map = {}

    # Check all calls and record changes to `call_map` if necessary.
    for call in calls:
        transform_sequence_association_append_map(call_map, call)

    # Fix sequence association in all calls in one go.
    if call_map:
        routine.body = Transformer(call_map).visit(routine.body)

def transform_sequence_association_append_map(call_map, call):
    """
    Check if `call` contains the sequence association pattern in one of the arguments,
    and if so, add the necessary transform data to `call_map`.
    """
    new_args = []
    found_scalar = False
    for dummy, arg in call.arg_map.items():
        if check_if_scalar_syntax(arg, dummy):
            found_scalar = True

            n_dims = len(dummy.shape)
            new_dims = []
            for s, lower in zip(arg.shape[:n_dims], arg.dimensions[:n_dims]):

                if isinstance(s, RangeIndex):
                    new_dims += [RangeIndex((lower, s.stop))]
                else:
                    new_dims += [RangeIndex((lower, s))]

            if len(arg.dimensions) > n_dims:
                new_dims += arg.dimensions[len(dummy.shape):]
            new_args += [arg.clone(dimensions=as_tuple(new_dims)),]
        else:
            new_args += [arg,]

    if found_scalar:
        call_map[call] = call.clone(arguments = as_tuple(new_args))
