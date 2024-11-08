# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import Array, RangeIndex
from loki.ir import nodes as ir, FindNodes, Transformer
from loki.tools import as_tuple
from loki.types import BasicType


__all__ = [
    'transform_sequence_association',
    'transform_sequence_association_append_map'
]


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
    calls = (c for c in FindNodes(ir.CallStatement).visit(routine.body) if not c.procedure_type is BasicType.DEFERRED)
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
