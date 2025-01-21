# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import Array, RangeIndex
from loki.ir import Transformer
from loki.tools import as_tuple
from loki.types import BasicType


__all__ = [
    'SequenceAssociationTransformation',
    'do_resolve_sequence_association',
    'SequenceAssociationTransformer'
]


class SequenceAssociationTransformation(Transformation):
    """
    :any:`Transformation` that resolves sequence association patterns
    in :any:`CallStatement` nodes.

    Parameters
    ----------
    resolve_sequence_associations : bool
        Flag to trigger or suppress resolution of sequence associations
    """

    def __init__(self, resolve_sequence_associations=True):
        self.resolve_sequence_associations = resolve_sequence_associations

    def transform_subroutine(self, routine, **kwargs):  # pylint: disable=unused-argument
        if self.resolve_sequence_associations:
            do_resolve_sequence_association(routine)


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


def do_resolve_sequence_association(routine):
    """
    Housekeeping routine to replace scalar syntax when passing arrays
    as arguments For example, a call like

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

    routine.body = SequenceAssociationTransformer(inplace=True).visit(routine.body)


class SequenceAssociationTransformer(Transformer):
    """
    Transformer that resolves sequence association patterns in
    :any:`CallStatement` nodes.
    """

    def visit_CallStatement(self, call, **kwargs):  # pylint: disable=unused-argument
        """
        Resolve sequence association patterns in arguments and return
        new :any:`CallStatement` object if any were found.
        """
        if call.procedure_type is BasicType.DEFERRED:
            return call

        new_args = []
        found_scalar = False
        for dummy, arg in call.arg_map.items():
            if check_if_scalar_syntax(arg, dummy):
                found_scalar = True

                n_dims = len(dummy.shape)
                new_dims = []

                if not arg.shape:
                    # Hack: If we don't have a shape, short-circuit here
                    new_dims = tuple(RangeIndex((None, None)) for _ in dummy.shape)
                else:
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
            return call.clone(arguments = as_tuple(new_args))

        return call
