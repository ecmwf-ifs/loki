# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Utility transformations to update or remove calls to DR_HOOK.
"""

from loki.batch import Transformation
from loki.expression import Literal
from loki.ir import (
    FindNodes, Transformer, CallStatement, Conditional, Import
)
from loki.tools import as_tuple


__all__ = ['DrHookTransformation']


def remove_unused_drhook_import(routine):
    """
    Remove unsed DRHOOK imports and corresponding handle.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine from which to remove DRHOOK import/handle.
    """

    mapper = {}
    for imp in FindNodes(Import).visit(routine.spec):
        if imp.module.lower() == 'yomhook':
            mapper[imp] = None

    if mapper:
        routine.spec = Transformer(mapper).visit(routine.spec)

    #Remove unused zhook_handle
    routine.variables = as_tuple(v for v in routine.variables if v != 'zhook_handle')


class DrHookTransformation(Transformation):
    """
    Re-write or remove the DrHook label markers in transformed
    kernel routines

    Parameters
    ----------
    suffix : str
        String suffix to append to DrHook labels
    remove : bool
        Remove calls to ``DR_HOOK``
    """

    def __init__(self, suffix=None, remove=False, **kwargs):
        self.remove = remove
        self.suffix = suffix
        super().__init__(**kwargs)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply transformation to subroutine object
        """
        role = kwargs['item'].role

        # Leave DR_HOOK annotations in driver routine
        if role == 'driver':
            return

        for r in routine.members:
            self.transform_subroutine(r, **kwargs)

        mapper = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            # Lazily changing the DrHook label in-place
            if call.name == 'DR_HOOK':
                if self.remove:
                    mapper[call] = None
                else:
                    new_label = f'{call.arguments[0].value.upper()}_{str(self.suffix).upper()}'
                    new_args = (Literal(value=new_label),) + call.arguments[1:]
                    mapper[call] = call.clone(arguments=new_args)

        if self.remove:
            for cond in FindNodes(Conditional).visit(routine.body):
                if cond.inline and 'LHOOK' in as_tuple(cond.condition):
                    mapper[cond] = None

        routine.body = Transformer(mapper).visit(routine.body)

        # Get rid of unused import and variable
        if self.remove:
            remove_unused_drhook_import(routine)
