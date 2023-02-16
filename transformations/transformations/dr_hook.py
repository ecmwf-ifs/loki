# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Utility transformations to update or remove calls to DR_HOOK
"""

from loki import (
    FindNodes, Transformer, Transformation, CallStatement, Conditional, as_tuple, Literal, Import
)

class DrHookTransformation(Transformation):
    """
    Re-write or remove the DrHook label markers in transformed
    kernel routines

    Parameters
    ----------
    remove : bool
        Remove calls to ``DR_HOOK``
    mode : str
        Transformation mode to insert into DrHook labels
    """
    def __init__(self, remove=False, mode=None, **kwargs):
        self.remove = remove
        self.mode = mode
        super().__init__(**kwargs)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply transformation to subroutine object
        """
        role = kwargs['item'].role

        # Leave DR_HOOK annotations in driver routine
        if role == 'driver':
            return

        mapper = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            # Lazily changing the DrHook label in-place
            if call.name == 'DR_HOOK':
                if self.remove:
                    mapper[call] = None
                else:
                    new_label = f'{call.arguments[0].value.upper()}_{str(self.mode).upper()}'
                    new_args = (Literal(value=new_label),) + call.arguments[1:]
                    mapper[call] = call.clone(arguments=new_args)

        if self.remove:
            for cond in FindNodes(Conditional).visit(routine.body):
                if cond.inline and 'LHOOK' in as_tuple(cond.condition):
                    mapper[cond] = None

        routine.body = Transformer(mapper).visit(routine.body)

        #Get rid of the imports, as imported variables may confuse the compiler
        if self.remove:
            for imp in FindNodes(Import).visit(routine.spec):
                if 'LHOOK' in [s.name for s in imp.symbols]:
                    mapper[imp] = None

            routine.spec = Transformer(mapper).visit(routine.spec)

            #Remove unused zhook_handle
            routine.variables = as_tuple(v for v in routine.variables if v.name != 'ZHOOK_HANDLE')
