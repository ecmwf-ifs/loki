"""
Utility transformations to update or remove calls to DR_HOOK
"""

from loki import (
    FindNodes, Transformer, Transformation, CallStatement, Conditional, as_tuple, Literal
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
