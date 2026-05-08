from loki.batch import SchedulerConfig, Transformation
from loki.ir import FindNodes, Transformer, nodes as ir
from loki.tools import as_tuple

__all__ = ['SanitiseUnusedRoutineTransformation']


class SanitiseUnusedRoutineTransformation(Transformation):
    """
    Sanitise configured routines that are no longer used but still need to compile.

    For selected routines this transformation:
    * rewrites array declarations in the routine spec to fully deferred shape
    * replaces the executable body with an ``ERROR STOP`` stub
    """

    def __init__(self, routines=None, raise_error=True):
        self.routines = as_tuple(routines)
        self.raise_error = raise_error

    @staticmethod
    def _matches_routine(routine, item=None, configured_routines=()):
        if item is not None:
            item_name = item.name
        elif routine.parent is not None:
            item_name = f'{routine.parent.name.lower()}#{routine.name.lower()}'
        else:
            item_name = routine.name.lower()

        return bool(SchedulerConfig.match_item_keys(item_name, configured_routines, match_item_parents=True))

    @staticmethod
    def _deferred_shape(symbol, routine):
        shape = getattr(symbol.type, 'shape', None)
        if not shape:
            return symbol

        deferred_shape = tuple(routine.parse_expr(':') for _ in shape)
        return symbol.clone(dimensions=deferred_shape, type=symbol.type.clone(shape=deferred_shape))

    @staticmethod
    def _should_keep_with_deferred_shape(symbol):
        return bool(symbol.type.intent is not None or symbol.type.pointer or symbol.type.allocatable)

    def transform_subroutine(self, routine, **kwargs):
        if not self._matches_routine(routine, item=kwargs.get('item'), configured_routines=self.routines):
            return

        decl_map = {}
        for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
            new_symbols = []
            for symbol in decl.symbols:
                shape = getattr(symbol.type, 'shape', None)
                if not shape:
                    new_symbols.append(symbol)
                    continue

                if self._should_keep_with_deferred_shape(symbol):
                    new_symbols.append(self._deferred_shape(symbol, routine))

            new_symbols = tuple(new_symbols)
            if not new_symbols:
                decl_map[decl] = None
            elif new_symbols != decl.symbols:
                decl_map[decl] = decl.clone(symbols=new_symbols)

        if decl_map:
            routine.spec = Transformer(decl_map).visit(routine.spec)

        if self.raise_error:
            routine.body = ir.Section(body=(
                ir.Intrinsic(text=f'error stop "Sanitised unused routine {routine.name} was called"'),
            ))
        else:
            routine.body = ir.Section(body=())
