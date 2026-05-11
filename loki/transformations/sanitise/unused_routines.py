# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import SchedulerConfig, Transformation
from loki.ir import FindNodes, Transformer, nodes as ir
from loki.tools import as_tuple

__all__ = ['SanitiseUnusedRoutineTransformation']


class SanitiseUnusedRoutineTransformation(Transformation):
    """
    Sanitise configured routines that are no longer used but still need to compile.

    For selected routines this transformation rewrites kept array declarations
    in the routine specification to fully deferred shape, removes local array
    declarations that no longer need to be represented, and replaces the
    executable body with a configurable stub.

    Parameters
    ----------
    routines : tuple or list of str, optional
        Routine names or qualified scheduler item names to sanitise. Matching
        uses :any:`SchedulerConfig.match_item_keys` with parent matching
        enabled, so both ``routine`` and ``module#routine`` forms are
        supported.
    stub_kind : str, optional
        The replacement body to insert for sanitised routines. Supported
        values are ``'error_stop'`` for a fail-loud stub and ``'empty'`` for
        an empty executable section. Defaults to ``'error_stop'``.
    """

    def __init__(self, routines=None, stub_kind='error_stop'):
        self.routines = as_tuple(routines)
        if stub_kind not in ('error_stop', 'empty'):
            raise ValueError(
                f'Invalid stub_kind {stub_kind!r}: expected one of ("error_stop", "empty")'
            )
        self.stub_kind = stub_kind

    @staticmethod
    def _matches_routine(routine, item=None, configured_routines=()):
        """Return whether one routine matches the configured sanitisation targets."""
        if item is not None:
            item_name = item.name
        elif routine.parent is not None:
            item_name = f'{routine.parent.name.lower()}#{routine.name.lower()}'
        else:
            item_name = routine.name.lower()

        return bool(SchedulerConfig.match_item_keys(item_name, configured_routines, match_item_parents=True))

    @staticmethod
    def _deferred_shape(symbol, routine):
        """Clone one array symbol with all declared dimensions rewritten to deferred shape."""
        shape = getattr(symbol.type, 'shape', None)
        if not shape:
            return symbol

        deferred_shape = tuple(routine.parse_expr(':') for _ in shape)
        return symbol.clone(dimensions=deferred_shape, type=symbol.type.clone(shape=deferred_shape))

    @staticmethod
    def _should_keep_with_deferred_shape(symbol):
        """Keep arguments, pointers, and allocatables while deferring their declared shape."""
        return bool(symbol.type.intent is not None or symbol.type.pointer or symbol.type.allocatable)

    def transform_subroutine(self, routine, **kwargs):
        """
        Sanitise one configured routine by shrinking its declarations and replacing its body.

        Only declarations that still matter to the routine interface or storage
        semantics are kept, and any retained arrays are rewritten to fully
        deferred shape so their original bounds expressions are no longer
        required.
        """
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

        if self.stub_kind == 'error_stop':
            routine.body = ir.Section(body=(
                ir.Intrinsic(text=f'error stop "Sanitised unused routine {routine.name} was called"'),
            ))
        else:
            routine.body = ir.Section(body=())
