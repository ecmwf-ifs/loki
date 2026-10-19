# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import ProcedureItem, Transformation
from loki.ir import FindNodes, FindVariables, Transformer, nodes as ir
from loki.logging import warning
from loki.tools import CaseInsensitiveDict, as_tuple
from loki.types import BasicType

__all__ = ['ReplaceKernels']


class ReplaceKernels(Transformation):
    """
    Replace configured kernel calls with alternative routines.

    This transformation supports:

    - replacing selected call targets by routine name;
    - remapping arguments with expressions over the dummy arguments of the
      original routine (renames, derived-type members, arithmetic, literals);
    - passing arguments by keyword or, where possible, by position;
    - rewriting ``USE`` imports and ``.intfb.h`` include-style imports;
    - optionally keeping the replacement routine out of scheduler traversal and
      CMake planning via ``block=True``.

    Conceptually, the transformation walks each call site in the processed
    routine, looks for names listed in ``replace_kernels_map``, resolves the
    replacement routine, and reconstructs the actual argument list against the
    replacement signature. The original routine is resolved to match the
    actual arguments of the call to its dummy arguments, which are then
    substituted in the rule expressions. Once the replacement call has been
    rebuilt, the transformation updates imports/includes to match the new
    callee and optionally marks the replacement routine as blocked for
    further scheduler traversal.

    The replacement map accepts entries of the form::

        {
            'old_kernel': {
                'routine': 'new_kernel',
                'args': {
                    'new_dummy': 'old_dummy',
                    'new_member': 'old_struct%MEMBER',
                    'new_end': 'MOD(old_geom%TOTAL%NGPTOT, old_geom%DIM%NPROMA)',
                    'new_flag': '.true.',
                    'new_start': {'expr': '1', 'positional': True},
                },
                'block': True,
            }
        }

    Parameters
    ----------
    replace_kernels_map : dict, optional
        Case-insensitive mapping from original routine name to replacement
        configuration. Each configuration entry supports the following keys:

        ``routine``
            Name of the replacement routine.

        ``args``
            Optional mapping from replacement dummy argument names to the
            expression that defines the actual argument of the replacement
            call. The expression is written in Fortran syntax in terms of the
            dummy arguments of the original routine, which are substituted by
            the actual arguments of the original call. Supported forms are:

            ``'new_dummy': 'expression'``
                Pass the expression by keyword. This covers renaming
                (``'old_dummy'``), derived-type members
                (``'old_dummy%MEMBER'``), arithmetic and intrinsics, and
                literals (``'.true.'``).

            ``'new_dummy': {'expr': 'expression', 'positional': True}``
                Pass the expression by position if all preceding arguments of
                the replacement call are passed by position, otherwise by keyword.

            Replacement dummies without an entry receive the actual argument of
            the original dummy with the same name. Optional dummies without
            a match are omitted.

        ``block``
            If true, mark the replacement routine/module as blocked in the
            current scheduler item so that it is not traversed or added to the
            generated plan.
    """

    creates_items = True

    def __init__(self, replace_kernels_map=None):
        self.replace_kernels_map = CaseInsensitiveDict(replace_kernels_map or {})

    @staticmethod
    def _get_replacement_item_from_cache(replacement_name, item_factory, config):
        """Return a cached procedure item for the requested replacement name."""
        item_cache = item_factory.item_cache
        replacement_name = replacement_name.lower()

        if isinstance(item_cache.get(replacement_name), ProcedureItem):
            return item_cache[replacement_name]

        procedure_items = [
            item for item in item_cache.values()
            if isinstance(item, ProcedureItem) and item.local_name.lower() == replacement_name
        ]
        if not procedure_items:
            # Look for a module procedure that has not been instantiated yet
            procedure_items = item_factory.get_or_create_module_definitions_from_candidates(
                replacement_name, config=config, only=(ProcedureItem,)
            )
        return procedure_items[0] if len(procedure_items) == 1 else None

    @staticmethod
    def _complete_replacement_source(replacement_item, scheduler_config, build_args):
        """Complete the parsed source for a replacement item with scheduler frontend settings."""
        frontend_args = {
            key: value for key, value in build_args.items()
            if key in ('definitions', 'preprocess', 'includes', 'defines', 'xmods', 'omni_includes', 'frontend')
        }
        frontend_args = scheduler_config.create_frontend_args(replacement_item.source.path, frontend_args)
        replacement_item.source.make_complete(**frontend_args)

    def _get_complete_replacement_item(self, replacement_name, **kwargs):
        """Resolve, load, and complete the replacement routine item if available."""
        item_factory = kwargs.get('item_factory')
        scheduler_config = kwargs.get('scheduler_config')
        build_args = dict(kwargs.get('build_args', {}))

        replacement_item = self._get_replacement_item_from_cache(replacement_name, item_factory,
                config=scheduler_config)
        if replacement_item is None or replacement_item.source is None:
            return None

        self._complete_replacement_source(replacement_item, scheduler_config, build_args)
        return replacement_item

    def _get_complete_item(self, routine_name, **kwargs):
        """Resolve, load, and complete an arbitrary routine item by name."""
        item_factory = kwargs.get('item_factory')
        scheduler_config = kwargs.get('scheduler_config')
        build_args = dict(kwargs.get('build_args', {}))

        routine_item = self._get_replacement_item_from_cache(routine_name, item_factory,
                config=scheduler_config)
        if routine_item is None or routine_item.source is None:
            return None

        self._complete_replacement_source(routine_item, scheduler_config, build_args)
        return routine_item

    @staticmethod
    def _mark_replacement_as_blocked(item, replacement_item):
        """Add the replacement routine and module names to the current scheduler block list."""
        block_entries = list(as_tuple(item.config.get('block', ())))
        candidates = [replacement_item.local_name]
        if replacement_item.scope_name:
            candidates.append(replacement_item.scope_name)

        for candidate in candidates:
            if candidate not in block_entries:
                block_entries.append(candidate)

        item.config['block'] = tuple(block_entries)

    @staticmethod
    def _normalise_argument_rules(replacement_args, new_routine):
        """
        Normalise the argument rules for one replacement routine.

        Parameters
        ----------
        replacement_args : dict
            The ``args`` configuration of a replacement, mapping replacement
            dummy argument names to an expression string or a dict with the
            keys ``expr`` and (optionally) ``positional``.
        new_routine : :any:`Subroutine`
            The replacement routine.

        Returns
        -------
        :any:`CaseInsensitiveDict`
            Mapping of replacement dummy argument names to tuples of
            the expression string and a flag for positional passing.
        """
        new_dummy_names = {arg.name.lower() for arg in new_routine.arguments}
        rules = CaseInsensitiveDict()

        for new_name, rule in CaseInsensitiveDict(replacement_args).items():
            if new_name.lower() not in new_dummy_names:
                raise RuntimeError(
                    f'Invalid replacement argument rule for {new_name}: '
                    f'{new_routine.name} has no such dummy argument'
                )
            if not isinstance(rule, dict):
                rule = {'expr': rule}
            if unknown := set(rule) - {'expr', 'positional'}:
                raise RuntimeError(
                    f'Invalid replacement argument rule for {new_name}: unknown keys {sorted(unknown)}'
                )
            if 'expr' not in rule:
                raise RuntimeError(f'Invalid replacement argument rule for {new_name}: expected expr')
            rules[new_name] = (str(rule['expr']), bool(rule.get('positional', False)))

        return rules

    def _get_original_arguments(self, call, **kwargs):
        """
        Determine the dummy arguments and the actual arguments of the original call.

        Parameters
        ----------
        call : :any:`CallStatement`
            The call to the original kernel.

        Returns
        -------
        tuple of (set, :any:`CaseInsensitiveDict`)
            The lower-case names of the original dummy arguments and a
            :any:`CaseInsensitiveDict` mapping them to the actual arguments.
        """
        if call.routine is not None and call.routine is not BasicType.DEFERRED:
            old_routine = call.routine
        else:
            # Current main does not always link the original callee in the IR,
            # so resolve it via the scheduler item
            old_routine_item = self._get_complete_item(str(call.name), **kwargs)
            if old_routine_item is None:
                raise RuntimeError(
                    f'Cannot resolve the original routine {call.name} to remap its arguments; '
                    'it must be available in the scheduler item cache'
                )
            old_routine = old_routine_item.ir

        dummy_names = tuple(arg.name for arg in old_routine.arguments)
        old_arg_map = CaseInsensitiveDict(zip(dummy_names, call.arguments))
        old_arg_map.update(call.kwarguments)
        return {name.lower() for name in dummy_names}, old_arg_map

    @staticmethod
    def _evaluate_rule(caller_routine, expr, old_arg_map, old_dummy_names):
        """
        Evaluate an argument rule expression in the scope of the caller.

        The original dummy arguments in the expression are substituted
        with the actual arguments of the call.

        Parameters
        ----------
        caller_routine : :any:`Subroutine`
            The routine containing the call.
        expr : str
            The rule expression, written in terms of the original dummy arguments.
        old_arg_map : :any:`CaseInsensitiveDict`
            Mapping of original dummy arguments to actual arguments.
        old_dummy_names : set of str
            The lower-case names of the original dummy arguments.
        """
        result = caller_routine.parse_expr(expr, evaluate=True, context=old_arg_map)

        symbol_names = {var.name.split('%')[0] for var in FindVariables().visit(result)}
        unbound = {name.lower() for name in symbol_names if name.lower() in old_dummy_names and name not in old_arg_map}
        if unbound:
            raise RuntimeError(
                f'Replacement expression "{expr}" uses original dummy arguments '
                f'without an actual argument in the call: {", ".join(sorted(unbound))}'
            )
        return result

    def _adapt_arguments(self, call, caller_routine, new_routine, **kwargs):
        """
        Rebuild the call arguments against the replacement routine signature.

        The replacement signature is treated as authoritative: each replacement
        dummy is visited in order and its actual argument is derived either from
        an explicit expression rule or from an implicit name-based match against
        the original call. Rule expressions are written in terms of the dummy
        arguments of the original routine, which are substituted by the actual
        arguments of the call.
        """
        replacement = self.replace_kernels_map[str(call.name).lower()]
        rules = self._normalise_argument_rules(replacement.get('args', {}), new_routine)
        old_dummy_names, old_arg_map = self._get_original_arguments(call, **kwargs)

        arguments = []
        kwarguments = []
        keep_positional = True

        for new_dummy in new_routine.arguments:
            new_name = new_dummy.name.lower()
            if new_name in rules:
                expr, positional = rules[new_name]
                actual = self._evaluate_rule(caller_routine, expr, old_arg_map, old_dummy_names)
                use_keyword = not (positional and keep_positional)
            else:
                actual = old_arg_map.get(new_name)
                use_keyword = not keep_positional

            if actual is None:
                if new_dummy.type.optional:
                    warning(
                        '[Loki::ReplaceKernels] Omitting optional replacement argument %s when replacing %s '
                        'with %s in %s', new_dummy.name, call.name, new_routine.name, caller_routine.name
                    )
                    continue
                raise RuntimeError(
                    f'Cannot map required replacement argument {new_dummy.name} when replacing '
                    f'{call.name} with {new_routine.name} in {caller_routine.name}'
                )

            if use_keyword:
                kwarguments.append((new_dummy.name, actual))
                keep_positional = False
            else:
                arguments.append(actual)

        return tuple(arguments), tuple(kwarguments)

    @staticmethod
    def _replace_import(scope, import_map, new_imports):
        """Rewrite matching imports in one scope and append any newly required module imports."""
        if import_map:
            scope.spec = Transformer(import_map).visit(scope.spec)
        if new_imports:
            use_positions = [
                index for index, node in enumerate(scope.spec.body)
                if isinstance(node, ir.Import) and not node.c_import and not node.f_include
            ]
            insert_pos = use_positions[-1] + 1 if use_positions else 0
            scope.spec.insert(insert_pos, new_imports)

    def _transform_subroutine(self, routine, **kwargs):
        """
        Replace matching calls in one routine and update the corresponding
        imports/includes.

        The transformation collects existing imports in both the routine scope
        and its parent scope, rewrites matching call statements, and then
        updates the relevant import location depending on whether the original
        dependency came from a local ``USE`` import, a parent-scope import, a
        ``.intfb.h`` include, or no explicit import at all.
        """
        item = kwargs.get('item')

        import_by_symbol = CaseInsensitiveDict()
        intfb_import_by_name = CaseInsensitiveDict()

        for imprt in FindNodes(ir.Import).visit(routine.spec):
            if imprt.c_import and imprt.module and '.intfb.h' in imprt.module:
                kernel_name = imprt.module.replace('.intfb.h', '')
                intfb_import_by_name[kernel_name] = ('routine', imprt)
            for symbol in imprt.symbols:
                import_by_symbol[str(symbol).lower()] = ('routine', imprt)

        parent_import_by_symbol = CaseInsensitiveDict()
        parent_intfb_import_by_name = CaseInsensitiveDict()
        if routine.parent is not None and getattr(routine.parent, 'spec', None) is not None:
            for imprt in FindNodes(ir.Import).visit(routine.parent.spec):
                if imprt.c_import and imprt.module and '.intfb.h' in imprt.module:
                    kernel_name = imprt.module.replace('.intfb.h', '')
                    parent_intfb_import_by_name[kernel_name] = imprt
                for symbol in imprt.symbols:
                    parent_import_by_symbol[str(symbol).lower()] = imprt

        call_map = {}
        new_imports = []
        parent_new_imports = []
        routine_import_map = {}
        parent_import_map = {}

        for call in FindNodes(ir.CallStatement).visit(routine.body):
            call_name = str(call.name).lower()
            if call_name not in self.replace_kernels_map:
                continue

            replacement = self.replace_kernels_map[call_name]
            replacement_name = replacement.get('routine')
            replacement_item = self._get_complete_replacement_item(replacement_name, **kwargs)
            if replacement_item is None:
                continue

            proc_symbol = replacement_item.ir.procedure_symbol.rescope(scope=routine)
            arguments, kwarguments = self._adapt_arguments(call, routine, replacement_item.ir, **kwargs)
            call_map[call] = call.clone(name=proc_symbol, arguments=arguments, kwarguments=kwarguments)

            # Prefer rewriting the import where the original dependency was
            # declared. Fall back to inserting a new module import if the call
            # was unresolved or came from an ``.intfb.h`` include.
            if call_name in import_by_symbol:
                imprt = import_by_symbol[call_name][1]
                routine_import_map[imprt] = imprt.clone(module=replacement_item.scope_name, symbols=(proc_symbol,))
            elif call_name in parent_import_by_symbol:
                imprt = parent_import_by_symbol[call_name]
                parent_import_map[imprt] = imprt.clone(module=replacement_item.scope_name, symbols=(proc_symbol,))
            elif call_name in intfb_import_by_name:
                imprt = intfb_import_by_name[call_name][1]
                routine_import_map[imprt] = None
                new_imports.append(ir.Import(module=replacement_item.scope_name, symbols=(proc_symbol,)))
            elif call_name in parent_intfb_import_by_name:
                imprt = parent_intfb_import_by_name[call_name]
                parent_import_map[imprt] = None
                parent_new_imports.append(ir.Import(module=replacement_item.scope_name, symbols=(proc_symbol,)))
            else:
                new_imports.append(ir.Import(module=replacement_item.scope_name, symbols=(proc_symbol,)))

            if item is not None and replacement.get('block', False):
                # Blocking the replacement item keeps it out of traversal and
                # plan generation while still allowing the caller rewrite.
                self._mark_replacement_as_blocked(item, replacement_item)

        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)
        self._replace_import(routine, routine_import_map, new_imports)
        if routine.parent is not None and getattr(routine.parent, 'spec', None) is not None:
            self._replace_import(routine.parent, parent_import_map, parent_new_imports)

    def transform_subroutine(self, routine, **kwargs):
        self._transform_subroutine(routine, **kwargs)

    def plan_subroutine(self, routine, **kwargs):
        self._transform_subroutine(routine, **kwargs)
