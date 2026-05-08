# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

from loki.batch import ModuleItem, ProcedureItem, Transformation
from loki.ir import FindNodes, Transformer, nodes as ir
from loki.logging import warning
from loki.tools import CaseInsensitiveDict, as_tuple
from loki.types import BasicType

__all__ = ['ReplaceKernels']


class ReplaceKernels(Transformation):
    """
    Replace configured kernel calls with alternative routines.

    This transformation supports:

    - replacing selected call targets by routine name;
    - remapping arguments by direct rename;
    - remapping arguments by extracting a member from a derived-type actual;
    - remapping arguments by formatting a custom expression template;
    - overriding an argument with a positional value;
    - overriding an argument with a literal or expression string;
    - rewriting ``USE`` imports and ``.intfb.h`` include-style imports;
    - optionally keeping the replacement routine out of scheduler traversal and
      CMake planning via ``block=True``.

    Conceptually, the transformation walks each call site in the processed
    routine, looks for names listed in ``replace_kernels_map``, resolves the
    replacement routine, and reconstructs the actual argument list against the
    replacement signature. When the original call target is unresolved in the
    IR, the transformation also tries to resolve the original routine from
    source so that remapping can still be done against the original dummy
    argument names instead of the raw actual expressions. Once the replacement
    call has been rebuilt, the transformation updates imports/includes to match
    the new callee and optionally marks the replacement routine as blocked for
    further scheduler traversal.

    The replacement map accepts entries of the form::

        {
            'old_kernel': {
                'routine': 'new_kernel',
                'args': {
                    'old_dummy': 'new_dummy',
                    'struct_arg': {'map_to': 'new_dummy', 'member': 'MEMBER'},
                    'old_end': {
                        'map_to': 'new_end',
                        'expr': 'MOD({geom}%TOTAL%NGPTOT, {geom}%DIM%NPROMA)',
                        'placeholders': {'geom': 'geom'},
                    },
                    'old_start': {'position': 1},
                    'new_flag': '.true.',
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
            Optional mapping that controls how actual arguments are rebuilt for
            the replacement call. Supported forms are:

            ``'old_dummy': 'new_dummy'``
                Map the actual argument of ``old_dummy`` onto the replacement
                dummy ``new_dummy``.

            ``'old_dummy': {'position': 1}``
                Pass a positional literal or expression to the replacement
                dummy named ``old_dummy``.

            ``'old_dummy': {'map_to': 'new_dummy', 'member': 'MEMBER'}``
                Pass ``actual%MEMBER`` from the original argument ``old_dummy``
                to the replacement dummy ``new_dummy``.

            ``'old_dummy': {'map_to': 'new_dummy', 'expr': '...',
            'placeholders': {'name': 'old_dummy_name'}}``
                Build an expression by formatting the template string with the
                resolved original actual arguments and pass the result to the
                replacement dummy ``new_dummy``.

            ``'new_dummy': '.true.'``
                Pass a literal or expression string directly to the replacement
                dummy ``new_dummy``.

        ``block``
            If true, mark the replacement routine/module as blocked in the
            current scheduler item so that it is not traversed or added to the
            generated plan.
    """

    creates_items = True

    def __init__(self, replace_kernels_map=None):
        self.replace_kernels_map = CaseInsensitiveDict(replace_kernels_map or {})

    @staticmethod
    def _get_replacement_item_from_cache(replacement_name, item_factory):
        """Return a cached procedure item for the requested replacement name."""
        item_cache = item_factory.item_cache

        if replacement_name in item_cache and isinstance(item_cache[replacement_name], ProcedureItem):
            return item_cache[replacement_name]

        qualified_matches = [
            item for item in item_cache.values()
            if isinstance(item, ProcedureItem) and item.local_name.lower() == replacement_name.lower()
        ]
        if len(qualified_matches) == 1:
            return qualified_matches[0]

        for key, item in item_cache.items():
            if isinstance(item, ProcedureItem) and replacement_name.lower() in key.lower():
                return item

        return None

    @staticmethod
    def _complete_replacement_source(replacement_item, scheduler_config, build_args):
        """Complete the parsed source for a replacement item with scheduler frontend settings."""
        frontend_args = {
            key: value for key, value in build_args.items()
            if key in ('definitions', 'preprocess', 'includes', 'defines', 'xmods', 'omni_includes', 'frontend')
        }
        frontend_args = scheduler_config.create_frontend_args(replacement_item.source.path, frontend_args)
        replacement_item.source.make_complete(**frontend_args)

    def _load_replacement_item_from_source(self, replacement_name, item_factory, scheduler_config, build_args):
        """Search configured source roots for a replacement routine and load its definition items."""
        paths = as_tuple(build_args.get('paths', ()))
        if not paths:
            paths = as_tuple(build_args.get('includes', ()))

        for root in paths:
            root = Path(root)
            if not root.exists():
                continue
            for path in root.glob('**/*'):
                if not path.is_file() or replacement_name.lower() not in path.stem.lower():
                    continue

                frontend_args = {
                    key: value for key, value in build_args.items()
                    if key in ('definitions', 'preprocess', 'includes', 'defines', 'xmods', 'omni_includes', 'frontend')
                }
                file_item = item_factory.get_or_create_file_item_from_path(path, scheduler_config, frontend_args)
                frontend_args = scheduler_config.create_frontend_args(path, frontend_args)
                file_item.source.make_complete(**frontend_args)
                definition_items = file_item.create_definition_items(
                    item_factory=item_factory, config=scheduler_config
                )
                for definition_item in definition_items:
                    if isinstance(definition_item, ModuleItem):
                        definition_item.create_definition_items(item_factory=item_factory, config=scheduler_config)

                replacement_item = self._get_replacement_item_from_cache(replacement_name, item_factory)
                if replacement_item is not None:
                    return replacement_item

        return None

    def _get_complete_replacement_item(self, replacement_name, **kwargs):
        """Resolve, load, and complete the replacement routine item if available."""
        item_factory = kwargs.get('item_factory')
        scheduler_config = kwargs.get('scheduler_config')
        build_args = dict(kwargs.get('build_args', {}))

        replacement_item = self._get_replacement_item_from_cache(replacement_name, item_factory)
        if replacement_item is None:
            replacement_item = self._load_replacement_item_from_source(
                replacement_name, item_factory, scheduler_config, build_args
            )
        if replacement_item is None:
            return None

        self._complete_replacement_source(replacement_item, scheduler_config, build_args)
        return replacement_item

    def _get_complete_item(self, routine_name, **kwargs):
        """Resolve, load, and complete an arbitrary routine item by name."""
        item_factory = kwargs.get('item_factory')
        scheduler_config = kwargs.get('scheduler_config')
        build_args = dict(kwargs.get('build_args', {}))

        routine_item = self._get_replacement_item_from_cache(routine_name, item_factory)
        if routine_item is None:
            routine_item = self._load_replacement_item_from_source(
                routine_name, item_factory, scheduler_config, build_args
            )
        if routine_item is None:
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
    def _split_argument_rules(replacement_args, new_routine):
        """
        Categorize argument remapping rules for one replacement routine.

        The returned dictionaries split the user-provided mapping into rename,
        literal override, positional override, member extraction, and
        expression-template based remapping so that the call reconstruction step
        can process the replacement signature in a single pass.
        """
        replacement_dummy_names = {arg.name.lower() for arg in new_routine.arguments}
        renamed_args = {}
        override_args = CaseInsensitiveDict()
        position_args = CaseInsensitiveDict()
        member_args = CaseInsensitiveDict()
        expr_args = CaseInsensitiveDict()

        for old_name, rule in CaseInsensitiveDict(replacement_args).items():
            if isinstance(rule, dict):
                position = rule.get('position')
                map_to = rule.get('map_to')
                member = rule.get('member')
                expr = rule.get('expr')
                placeholders = rule.get('placeholders')

                if position is not None:
                    if any(value is not None for value in (map_to, member, expr, placeholders)):
                        raise RuntimeError(
                            f'Invalid replacement argument rule for {old_name}: position cannot be combined '
                            'with map_to/member/expr/placeholders'
                        )
                    position_args[old_name.lower()] = str(position)
                    continue

                if map_to is None:
                    raise RuntimeError(
                        f'Invalid replacement argument rule for {old_name}: expected map_to'
                    )

                if member is not None and expr is not None:
                    raise RuntimeError(
                        f'Invalid replacement argument rule for {old_name}: member and expr are mutually exclusive'
                    )

                if member is not None:
                    member_args[old_name.lower()] = {
                        'map_to': str(map_to).lower(),
                        'member': str(member),
                    }
                elif expr is not None:
                    if not isinstance(placeholders, dict) or not placeholders:
                        raise RuntimeError(
                            f'Invalid replacement argument rule for {old_name}: expected expr/placeholders'
                        )
                    expr_args[old_name.lower()] = {
                        'map_to': str(map_to).lower(),
                        'expr': str(expr),
                        'placeholders': CaseInsensitiveDict(
                            (str(name), str(dummy).lower()) for name, dummy in placeholders.items()
                        ),
                    }
                else:
                    raise RuntimeError(
                        f'Invalid replacement argument rule for {old_name}: expected member or expr'
                    )
            elif str(rule).lower() in replacement_dummy_names:
                renamed_args[old_name.lower()] = str(rule).lower()
            else:
                override_args[old_name.lower()] = str(rule)

        return renamed_args, override_args, position_args, member_args, expr_args

    @staticmethod
    def _build_member_expr(routine, actual, member):
        """Build a parsed member-access expression from an original actual argument."""
        member = member.lstrip('%')
        return routine.parse_expr(f'{actual}%{member}')

    @staticmethod
    def _build_template_expr(routine, old_arg_map, expr_rule):
        """Format and parse a replacement expression template from original actuals."""
        resolved_placeholders = {}
        for placeholder, old_name in expr_rule['placeholders'].items():
            actual = old_arg_map.get(old_name)
            if actual is None:
                raise RuntimeError(
                    f'Cannot resolve placeholder {placeholder} from original argument {old_name}'
                )
            resolved_placeholders[placeholder] = str(actual)

        try:
            expr = expr_rule['expr'].format(**resolved_placeholders)
        except KeyError as exc:
            raise RuntimeError(
                f'Unknown placeholder {exc.args[0]} in replacement expression {expr_rule["expr"]}'
            ) from exc

        return routine.parse_expr(expr)

    def _adapt_arguments(self, call, caller_routine, new_routine, **kwargs):
        """
        Rebuild the call arguments against the replacement routine signature.

        The replacement signature is treated as authoritative: each replacement
        dummy is visited in order and its actual argument is derived either from
        an explicit remapping rule or from an implicit name-based match against
        the original call. When the original callee is unresolved, the
        transformation tries to resolve that routine from source first so that
        remapping still uses the original dummy names instead of the rendered
        actual expressions.
        """
        replacement = self.replace_kernels_map[str(call.name).lower()]
        replacement_args = replacement.get('args', {})
        renamed_args, override_args, position_args, member_args, expr_args = self._split_argument_rules(
            replacement_args, new_routine
        )

        if call.routine is not None and call.routine is not BasicType.DEFERRED:
            old_arg_map = CaseInsensitiveDict((dummy.name, actual) for dummy, actual in call.arg_map.items())
        else:
            # Current main does not always link the original callee in the IR,
            # so resolve it from source if possible and rebuild the original
            # dummy-to-actual mapping from its signature.
            old_routine_item = self._get_complete_item(str(call.name), **kwargs)
            if old_routine_item is not None:
                old_arg_map = CaseInsensitiveDict()
                positional_arguments = iter(call.arguments)
                for old_dummy in old_routine_item.ir.arguments:
                    kwarg_actual = next(
                        (
                            value for name, value in call.kwarguments
                            if name.lower() == old_dummy.name.lower()
                        ),
                        None,
                    )
                    if kwarg_actual is not None:
                        old_arg_map[old_dummy.name] = kwarg_actual
                    else:
                        actual = next(positional_arguments, None)
                        if actual is not None:
                            old_arg_map[old_dummy.name] = actual
            else:
                old_arg_map = CaseInsensitiveDict((kw[0], kw[1]) for kw in call.kwarguments)
                for actual in call.arguments:
                    old_arg_map.setdefault(str(actual).lower(), actual)

        reverse_renamed_args = {new_name: old_name for old_name, new_name in renamed_args.items()}
        reverse_member_args = {rule['map_to']: old_name for old_name, rule in member_args.items()}
        reverse_expr_args = {rule['map_to']: old_name for old_name, rule in expr_args.items()}
        arguments = []
        kwarguments = []
        keep_positional = True

        for new_dummy in new_routine.arguments:
            new_name = new_dummy.name.lower()
            actual = None
            use_keyword = not keep_positional
            override_name = next(
                (old_name for old_name, mapped_name in renamed_args.items() if mapped_name == new_name), None
            )

            if new_name in reverse_member_args:
                old_name = reverse_member_args[new_name]
                actual = self._build_member_expr(
                    caller_routine, old_arg_map[old_name], member_args[old_name]['member']
                )
                use_keyword = True
            elif new_name in reverse_expr_args:
                old_name = reverse_expr_args[new_name]
                actual = self._build_template_expr(caller_routine, old_arg_map, expr_args[old_name])
                use_keyword = True
            elif new_name in position_args:
                actual = caller_routine.parse_expr(position_args[new_name])
            elif override_name and override_name in override_args:
                actual = caller_routine.parse_expr(override_args[override_name])
                use_keyword = True
            elif new_name in override_args:
                actual = caller_routine.parse_expr(override_args[new_name])
                use_keyword = True
            elif new_name in reverse_renamed_args:
                actual = old_arg_map.get(reverse_renamed_args[new_name])
                use_keyword = True
            elif new_name in old_arg_map:
                actual = old_arg_map[new_name]

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
