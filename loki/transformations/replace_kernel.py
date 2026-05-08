from pathlib import Path

from loki.batch import ModuleItem, ProcedureItem, Transformation
from loki.ir import FindNodes, Transformer, nodes as ir
from loki.logging import warning
from loki.tools import CaseInsensitiveDict, as_tuple
from loki.types import BasicType

__all__ = ['ReplaceKernels', 'ReplaceKernels2']


class ReplaceKernels(Transformation):
    """
    Replace calls to one kernel with another kernel.

    The replacement map accepts entries of the form::

        {
            'old_kernel': {
                'routine': 'new_kernel',
                'args': {'old_dummy': 'new_dummy_or_expr'},
                'ignore': True,
            }
        }
    """

    creates_items = True

    def __init__(self, replace_kernels_map=None):
        self.replace_kernels_map = CaseInsensitiveDict(replace_kernels_map or {})

    @staticmethod
    def _get_repl_item_from_cache(repl_name, item_factory):
        item_cache = item_factory.item_cache

        if repl_name in item_cache and isinstance(item_cache[repl_name], ProcedureItem):
            return item_cache[repl_name]

        qualified_matches = [
            item for item in item_cache.values()
            if isinstance(item, ProcedureItem) and item.local_name.lower() == repl_name.lower()
        ]
        if len(qualified_matches) == 1:
            return qualified_matches[0]

        for key, item in item_cache.items():
            if isinstance(item, ProcedureItem) and repl_name.lower() in key.lower():
                return item

        return None

    @staticmethod
    def _complete_repl_source(repl_item, scheduler_config, build_args):
        frontend_args = {
            key: value for key, value in build_args.items()
            if key in ('definitions', 'preprocess', 'includes', 'defines', 'xmods', 'omni_includes', 'frontend')
        }
        frontend_args = scheduler_config.create_frontend_args(repl_item.source.path, frontend_args)
        repl_item.source.make_complete(**frontend_args)

    def _load_repl_item_from_source(self, repl_name, item_factory, scheduler_config, build_args):
        paths = as_tuple(build_args.get('paths', ()))
        if not paths:
            paths = as_tuple(build_args.get('includes', ()))

        for root in paths:
            root = Path(root)
            if not root.exists():
                continue
            for path in root.glob('**/*'):
                if not path.is_file():
                    continue
                if repl_name.lower() not in path.stem.lower():
                    continue
                frontend_args = {
                    key: value for key, value in build_args.items()
                    if key in ('definitions', 'preprocess', 'includes', 'defines', 'xmods', 'omni_includes', 'frontend')
                }
                file_item = item_factory.get_or_create_file_item_from_path(path, scheduler_config, frontend_args)
                frontend_args = scheduler_config.create_frontend_args(path, frontend_args)
                file_item.source.make_complete(**frontend_args)
                definition_items = file_item.create_definition_items(item_factory=item_factory, config=scheduler_config)
                for definition_item in definition_items:
                    if isinstance(definition_item, ModuleItem):
                        definition_item.create_definition_items(item_factory=item_factory, config=scheduler_config)
                repl_item = self._get_repl_item_from_cache(repl_name, item_factory)
                if repl_item is not None:
                    return repl_item
        return None

    def _get_complete_repl_item(self, repl_name, **kwargs):
        item_factory = kwargs.get('item_factory')
        scheduler_config = kwargs.get('scheduler_config')
        build_args = dict(kwargs.get('build_args', {}))

        repl_item = self._get_repl_item_from_cache(repl_name, item_factory)
        if repl_item is None:
            repl_item = self._load_repl_item_from_source(repl_name, item_factory, scheduler_config, build_args)
        if repl_item is None:
            return None

        self._complete_repl_source(repl_item, scheduler_config, build_args)
        return repl_item

    @staticmethod
    def _extend_block_list(item, repl_item):
        """
        Add the replacement item to the caller's ``block`` list.

        Using ``block`` (rather than ``ignore``) prevents the replacement
        from ever entering the scheduler graph, which avoids complex
        interactions with mode propagation and multi-pipeline processing.
        """
        block_entries = list(as_tuple(item.config.get('block', ())))
        candidates = [repl_item.local_name]
        if repl_item.scope_name:
            candidates.append(repl_item.scope_name)

        for candidate in candidates:
            if candidate not in block_entries:
                block_entries.append(candidate)

        item.config['block'] = tuple(block_entries)

    @staticmethod
    def _split_arg_rules(replacement_args, new_routine):
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
        member = member.lstrip('%')
        return routine.parse_expr(f'{actual}%{member}')

    @staticmethod
    def _build_template_expr(routine, old_arg_map, expr_rule):
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

    def _adapt_arguments(self, call, caller_routine, new_routine):
        """Build argument list for the replacement call.
        
        Parameters
        ----------
        call : CallStatement - the original call
        caller_routine : Subroutine - the routine containing the call (for parse_expr)
        new_routine : Subroutine - the replacement routine
        """
        replacement = self.replace_kernels_map[str(call.name).lower()]
        replacement_args = replacement.get('args', {})
        renamed_args, override_args, position_args, member_args, expr_args = \
            self._split_arg_rules(replacement_args, new_routine)

        if call.routine is not None and call.routine is not BasicType.DEFERRED:
            old_arg_map = CaseInsensitiveDict((dummy.name, actual) for dummy, actual in call.arg_map.items())
        else:
            # Routine not resolved (e.g. imported via #include intfb.h)
            # Build from keyword args + positional args keyed by expression name
            old_arg_map = CaseInsensitiveDict((kw[0], kw[1]) for kw in call.kwarguments)
            for actual in call.arguments:
                old_arg_map.setdefault(str(actual).lower(), actual)

        reverse_renamed_args = {new_name: old_name for old_name, new_name in renamed_args.items()}
        reverse_member_args = {
            rule['map_to']: old_name for old_name, rule in member_args.items()
        }
        reverse_expr_args = {
            rule['map_to']: old_name for old_name, rule in expr_args.items()
        }
        arguments = []
        kwarguments = []
        keep_positional = True

        for new_dummy in new_routine.arguments:
            new_name = new_dummy.name.lower()
            actual = None
            use_keyword = not keep_positional
            override_name = next((old_name for old_name, mapped_name in renamed_args.items() if mapped_name == new_name), None)

            if new_name in reverse_member_args:
                old_name = reverse_member_args[new_name]
                actual = self._build_member_expr(caller_routine, old_arg_map[old_name], member_args[old_name]['member'])
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

    def _transform_subroutine(self, routine, **kwargs):
        item = kwargs.get('item')

        import_by_symbol = CaseInsensitiveDict()
        # Map kernel names to c_import includes (e.g. #include "kernel_a1.intfb.h")
        intfb_import_by_name = CaseInsensitiveDict()
        routine_import_map = {}
        parent_import_map = {}

        for imprt in FindNodes(ir.Import).visit(routine.spec):
            if imprt.c_import and imprt.module and '.intfb.h' in imprt.module:
                # Extract kernel name from "kernel_name.intfb.h"
                kernel_name = imprt.module.replace('.intfb.h', '')
                intfb_import_by_name[kernel_name] = ('routine', imprt)
            for symbol in imprt.symbols:
                import_by_symbol[str(symbol).lower()] = ('routine', imprt)

        if routine.parent is not None and getattr(routine.parent, 'spec', None) is not None:
            for imprt in FindNodes(ir.Import).visit(routine.parent.spec):
                if imprt.c_import and imprt.module and '.intfb.h' in imprt.module:
                    kernel_name = imprt.module.replace('.intfb.h', '')
                    intfb_import_by_name.setdefault(kernel_name, ('parent', imprt))
                for symbol in imprt.symbols:
                    import_by_symbol.setdefault(str(symbol).lower(), ('parent', imprt))

        call_map = {}
        new_imports = []

        for call in FindNodes(ir.CallStatement).visit(routine.body):
            call_name = str(call.name).lower()
            if call_name not in self.replace_kernels_map:
                continue

            replacement = self.replace_kernels_map[call_name]
            repl_name = replacement.get('routine')
            repl_item = self._get_complete_repl_item(repl_name, **kwargs)
            if repl_item is None:
                continue

            proc_symbol = repl_item.ir.procedure_symbol.rescope(scope=routine)
            arguments, kwarguments = self._adapt_arguments(call, routine, repl_item.ir)
            call_map[call] = call.clone(name=proc_symbol, arguments=arguments, kwarguments=kwarguments)

            if call_name in import_by_symbol:
                location, imprt = import_by_symbol[call_name]
                if location == 'routine':
                    routine_import_map[imprt] = imprt.clone(module=repl_item.scope_name, symbols=(proc_symbol,))
                else:
                    parent_import_map[imprt] = imprt.clone(module=repl_item.scope_name, symbols=(proc_symbol,))
            elif call_name in intfb_import_by_name:
                # Remove the #include and collect a USE statement for later insertion
                location, imprt = intfb_import_by_name[call_name]
                if location == 'routine':
                    routine_import_map[imprt] = None  # delete
                else:
                    parent_import_map[imprt] = None  # delete
                new_imports.append(ir.Import(module=repl_item.scope_name, symbols=(proc_symbol,)))
            else:
                # No existing import found — inject a new USE in routine spec
                new_imports.append(ir.Import(module=repl_item.scope_name, symbols=(proc_symbol,)))

            if item is not None and replacement.get('ignore', False):
                self._extend_block_list(item, repl_item)

        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)
        if routine_import_map:
            routine.spec = Transformer(routine_import_map).visit(routine.spec)
        if parent_import_map:
            routine.parent.spec = Transformer(parent_import_map).visit(routine.parent.spec)
        if new_imports:
            # Insert after the last existing USE statement, or prepend if none exist
            use_positions = [
                i for i, node in enumerate(routine.spec.body)
                if isinstance(node, ir.Import) and not node.c_import and not node.f_include
            ]
            insert_pos = (use_positions[-1] + 1) if use_positions else 0
            routine.spec.insert(insert_pos, new_imports)

    def transform_subroutine(self, routine, **kwargs):
        self._transform_subroutine(routine, **kwargs)

    def plan_subroutine(self, routine, **kwargs):
        self._transform_subroutine(routine, **kwargs)


class ReplaceKernels2(ReplaceKernels):
    pass
