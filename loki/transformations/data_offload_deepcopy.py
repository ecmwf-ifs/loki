# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import yaml
import re

from collections import defaultdict

from loki.backend import fgen
from loki.batch import Transformation, TypeDefItem, ProcedureItem
from loki.ir import (
    nodes as ir, FindNodes, pragma_regions_attached, get_pragma_parameters, Transformer,
    SubstitutePragmaStrings
)
from loki.expression import symbols as sym, SubstituteExpressions, FindLiterals
from loki.types import BasicType, DerivedType, SymbolAttributes
from loki.analyse import dataflow_analysis_attached
from loki.tools import as_tuple
from loki.transformations import find_driver_loops
from loki.logging import warning

__all__ = ['DataOffloadDeepcopyAnalysis', 'DataOffloadDeepcopyTransformation']

class DataOffloadDeepcopyAnalysis(Transformation):
    """
    A transformation pass to analyse the usage of subroutine arguments in a call-tree.
    """

    _key = 'DataOffloadDeepcopyAnalysis'

    reverse_traversal = True
    """Traversal from the leaves upwards"""

    item_filter = (ProcedureItem, TypeDefItem)
    process_ignored_items = True

    def __init__(self, debug=False, **kwargs):
        self.debug = debug

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        targets = kwargs['targets']
        successors = kwargs.get('successors', ())

        if not (item := kwargs.get('item', None)):
            raise RuntimeError('Cannot apply DataOffloadAnalysis without item to store analysis.')

        if role == 'driver':
           self.process_driver(routine, item, successors, role, targets)
        if role == 'kernel':
            self.process_kernel(routine, item, successors, role, targets)

    @classmethod
    def _resolve_nesting(cls, k, v):
        name_parts = k.split('%')
        if len(name_parts) > 1:
            v = cls._resolve_nesting('%'.join(name_parts[1:]), v)

        return {name_parts[0]: v}

    @classmethod
    def _nested_merge(cls, ref_dict, temp_dict, force=False):
        for key in temp_dict:
            if key in ref_dict:
                if isinstance(temp_dict[key], dict) and isinstance(ref_dict[key], dict):
                    ref_dict[key] = cls._nested_merge(ref_dict[key], temp_dict[key], force=force)
                elif not isinstance(ref_dict[key], dict) and not isinstance(temp_dict[key], dict) and not force:
                    ref_dict[key] += temp_dict[key]
                elif not isinstance(ref_dict[key], dict):
                    ref_dict[key] = temp_dict[key]
            else:
                ref_dict.update({key: temp_dict[key]})

        return ref_dict

    def process_driver(self, routine, item, successors, role, targets):

        item.trafo_data[self._key] = defaultdict(dict)
        exclude_vars = item.config.get('exclude_offload_vars', [])

        for loop in find_driver_loops(routine.body, targets):
            calls = FindNodes(ir.CallStatement).visit(loop.body)
            call_routines = [call.routine for call in calls]
            _successors = [s for s in successors if s.ir in call_routines or not isinstance(s, ProcedureItem)]

            #gather analysis from children
            self._gather_from_children(routine, item, _successors, role)

            layered_dict = {}
            for k, v in item.trafo_data[self._key]['analysis'].items():
                _temp_dict = self._resolve_nesting(k, v)
                layered_dict = self._nested_merge(layered_dict, _temp_dict)
            item.trafo_data[self._key]['analysis'][loop] = layered_dict

            # filter out explicitly exlucded vars
            if exclude_vars:
                item.trafo_data[self._key]['analysis'][loop] = {k: v for k, v in
                                                                item.trafo_data[self._key]['analysis'][loop].items()
                                                                if not k in exclude_vars
                }

            if self.debug:
                _successors = [s for s in _successors if isinstance(s, ProcedureItem)]
                with open(f'driver_{_successors[0].ir.name}_dataoffload_analysis.yaml', 'w') as file:
                    yaml.dump(item.trafo_data[self._key]['analysis'][loop], file)

    def process_kernel(self, routine, item, successors, role, targets):

        #gather analysis from children
        item.trafo_data[self._key] = defaultdict(dict)
        self._gather_from_children(routine, item, successors, role)
        variable_map = routine.variable_map

        pointers = [a for a in FindNodes(ir.Assignment).visit(routine.body) if a.ptr]
        if pointers:
            warning(f'[Loki] Data offload deepcopy: pointer associations found in {routine.name}')

        with dataflow_analysis_attached(routine, targets=targets):
            #gather used symbols in specification
            for v in routine.spec.uses_symbols:
                if v.name_parts[0].lower() in routine._dummies:
                    stat = item.trafo_data[self._key]['analysis'].get(v.name.lower(), 'read')
                    if stat == 'write':
                        stat = 'readwrite'
                    item.trafo_data[self._key]['analysis'].update({v.name.lower(): stat})

            #gather used and defined symbols in body
            for v in routine.body.uses_symbols:
                if v.name_parts[0].lower() in routine._dummies:
                    stat = item.trafo_data[self._key]['analysis'].get(v.name.lower(), 'read')
                    if stat == 'write':
                        stat = 'readwrite'
                    item.trafo_data[self._key]['analysis'].update({v.name.lower(): stat})

            for v in routine.body.defines_symbols:
                if v.name_parts[0].lower() in routine._dummies:
                    if v in routine.body.uses_symbols:
                        item.trafo_data[self._key]['analysis'].update({v.name.lower(): 'readwrite'})
                    else:
                        item.trafo_data[self._key]['analysis'].update({v.name.lower(): 'write'})

        item.trafo_data[self._key]['analysis'] = {k: v
            for k, v in item.trafo_data[self._key]['analysis'].items()
            if not isinstance(getattr(getattr(variable_map.get(k, k), 'type', None), 'dtype', None), DerivedType)
        }


        if self.debug:
            layered_dict = {}
            for k, v in item.trafo_data[self._key]['analysis'].items():
                _temp_dict = self._resolve_nesting(k, v)
                layered_dict = self._nested_merge(layered_dict, _temp_dict)

            with open(f'{routine.name.lower()}_dataoffload_analysis.yaml', 'w') as file:
                yaml.dump(layered_dict, file)

    @staticmethod
    def _sanitise_args(arg_map):

        _arg_map = {}
        for dummy, arg in arg_map.items():
            if isinstance(arg, sym._Literal):
                continue
            if isinstance(arg, sym.LogicalNot):
                arg = arg.child

            _arg_map.update({dummy.name.lower(): arg.name.lower()})

        return _arg_map

    def _gather_from_children(self, routine, item, successors, role):
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            child = [child for child in successors if child.ir == call.routine]
            if child:
                child = child[0]

                if call.routine is BasicType.DEFERRED:
                    raise RuntimeError('Cannot apply DataOffloadAnalysis without enriching calls.')

                _arg_map = self._sanitise_args(call.arg_map)

                _child_analysis = {'%'.join([_arg_map.get(n, n) for n in k.split('%')]): v
                                   for k, v in child.trafo_data[self._key]['analysis'].items()}
                if role == 'kernel':
                    _child_analysis = {k: v for k, v in _child_analysis.items()
                                       if k.split('%')[0].lower() in routine._dummies}

                for k, v in _child_analysis.items():
                    _v = item.trafo_data[self._key]['analysis'].get(k, v)
                    if _v != v:
                        if _v == 'write':
                            item.trafo_data[self._key]['analysis'].update({k: 'write'})
                        elif _v == 'read' and 'write' in v:
                            item.trafo_data[self._key]['analysis'].update({k: 'readwrite'})
                    else:
                        item.trafo_data[self._key]['analysis'].update({k: _v})

        self._gather_typedefs_from_children(successors, item.trafo_data[self._key]['typedef_configs'])

    def _gather_typedefs_from_children(self, successors, typedef_configs):
        for child in successors:
            if isinstance(child, TypeDefItem) and child.trafo_data.get(self._key, None):
                for k, v in child.trafo_data[self._key]['typedef_configs'].items():
                    typedef_configs[k] = v

    def transform_typedef(self, typedef, **kwargs):
        successors = kwargs.get('successors', [])
        item = kwargs['item']

        item.trafo_data[self._key] = defaultdict(dict)
        item.trafo_data[self._key]['typedef_configs'][typedef.name.lower()] = item.config
        self._gather_typedefs_from_children(successors, item.trafo_data[self._key]['typedef_configs'])


class DataOffloadDeepcopyTransformation(Transformation):
    """
    A transformation to generate deepcopies of derived-types.
    """

    _key = 'DataOffloadDeepcopyAnalysis'

    def transform_subroutine(self, routine, **kwargs):

        if not (item := kwargs.get('item', None)):
            raise RuntimeError('Cannot apply DataOffloadDeepcopyTransformation without item containing analysis.')

        if not item.trafo_data[self._key]:
            raise RuntimeError('Must run DataOffloadDeepcopyAnalysis before the transformation.')

        role = kwargs['role']
        targets = kwargs['targets']

        if role == 'driver':
            self.process_driver(routine, item.trafo_data[self._key]['analysis'],
                                item.trafo_data[self._key]['typedef_configs'], targets)

    @staticmethod
    def _is_active_loki_data_region(region):
        if region.pragma.keyword.lower() != 'loki':
            return False
        if 'data' not in region.pragma.content.lower():
            return False

        if 'set_pointers' in region.pragma.content.lower():
            return 'set_pointers'
        if 'offload' in region.pragma.content.lower():
            return 'offload'

        return False

    @staticmethod
    def _init_kwargs(mode, analysis, typedef_configs, parameters):
        return {'new_vars': (),
                'mode': mode,
                'parent': None,
                'analysis': analysis,
                'typedef_configs': typedef_configs,
                'present': [p.lower() for p in parameters.get('present', '').split(',')],
                'private': [p.lower() for p in parameters.get('private', '').split(',')],
                'temporary': [p.lower() for p in parameters.get('temporary', '').split(',')],
                'device_resident': [p.lower() for p in parameters.get('device_resident', '').split(',')],
        }

    @staticmethod
    def _update_with_manual_overrides(key, parameters, analysis):
        override_vars = parameters.get(key, [])
        if override_vars:
            override_vars = [p.strip().lower() for p in override_vars.split(',')]

        for v in override_vars:
            _temp_dict = DataOffloadDeepcopyAnalysis._resolve_nesting(v, key)
            analysis = DataOffloadDeepcopyAnalysis._nested_merge(analysis, _temp_dict, force=True)

        return analysis

    def process_driver(self, routine, analysis, typedef_configs, targets):

        pragma_map = {}
        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):

                # Only work on active `!$loki data` regions
                if not (mode := self._is_active_loki_data_region(region)):
                    continue

                parameters = get_pragma_parameters(region.pragma, starts_with='data')

                copy, host, wipe = (), (), ()

                driver_loops = find_driver_loops(region.body, targets)
                assert len(driver_loops) == 1

                for loop in driver_loops:

                    _analysis = analysis[loop]

                    #update analysis with manual overrides
                    _analysis = self._update_with_manual_overrides('read', parameters, _analysis)
                    _analysis = self._update_with_manual_overrides('write', parameters, _analysis)
                    _analysis = self._update_with_manual_overrides('readwrite', parameters, _analysis)

                    kwargs = self._init_kwargs(mode, _analysis, typedef_configs, parameters)
                    _copy, _host, _wipe = self.generate_deepcopy(routine, routine.symbol_map, **kwargs)

                    copy += _copy
                    host += _host
                    wipe += _wipe

                    if (new_vars := set(kwargs['new_vars'])):
                        routine.variables += as_tuple(new_vars)

                    if mode == 'offload':
                        if (private_vars := kwargs.get('private', None)):
                            pragma = ir.Pragma(keyword='loki', content=f"loop driver private({','.join(private_vars)})")
                            loop._update(pragma=as_tuple(pragma))

                    present_vars = [v.upper() for v in _analysis if not v in kwargs['private']]

                    # add create directives for unused arguments
                    arguments = [arg for call in FindNodes(ir.CallStatement).visit(loop.body) for arg in call.arguments]
                    create_vars = [arg.name.lower() for arg in arguments if not arg.name.lower() in _analysis]

                if mode == 'offload':
                    # wrap in acc data pragma
                    content = f"data present({','.join(present_vars)})"
                    if create_vars:
                        content += f" create({','.join(create_vars)})"
                    acc_data_pragma = ir.Pragma(keyword='acc', content=content)
                    acc_data_pragma_post = ir.Pragma(keyword='acc', content="end data")

                    pragma_map.update({region.pragma: (copy, acc_data_pragma),
                                       region.pragma_post: (acc_data_pragma_post, host, wipe)})
                else:
                    pragma_map.update({region.pragma: host, region.pragma_post: None})


        routine.body = Transformer(pragma_map).visit(routine.body)

    @staticmethod
    def create_field_api_call(field_object, dest, access, ptr):
        return ir.CallStatement(name=sym.Variable(name='GET_' + dest.upper() + '_DATA_' + access, parent=field_object),
                                arguments=as_tuple(ptr))

    @staticmethod
    def create_memory_status_test(check, var, body):

        condition = sym.IntrinsicLiteral(value=f'{check}({var.name})')
        return as_tuple(ir.Conditional(condition=condition, body=body))

    @staticmethod
    def create_aliased_ptr_assignment(ptr, alias):
        dims = [sym.InlineCall(function=sym.ProcedureSymbol('LBOUND', scope=ptr.scope),
                               parameters=(ptr, sym.IntLiteral(r+1))) for r in range(len(ptr.shape))]

        lhs = ptr.parent.type.dtype.typedef.variable_map[alias].clone(parent=ptr.parent,
            dimensions=as_tuple([sym.RangeIndex(children=(d, None)) for d in dims]))
        return ir.Assignment(lhs=lhs, rhs=ptr, ptr=True)

    def _set_field_api_ptrs(self, var, stat, symbol_map, typedef_config, parent, mode):

        _var = var
        if (view_prefix := typedef_config.get('view_prefix', None)):
            _var = re.sub(f'^{view_prefix}', '', var.lower())

        aliased_ptrs = typedef_config.get('aliased_ptrs', {})
        reverse_alias_map = dict((v, k) for k, v in aliased_ptrs.items())

        field_object = typedef_config['field_prefix'].lower()
        field_object += reverse_alias_map.get(_var.lower(), _var.lower()).replace('_field', '')
        field_object = symbol_map[field_object].clone(parent=parent)
        field_ptr = symbol_map[var].clone(dimensions=None, parent=parent)

        if stat == 'read':
            access = 'RDONLY'
        elif stat == 'readwrite':
            access = 'RDWR'
        else:
            access = 'WRONLY'

        device = as_tuple(self.create_field_api_call(field_object, 'DEVICE', access,
                                                     field_ptr))
        device += as_tuple(ir.Pragma(keyword='acc', content=f'enter data attach({field_ptr})'))
        if (alias := aliased_ptrs.get(field_ptr.name_parts[-1].lower(), None)):
            alias_var = symbol_map[alias].clone(parent=parent, dimensions=None)
            device += as_tuple(self.create_aliased_ptr_assignment(field_ptr, alias))
            device += as_tuple(ir.Pragma(keyword='acc', content=f'enter data attach({alias_var})'))


        host = as_tuple(self.create_field_api_call(field_object, 'HOST', 'RDWR',
                                                   field_ptr))
        wipe = ()

        if mode == 'offload':
            if alias:
                host += as_tuple(self.create_aliased_ptr_assignment(field_ptr, alias))
                wipe += as_tuple(ir.Pragma(keyword='acc', content=f'exit data detach({alias_var})'))
            wipe += as_tuple(ir.Pragma(keyword='acc', content=f'exit data detach({field_ptr})'))
            wipe += as_tuple(ir.CallStatement(name=sym.Variable(name='DELETE_DEVICE_DATA', parent=field_object),
                                                  arguments=()))
        elif alias:
                host += as_tuple(self.create_aliased_ptr_assignment(field_ptr, alias))

        device = self.create_memory_status_test('ASSOCIATED', field_object, device)
        host = self.create_memory_status_test('ASSOCIATED', field_object, host)
        wipe = self.create_memory_status_test('ASSOCIATED', field_object, wipe)

        return device, host, wipe

    @staticmethod
    def _map_memory_status_checks(str_map, body):
        literals = FindLiterals().visit(body)
        literals = [l for l in literals if isinstance(l, sym.IntrinsicLiteral)]

        _str_map = {SubstitutePragmaStrings._sanitise(k): v for k, v in str_map.items()}
        _literal_map = {}
        for l in literals:
            _l = l.value
            for k, v in _str_map.items():
                _l = re.sub(k, v, _l, flags=re.IGNORECASE)
            _literal_map[l] = sym.IntrinsicLiteral(value=_l)

        return _literal_map

    def _wrap_in_loopnest(self, routine, var, parent, body):

        loopbody = ()
        loop_vars = []
        for dim in range(len(var.type.shape)):
            loop_vars += as_tuple(sym.Variable(name=f'J{dim+1}', type=SymbolAttributes(dtype=BasicType.INTEGER)))

            lstart = sym.IntrinsicLiteral(value=f'LBOUND({parent.name}%{var.name},{dim+1})')
            lend = sym.IntrinsicLiteral(value=f'UBOUND({parent.name}%{var.name},{dim+1})')
            bounds = sym.LoopRange((lstart, lend))

            if not loopbody:
                vmap = {var.clone(parent=parent):
                        var.clone(parent=parent, dimensions=as_tuple(sym.Variable(name=f'J{dim+1}')))}
                str_map = {fgen(k): fgen(v) for k, v in vmap.items()}
                loopbody = as_tuple(SubstitutePragmaStrings(str_map).visit(body))

                vmap.update({var.clone(parent=parent, dimensions=None):
                             var.clone(parent=parent, dimensions=as_tuple(sym.Variable(name=f'J{dim+1}')))})
                str_map = {fgen(k): fgen(v) for k, v in vmap.items()}
                vmap.update(self._map_memory_status_checks(str_map, body))

                loopbody = as_tuple(SubstituteExpressions(vmap).visit(loopbody))
            else:
                vmap = {sym.Variable(name=f'J{dim}'): sym.Variable(name=f'J{dim}, J{dim+1}')}
                str_map = {fgen(k): fgen(v) for k, v in vmap.items()}

                vmap.update(self._map_memory_status_checks(str_map, loopbody))

                loopbody = as_tuple(SubstituteExpressions(vmap).visit(loopbody))
                loopbody = as_tuple(SubstitutePragmaStrings(str_map).visit(loopbody))

            loop = ir.Loop(variable=loop_vars[-1], bounds=bounds, body=loopbody)
            loopbody = loop

        routine.variables += as_tuple(loop_vars)
        return as_tuple(loop)

    def generate_deepcopy(self, routine, symbol_map, **kwargs):

        copy = ()
        host = ()
        wipe = ()
        analysis = kwargs.pop('analysis')

        for var in analysis:

            if var in kwargs['present'] or var in kwargs['private']:
                continue

            delete = not var in kwargs['device_resident']
            temporary = var in kwargs['temporary']
            if (parent := kwargs['parent']):
                delete = not parent.name_parts[0].lower() in kwargs['device_resident']
                temporary = parent.name_parts[0].lower() in kwargs['temporary']

            if isinstance(symbol_map[var].type.dtype, DerivedType):

                _parent = kwargs.pop('parent')
                parent = symbol_map[var].clone(parent=_parent)

                _copy, _host, _wipe = self.generate_deepcopy(routine, symbol_map[var].type.dtype.typedef.variable_map,
                                                             analysis=analysis[var], parent=parent, **kwargs)

                #wrap in loop
                if symbol_map[var].type.shape:
                    _copy = self._wrap_in_loopnest(routine, symbol_map[var], _parent, _copy)
                    _host = self._wrap_in_loopnest(routine, symbol_map[var], _parent, _host)
                    _wipe = self._wrap_in_loopnest(routine, symbol_map[var], _parent, _wipe)

                if kwargs['mode'] == 'offload':
                    if not _parent or (symbol_map[var].type.allocatable or symbol_map[var].type.pointer):
                        _copy = as_tuple(ir.Pragma(keyword='acc',
                        content=f'enter data copyin({symbol_map[var].clone(parent=_parent, dimensions=None)})')) + _copy
                        _wipe += as_tuple(ir.Pragma(keyword='acc',
                        content=f'exit data delete({symbol_map[var].clone(parent=_parent, dimensions=None)}) finalize'))

                #wrap in memory status check
                check = 'ASSOCIATED' if symbol_map[var].type.pointer else None
                check = 'ALLOCATED' if symbol_map[var].type.allocatable else None
                if check:
                    _copy = self.create_memory_status_test(check, symbol_map[var].clone(parent=_parent), _copy)
                    _host = self.create_memory_status_test(check, symbol_map[var].clone(parent=_parent), _host)
                    _wipe = self.create_memory_status_test(check, symbol_map[var].clone(parent=_parent), _wipe)

                kwargs['parent'] = _parent
            else:
                _copy = ()
                _host = ()
                _wipe = ()

                stat = analysis[var]
                parent = kwargs['parent']
                mode = kwargs['mode']
                field = False

                if re.search('^field_[0-9][a-z][a-z]_array', parent.type.dtype.typedef.name.lower()):
                    typedef_config = {'field_prefix': 'F_'}
                    field = True
                else:
                    typedef_config = kwargs['typedef_configs'][parent.type.dtype.typedef.name.lower()] if parent else None

                # check for FIELD_API pointer
                if re.search('_field$', var, re.IGNORECASE) or field:
                    _copy, _host, _wipe = self._set_field_api_ptrs(var, stat, symbol_map, typedef_config, parent, mode)

                elif mode == 'offload':
                # if not we have a regular variable
                    check = 'ASSOCIATED' if symbol_map[var].type.pointer else None
                    check = 'ALLOCATED' if symbol_map[var].type.allocatable else None
                    if check:
                        _copy = as_tuple(ir.Pragma(keyword='acc',
                                    content=f'enter data copyin({symbol_map[var].clone(parent=parent)})'))

                    if stat != 'read':
                        _host = as_tuple(ir.Pragma(keyword='acc',
                                    content=f'update self({symbol_map[var].clone(parent=parent)})'))
                    if check:
                        _wipe = as_tuple(ir.Pragma(keyword='acc',
                        content=f'exit data delete finalize({symbol_map[var].clone(parent=parent, dimensions=None)})'))

            copy += as_tuple(_copy)
            if delete and not temporary:
                host += as_tuple(_host)
            if delete:
                wipe += as_tuple(_wipe)

        return copy, host, wipe
