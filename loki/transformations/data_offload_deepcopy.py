# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import yaml
import re
import pdb

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
        successors = kwargs.get('successors', ())

        if not (item := kwargs.get('item', None)):
            raise RuntimeError('Cannot apply DataOffloadAnalysis without item to store analysis.')

        if role == 'driver':
           self.process_driver(routine, item, successors, 'driver')
        if role == 'kernel':
            self.process_kernel(routine, item, successors, 'kernel')

    def _resolve_nesting(self, k, v):
        name_parts = k.split('%')
        if len(name_parts) > 1:
            v = self._resolve_nesting('%'.join(name_parts[1:]), v)

        return {name_parts[0]: v}

    def _nested_merge(self, ref_dict, temp_dict):
        for key in temp_dict:
            if key in ref_dict:
                if isinstance(temp_dict[key], dict) and isinstance(ref_dict[key], dict):
                    ref_dict[key] = self._nested_merge(ref_dict[key], temp_dict[key])
                elif not isinstance(ref_dict[key], dict) and not isinstance(temp_dict[key], dict):
                    ref_dict[key] += temp_dict[key]
                elif not isinstance(ref_dict[key], dict):
                    ref_dict[key] = temp_dict[key]
            else:
                ref_dict.update({key: temp_dict[key]})

        return ref_dict

    def process_driver(self, routine, item, successors, role):
        #gather analysis from children
        item.trafo_data[self._key] = defaultdict(dict)
        self._gather_from_children(routine, item, successors, role)

        layered_dict = {}
        for k, v in item.trafo_data[self._key]['analysis'].items():
            _temp_dict = self._resolve_nesting(k, v)
            layered_dict = self._nested_merge(layered_dict, _temp_dict)
        item.trafo_data[self._key]['analysis'] = layered_dict

        # filter out explicitly exlucded vars
        if (exclude_vars := item.config.get('exclude_offload_vars', [])):
            item.trafo_data[self._key]['analysis'] = {k: v for k, v in item.trafo_data[self._key]['analysis'].items()
                                                      if not k in exclude_vars
            }

        if self.debug:
            with open('dataoffload_analysis.yaml', 'w') as file:
                yaml.dump(item.trafo_data[self._key]['analysis'], file)

    def process_kernel(self, routine, item, successors, role):

        #gather analysis from children
        item.trafo_data[self._key] = defaultdict(dict)
        self._gather_from_children(routine, item, successors, role)

        with dataflow_analysis_attached(routine, exclude_calls=True):
            #gather used symbols in specification
            item.trafo_data[self._key]['analysis'].update({v.name.lower(): 'read' for v in routine.spec.uses_symbols
                                               if v.name_parts[0].lower() in routine._dummies})

            #gather used and defined symbols in body
            item.trafo_data[self._key]['analysis'].update({v.name.lower(): 'read' for v in routine.body.uses_symbols
                                               if v.name_parts[0].lower() in routine._dummies})
            _defined_symbols = {v for v in routine.body.defines_symbols if v.name_parts[0].lower() in routine._dummies}
            item.trafo_data[self._key]['analysis'].update({v.name.lower(): 'readwrite'
                                               if item.trafo_data[self._key].get(v.name.lower(), '') == 'read' else 'write'
                                               for v in _defined_symbols})

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
                item.trafo_data[self._key]['analysis'].update({k: v
                                                   if v == item.trafo_data[self._key]['analysis'].get(v, v) else 'readwrite'
                                                   for k, v in _child_analysis.items()})

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

    def process_driver(self, routine, analysis, typedef_configs, targets):

        pragma_map = {}
        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):

                # Only work on active `!$loki data` regions
                if not (mode := self._is_active_loki_data_region(region)):
                    continue

                parameters = get_pragma_parameters(region.pragma, starts_with='data')

                kwargs = self._init_kwargs(mode, analysis, typedef_configs, parameters)
                copy, host, wipe = self.generate_deepcopy(routine, routine.symbol_map, **kwargs)

                if mode == 'offload':
                    if (private_vars := kwargs.get('private', None)):
                        pragma = ir.Pragma(keyword='loki', content=f"loop driver private({','.join(private_vars)})")
                        for loop in find_driver_loops(region.body, targets):
                            loop._update(pragma=as_tuple(pragma))

                    # wrap in acc data pragma
                    present_vars = [v.upper() for v in analysis if not v in kwargs['private']]
                    acc_data_pragma = ir.Pragma(keyword='acc', content=f"data present({','.join(present_vars)})")
                    acc_data_pragma_post = ir.Pragma(keyword='acc', content="end data")

                    pragma_map.update({region.pragma: (copy, acc_data_pragma),
                                       region.pragma_post: (acc_data_pragma_post, host, wipe)})
                else:
                    pragma_map.update({region.pragma: host, region.pragma_post: None})

                if (new_vars := set(kwargs['new_vars'])):
                    routine.variables += as_tuple(new_vars)

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

        field_object = typedef_config['field_prefix'].lower() + var.lower().replace('_field', '')
        field_object = symbol_map[field_object].clone(parent=parent)
        field_ptr = symbol_map[var].clone(dimensions=None, parent=parent)
        aliased_ptrs = typedef_config.get('aliased_ptrs', [])

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

                typedef_config = kwargs['typedef_configs'][parent.type.dtype.typedef.name.lower()] if parent else None

                # check for FIELD_API pointer
                if re.search('_field$', var, re.IGNORECASE):
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
