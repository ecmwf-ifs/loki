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
    SubstitutePragmaStrings, SubstituteExpressions
)
from loki.expression import symbols as sym
from loki.types import BasicType, DerivedType, SymbolAttributes
from loki.analyse import DataflowAnalysisAttacher, DataflowAnalysisDetacher
from loki.tools import as_tuple
from loki.transformations import find_driver_loops
from loki.logging import warning

__all__ = ['DataOffloadDeepcopyAnalysis', 'DataOffloadDeepcopyTransformation']


def strip_nested_dimensions(expr):
    """
    Strip dimensions from array expressions of arbitrary derived-type
    nesting depth.
    """

    parent = expr.parent
    if parent:
        parent = strip_nested_dimensions(parent)
    return expr.clone(dimensions=None, parent=parent)


def get_sanitised_arg_map(arg_map):
    """
    Return sanitised mapping of dummy argument names to arguments.
    """

    _arg_map = {}
    for dummy, arg in arg_map.items():
        if isinstance(arg, sym._Literal):
            continue
        if isinstance(arg, sym.LogicalNot):
            arg = arg.child

        _arg_map[dummy.clone(dimensions=None)] = strip_nested_dimensions(arg)

    return _arg_map


def get_root_var(var):
    """Get the root variable of a derived-type expression."""

    if var.parent:
        root = get_root_var(var.parent)
        return root
    else:
        return var


def map_derived_type_arguments(arg_map, analysis):
    """
    Map the root variable of derived-type dummy argument components
    to the corresponding argument.
    """

    _analysis = {}
    for k, v in analysis.items():

        dummy_root = get_root_var(k)
        if not (arg := arg_map.get(dummy_root, None)):
            continue

        expr_map = {dummy_root: arg}
        var = SubstituteExpressions(expr_map).visit(k)

        _analysis[var] = v

    return _analysis


class DeepcopyDataflowAnalysisAttacher(DataflowAnalysisAttacher):

    def visit_CallStatement(self, o, **kwargs):

        successors = kwargs['successors']
        routine = kwargs['routine']

        if not o.routine:
            raise RuntimeError('Cannot apply DataOffloadAnalysis without enriching calls.')

        child = [child for child in successors if child.ir == o.routine]
        if not child:
            return self.visit_Node(o, **kwargs)

        child = child[0]

        # remap root variable names to current scope
        arg_map = get_sanitised_arg_map(o.arg_map)
        child_analysis = child.trafo_data['DataOffloadDeepcopyAnalysis']['analysis']
        child_analysis = map_derived_type_arguments(arg_map, child_analysis)

        defines, uses = set(), set()
        for k, v in child_analysis.items():

            if 'read' in v:
                uses |= {k}
            if 'write' in v:
                defines |= {k}

        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

class DataOffloadDeepcopyAnalysis(Transformation):
    """
    A transformation pass to analyse the usage of subroutine arguments in a call-tree.
    """

    _key = 'DataOffloadDeepcopyAnalysis'

    reverse_traversal = True
    """Traversal from the leaves upwards"""

    item_filter = (ProcedureItem, TypeDefItem)
    # Modules (correctly) placed in the ignore list contain type definitions and must
    # therefore be processed.
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
           self.process_driver(routine, item, successors, targets)
        if role == 'kernel':
            self.process_kernel(routine, item, successors)

    @classmethod
    def _resolve_nesting(cls, k, v, variable_map):
        name_parts = k.name.split('%', maxsplit=1)
        parent = variable_map[name_parts[0]].clone(dimensions=None)
        if len(name_parts) > 1:
            child_name_parts = name_parts[1].split('%', maxsplit=1)
            child = parent.type.dtype.typedef.variable_map[child_name_parts[0]]
            if len(child_name_parts) > 1:
                child = child.get_derived_type_member(child_name_parts[1])
            v = cls._resolve_nesting(child, v, parent.type.dtype.typedef.variable_map)

        return {parent: v}

    @classmethod
    def _nested_merge(cls, ref_dict, temp_dict, force=False):
        for key in temp_dict:
            if key in ref_dict:
                if isinstance(temp_dict[key], dict) and isinstance(ref_dict[key], dict):
                    ref_dict[key] = cls._nested_merge(ref_dict[key], temp_dict[key], force=force)
                elif not isinstance(ref_dict[key], dict) and not isinstance(temp_dict[key], dict) and not force:
                    if ref_dict[key] == temp_dict[key]:
                        continue
                    else:
                        raise RuntimeError(f'[Loki::DataOffloadDeepcopyAnalysis] conflicting dataflow analysis for {key}')
                elif not isinstance(ref_dict[key], dict):
                    ref_dict[key] = temp_dict[key]
            else:
                ref_dict.update({key: temp_dict[key]})

        return ref_dict

    def stringify_dict(self, _dict):
        """
        Stringify expression keys of a nested dict.
        """

        stringified_dict = {}
        for k, v in _dict.items():
            if isinstance(v, dict):
                stringified_dict[k.name.lower()] = self.stringify_dict(v)
            else:
                stringified_dict[k.name.lower()] = v

        return stringified_dict

    def process_driver(self, routine, item, successors, targets):

        item.trafo_data[self._key] = defaultdict(dict)
        exclude_vars = item.config.get('exclude_offload_vars', [])

        for loop in find_driver_loops(routine.body, targets):

            calls = FindNodes(ir.CallStatement).visit(loop.body)
            call_routines = [call.routine for call in calls]
            _successors = [s for s in successors if s.ir in call_routines or not isinstance(s, ProcedureItem)]

            #gather analysis from children
            analysis = {}
            self._gather_from_children(loop, item, analysis, _successors)

            layered_dict = {}
            for k, v in analysis.items():
                _temp_dict = self._resolve_nesting(k, v, routine.symbol_map)
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
                str_layered_dict = self.stringify_dict(layered_dict)
                with open(f'driver_{_successors[0].ir.name}_dataoffload_analysis.yaml', 'w') as file:
                    yaml.dump(str_layered_dict, file)

    def process_kernel(self, routine, item, successors):

        #gather analysis from children
        item.trafo_data[self._key] = defaultdict(dict)
        self._gather_typedefs_from_children(successors, item.trafo_data[self._key]['typedef_configs'])
        variable_map = routine.variable_map

        pointers = [a for a in FindNodes(ir.Assignment).visit(routine.body) if a.ptr]
        if pointers:
            warning(f'[Loki] Data offload deepcopy: pointer associations found in {routine.name}')

        DeepcopyDataflowAnalysisAttacher().visit(routine.spec, variable_map=variable_map, successors=successors,
                                                 dummies=routine._dummies, routine=routine)
        DeepcopyDataflowAnalysisAttacher().visit(routine.body, variable_map=variable_map, successors=successors,
                                                 dummies=routine._dummies, routine=routine)

        #gather used symbols in specification
        for v in routine.spec.uses_symbols:
            if v.name_parts[0].lower() in routine._dummies:
                item.trafo_data[self._key]['analysis'][v.clone(dimensions=None)] = 'read'

        #gather used and defined symbols in body
        for v in routine.body.uses_symbols:
            if v.name_parts[0].lower() in routine._dummies:
                item.trafo_data[self._key]['analysis'][v.clone(dimensions=None)] = 'read'

        for v in routine.body.defines_symbols:
            if v.name_parts[0].lower() in routine._dummies:
                if v in (routine.spec.uses_symbols | routine.body.uses_symbols):
                    item.trafo_data[self._key]['analysis'][v.clone(dimensions=None)] = 'readwrite'
                else:
                    item.trafo_data[self._key]['analysis'][v.clone(dimensions=None)] = 'write'

        DataflowAnalysisDetacher().visit(routine.spec)
        DataflowAnalysisDetacher().visit(routine.body)

        if self.debug:
            layered_dict = {}
            for k, v in item.trafo_data[self._key]['analysis'].items():
                _temp_dict = self._resolve_nesting(k, v, routine.symbol_map)
                layered_dict = self._nested_merge(layered_dict, _temp_dict)

            with open(f'{routine.name.lower()}_dataoffload_analysis.yaml', 'w') as file:
                str_layered_dict = self.stringify_dict(layered_dict)
                yaml.dump(str_layered_dict, file)

    def _gather_from_children(self, routine, item, analysis, successors):
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            child = [child for child in successors if child.ir == call.routine]
            if child:
                child = child[0]

                if call.routine is BasicType.DEFERRED:
                    raise RuntimeError('Cannot apply DataOffloadAnalysis without enriching calls.')

                arg_map = get_sanitised_arg_map(call.arg_map)
                child_analysis = child.trafo_data[self._key]['analysis']
                child_analysis = map_derived_type_arguments(arg_map, child_analysis)

                for k, v in child_analysis.items():
                    _v = analysis.get(k, v)
                    if _v != v:
                        if _v == 'write':
                            continue
                        elif _v == 'read' and 'write' in v:
                            analysis.update({k: 'readwrite'})
                    else:
                        analysis.update({k: _v})

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
                'parent_present': [p.lower() for p in parameters.get('parent_present', '').split(',')]
        }

    @staticmethod
    def _update_with_manual_overrides(key, parameters, analysis, routine):
        override_vars = parameters.get(key, [])
        if override_vars:
            override_vars = [p.strip().lower() for p in override_vars.split(',')]

        variable_map = routine.variable_map
        for v in override_vars:
            name_parts = v.split('%', maxsplit=1)
            var = variable_map[name_parts[0]]
            if len(name_parts) > 1:
                var = var.get_derived_type_member(name_parts[1])
            _temp_dict = DataOffloadDeepcopyAnalysis._resolve_nesting(var, key, variable_map)
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

                symbol_map = routine.symbol_map

                for loop in driver_loops:

                    _analysis = analysis[loop]

                    #update analysis with manual overrides
                    _analysis = self._update_with_manual_overrides('read', parameters, _analysis, routine)
                    _analysis = self._update_with_manual_overrides('write', parameters, _analysis, routine)
                    _analysis = self._update_with_manual_overrides('readwrite', parameters, _analysis, routine)

                    kwargs = self._init_kwargs(mode, _analysis, typedef_configs, parameters)
                    _copy, _host, _wipe = self.generate_deepcopy(routine, symbol_map, **kwargs)

                    copy += _copy
                    host += _host
                    wipe += _wipe

                    if (new_vars := set(kwargs['new_vars'])):
                        routine.variables += as_tuple(new_vars)

#                    if mode == 'offload':
#                        if (private_vars := kwargs.get('private', None)):
#                            pragma = ir.Pragma(keyword='loki', content=f"loop driver private({','.join(private_vars)})")
#                            loop._update(pragma=as_tuple(pragma))

                    present_vars = [v.name for v in _analysis if not v in kwargs['private'] and not
                            (isinstance(v, sym.Scalar) and isinstance(v.type.dtype, BasicType))]

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

        condition = sym.InlineCall(function=sym.ProcedureSymbol(check, scope=var.scope),
                                   parameters=as_tuple(var.clone(dimensions=None)))
        return as_tuple(ir.Conditional(condition=condition, body=body))

    @staticmethod
    def create_aliased_ptr_assignment(ptr, alias):
        dims = [sym.InlineCall(function=sym.ProcedureSymbol('LBOUND', scope=ptr.scope),
                               parameters=(ptr, sym.IntLiteral(r+1))) for r in range(len(ptr.shape))]

        lhs = ptr.parent.type.dtype.typedef.variable_map[alias].clone(parent=ptr.parent,
            dimensions=as_tuple([sym.RangeIndex(children=(d, None)) for d in dims]))
        return ir.Assignment(lhs=lhs, rhs=ptr, ptr=True)

    def _set_field_api_ptrs(self, var, stat, typedef_config, parent, mode):

        _var = var.name.lower()
        if (view_prefix := typedef_config.get('view_prefix', None)):
            _var = re.sub(f'^{view_prefix}', '', _var)

        aliased_ptrs = typedef_config.get('aliased_ptrs', {})
        reverse_alias_map = dict((v, k) for k, v in aliased_ptrs.items())

        field_object = typedef_config['field_prefix'].lower()
        field_object += reverse_alias_map.get(_var, _var).replace('_field', '')
        field_object = parent.type.dtype.typedef.variable_map[field_object].clone(parent=parent)
        field_ptr = var.clone(dimensions=None, parent=parent)

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
            alias_var = parent.type.dtype.typedef.variable_map[alias].clone(parent=parent, dimensions=None)
            device += as_tuple(self.create_aliased_ptr_assignment(field_ptr, alias))
            device += as_tuple(ir.Pragma(keyword='acc', content=f'enter data attach({alias_var})'))


        host = as_tuple(self.create_field_api_call(field_object, 'HOST', 'RDWR',
                                                   field_ptr))
        wipe = ()

        if mode == 'offload':
            if alias:
                host += as_tuple(self.create_aliased_ptr_assignment(field_ptr, alias))
                wipe += as_tuple(ir.Pragma(keyword='acc', content=f'exit data detach({alias_var}) finalize'))
            wipe += as_tuple(ir.Pragma(keyword='acc', content=f'exit data detach({field_ptr}) finalize'))
            wipe += as_tuple(ir.CallStatement(name=sym.Variable(name='DELETE_DEVICE_DATA', parent=field_object),
                                                  arguments=()))
        elif alias:
                host += as_tuple(self.create_aliased_ptr_assignment(field_ptr, alias))

        device = self.create_memory_status_test('ASSOCIATED', field_object, device)
        host = self.create_memory_status_test('ASSOCIATED', field_object, host)
        wipe = self.create_memory_status_test('ASSOCIATED', field_object, wipe)

        return device, host, wipe

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
                str_map = {str(k): str(v) for k, v in vmap.items()}

                loopbody = as_tuple(SubstitutePragmaStrings(str_map).visit(body))
                loopbody = as_tuple(SubstituteExpressions(vmap).visit(loopbody))
            else:
                vmap = {sym.Variable(name=f'J{dim}'): sym.Variable(name=f'J{dim}, J{dim+1}')}
                str_map = {str(k): str(v) for k, v in vmap.items()}

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

            if isinstance(var.type.dtype, DerivedType):

                _parent = kwargs.pop('parent')
                parent = var.clone(parent=_parent)

                _copy, _host, _wipe = None, None, None
                if not analysis[var] in ['read', 'readwrite', 'write']:
                    _copy, _host, _wipe = self.generate_deepcopy(routine, var.type.dtype.typedef.variable_map,
                                                                 analysis=analysis[var], parent=parent, **kwargs)

                #wrap in loop
                if var.type.shape:
                    _copy = self._wrap_in_loopnest(routine, var, _parent, _copy)
                    _host = self._wrap_in_loopnest(routine, var, _parent, _host)
                    _wipe = self._wrap_in_loopnest(routine, var, _parent, _wipe)

                if kwargs['mode'] == 'offload':
                    if (not _parent and not var in kwargs['parent_present']) or \
                        (var.type.allocatable or var.type.pointer):
                        _copy = as_tuple(ir.Pragma(keyword='acc',
                        content=f'enter data copyin({var.clone(parent=_parent, dimensions=None)})')) + _copy
                        if delete:
                            _wipe += as_tuple(ir.Pragma(keyword='acc',
                            content=f'exit data delete({var.clone(parent=_parent, dimensions=None)}) finalize'))

                #wrap in memory status check
                check = 'ASSOCIATED' if var.type.pointer else None
                check = 'ALLOCATED' if var.type.allocatable else None
                if check:
                    _copy = self.create_memory_status_test(check, var.clone(parent=_parent), _copy)
                    _host = self.create_memory_status_test(check, var.clone(parent=_parent), _host)
                    _wipe = self.create_memory_status_test(check, var.clone(parent=_parent), _wipe)

                kwargs['parent'] = _parent
            else:
                _copy = ()
                _host = ()
                _wipe = ()

                stat = analysis[var]
                parent = kwargs['parent']
                mode = kwargs['mode']
                typedef_config = kwargs['typedef_configs'].get(parent.type.dtype.typedef.name.lower(), None) \
                                 if parent else None
                field = False
                if typedef_config:
                    field = var in typedef_config.get('field_ptrs', [])

                if parent and re.search('^field_[0-9][a-z][a-z]_array', parent.type.dtype.typedef.name.lower()):
                    typedef_config = {'field_prefix': 'F_'}
                    field = True

                # check for FIELD_API pointer
                if re.search('_field$', var.name, re.IGNORECASE) or field:
                    _copy, _host, _wipe = self._set_field_api_ptrs(var, stat, typedef_config, parent, mode)

                elif mode == 'offload':
                # if not we have a regular variable
                    check = 'ASSOCIATED' if var.type.pointer else None
                    check = 'ALLOCATED' if var.type.allocatable else None
                    if check:
                        _copy = as_tuple(ir.Pragma(keyword='acc',
                                    content=f'enter data copyin({var.clone(parent=parent)})'))

                    if stat != 'read':
                        _host = as_tuple(ir.Pragma(keyword='acc',
                                    content=f'update self({var.clone(parent=parent)})'))
                    if check:
                        _wipe = as_tuple(ir.Pragma(keyword='acc',
                        content=f'exit data delete finalize({var.clone(parent=parent, dimensions=None)})'))

            copy += as_tuple(_copy)
            if delete and not temporary:
                host += as_tuple(_host)
            if delete:
                wipe += as_tuple(_wipe)

        return copy, host, wipe
