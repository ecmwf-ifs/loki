# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from pathlib import Path

import re
import yaml

from loki.batch import Transformation, TypeDefItem, ProcedureItem
from loki.ir import (
        nodes as ir, FindNodes, SubstituteExpressions, Transformer,
        pragma_regions_attached, get_pragma_parameters, SubstitutePragmaStrings
)
from loki.expression import symbols as sym
from loki.analyse.analyse_dataflow import DataflowAnalysisAttacher, DataflowAnalysisDetacher
from loki.transformations.utilities import find_driver_loops
from loki.logging import warning
from loki.tools import as_tuple
from loki.types import BasicType, DerivedType, SymbolAttributes

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


def map_derived_type_arguments(arg_map, analysis):
    """
    Map the root variable of derived-type dummy argument components
    to the corresponding argument.
    """

    _analysis = {}
    for k, v in analysis.items():

        dummy_root = k.parents[0] if k.parents else k
        if not (arg := arg_map.get(dummy_root, None)):
            continue

        expr_map = {dummy_root: arg}
        var = SubstituteExpressions(expr_map).visit(k)

        _analysis[var] = v

    return _analysis


def create_nested_dict(k, v, variable_map):
    """Create nested dict from derived-type expression."""

    name_parts = k.name.split('%', maxsplit=1)
    parent = variable_map[name_parts[0]].clone(dimensions=None)
    if len(name_parts) > 1:
        child_name_parts = name_parts[1].split('%', maxsplit=1)
        child = parent.type.dtype.typedef.variable_map[child_name_parts[0]]
        if len(child_name_parts) > 1:
            child = child.get_derived_type_member(child_name_parts[1])
        v = create_nested_dict(child, v, parent.type.dtype.typedef.variable_map)

    return {parent: v}


def merge_nested_dict(ref_dict, temp_dict, force=False):
    """Merge nested dicts."""

    for key in temp_dict:
        if key in ref_dict:
            if isinstance(temp_dict[key], dict) and isinstance(ref_dict[key], dict):
                ref_dict[key] = merge_nested_dict(ref_dict[key], temp_dict[key], force=force)
            elif force:
                ref_dict[key] = temp_dict[key]
        else:
            ref_dict.update({key: temp_dict[key]})

    return ref_dict


class DeepcopyDataflowAnalysisAttacher(DataflowAnalysisAttacher):
    """
    Dummy argument intents in Fortran also have implications on memory status, and `INTENT(OUT)`
    is therefore fundamentally unsafe for allocatables and pointers. Therefore in order to discern
    write-only accesses to arguments, we have to bypass the intent. This is achieved here by importing
    the dataflow analysis of the child :any:`Subroutine` and ignoring the intents altogether.
    """

    def visit_CallStatement(self, o, **kwargs):

        successor_map = kwargs['successor_map']

        if not o.routine:
            msg = f'[Loki::DataOffloadDeepcopyAnalysis] Cannot apply transformation without enriching calls: {o}.'
            raise RuntimeError(msg)

        child = successor_map.get(o, None)
        if not child:
            return self.visit_Node(o, **kwargs)

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

    The resulting analysis is a nested dict, of nesting depth equal to the longest 
    derived-type expression, containing the access mode of all the arguments used 
    in a call-tree. For example, the following assignments:

    .. code-block:: fortran
       a%b%c = a%b%c + 1
       d = e


    would yield the following analysis:

    .. code-block:: python
       {
          a: {
            b: {
               c: 'readwrite'
            }
          },
          d: 'write',
          e: 'read' 
       }

    The analysis is stored in the :any:`Item.trafo_data` of the :any:`Item` corresponding to the driver layer
    :any:`Subroutine`. It should be noted that the analysis is stored per driver-layer loop. The driver's
    :any:`Item.trafo_data` also contains :any:`Scheduler` config entries corresponding to the derived-types
    used throughout the call-tree in a :data:`typedef_configs` dict.

    Parameters
    ----------
    output_analysis : bool
       If enabled, the analysis is written to disk as yaml files. For kernels, the files are named
       routine.name_dataoffload_analysis.yaml. For drivers, the files are named 
       driver_target-name_offload_analysis.yaml, where "target-name" is the name of the first target
       routine in a given driver loop.
    """

    _key = 'DataOffloadDeepcopyAnalysis'

    reverse_traversal = True
    """Traversal from the leaves upwards"""

    item_filter = (ProcedureItem, TypeDefItem)
    # Modules (correctly) placed in the ignore list contain type definitions and must
    # therefore be processed.
    process_ignored_items = True

    def __init__(self, output_analysis=False):
        self.output_analysis = output_analysis

    def transform_subroutine(self, routine, **kwargs):

        if not (item := kwargs.pop('item', None)):
            msg = f'[Loki::DataOffloadDeepcopyAnalysis] Cannot apply transformation without item: {routine}.'
            raise RuntimeError(msg)

        role = kwargs.pop('role')
        targets = kwargs.pop('targets')
        sgraph = kwargs.pop('sub_sgraph')
        successors = sgraph.successors(item=item)


        if role == 'driver':
            self.process_driver(routine, item, successors, targets, **kwargs)
        if role == 'kernel':
            self.process_kernel(routine, item, successors, sgraph, **kwargs)

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

    def process_driver(self, routine, item, successors, targets, **kwargs):

        item.trafo_data[self._key] = defaultdict(dict)

        for loop in find_driver_loops(routine.body, targets):

            # We can't simply map successor.ir: successor here because we may call a routine twice with different
            # arguments
            successor_map = {}
            calls = FindNodes(ir.CallStatement).visit(loop.body)
            for call in calls:
                if (successor := [s for s in successors if call.routine == s.ir]):
                    successor_map[call] = successor[0]

            # gather analysis from children
            analysis = self.gather_analysis_from_children(successor_map)
            # gather typedef configs from children
            self.gather_typedef_configs_from_callees(successors, item.trafo_data[self._key]['typedef_configs'])

            layered_dict = {}
            for k, v in analysis.items():
                _temp_dict = create_nested_dict(k, v, routine.symbol_map)
                layered_dict = merge_nested_dict(layered_dict, _temp_dict)

            item.trafo_data[self._key]['analysis'][loop] = layered_dict

            if self.output_analysis:
                str_layered_dict = self.stringify_dict(layered_dict)
                base_dir = Path(kwargs['build_args']['output_dir'])
                with open(base_dir/f'driver_{list(successor_map.keys())[0].name}_dataoffload_analysis.yaml', 'w') as f:
                    yaml.dump(str_layered_dict, f)

    def process_kernel(self, routine, item, successors, sgraph, **kwargs):

        item.trafo_data[self._key] = defaultdict(dict)

        # gather typedef config overrides
        for child in successors:
            if isinstance(child, TypeDefItem):
                self.gather_typedef_configs(child, sgraph, item.trafo_data[self._key]['typedef_configs'])

        # gather typedef configs from callees
        self.gather_typedef_configs_from_callees(successors, item.trafo_data[self._key]['typedef_configs'])

        # Pointer indirection completely breaks the dataflow analysis, as the target
        # simply appears as if its being "read", regardless of how the pointer is used.
        # Since resolving pointer association is (super) hard, we just warn the user
        # here to double check the dataflow and provide overrides if necessary.
        pointers = any(a.ptr for a in FindNodes(ir.Assignment).visit(routine.body))
        if pointers:
            warning(f'[Loki::DataOffloadDeepcopyAnalysis] Pointer associations found in {routine.name}')

        # We can't simply map successor.ir: successor here because we may call a routine twice with different
        # arguments
        successor_map = {}
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if (successor := [s for s in successors if call.routine == s.ir]):
                successor_map[call] = successor[0]

        # We make do here (lazily) without a context manager, as this override of the
        # DataflowAnalysisAttacher is not meant for use outside of the current module.
        DeepcopyDataflowAnalysisAttacher().visit(routine.spec, successor_map=successor_map)
        DeepcopyDataflowAnalysisAttacher().visit(routine.body, successor_map=successor_map)

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

        if self.output_analysis:
            layered_dict = {}
            for k, v in item.trafo_data[self._key]['analysis'].items():
                _temp_dict = create_nested_dict(k, v, routine.symbol_map)
                layered_dict = merge_nested_dict(layered_dict, _temp_dict)

            base_dir = Path(kwargs['build_args']['output_dir'])
            with open(base_dir/f'{routine.name.lower()}_dataoffload_analysis.yaml', 'w') as file:
                str_layered_dict = self.stringify_dict(layered_dict)
                yaml.dump(str_layered_dict, file)

    def gather_analysis_from_children(self, successor_map):
        """Gather analysis from callees."""

        analysis = {}
        for call, child in successor_map.items():

            arg_map = get_sanitised_arg_map(call.arg_map)
            child_analysis = child.trafo_data[self._key]['analysis']
            child_analysis = map_derived_type_arguments(arg_map, child_analysis)

            for k, v in child_analysis.items():
                _v = analysis.get(k, v)
                if _v != v:
                    if _v == 'write':
                        continue
                    analysis[k] = 'readwrite'
                else:
                    analysis[k] = _v

        return analysis

    def gather_typedef_configs_from_callees(self, successors, typedef_configs):
        """Gather typedef configs from children."""

        for child in successors:
            if isinstance(child, ProcedureItem) and child.trafo_data.get(self._key, None):
                typedef_configs.update(child.trafo_data[self._key]['typedef_configs'])

    def gather_typedef_configs(self, item, sgraph, typedef_configs):
        """Gather typdef configs."""

        typedef_configs.update({item.ir.name.lower(): item.config})
        for child in sgraph.successors(item=item):
            if isinstance(child, TypeDefItem):
                self.gather_typedef_configs(child, sgraph, typedef_configs)


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
            _temp_dict = create_nested_dict(var, key, variable_map)
            analysis = merge_nested_dict(analysis, _temp_dict, force=True)

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
