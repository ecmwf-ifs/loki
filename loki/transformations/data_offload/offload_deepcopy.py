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
        pragma_regions_attached, get_pragma_parameters, SubstitutePragmaStrings,
        is_loki_pragma
)
from loki.expression import symbols as sym
from loki.analyse.analyse_dataflow import DataflowAnalysisAttacher, DataflowAnalysisDetacher
from loki.transformations.utilities import find_driver_loops
from loki.logging import warning
from loki.tools import as_tuple
from loki.types import BasicType, DerivedType, SymbolAttributes
from loki.transformations.field_api import (
        FieldAPITransferType, field_get_device_data, field_get_host_data, field_delete_device_data
)

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
            self.process_kernel(routine, item, successors, **kwargs)

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
            self.gather_typedef_configs(successors, item.trafo_data[self._key]['typedef_configs'])

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

    def process_kernel(self, routine, item, successors, **kwargs):

        item.trafo_data[self._key] = defaultdict(dict)

        # gather typedef configs from successors
        self.gather_typedef_configs(successors, item.trafo_data[self._key]['typedef_configs'])

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

    def gather_typedef_configs(self, successors, typedef_configs):
        """Gather typedef configs from children."""

        for child in successors:
            if isinstance(child, TypeDefItem) and child.trafo_data.get(self._key, None):
                typedef_configs.update(child.trafo_data[self._key]['typedef_configs'])

    def transform_module(self, module, **kwargs): # pylint: disable=unused-argument
        """Cache the current type definition config for later reuse."""

        item = kwargs['item']
        successors = kwargs['sub_sgraph'].successors(item=item)
        item.trafo_data[self._key] = defaultdict(dict)

        item.trafo_data[self._key]['typedef_configs'][item.ir.name.lower()] = item.config
        self.gather_typedef_configs(successors, item.trafo_data[self._key]['typedef_configs'])


class DataOffloadDeepcopyTransformation(Transformation):
    """
    A transformation to generate deepcopies of derived-types.
    """

    _key = 'DataOffloadDeepcopyAnalysis'
    field_array_match_pattern = re.compile('^field_[0-9][a-z][a-z]_array')

    def transform_subroutine(self, routine, **kwargs):

        if not (item := kwargs.get('item', None)):
            msg = '[Loki::DataOffloadDeepcopyTransformation] can only be applied by the Scheduler.'
            raise RuntimeError(msg)

        if not item.trafo_data[self._key]:
            raise RuntimeError(f'[Loki::DataOffloadDeepcopyTransformation] item missing analysis: {item.name}.')

        role = kwargs['role']
        targets = kwargs['targets']

        if role == 'driver':
            self.process_driver(routine, item.trafo_data[self._key]['analysis'],
                                item.trafo_data[self._key]['typedef_configs'], targets)

    @staticmethod
    def _is_active_loki_data_region(region):
        """Determine if we are in an active loki data region and if so return the deepcopy mode."""

        if is_loki_pragma(region.pragma, starts_with='data offload'):
            return 'offload'
        if is_loki_pragma(region.pragma, starts_with='data set_pointers'):
            return 'set_pointers'

        return False

    @staticmethod
    def update_with_manual_overrides(parameters, analysis, variable_map):
        """Update analysis with manual overrides specified in !loki data pragma."""

        override_map = {}
        for key in ['write', 'read', 'readwrite']:
            _vars = parameters.get(key, None)
            if _vars:
                _vars = [v.strip() for v in _vars.split(',')]
                override_map.update({var: key for var in _vars})

        for v, override in override_map.items():
            name_parts = v.split('%', maxsplit=1)
            var = variable_map[name_parts[0]]
            if len(name_parts) > 1:
                var = var.get_derived_type_member(name_parts[1])
            temp_dict = create_nested_dict(var, override, variable_map)
            analysis = merge_nested_dict(analysis, temp_dict, force=True)

        return analysis

    @staticmethod
    def get_pragma_vars(parameters, category):
        return [v.strip() for v in parameters.get(category, '').split(',')]

    def insert_deepcopy_instructions(self, region, mode, copy, host, wipe, present_vars):
        """Insert the generated deepcopy instructions and wrap the driver loop in 
           a `data present` pragma region if applicable."""

        if mode == 'offload':
            # wrap in acc data present pragma
            content = f"data present({', '.join(present_vars)})"
            acc_data_pragma = ir.Pragma(keyword='acc', content=content)
            acc_data_pragma_post = ir.Pragma(keyword='acc', content="end data")

            pragma_map = {region.pragma: (copy, acc_data_pragma)}
            pragma_map.update({region.pragma_post: (acc_data_pragma_post, host, wipe)})
        else:
            # We remove all offload instructions first and non F-API related boiler plate
            vmap = {}

            conds = FindNodes((ir.Conditional, ir.Loop), greedy=True).visit(host)
            for cond in conds:
                calls = FindNodes(ir.CallStatement).visit(cond.body)
                get_host_call = any('get_host_data_rdwr' in v.name.name.lower() for v in calls)

                if not get_host_call:
                    vmap[cond] = None

            host_pragmas = FindNodes(ir.Pragma).visit(host)
            vmap.update({p: None for p in host_pragmas})
            host = Transformer(vmap).visit(host)

            # Now we insert the updated "host" body in the driver layer
            pragma_map = {region.pragma: host, region.pragma_post: None}

        return pragma_map

    def process_driver(self, routine, analyses, typedef_configs, targets):

        pragma_map = {}
        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):

                # Only work on active `!$loki data` regions
                if not is_loki_pragma(region.pragma, starts_with='data'):
                    continue

                parameters = get_pragma_parameters(region.pragma, starts_with='data')
                driver_loops = find_driver_loops(region.body, targets)

                # skip the deepcopy for variables previously marked as present/private
                present = self.get_pragma_vars(parameters, 'present')
                private = self.get_pragma_vars(parameters, 'private')

                # temporary variables are not copied back to host and are wiped from device memory
                temporary = self.get_pragma_vars(parameters, 'temporary')

                # device_resident variables are left on device (i.e. neither copied back to host nor deleted)
                device_resident = self.get_pragma_vars(parameters, 'device_resident')

                copy, host, wipe = (), (), ()
                present_vars = ()
                for loop in driver_loops:

                    analysis = analyses[loop]

                    # update analysis with manual overrides
                    analysis = self.update_with_manual_overrides(parameters, analysis, routine.symbol_map)

                    # recursively traverse analysis and generate deepcopy
                    _copy, _host, _wipe = self.generate_deepcopy(routine, analysis=analysis, present=present,
                                                                 private=private, temporary=temporary,
                                                                 device_resident=device_resident,
                                                                 typedef_configs=typedef_configs)

                    copy += _copy
                    host += _host
                    wipe += _wipe

                    present_vars += as_tuple(v.name for v in analysis if not v in private)

                # replace the `!$loki data` PragmaRegion with the generated deepcopy instructions
                pragma_map.update(self.insert_deepcopy_instructions(region, mode, copy, host, wipe, present_vars))

        routine.body = Transformer(pragma_map).visit(routine.body)

    def wrap_in_loopnest(self, var, body, routine):
        """Wrap body in loop nest corresponding to the shape of var."""

        # Don't wrap an empty body
        if not body:
            return ()

        loopbody = ()
        loop_vars = []
        variable_map = routine.variable_map

        for dim in range(len(var.type.shape)):
            if f'j{dim+1}' in variable_map:
                loop_vars += [variable_map[f'j{dim+1}']]
            else:
                loop_vars += [sym.Variable(name=f'J{dim+1}', type=SymbolAttributes(dtype=BasicType.INTEGER),
                                           scope=routine)]
                routine.variables += as_tuple(loop_vars[-1])

            # Create loop bounds
            lstart = sym.InlineCall(function=sym.ProcedureSymbol('LBOUND', scope=routine),
                                    parameters=(var, sym.IntLiteral(dim+1)))
            lend = sym.InlineCall(function=sym.ProcedureSymbol('UBOUND', scope=routine),
                                    parameters=(var, sym.IntLiteral(dim+1)))
            bounds = sym.LoopRange((lstart, lend))

            if not loopbody:
                # Create first layer of loop nest
                vmap = {var: var.clone(dimensions=as_tuple(loop_vars[-1]))}
                str_map = {str(k): str(v) for k, v in vmap.items()}

                SubstitutePragmaStrings(str_map).visit(body)
                loopbody = as_tuple(SubstituteExpressions(vmap).visit(body))
            else:
                # Add subsequent layers
                vmap = {loop_vars[-2]: (loop_vars[-2], loop_vars[-1])}
                str_map = {str(k): str(v) for k, v in vmap.items()}

                SubstitutePragmaStrings(str_map).visit(loopbody)
                loopbody = as_tuple(SubstituteExpressions(vmap).visit(loopbody))

            loop = ir.Loop(variable=loop_vars[-1], bounds=bounds, body=loopbody)
            loopbody = loop

        return as_tuple(loop)

    @staticmethod
    def create_memory_status_test(check, var, body, scope):
        """Wrap a given body in a memory status check."""

        # Don't wrap an empty body
        if not body:
            return ()

        condition = sym.InlineCall(function=sym.ProcedureSymbol(check, scope=scope),
                                   parameters=as_tuple(var))
        return as_tuple(ir.Conditional(condition=condition, body=body))

    @staticmethod
    def enter_data_copyin(var):
        """Generate unstructured data copyin instruction."""
        return as_tuple(ir.Pragma(keyword='acc', content=f'enter data copyin({var})'))

    @staticmethod
    def enter_data_create(var):
        """Generate unstructured data create instruction."""
        return as_tuple(ir.Pragma(keyword='loki', content=f'unstructured-data create({var})'))

    @staticmethod
    def enter_data_attach(var):
        """Generate unstructured data attach instruction."""
        return as_tuple(ir.Pragma(keyword='acc', content=f'enter data attach({var})'))

    @staticmethod
    def exit_data_detach(var):
        """Generate unstructured data detach instruction."""
        return as_tuple(ir.Pragma(keyword='acc', content=f'exit data detach({var}) finalize'))

    @staticmethod
    def exit_data_delete(var):
        """Generate unstructured data delete instruction."""
        return as_tuple(ir.Pragma(keyword='acc', content=f'exit data delete({var}) finalize'))

    @staticmethod
    def update_self(var):
        """Pull back data to host."""
        return as_tuple(ir.Pragma(keyword='acc', content=f'update self({var})'))

    @staticmethod
    def create_aliased_ptr_assignment(ptr, alias):
        """Associate an aliased pointer to its target."""

        dims = [sym.InlineCall(function=sym.ProcedureSymbol('LBOUND', scope=ptr.scope),
                               parameters=(ptr, sym.IntLiteral(r+1))) for r in range(len(ptr.shape))]

        alias_ptr = ptr.parent.type.dtype.typedef.variable_map[alias]
        lhs = alias_ptr.clone(parent=ptr.parent,
                              dimensions=as_tuple([sym.RangeIndex(children=(d, None)) for d in dims]))

        return ir.Assignment(lhs=lhs, rhs=ptr, ptr=True)

    def create_field_api_offload(self, var, analysis, typedef_config, parent, scope):

        #TODO: currently this assumes FIELD objects and their associated pointers are
        # components of the same derived-type. This should be generalised for the case
        # where the two are declared separately.

        # Strip view pointer prefix
        var_name = var.name.lower()

        # Get FIELD object name
        if not (field_object_name := typedef_config['field_ptr_map'].get(var_name, None)):
            field_object_name = typedef_config['field_prefix'] + var_name.replace('_field', '')

        # Create FIELD object
        variable_map = parent.type.dtype.typedef.variable_map
        field_object = variable_map[field_object_name].clone(parent=parent)
        field_ptr = var.clone(dimensions=None, parent=parent)

        if analysis == 'read':
            access_mode = FieldAPITransferType.READ_ONLY
        elif analysis == 'readwrite':
            access_mode = FieldAPITransferType.READ_WRITE
        else:
            access_mode = FieldAPITransferType.WRITE_ONLY

        device = as_tuple(field_get_device_data(field_object, field_ptr, access_mode, scope))
        device += self.enter_data_attach(field_ptr)
        host = as_tuple(field_get_host_data(field_object, field_ptr, FieldAPITransferType.READ_WRITE, scope))
        wipe = self.exit_data_detach(field_ptr)
        wipe += as_tuple(field_delete_device_data(field_object, scope))

        device = self.create_memory_status_test('ASSOCIATED', field_object, device, scope)
        host = self.create_memory_status_test('ASSOCIATED', field_object, host, scope)
        wipe = self.create_memory_status_test('ASSOCIATED', field_object, wipe, scope)

        return device, host, wipe

    def create_dummy_field_array_typedef_config(self, parent):
        """The scheduler will never traverse the FIELD_RANKSUFF_ARRAY type definitions,
           so we create a dummy typedef config here."""

        if self.field_array_match_pattern.match(parent.type.dtype.typedef.name.lower()):
            typedef_config = {
                'field_prefix': 'F_',
                'field_ptr_suffix': '_FIELD',
                'field_ptr_map': {}
            }
            return typedef_config
        return None

    def generate_deepcopy(self, routine, **kwargs):
        """Recursively traverse the deepcopy analysis to generate the deepcopy instructions."""

        # initialise tuples used to store the deepcopy instructions
        copy, host, wipe = (), (), ()

        analysis = kwargs.pop('analysis')
        parent = kwargs.pop('parent', None)

        for var in analysis:

            _copy, _host, _wipe = (), (), ()

            # Don't generate a deepcopy for variables marked as present or private
            if var in kwargs['present'] or var in kwargs['private']:
                continue

            # determine if var should be kept on device
            delete = not var in kwargs['device_resident']
            # determine if this is a temporary variable
            temporary = var in kwargs['temporary']

            check = 'ASSOCIATED' if var.type.pointer else None
            check = 'ALLOCATED' if var.type.allocatable else None

            if isinstance(var.type.dtype, DerivedType):

                var_with_parent = var.clone(parent=parent)
                _copy, _host, _wipe = self.generate_deepcopy(routine, analysis=analysis[var], parent=var_with_parent,
                                                             **kwargs)

                #wrap in loop
                if var.type.shape:
                    _copy = self.wrap_in_loopnest(var_with_parent, _copy, routine)
                    _host = self.wrap_in_loopnest(var_with_parent, _host, routine)
                    _wipe = self.wrap_in_loopnest(var_with_parent, _wipe, routine)

                # var must be allocated/deallocated on device
                if not parent or check:
                    _copy = self.enter_data_copyin(var_with_parent) + _copy
                    _wipe += self.exit_data_delete(var_with_parent)

                # wrap in memory status check
                if check:
                    _copy = self.create_memory_status_test(check, var_with_parent, _copy, routine)
                    _host = self.create_memory_status_test(check, var_with_parent, _host, routine)
                    _wipe = self.create_memory_status_test(check, var_with_parent, _wipe, routine)

            else:

                # First determine whether we have a field pointer or a regular array/scalar
                typedef_config = None
                if parent:
                    typedef_config = kwargs['typedef_configs'].get(parent.type.dtype.typedef.name.lower(), None)

                # Create a dummy typedef config for FIELD_RANKSUFF_ARRAY types
                if parent and not typedef_config:
                    typedef_config = self.create_dummy_field_array_typedef_config(parent)

                field = False
                if typedef_config:
                    # Is our pointer in the given list of field ptrs or has the right suffix?
                    suffix = typedef_config['field_ptr_suffix']
                    field = var in typedef_config.get('field_ptrs', [])
                    field = field or re.search(f'{suffix}$', var.name, re.IGNORECASE)

                if field:
                    _copy, _host, _wipe = self.create_field_api_offload(var, analysis[var], typedef_config,
                                                                        parent, routine)
                else:
                    # We have a regular array/scalar
                    if not parent or check:
                        if analysis[var] == 'write':
                            _copy = self.enter_data_create(var.clone(parent=parent))
                        else:
                            _copy = self.enter_data_copyin(var.clone(parent=parent))
                        _wipe = self.exit_data_delete(var.clone(parent=parent))

                    # Copy back to host if necessary
                    if analysis[var] != 'read':
                        _host = self.update_self(var.clone(parent=parent))

                    # wrap in memory status check
                    if check:
                        _copy = self.create_memory_status_test(check, var.clone(parent=parent), _copy, routine)
                        _host = self.create_memory_status_test(check, var.clone(parent=parent), _host, routine)
                        _wipe = self.create_memory_status_test(check, var.clone(parent=parent), _wipe, routine)

            copy += as_tuple(_copy)
            if delete and not temporary:
                host += as_tuple(_host)
            if delete:
                wipe += as_tuple(_wipe)

        return copy, host, wipe
