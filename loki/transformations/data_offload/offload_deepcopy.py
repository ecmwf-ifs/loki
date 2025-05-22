# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from pathlib import Path

import yaml

from loki.batch import Transformation, TypeDefItem, ProcedureItem
from loki.ir import nodes as ir, FindNodes, SubstituteExpressions
from loki.expression import symbols as sym
from loki.analyse.analyse_dataflow import DataflowAnalysisAttacher, DataflowAnalysisDetacher
from loki.transformations.utilities import find_driver_loops
from loki.logging import warning

__all__ = ['DataOffloadDeepcopyAnalysis']


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
    A transformation pass to analyse the usage of subroutine arguments in a call-tree. The resulting analysis is a
    nested dict, of nesting depth equal to the longest derived-type expression, containing the access
    mode of all the arguments used in a call-tree. For example, the following assignments:

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
    :any:`Subroutine`. It should be noted that the analysis is stored per driver-layer loop.

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
        successors = kwargs.pop('sub_sgraph').successors(item=item)


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

            #gather analysis from children
            analysis = self.gather_analysis_from_children(successor_map)
            self.gather_typedefs_from_children(successors, item.trafo_data[self._key]['typedef_configs'])

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

        #gather analysis from children
        item.trafo_data[self._key] = defaultdict(dict)
        self.gather_typedefs_from_children(successors, item.trafo_data[self._key]['typedef_configs'])

        pointers = any(a.ptr for a in FindNodes(ir.Assignment).visit(routine.body))
        if pointers:
            warning(f'[Loki::DataOffloadDeepcopyAnalysis] Pointer associations found in {routine.name}')

        # We can't simply map successor.ir: successor here because we may call a routine twice with different
        # arguments
        successor_map = {}
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if (successor := [s for s in successors if call.routine == s.ir]):
                successor_map[call] = successor[0]

        # We make do here (lazily) without a context manager, as this override of the
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

    def gather_typedefs_from_children(self, successors, typedef_configs):
        """Gather type definitions imported in children."""

        for child in successors:
            if isinstance(child, TypeDefItem) and child.trafo_data.get(self._key, None):
                for k, v in child.trafo_data[self._key]['typedef_configs'].items():
                    typedef_configs[k] = v

    def transform_typedef(self, typedef, **kwargs):
        """Cache the current type definition for later reuse."""

        item = kwargs['item']
        successors = kwargs['sub_sgraph'].successors(item=item)

        item.trafo_data[self._key] = defaultdict(dict)
        item.trafo_data[self._key]['typedef_configs'][typedef.name.lower()] = item.config
        self.gather_typedefs_from_children(successors, item.trafo_data[self._key]['typedef_configs'])
