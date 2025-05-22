# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import yaml
import re

from collections import defaultdict

from loki.batch import Transformation, TypeDefItem, ProcedureItem
from loki.ir import (
    nodes as ir, FindNodes, pragma_regions_attached, get_pragma_parameters, Transformer,
    SubstitutePragmaStrings, SubstituteExpressions
)
from loki.expression import symbols as sym
from loki.types import BasicType, DerivedType, SymbolAttributes
from loki.analyse.analyse_dataflow import DataflowAnalysisAttacher, DataflowAnalysisDetacher
from loki.tools import as_tuple
from loki.transformations.utilities import find_driver_loops
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

        if not (item := kwargs.get('item', None)):
            raise RuntimeError('Cannot apply DataOffloadAnalysis without item to store analysis.')

        role = kwargs['role']
        targets = kwargs['targets']
        successors = kwargs['sub_sgraph'].successors(item=item)


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
        item = kwargs['item']
        successors = kwargs['sub_sgraph'].successors(item=item)

        item.trafo_data[self._key] = defaultdict(dict)
        item.trafo_data[self._key]['typedef_configs'][typedef.name.lower()] = item.config
        self._gather_typedefs_from_children(successors, item.trafo_data[self._key]['typedef_configs'])
