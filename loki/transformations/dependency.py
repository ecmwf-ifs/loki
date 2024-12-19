# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.ir import nodes as ir, Transformer, FindNodes
from loki.tools.util import as_tuple

__all__ = ['DuplicateKernel', 'RemoveKernel']


class DuplicateKernel(Transformation):

    creates_items = True

    def __init__(self, kernels=None, duplicate_suffix='duplicated',
                 duplicate_module_suffix=None):
        self.suffix = duplicate_suffix
        self.module_suffix = duplicate_module_suffix or duplicate_suffix
        print(f"suffix: {self.suffix}")
        print(f"module_suffix: {self.module_suffix}")
        self.kernels = tuple(kernel.lower() for kernel in as_tuple(kernels))

    def transform_subroutine(self, routine, **kwargs):

        item = kwargs.get('item', None)
        item_factory = kwargs.get('item_factory', None)
        if not item and 'items' in kwargs:
            if kwargs['items']:
                item = kwargs['items'][0]

        successors = as_tuple(kwargs.get('successors'))
        item.plan_data['additional_dependencies'] = ()
        new_deps = {}
        for child in successors:
            if child.local_name.lower() in self.kernels:
                new_dep = item_factory.clone_procedure_item(child, self.suffix, self.module_suffix)
                new_deps[new_dep.name.lower()] = new_dep

        imports = as_tuple(FindNodes(ir.Import).visit(routine.spec))
        parent_imports = as_tuple(FindNodes(ir.Import).visit(routine.parent.ir)) if routine.parent is not None else ()
        all_imports = imports + parent_imports
        import_map = {}
        for _imp in all_imports:
            for symbol in _imp.symbols:
                import_map[symbol] = _imp

        calls = FindNodes(ir.CallStatement).visit(routine.body)
        call_map = {}
        for call in calls:
            if str(call.name).lower() in self.kernels:
                new_call_name = f'{str(call.name)}_{self.suffix}'.lower()
                call_map[call] = (call, call.clone(name=new_deps[new_call_name].procedure_symbol))
                if call.name in import_map:
                    new_import_module = \
                            import_map[call.name].module.upper().replace('MOD', f'{self.module_suffix.upper()}_MOD')
                    new_symbols = [symbol.clone(name=f"{symbol.name}_{self.suffix}")
                                   for symbol in import_map[call.name].symbols]
                    new_import = ir.Import(module=new_import_module, symbols=as_tuple(new_symbols))
                    routine.spec.append(new_import)
        routine.body = Transformer(call_map).visit(routine.body)

    def plan_subroutine(self, routine, **kwargs):
        item = kwargs.get('item', None)
        item_factory = kwargs.get('item_factory', None)
        if not item and 'items' in kwargs:
            if kwargs['items']:
                item = kwargs['items'][0]

        successors = as_tuple(kwargs.get('successors'))
        item.plan_data['additional_dependencies'] = ()
        for child in successors:
            if child.local_name.lower() in self.kernels:
                new_dep = item_factory.clone_procedure_item(child, self.suffix, self.module_suffix)
                item.plan_data['additional_dependencies'] += as_tuple(new_dep)

class RemoveKernel(Transformation):

    creates_items = True

    def __init__(self, kernels=None):
        self.kernels = tuple(kernel.lower() for kernel in as_tuple(kernels))

    def transform_subroutine(self, routine, **kwargs):
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        call_map = {}
        for call in calls:
            if str(call.name).lower() in self.kernels:
                call_map[call] = None
        routine.body = Transformer(call_map).visit(routine.body)

    def plan_subroutine(self, routine, **kwargs):
        item = kwargs.get('item', None)
        item_factory = kwargs.get('item_factory', None)
        if not item and 'items' in kwargs:
            if kwargs['items']:
                item = kwargs['items'][0]

        successors = as_tuple(kwargs.get('successors'))
        item.plan_data['removed_dependencies'] = ()
        for child in successors:
            if child.local_name.lower() in self.kernels:
                item.plan_data['removed_dependencies'] += (child.local_name.lower(),)
        # propagate 'removed_dependencies' to corresponding module (if it exists)
        module_name = item.name.split('#')[0]
        if module_name:
            module_item = item_factory.item_cache[item.name.split('#')[0]]
            module_item.plan_data['removed_dependencies'] = item.plan_data['removed_dependencies']
