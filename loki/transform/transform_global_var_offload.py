from loki.transform.transformation import Transformation
from loki.bulk.item import GlobalVarImportItem
from loki.analyse import dataflow_analysis_attached
from loki.ir import Pragma, CallStatement, Import
from loki.visitors.find import FindNodes
from loki.visitors.transform import Transformer
from loki.expression.expr_visitors import FindInlineCalls
import re

__all__ = ['GlobalVarOffloadTransformation']

class GlobalVarOffloadTransformation(Transformation):

    def __init__(self):

        self._acc_copyin = set()
        self._acc_copyout = set()
        self._var_set = set()
#        self._imports = set()

    def transform_module(self, module, **kwargs):

        item = kwargs['item']

        #... bail if not global variable import
        if not isinstance(item, GlobalVarImportItem):
            return

        #... confirm that var to be offloaded is declared in module
        _symbol = item.name.split('#')[-1].lower()
        assert _symbol in [s.name.lower() for s in module.variables]

        sym_dict = {s.name.lower(): s for s in module.variables}
        pragmas = [p for p in FindNodes(Pragma).visit(module.spec) if p.keyword.lower() == 'acc']

        #... do nothing if var is a parameter
        if sym_dict[_symbol].type.parameter:
            return

        #... check if var is already declared
        for p in pragmas:
            if re.search(fr'\b{_symbol}\b', p.content.lower()):
                return

        self._var_set.add(_symbol)
        module.spec.append(Pragma(keyword='acc', content=f'declare create({_symbol})'))

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs.get('role')
        if role == 'driver':
            self.process_driver(routine, **kwargs)
        elif role == 'kernel':
            self.process_kernel(routine, **kwargs)

    def process_driver(self, routine, **kwargs):

        pragma_map = {}
        for pragma in [pragma for pragma in FindNodes(Pragma).visit(routine.body) if pragma.keyword == 'loki']:
            if 'update_device' in pragma.content:
                if self._acc_copyin:
                    pragma_map.update({pragma: Pragma(keyword='acc', content='update device(' + ','.join(self._acc_copyin) + ')')})
                else:
                    pragma_map.update({pragma: None})
            if 'update_host' in pragma.content:
                if self._acc_copyout:
                    pragma_map.update({pragma: Pragma(keyword='acc', content='update self(' + ','.join(self._acc_copyout) + ')')})
                else:
                    pragma_map.update({pragma: None})

        routine.body = Transformer(pragma_map).visit(routine.body)

    def process_kernel(self, routine, **kwargs):

        _imports = FindNodes(Import).visit(routine.spec)
        _calls = [c.name for c in FindNodes(CallStatement).visit(routine.body)]
        _icalls = FindInlineCalls().visit(routine.body)

        _imports = [i for i in _imports if i.symbols not in _calls]
        _imports = [i for i in _imports if i.symbols not in _icalls]

        _import_dict = {s.name.lower(): i.module for i in _imports for s in i.symbols}
        _imports = [s.name.lower() for i in _imports for s in i.symbols]

        with dataflow_analysis_attached(routine):
            for item in [i for i in _imports if i in kwargs.get('targets')]:
                if item not in self._var_set:
                    continue
                if item in routine.body.uses_symbols:
                    self._acc_copyin.add(item)
                if item in routine.body.defines_symbols:
                    self._acc_copyout.add(item)
