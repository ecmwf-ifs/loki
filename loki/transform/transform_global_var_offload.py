from loki.transform.transformation import Transformation
from loki.bulk.item import GlobalVarImportItem
from loki.analyse import dataflow_analysis_attached
from loki.tools.util import as_tuple
from loki.ir import VariableDeclaration, Pragma
from loki.visitors.find import FindNodes
from loki.visitors.transform import Transformer

__all__ = ['GlobalVarImportAnalysis', 'GlobalVarOffloadTransformation']

class GlobalVarImportAnalysis(Transformation):

    def transform_subroutine(self, routine, **kwargs):
        with dataflow_analysis_attached(routine):
            for item in [s for s in kwargs.get('successors', None) if isinstance(s, GlobalVarImportItem)]:
                _symbol = item.name.split('#')[-1]
                if _symbol in routine.body.uses_symbols:
                    item.trafo_data.update({'acc_copyin': as_tuple(_symbol.lower())})
                if _symbol in routine.body.defines_symbols:
                    item.trafo_data.update({'acc_copyout': as_tuple(_symbol.lower())})

class GlobalVarOffloadTransformation(Transformation):

    def __init__(self):
        self._acc_copyin = set()
        self._acc_copyout = set()

    def transform_module(self, module, **kwargs):
        item = kwargs['item']

        copyin = item.trafo_data.get('acc_copyin')
        copyout = item.trafo_data.get('acc_copyout')

        #... skip transformation if symbol was imported but never used
        if not copyin and not copyout:
            return

        #... confirm that any vars to be offloaded are declared in module
        decl_symbs = [s.name.lower() for s in module.variables]
        for var in [v for v in [copyin, copyout] if v]:
            assert var[0] in decl_symbs

        sym_dict = {s.name.lower(): s for s in module.variables}
        pragmas = [p for p in FindNodes(Pragma).visit(module.spec) if p.keyword.lower() == 'acc']

        for var in [v for v in [copyin, copyout] if v]:

            #... do nothing if var is a parameter
            if sym_dict[var[0]].type.parameter:
                continue

            #... check if var is already declared
            for p in pragmas:
                if var[0] in p.content.lower():
                    continue

            module.spec.append(Pragma(keyword='acc', content=f'declare create({var[0].lower()})'))
            if copyin:
                if var[0] in copyin:
                    self._acc_copyin.add(var[0])
            if copyout:
                if var[0] in copyout:
                    self._acc_copyout.add(var[0])

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs.get('role')
        if role == 'driver':
            self.process_driver(routine, **kwargs)

    def process_driver(self, routine, **kwargs):

        pragma_map = {}
        for pragma in [pragma for pragma in FindNodes(Pragma).visit(routine.body) if pragma.keyword == 'loki']:
            if 'update_device' in pragma.content and self._acc_copyin:
                pragma_map.update({pragma: Pragma(keyword='acc', content='update device(' + ','.join(self._acc_copyin) + ')')})
            if 'update_host' in pragma.content and self._acc_copyout:
                pragma_map.update({pragma: Pragma(keyword='acc', content='update self(' + ','.join(self._acc_copyout) + ')')})

        routine.body = Transformer(pragma_map).visit(routine.body)
