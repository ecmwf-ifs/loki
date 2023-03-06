from loki.transform.transformation import Transformation
from loki.bulk.item import GlobalVarImportItem
from loki.analyse import dataflow_analysis_attached
from loki.backend import fgen
from loki.tools.util import as_tuple
from loki.ir import VariableDeclaration, Pragma
from loki.visitors.find import FindNodes

__all__ = ['GlobalVarImportAnalysis', 'GlobalVarOffloadTransformation']

class GlobalVarImportAnalysis(Transformation):

    def transform_subroutine(self, routine, **kwargs):
        with dataflow_analysis_attached(routine):
            for item in [s for s in kwargs.get('successors', None) if isinstance(s, GlobalVarImportItem)]:
                _symbol = item.name.split('#')[-1]
                if _symbol in routine.body.uses_symbols:
                    item.trafo_data.update({'acc_copyin': as_tuple(_symbol)})
                if _symbol in routine.body.defines_symbols:
                    item.trafo_data.update({'acc_copyout': as_tuple(_symbol)})

class GlobalVarOffloadTransformation(Transformation):

    def transform_module(self, module, **kwargs):
        print(f'Got here module: {module.name}')
        item = kwargs['item']

        copy = item.trafo_data.get('acc_copy')
        copyin = item.trafo_data.get('acc_copyin')
        copyout = item.trafo_data.get('acc_copyout')

        #... check that any vars to be offloaded are declared in module
#        all_vars = copy + copyin + copyout
        decl_symbs = [s.name.lower() for d in FindNodes(VariableDeclaration).visit(module.spec) for s in d.symbols]
        for var in [copy, copyin, copyout]:
            if var:
                assert var[0].lower() in decl_symbs

#        print(f'copy: {copy}')
#        print(f'copyin: {copyin}')
#        print(f'copyout: {copyout}')
