from loki.transform.transformation import Transformation
from loki.bulk.item import GlobalVarImportItem
from loki.analyse import dataflow_analysis_attached
from loki.ir import Pragma, CallStatement, Import, Comment
from loki.visitors.find import FindNodes
from loki.visitors.transform import Transformer
from loki.expression.expr_visitors import FindInlineCalls
from loki.expression.symbols import Variable
from loki.tools.util import as_tuple
import re

__all__ = ['GlobalVarOffloadTransformation']

class GlobalVarOffloadTransformation(Transformation):
    """
    :any:`Transformation` class that facilitates insertion of offload directives
    for module variable imports. The following offload paradigms are currently supported:

    * OpenACC

    It comprises of three main components. ``process_kernel`` which collects a set of
    imported variables to offload, ``transform_module`` which adds device-side declarations
    for the imported variables to the relevant modules, and ``process_driver`` which adds
    offload instructions at the driver-layer. ``!$loki update_device`` and ``!$loki update_host``
    pragmas are needed in the ``driver`` source to insert offload and/or copy-back directives.

    Nested Fortran derived-types are not currently supported. If such an import is encountered,
    only the device-side declaration will be added to the relevant module file, and the offload
    instructions will have to manually be added afterwards.

    NB: This transformation should only be used as part of a :any:`Scheduler` traversal, and
    **must** be run in reverse, e.g.:
    scheduler.process(transformation=GlobalVarOffloadTransformation(), reverse=True)
    """

    def __init__(self):
        self._acc_copyin = set()
        self._acc_copyout = set()
        self._var_set = set()
        self._imports = {}

    def transform_module(self, module, **kwargs):
        """
        Add device-side declarations for imported variables.
        """

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
        """
        Add offload and/or copy-back directives for the imported variables.
        """

        pragma_map = {}
        _symbols = set()
        for pragma in [pragma for pragma in FindNodes(Pragma).visit(routine.body) if pragma.keyword == 'loki']:
            if 'update_device' in pragma.content:
                if self._acc_copyin:
                    pragma_map.update({pragma: Pragma(keyword='acc', content='update device(' + ','.join(self._acc_copyin) + ')')})
                    _symbols = _symbols | self._acc_copyin
                else:
                    pragma_map.update({pragma: None})
            if 'update_host' in pragma.content:
                if self._acc_copyout:
                    pragma_map.update({pragma: Pragma(keyword='acc', content='update self(' + ','.join(self._acc_copyout) + ')')})
                    _symbols = _symbols | self._acc_copyout
                else:
                    pragma_map.update({pragma: None})

        routine.body = Transformer(pragma_map).visit(routine.body)

        _import_dict = {}
        _old_imports = FindNodes(Import).visit(routine.spec)
        _imported_symbols = [s.name.lower() for i in _old_imports for s in i.symbols]
        for s in _symbols:
            if s in routine.variables:
                continue
            if s in _imported_symbols:
                continue

            if _import_dict.get(self._imports[s], None):
                _import_dict[self._imports[s]] += as_tuple(s)
            else:
                _import_dict.update({self._imports[s]: as_tuple(s)})

        _new_imports = ()
        for k, v in _import_dict.items():
            _new_imports += as_tuple(Import(k, symbols=tuple(Variable(name=s, scope=routine) for s in v)))

        import_pos = routine.spec.body.index(_old_imports[-1]) + 1
        if _new_imports:
            routine.spec.insert(import_pos, Comment(text='!.....Adding global variables to driver symbol table for offload instructions'))
        import_pos += 1
        for index, i in enumerate(_new_imports):
            routine.spec.insert(import_pos + index, i)

    def process_kernel(self, routine, **kwargs):
        """
        Collect the set of module variables to be offloaded.
        """

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
                    self._imports.update({item: _import_dict[item]})
                if item in routine.body.defines_symbols:
                    self._acc_copyout.add(item)
                    self._imports.update({item: _import_dict[item]})
