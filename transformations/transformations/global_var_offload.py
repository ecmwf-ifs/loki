import re

from loki.transform.transformation import Transformation
from loki.bulk.item import GlobalVarImportItem
from loki.analyse import dataflow_analysis_attached
from loki.ir import Pragma, CallStatement, Import, Comment
from loki.visitors.find import FindNodes
from loki.visitors.transform import Transformer
from loki.expression.expr_visitors import FindInlineCalls
from loki.expression.symbols import Variable, Array
from loki.tools.util import as_tuple
from loki.types import DerivedType, BasicType
from loki.logging import warning

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
    The functionality is illustrated in the example below.

    E.g., the following code:

..    code-block:: fortran

      module moduleB
         real :: var2
         real :: var3
      end module moduleB

      module moduleC
         real :: var4
         real :: var5
      end module moduleC

      subroutine driver()
      implicit none

      !$loki update_device
      !$acc serial
      call kernel()
      !$acc end serial
      !$loki update_host

      end subroutine driver

      subroutine kernel()
      use moduleB, only: var2,var3
      use moduleC, only: var4,var5
      implicit none
      !$acc routine seq

      var4 = var2
      var5 = var3

      end subroutine kernel

    is transformed to:

..    code-block:: fortran

      module moduleB
         real :: var2
         real :: var3
        !$acc declare create(var2)
        !$acc declare create(var3)
      end module moduleB

      module moduleC
         real :: var4
         real :: var5
        !$acc declare create(var4)
        !$acc declare create(var5)
      end module moduleC

      subroutine driver()
      implicit none

      !$acc update device( var2,var3 )
      !$acc serial
      call kernel()
      !$acc end serial
      !$acc update self( var4,var5 )

      end subroutine driver

    Nested Fortran derived-types and arrays of derived-types are not currently supported.
    If such an import is encountered, only the device-side declaration will be added to the
    relevant module file, and the offload instructions will have to manually be added afterwards.

    NB: This transformation should only be used as part of a :any:`Scheduler` traversal, and
    **must** be run in reverse, e.g.:
    scheduler.process(transformation=GlobalVarOffloadTransformation(), reverse=True)
    """

    def __init__(self):
        self._enter_data_copyin = set()
        self._enter_data_create = set()
        self._exit_data = set()
        self._acc_copyin = set()
        self._acc_copyout = set()
        self._var_set = set()
        self._modules = {}

    def transform_module(self, module, **kwargs):
        """
        Add device-side declarations for imported variables.
        """

        item = kwargs['item']

        # bail if not global variable import
        if not isinstance(item, GlobalVarImportItem):
            return

        # confirm that var to be offloaded is declared in module
        symbol = item.name.split('#')[-1].lower()
        assert symbol in [s.name.lower() for s in module.variables]

        # do nothing if var is a parameter
        if module.symbol_map[symbol].type.parameter:
            return

        # check if var is already declared
        pragmas = [p for p in FindNodes(Pragma).visit(module.spec) if p.keyword.lower() == 'acc']
        for p in pragmas:
            if re.search(fr'\b{symbol}\b', p.content.lower()):
                return

        # Update the set of variables to be offloaded
        self._var_set.add(symbol)

        # Write ACC declare pragma
        module.spec.append(Pragma(keyword='acc', content=f'declare create({symbol})'))

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs.get('role')
        if role == 'driver':
            self.process_driver(routine)
        elif role == 'kernel':
            self.process_kernel(routine, **kwargs)

    def process_driver(self, routine):
        """
        Add offload and/or copy-back directives for the imported variables.
        """

        update_device = ()
        update_host = ()

        # build offload pragmas
        if self._acc_copyin:
            update_device += as_tuple(Pragma(keyword='acc',
                                             content='update device(' + ','.join(self._acc_copyin) + ')'),)
        if self._enter_data_copyin:
            update_device += as_tuple(Pragma(keyword='acc',
                                             content='enter data copyin(' + ','.join(self._enter_data_copyin) + ')'),)
        if self._enter_data_create:
            update_device += as_tuple(Pragma(keyword='acc',
                                             content='enter data create(' + ','.join(self._enter_data_create) + ')'),)
        if self._exit_data:
            update_host += as_tuple(Pragma(keyword='acc',
                                           content='exit data copyout(' + ','.join(self._exit_data) + ')'),)
        if self._acc_copyout:
            update_host += as_tuple(Pragma(keyword='acc', content='update self(' + ','.join(self._acc_copyout) + ')'),)

        # replace Loki pragmas with acc data/update pragmas
        pragma_map = {}
        for pragma in [pragma for pragma in FindNodes(Pragma).visit(routine.body) if pragma.keyword == 'loki']:
            if 'update_device' in pragma.content:
                if update_device:
                    pragma_map.update({pragma: update_device})
                else:
                    pragma_map.update({pragma: None})

            if 'update_host' in pragma.content:
                if update_host:
                    pragma_map.update({pragma: update_host})
                else:
                    pragma_map.update({pragma: None})
        routine.body = Transformer(pragma_map).visit(routine.body)

        # build new imports to add offloaded global vars to driver symbol table
        new_import_map = {}
        for s in self._var_set:
            if s in routine.symbol_map:
                continue

            if new_import_map.get(self._modules[s], None):
                new_import_map[self._modules[s]] += as_tuple(s)
            else:
                new_import_map.update({self._modules[s]: as_tuple(s)})

        new_imports = ()
        for k, v in new_import_map.items():
            new_imports += as_tuple(Import(k, symbols=tuple(Variable(name=s, scope=routine) for s in v)))

        # add new imports to driver subroutine sepcification
        import_pos = 0
        if (old_imports := FindNodes(Import).visit(routine.spec)):
            import_pos = routine.spec.body.index(old_imports[-1]) + 1
        if new_imports:
            routine.spec.insert(import_pos, Comment(text=
                                '!.....Adding global variables to driver symbol table for offload instructions'))
        import_pos += 1
        for index, i in enumerate(new_imports):
            routine.spec.insert(import_pos + index, i)

    def process_kernel(self, routine, **kwargs):
        """
        Collect the set of module variables to be offloaded.
        """

        # filter out procedure imports
        imports = FindNodes(Import).visit(routine.spec)
        calls = [c.name for c in FindNodes(CallStatement).visit(routine.body)]

        imports = [i for i in imports if i.symbols not in calls]
        imports = [i for i in imports if i.symbols not in FindInlineCalls().visit(routine.body)]

        # build map of modules corresponding to imports
        import_mod = {s.name.lower(): i.module for i in imports for s in i.symbols}

        # separate out derived and basic types
        basic_types = [s.name.lower() for i in imports for s in i.symbols if s in kwargs['targets']
                       if isinstance(s.type.dtype, BasicType) and s.name.lower() in self._var_set]
        deriv_types = [s for i in imports for s in i.symbols if s in kwargs['targets']
                       if isinstance(s.type.dtype, DerivedType) and s.name.lower() in self._var_set]

        with dataflow_analysis_attached(routine):

            # collect symbols to add to acc update pragmas in driver layer
            for item in basic_types:
                if item in routine.body.uses_symbols:
                    self._acc_copyin.add(item)
                    self._modules.update({item: import_mod[item]})
                if item in routine.body.defines_symbols:
                    self._acc_copyout.add(item)
                    self._modules.update({item: import_mod[item]})

            # collect symbols to add to acc enter/exit data pragmas in driver layer
            for item in deriv_types:
                item_vars = item.type.dtype.typedef.variables
                if isinstance(item, Array):
                    # pylint: disable-next=line-too-long
                    warning(f'[Loki::GlobalVarOffload] Arrays of derived-types must be offloaded manually - {item} in {routine}')
                    self._var_set.remove(item.name.lower())
                elif any(isinstance(v.type.dtype, DerivedType) for v in item_vars):
                    # pylint: disable-next=line-too-long
                    warning(f'[Loki::GlobalVarOffload] Nested derived-types must be offloaded manually - {item} in {routine}')
                    self._var_set.remove(item.name.lower())
                else:
                    for var in item_vars:
                        symbol = f'{item.name.lower()}%{var.name.lower()}'

                        if symbol in routine.body.uses_symbols:
                            self._enter_data_copyin.add(symbol)
                            self._modules.update({item: import_mod[item.name.lower()]})

                        if symbol in routine.body.defines_symbols:
                            self._exit_data.add(symbol)

                            if not symbol in self._enter_data_copyin and var.type.allocatable:
                                self._enter_data_create.add(symbol)
                            self._modules.update({item: import_mod[item.name.lower()]})
