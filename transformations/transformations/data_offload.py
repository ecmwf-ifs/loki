# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import reduce
import operator

from loki import (
    pragma_regions_attached, PragmaRegion, Transformation, FindNodes,
    CallStatement, Pragma, Array, as_tuple, Transformer, warning, BasicType,
    GlobalVarImportItem, SubroutineItem, dataflow_analysis_attached, Import,
    Comment, FindInlineCalls, Variable, flatten, DerivedType, get_pragma_parameters,
    CaseInsensitiveDict
)


__all__ = ['DataOffloadTransformation', 'GlobalVarOffloadTransformation']


class DataOffloadTransformation(Transformation):
    """
    Utility transformation to insert data offload regions for GPU devices
    based on marked ``!$loki data`` regions. In the first instance this
    will insert OpenACC data offload regions, but can be extended to other
    offload region semantics (eg. OpenMP-5) in the future.

    Parameters
    ----------
    remove_openmp : bool
        Remove any existing OpenMP pragmas inside the marked region.
    """

    def __init__(self, **kwargs):
        # We need to record if we actually added any, so
        # that down-stream processing can use that info
        self.has_data_regions = False
        self.remove_openmp = kwargs.get('remove_openmp', False)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply the transformation to a `Subroutine` object.

        Parameters
        ----------
        routine : `Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the `routine` in the scheduler call tree.
            This transformation will only apply at the ``'driver'`` level.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """
        role = kwargs.get('role')
        targets = as_tuple(kwargs.get('targets', None))

        if targets:
            targets = tuple(t.lower() for t in targets)

        if role == 'driver':
            self.remove_openmp_pragmas(routine, targets)
            self.insert_data_offload_pragmas(routine, targets)

    @staticmethod
    def _is_active_loki_data_region(region, targets):
        """
        Utility to decide if a ``PragmaRegion`` is of type ``!$loki data``
        and has active target routines.
        """
        if region.pragma.keyword.lower() != 'loki':
            return False
        if 'data' not in region.pragma.content.lower():
            return False

        # Find all targeted kernel calls
        calls = FindNodes(CallStatement).visit(region)
        calls = [c for c in calls if str(c.name).lower() in targets]
        if len(calls) == 0:
            return False

        return True

    def insert_data_offload_pragmas(self, routine, targets):
        """
        Find ``!$loki data`` pragma regions and create according
        ``!$acc udpdate`` regions.

        Parameters
        ----------
        routine : `Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """
        pragma_map = {}
        with pragma_regions_attached(routine):
            for region in FindNodes(PragmaRegion).visit(routine.body):
                # Only work on active `!$loki data` regions
                if not self._is_active_loki_data_region(region, targets):
                    continue

                # Find all targeted kernel calls
                calls = FindNodes(CallStatement).visit(region)
                calls = [c for c in calls if str(c.name).lower() in targets]

                # Collect the three types of device data accesses from calls
                inargs = ()
                inoutargs = ()
                outargs = ()

                for call in calls:
                    if call.routine is BasicType.DEFERRED:
                        warning(f'[Loki] Data offload: Routine {routine.name} has not been enriched with ' +
                                f'in {str(call.name).lower()}')

                        continue

                    for param, arg in call.arg_iter():
                        if isinstance(param, Array) and param.type.intent.lower() == 'in':
                            inargs += (str(arg.name).lower(),)
                        if isinstance(param, Array) and param.type.intent.lower() == 'inout':
                            inoutargs += (str(arg.name).lower(),)
                        if isinstance(param, Array) and param.type.intent.lower() == 'out':
                            outargs += (str(arg.name).lower(),)

                # Sanitize data access categories to avoid double-counting variables
                inoutargs += tuple(v for v in inargs if v in outargs)
                inargs = tuple(v for v in inargs if v not in inoutargs)
                outargs = tuple(v for v in outargs if v not in inoutargs)

                # Filter for duplicates
                inargs = tuple(dict.fromkeys(inargs))
                outargs = tuple(dict.fromkeys(outargs))
                inoutargs = tuple(dict.fromkeys(inoutargs))

                # Now geenerate the pre- and post pragmas (OpenACC)
                copyin = f'copyin({", ".join(inargs)})' if inargs else ''
                copy = f'copy({", ".join(inoutargs)})' if inoutargs else ''
                copyout = f'copyout({", ".join(outargs)})' if outargs else ''
                pragma = Pragma(keyword='acc', content=f'data {copyin} {copy} {copyout}')
                pragma_post = Pragma(keyword='acc', content='end data')
                pragma_map[region.pragma] = pragma
                pragma_map[region.pragma_post] = pragma_post

                # Record that we actually created a new region
                if not self.has_data_regions:
                    self.has_data_regions = True

        routine.body = Transformer(pragma_map).visit(routine.body)

    def remove_openmp_pragmas(self, routine, targets):
        """
        Remove any existing OpenMP pragmas in the offload regions that
        will have been intended for OpenMP threading rather than
        offload.

        Parameters
        ----------
        routine : `Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """
        pragma_map = {}
        with pragma_regions_attached(routine):
            for region in FindNodes(PragmaRegion).visit(routine.body):
                # Only work on active `!$loki data` regions
                if not self._is_active_loki_data_region(region, targets):
                    continue

                for p in FindNodes(Pragma).visit(routine.body):
                    if p.keyword.lower() == 'omp':
                        pragma_map[p] = None
                for r in FindNodes(PragmaRegion).visit(region):
                    if r.pragma.keyword.lower() == 'omp':
                        pragma_map[r.pragma] = None
                        pragma_map[r.pragma_post] = None

        routine.body = Transformer(pragma_map).visit(routine.body)


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

    .. code-block:: fortran

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

    .. code-block:: fortran

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

    Parameters
    ----------
    key : str, optional
        Overwrite the key that is used to store analysis results in ``trafo_data``.
    """

    _key = 'GlobalVarOffloadTransformation'

    def __init__(self, key=None):
        if key:
            self._key = key

    def transform_module(self, module, **kwargs):
        """
        Add device-side declarations for imported variables.
        """

        item = kwargs['item']

        # bail if not global variable import
        if not isinstance(item, GlobalVarImportItem):
            return

        # confirm that var to be offloaded is declared in module
        symbol = item.local_name
        assert symbol in [s.name.lower() for s in module.variables]

        if not item.trafo_data.get(self._key, None):
            item.trafo_data[self._key] = {'var_set': set()}

        # do nothing if var is a parameter
        if module.symbol_map[symbol].type.parameter:
            return

        # check if var is already declared
        pragmas = [p for p in FindNodes(Pragma).visit(module.spec) if p.keyword.lower() == 'acc']
        acc_pragma_parameters = get_pragma_parameters(pragmas, starts_with='declare', only_loki_pragmas=False)
        if acc_pragma_parameters:
            if symbol in flatten([v.replace(' ','').lower().split(',') for v in acc_pragma_parameters['create']]):
                return

        # Update the set of variables to be offloaded
        item.trafo_data[self._key]['var_set'].add(symbol)

        # Write ACC declare pragma
        module.spec.append(Pragma(keyword='acc', content=f'declare create({symbol})'))

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs.get('role')
        item = kwargs['item']

        # Bail if this routine is not part of a scheduler traversal
        if item and not item.local_name == routine.name.lower():
            return

        if not item.trafo_data.get(self._key, None):
            item.trafo_data[self._key] = {}

        # Initialize sets/maps to store analysis
        item.trafo_data[self._key]['modules'] = {}
        item.trafo_data[self._key]['enter_data_copyin'] = set()
        item.trafo_data[self._key]['enter_data_create'] = set()
        item.trafo_data[self._key]['exit_data'] = set()
        item.trafo_data[self._key]['acc_copyin'] = set()
        item.trafo_data[self._key]['acc_copyout'] = set()

        successors = kwargs.get('successors', ())
        if role == 'driver':
            self.process_driver(routine, successors)
        if role == 'kernel':
            self.process_kernel(routine, successors, item, kwargs['targets'])

    def process_driver(self, routine, successors):
        """
        Add offload and/or copy-back directives for the imported variables.
        """

        update_device = ()
        update_host = ()

        # build offload pragmas
        _acc_copyin = reduce(operator.or_, [s.trafo_data[self._key]['acc_copyin']
                             for s in successors if isinstance(s, SubroutineItem)], set())
        if _acc_copyin:
            update_device += as_tuple(Pragma(keyword='acc',
                                             content='update device(' + ','.join(_acc_copyin) + ')'),)
        _enter_data_copyin = reduce(operator.or_, [s.trafo_data[self._key]['enter_data_copyin']
                             for s in successors if isinstance(s, SubroutineItem)], set())
        if _enter_data_copyin:
            update_device += as_tuple(Pragma(keyword='acc',
                                             content='enter data copyin(' + ','.join(_enter_data_copyin) + ')'),)
        _enter_data_create = reduce(operator.or_, [s.trafo_data[self._key]['enter_data_create']
                             for s in successors if isinstance(s, SubroutineItem)], set())
        if _enter_data_create:
            update_device += as_tuple(Pragma(keyword='acc',
                                             content='enter data create(' + ','.join(_enter_data_create) + ')'),)
        _exit_data = reduce(operator.or_, [s.trafo_data[self._key]['exit_data']
                            for s in successors if isinstance(s, SubroutineItem)], set())
        if _exit_data:
            update_host += as_tuple(Pragma(keyword='acc',
                                           content='exit data copyout(' + ','.join(_exit_data) + ')'),)
        _acc_copyout = reduce(operator.or_, [s.trafo_data[self._key]['acc_copyout']
                            for s in successors if isinstance(s, SubroutineItem)], set())
        if _acc_copyout:
            update_host += as_tuple(Pragma(keyword='acc', content='update self(' + ','.join(_acc_copyout) + ')'),)

        # replace Loki pragmas with acc data/update pragmas
        pragma_map = {}
        for pragma in FindNodes(Pragma).visit(routine.body):
            if pragma.keyword == 'loki':
                if 'update_device' in pragma.content:
                    if update_device:
                        pragma_map[pragma] = update_device
                    else:
                        pragma_map[pragma] = None

                if 'update_host' in pragma.content:
                    if update_host:
                        pragma_map[pragma] = update_host
                    else:
                        pragma_map[pragma] = None

        routine.body = Transformer(pragma_map).visit(routine.body)

        # build set of symbols to be offloaded
        _var_set = reduce(operator.or_, [s.trafo_data[self._key]['var_set']
                          for s in successors], set())
        #build map of module imports corresponding to offloaded symbols
        _modules = {}
        _modules.update({k: v
                         for s in successors if isinstance(s, SubroutineItem)
                         for k, v in s.trafo_data[self._key]['modules'].items()})

        # build new imports to add offloaded global vars to driver symbol table
        new_import_map = {}
        for s in _var_set:
            if s in routine.symbol_map:
                continue

            if new_import_map.get(_modules[s], None):
                new_import_map[_modules[s]] += as_tuple(s)
            else:
                new_import_map.update({_modules[s]: as_tuple(s)})

        new_imports = ()
        for k, v in new_import_map.items():
            new_imports += as_tuple(Import(k, symbols=tuple(Variable(name=s, scope=routine) for s in v)))

        # add new imports to driver subroutine sepcification
        import_pos = 0
        if (old_imports := FindNodes(Import).visit(routine.spec)):
            import_pos = routine.spec.body.index(old_imports[-1]) + 1
        if new_imports:
            routine.spec.insert(import_pos, Comment(text=
               '![Loki::GlobalVarOffload].....Adding global variables to driver symbol table for offload instructions'))
            import_pos += 1
            routine.spec.insert(import_pos, new_imports)

    def process_kernel(self, routine, successors, item, targets):
        """
        Collect the set of module variables to be offloaded.
        """

        # build map of modules corresponding to imported symbols
        import_mod = CaseInsensitiveDict((s.name, i.module) for i in routine.imports for s in i.symbols)

        #build set of offloaded symbols
        item.trafo_data[self._key]['var_set'] = reduce(operator.or_,
                                                       [s.trafo_data[self._key]['var_set'] for s in successors], set())

        #build map of module imports corresponding to offloaded symbols
        item.trafo_data[self._key]['modules'].update({k: v
                                                     for s in successors if isinstance(s, SubroutineItem)
                                                     for k, v in s.trafo_data[self._key]['modules'].items()})

        # separate out derived and basic types
        imported_vars = [var for var in routine.imported_symbols if var in item.trafo_data[self._key]['var_set']]
        basic_types = [var.name.lower() for var in imported_vars if isinstance(var.type.dtype, BasicType)]
        deriv_types = [var              for var in imported_vars if isinstance(var.type.dtype, DerivedType)]

        # accumulate contents of acc directives
        item.trafo_data[self._key]['enter_data_copyin'] = reduce(operator.or_,
                                                       [s.trafo_data[self._key]['enter_data_copyin']
                                                       for s in successors if isinstance(s, SubroutineItem)], set())
        item.trafo_data[self._key]['enter_data_create'] = reduce(operator.or_,
                                                       [s.trafo_data[self._key]['enter_data_create']
                                                       for s in successors if isinstance(s, SubroutineItem)], set())
        item.trafo_data[self._key]['exit_data'] = reduce(operator.or_,
                                                       [s.trafo_data[self._key]['exit_data']
                                                       for s in successors if isinstance(s, SubroutineItem)], set())
        item.trafo_data[self._key]['acc_copyin'] = reduce(operator.or_,
                                                          [s.trafo_data[self._key]['acc_copyin']
                                                          for s in successors if isinstance(s, SubroutineItem)], set())
        item.trafo_data[self._key]['acc_copyout'] = reduce(operator.or_,
                                                           [s.trafo_data[self._key]['acc_copyout']
                                                           for s in successors if isinstance(s, SubroutineItem)], set())

        with dataflow_analysis_attached(routine):

            # collect symbols to add to acc update pragmas in driver layer
            for basic in basic_types:
                if basic in routine.body.uses_symbols:
                    item.trafo_data[self._key]['acc_copyin'].add(basic)
                    item.trafo_data[self._key]['modules'].update({basic: import_mod[basic]})
                if basic in routine.body.defines_symbols:
                    item.trafo_data[self._key]['acc_copyout'].add(basic)
                    item.trafo_data[self._key]['modules'].update({basic: import_mod[basic]})

            # collect symbols to add to acc enter/exit data pragmas in driver layer
            for deriv in deriv_types:
                deriv_vars = deriv.type.dtype.typedef.variables
                if isinstance(deriv, Array):
                    # pylint: disable-next=line-too-long
                    warning(f'[Loki::GlobalVarOffload] Arrays of derived-types must be offloaded manually - {deriv} in {routine}')
                    item.trafo_data[self._key]['var_set'].remove(deriv.name.lower())
                elif any(isinstance(v.type.dtype, DerivedType) for v in deriv_vars):
                    # pylint: disable-next=line-too-long
                    warning(f'[Loki::GlobalVarOffload] Nested derived-types must be offloaded manually - {deriv} in {routine}')
                    item.trafo_data[self._key]['var_set'].remove(deriv.name.lower())
                else:
                    for var in deriv_vars:
                        symbol = f'{deriv.name.lower()}%{var.name.lower()}'

                        if symbol in routine.body.uses_symbols:
                            item.trafo_data[self._key]['enter_data_copyin'].add(symbol)
                            item.trafo_data[self._key]['modules'].update({deriv: import_mod[deriv.name.lower()]})

                        if symbol in routine.body.defines_symbols:
                            item.trafo_data[self._key]['exit_data'].add(symbol)

                            if not symbol in item.trafo_data[self._key]['enter_data_copyin'] and var.type.allocatable:
                                item.trafo_data[self._key]['enter_data_create'].add(symbol)
                                item.trafo_data[self._key]['modules'].update({deriv: import_mod[deriv.name.lower()]})
