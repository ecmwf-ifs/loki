# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from itertools import chain
from loki import (
    pragma_regions_attached, PragmaRegion, Transformation, FindNodes,
    CallStatement, Pragma, Scalar, Array, as_tuple, Transformer, warning, BasicType,
    SubroutineItem, GlobalVarImportItem, dataflow_analysis_attached, Import,
    Comment, Variable, flatten, DerivedType, get_pragma_parameters, CaseInsensitiveDict
)


__all__ = [
    'DataOffloadTransformation', 'GlobalVariableAnalysis',
    'NewGlobalVarOffloadTransformation', 'GlobalVarOffloadTransformation'
]


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
    assume_deviceptr : bool
        Mark all offloaded arrays as true device-pointers if data offload
        is being managed outside of structured OpenACC data regions.
    """

    def __init__(self, **kwargs):
        # We need to record if we actually added any, so
        # that down-stream processing can use that info
        self.has_data_regions = False
        self.remove_openmp = kwargs.get('remove_openmp', False)
        self.assume_deviceptr = kwargs.get('assume_deviceptr', False)

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
                if self.assume_deviceptr:
                    offload_args = inargs + outargs + inoutargs
                    if offload_args:
                        deviceptr = f' deviceptr({", ".join(offload_args)})'
                    else:
                        deviceptr = ''
                    pragma = Pragma(keyword='acc', content=f'data{deviceptr}')
                else:
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


class GlobalVariableAnalysis(Transformation):
    """
    Transformation pass to analyse the declaration and use of (global) module variables.

    This analysis is a requirement before applying :any:`GlobalVarOffloadTransformation`.

    Collect data in :any:`Item.trafo_data` for :any:`SubroutineItem` and
    :any:`GlobalVarImportItem` items and store analysis results under the
    provided :data:`key` (default: ``'GlobalVariableAnalysis'``) in the
    items' ``trafo_data``.

    For procedures, use the the Loki dataflow analysis functionality to compile
    a list of used and/or defined variables (i.e., read and/or written).
    Store these under the keys ``'uses_symbols'`` and ``'defines_symbols'``, respectively.

    For modules/:any:`GlobalVarImportItem`, store the list of variables declared in the
    module under the key ``'declares'`` and out of this the subset of variables that
    need offloading to device under the key ``'offload'``.

    Note that in every case, the full variable symbols are stored to allow access to
    type information in transformations using the analysis data.

    The generated trafo_data has the following schema::

        GlobalVarImportItem: {
            'declares': set(Variable, Variable, ...),
            'offload': set(Variable, ...)
        }

        SubroutineItem: {
            'uses_symbols': set( (Variable, '<module_name>'), (Variable, '<module_name>'), ...),
            'defines_symbols': set((Variable, '<module_name>'), (Variable, '<module_name>'), ...)
        }

    Parameters
    ----------
    key : str, optional
        Specify a different identifier under which trafo_data is stored
    """

    _key = 'GlobalVariableAnalysis'
    """Default identifier for trafo_data entry"""

    reverse_traversal = True
    """Traversal from the leaves upwards, i.e., modules with global variables are processed first,
    then kernels using them before the driver."""

    item_filter = (SubroutineItem, GlobalVarImportItem)
    """Process procedures and modules with global variable declarations."""

    def __init__(self, key=None):
        if key:
            self._key = key

    def transform_module(self, module, **kwargs):
        if 'item' not in kwargs:
            raise RuntimeError('Cannot apply GlobalVariableAnalysis without item to store analysis data')

        item = kwargs['item']

        if not isinstance(item, GlobalVarImportItem):
            raise RuntimeError('Module transformation applied for non-module item')

        # Gather all module variables and filter out parameters
        variables = {var for var in module.variables if not var.type.parameter}

        # Initialize and store trafo data
        item.trafo_data[self._key] = {
            'declares': variables,
            'offload': set()
        }

    def transform_subroutine(self, routine, **kwargs):
        if 'item' not in kwargs:
            raise RuntimeError('Cannot apply GlobalVariableAnalysis without item to store analysis data')
        if 'successors' not in kwargs:
            raise RuntimeError('Cannot apply GlobalVariableAnalysis without successors to store offload analysis data')

        item = kwargs['item']
        successors = kwargs['successors']

        # Gather all symbols imported in this routine or parent scopes
        import_map = CaseInsensitiveDict()
        scope = routine
        while scope:
            import_map.update(scope.import_map)
            scope = scope.parent

        with dataflow_analysis_attached(routine):
            # Gather read and written symbols that have been imported
            uses_imported_symbols = {
                var for var in routine.body.uses_symbols
                if var.name in import_map or (var.parent and var.parents[0].name in import_map)
            }
            defines_imported_symbols = {
                var for var in routine.body.defines_symbols
                if var.name in import_map or (var.parent and var.parents[0].name in import_map)
            }

            # Filter out type and procedure imports by restricting to Scalar and Array symbols
            uses_imported_symbols = {var for var in uses_imported_symbols if isinstance(var, (Scalar, Array))}
            defines_imported_symbols = {var for var in defines_imported_symbols if isinstance(var, (Scalar, Array))}

            # Discard parameters (which are read-only by definition)
            uses_imported_symbols = {var for var in uses_imported_symbols if not var.type.parameter}

            def _map_var_to_module(var):
                if var.parent:
                    module = var.parents[0].type.module
                    module_var = module.variable_map[var.parents[0].name]
                    dimensions = getattr(module_var, 'dimensions', None)
                    for child in chain(var.parents[1:], (var,)):
                        module_var = child.clone(
                            name=f'{module_var.name}%{child.name}',
                            parent=module_var,
                            scope=module_var.scope
                        )
                    return (module_var.clone(dimensions=dimensions), module.name.lower())
                module = var.type.module
                return (module.variable_map[var.name], module.name.lower())

            # Store symbol lists in trafo data
            item.trafo_data[self._key] = {}
            item.trafo_data[self._key]['uses_symbols'] = {
                _map_var_to_module(var) for var in uses_imported_symbols
            }
            item.trafo_data[self._key]['defines_symbols'] = {
                _map_var_to_module(var) for var in defines_imported_symbols
            }

        # Propagate offload requirement to the items of the global variables
        successors_map = CaseInsensitiveDict(
            (item.name, item) for item in successors if isinstance(item, GlobalVarImportItem)
        )
        for var, module in chain(
            item.trafo_data[self._key]['uses_symbols'],
            item.trafo_data[self._key]['defines_symbols']
        ):
            if var.parent:
                successor = successors_map[f'{module}#{var.parents[0].name}']
            else:
                successor = successors_map[f'{module}#{var.name}']
            successor.trafo_data[self._key]['offload'].add(var)

        # Amend analysis data with data from successors
        # Note: This is a temporary workaround for the incomplete list of successor items
        # provided by the current scheduler implementation
        for successor in successors:
            if isinstance(successor, SubroutineItem):
                item.trafo_data[self._key]['uses_symbols'] |= successor.trafo_data[self._key]['uses_symbols']
                item.trafo_data[self._key]['defines_symbols'] |= successor.trafo_data[self._key]['defines_symbols']


class NewGlobalVarOffloadTransformation(Transformation):
    """
    Transformation to insert offload directives for module variables used in device routines

    Currently, only OpenACC data offloading is supported.

    This requires a prior analysis pass with :any:`GlobalVariableAnalysis` to collect
    the relevant global variable use information.

    The offload directives are inserted by replacing ``!$loki update_device`` and
    ``!$loki update_host`` pragmas in the driver's source code. Importantly, no offload
    directives are added if these pragmas have not been added to the original source code!

    For global variables, the device-side declarations are added in :meth:`transform_module`.
    For driver procedures, the data offload and pull-back directives are added in
    the utility method :meth:`process_driver`, which is invoked by :meth:`transform_subroutine`.

    For example, the following code:

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
    """

    # Include module variable imports in the underlying graph
    # connectivity for traversal with the Scheduler
    item_filter = (SubroutineItem, GlobalVarImportItem)

    def __init__(self, key=None):
        self._key = key or GlobalVariableAnalysis._key

    def transform_module(self, module, **kwargs):
        """
        Add device-side declarations for imported variables
        """
        if 'item' not in kwargs:
            raise RuntimeError('Cannot apply GlobalVarOffloadTransformation without trafo_data in item')

        item = kwargs['item']

        if not isinstance(item, GlobalVarImportItem):
            raise RuntimeError('Module transformation applied for non-module item')

        # Check for already declared offloads
        acc_pragmas = [pragma for pragma in FindNodes(Pragma).visit(module.spec) if pragma.keyword.lower() == 'acc']
        acc_pragma_parameters = get_pragma_parameters(acc_pragmas, starts_with='declare', only_loki_pragmas=False)
        declared_variables = set(flatten([
            v.replace(' ','').lower().split()
            for v in as_tuple(acc_pragma_parameters.get('create'))
        ]))

        # Build list of symbols to be offloaded
        offload_variables = {
            var.parents[0] if var.parent else var
            for var in item.trafo_data[self._key].get('offload', ())
        }

        if (invalid_vars := offload_variables - set(module.variables)):
            raise RuntimeError(f'Invalid variables in offload analysis: {", ".join(v.name for v in invalid_vars)}')

        # Add ACC declare pragma for offload variables that are not yet declared
        offload_variables = offload_variables - declared_variables
        if offload_variables:
            module.spec.append(
                Pragma(keyword='acc', content=f'declare create({",".join(v.name for v in offload_variables)})')
            )

    def transform_subroutine(self, routine, **kwargs):
        """
        Add data offload and pull-back directives to the driver
        """
        role = kwargs.get('role')
        successors = kwargs.get('successors', ())

        if role == 'driver':
            self.process_driver(routine, successors)

    def process_driver(self, routine, successors):
        """
        Add data offload and pullback directives

        List of variables that requires offloading is obtained from the analysis data
        stored for each successor in :data:`successors`.
        """
        # Empty lists for update directives
        update_device = ()
        update_host = ()

        # Combine analysis data across successor items
        defines_symbols = set()
        uses_symbols = set()
        for item in successors:
            defines_symbols |= item.trafo_data.get(self._key, {}).get('defines_symbols', set())
            uses_symbols |= item.trafo_data.get(self._key, {}).get('uses_symbols', set())

        # Filter out arrays of derived types and nested derived types
        # For these, automatic offloading is currently not supported
        exclude_symbols = set()
        for var_, module in chain(defines_symbols, uses_symbols):
            var = var_.parents[0] if var_.parent else var_
            if not isinstance(var.type.dtype, DerivedType):
                continue
            if isinstance(var, Array):
                exclude_symbols.add(var)
                warning((
                    '[Loki::GlobalVarOffloadTransformation] '
                    f'Automatic offloading of derived type arrays not implemented: {var} in {routine.name}'
                ))
            if any(isinstance(v.type.dtype, DerivedType) for v in var.type.dtype.typedef.variables):
                exclude_symbols.add(var)
                warning((
                    '[Loki::GlobalVarOffloadTransformation] '
                    f'Automatic offloading of nested derived types not implemented: {var} in {routine.name}'
                ))

        uses_symbols = {
            (var, module) for var, module in uses_symbols
            if var not in exclude_symbols and not (var.parent and var.parents[0] in exclude_symbols)
        }
        defines_symbols = {
            (var, module) for var, module in defines_symbols
            if var not in exclude_symbols and not (var.parent and var.parents[0] in exclude_symbols)
        }

        # All variables that are used in a kernel need a host-to-device transfer
        if uses_symbols:
            update_variables = {
                v for v, _ in uses_symbols
                if not (v.parent or isinstance(v.type.dtype, DerivedType))
            }
            copyin_variables = {v for v, _ in uses_symbols if v.parent}
            if update_variables:
                update_device += (
                    Pragma(keyword='acc', content=f'update device({",".join(v.name for v in update_variables)})'),
                )
            if copyin_variables:
                update_device += (
                    Pragma(keyword='acc', content=f'enter data copyin({",".join(v.name for v in copyin_variables)})'),
                )

        # All variables that are written in a kernel need a device-to-host transfer
        if defines_symbols:
            update_variables = {v for v, _ in defines_symbols if not v.parent}
            copyout_variables = {v for v, _ in defines_symbols if v.parent}
            create_variables = {
                v for v in copyout_variables
                if v not in uses_symbols and v.type.allocatable
            }
            if update_variables:
                update_host += (
                    Pragma(keyword='acc', content=f'update self({",".join(v.name for v in update_variables)})'),
                )
            if copyout_variables:
                update_host += (
                    Pragma(keyword='acc', content=f'exit data copyout({",".join(v.name for v in copyout_variables)})'),
                )
            if create_variables:
                update_device += (
                    Pragma(keyword='acc', content=f'enter data create({",".join(v.name for v in create_variables)})'),
                )

        # Replace Loki pragmas with acc data/update pragmas
        pragma_map = {}
        for pragma in FindNodes(Pragma).visit(routine.body):
            if pragma.keyword == 'loki':
                if 'update_device' in pragma.content:
                    pragma_map[pragma] = update_device or None
                if 'update_host' in pragma.content:
                    pragma_map[pragma] = update_host or None

        routine.body = Transformer(pragma_map).visit(routine.body)

        # Add imports for offload variables
        offload_map = defaultdict(set)
        for var, module in chain(uses_symbols, defines_symbols):
            offload_map[module].add(var.parents[0] if var.parent else var)

        import_map = CaseInsensitiveDict()
        scope = routine
        while scope:
            import_map.update(scope.import_map)
            scope = scope.parent

        missing_imports_map = defaultdict(set)
        for module, variables in offload_map.items():
            missing_imports_map[module] |= {var for var in variables if var.name not in import_map}

        if missing_imports_map:
            routine.spec.prepend(Comment(text=(
                '![Loki::GlobalVarOffloadTransformation] ---------------------------------------'
            )))
            for module, variables in missing_imports_map.items():
                symbols = tuple(var.rescope(scope=routine) for var in variables)
                routine.spec.prepend(Import(module=module, symbols=symbols))

            routine.spec.prepend(Comment(text=(
                '![Loki::GlobalVarOffloadTransformation] '
                '-------- Added global variable imports for offload directives -----------'
            )))


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

    # Traverse call tree in reverse when using Scheduler
    reverse_traversal = True

    # Include module variable imports in the underlying graph
    # connectivity for traversal with the Scheduler
    item_filter = (SubroutineItem, GlobalVarImportItem)

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

        item.trafo_data.setdefault(self._key, {})
        item.trafo_data[self._key].setdefault('var_set', set())

        # do nothing if var is a parameter
        if module.symbol_map[symbol].type.parameter:
            return

        # check if var is already declared
        pragmas = [p for p in FindNodes(Pragma).visit(module.spec) if p.keyword.lower() == 'acc']
        acc_pragma_parameters = get_pragma_parameters(pragmas, starts_with='declare', only_loki_pragmas=False)
        if acc_pragma_parameters:
            if symbol in flatten([v.replace(' ','').lower().split(',')
                                 for v in as_tuple(acc_pragma_parameters['create'])]):
                return

        # Update the set of variables to be offloaded
        item.trafo_data[self._key]['var_set'].add(symbol)

        # Write ACC declare pragma
        module.spec.append(Pragma(keyword='acc', content=f'declare create({symbol})'))

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs.get('role')
        item = kwargs['item']

        item.trafo_data.setdefault(self._key, {})

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
            self.process_kernel(routine, successors, item)

    def process_driver(self, routine, successors):
        """
        Add offload and/or copy-back directives for the imported variables.
        """

        update_device = ()
        update_host = ()

        # build offload pragmas
        key_directive_map = {
            'acc_copyin': 'update device',
            'enter_data_copyin': 'enter data copyin',
            'enter_data_create': 'enter data create',
        }
        for key, directive in key_directive_map.items():
            variables = set.union(*[s.trafo_data.get(self._key, {}).get(key) for s in successors], set())
            if variables:
                update_device += (Pragma(keyword='acc', content=f'{directive}({",".join(variables)})'),)

        key_directive_map = {
            'exit_data': 'exit data copyout',
            'acc_copyout': 'update self'
        }
        for key, directive in key_directive_map.items():
            variables = set.union(*[s.trafo_data.get(self._key, {}).get(key) for s in successors], set())
            if variables:
                update_host += (Pragma(keyword='acc', content=f'{directive}({",".join(variables)})'),)

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
        _var_set = set.union(
            *[s.trafo_data.get(self._key, {}).get('var_set', set()) for s in successors],
            set()
        )
        # build map of module imports corresponding to offloaded symbols
        _modules = {
            k: v
            for s in successors
            for k, v in s.trafo_data.get(self._key, {}).get('modules', {}).items()
        }

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

    def process_kernel(self, routine, successors, item):
        """
        Collect the set of module variables to be offloaded.
        """

        # build map of modules corresponding to imported symbols
        import_mod = CaseInsensitiveDict((s.name, i.module) for i in routine.imports for s in i.symbols)

        #build set of offloaded symbols
        item.trafo_data[self._key]['var_set'] = set.union(
            *[s.trafo_data.get(self._key, {}).get('var_set', set()) for s in successors],
            set()
        )

        #build map of module imports corresponding to offloaded symbols
        item.trafo_data[self._key]['modules'].update({
            k: v
            for s in successors
            for k, v in s.trafo_data.get(self._key, {}).get('modules', {}).items()
        })

        # separate out derived and basic types
        imported_vars = [var for var in routine.imported_symbols if var in item.trafo_data[self._key]['var_set']]
        basic_types = [var.name.lower() for var in imported_vars if isinstance(var.type.dtype, BasicType)]
        deriv_types = [var              for var in imported_vars if isinstance(var.type.dtype, DerivedType)]

        # accumulate contents of acc directives
        keys = ('enter_data_copyin', 'enter_data_create', 'exit_data', 'acc_copyin', 'acc_copyout')
        for key in keys:
            item.trafo_data[self._key][key] = set.union(
                *[s.trafo_data.get(self._key, {}).get(key, set()) for s in successors], set()
            )

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
