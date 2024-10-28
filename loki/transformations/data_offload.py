# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from itertools import chain

from loki.analyse import dataflow_analysis_attached
from loki.batch import Transformation, ProcedureItem, ModuleItem
from loki.expression import Scalar, Array
from loki.ir import (
    FindNodes, PragmaRegion, CallStatement, Pragma, Import, Comment,
    Transformer, pragma_regions_attached, get_pragma_parameters,
    FindInlineCalls, SubstituteExpressions
)
from loki.logging import warning
from loki.tools import as_tuple, flatten, CaseInsensitiveDict, CaseInsensitiveDefaultDict
from loki.types import BasicType, DerivedType


__all__ = [
    'DataOffloadTransformation', 'GlobalVariableAnalysis',
    'GlobalVarOffloadTransformation', 'GlobalVarHoistTransformation'
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
                        warning(f'[Loki] Data offload: Routine {routine.name} has not been enriched ' +
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
                pragma_map[region.pragma] = (region.pragma.clone(), pragma)
                pragma_map[region.pragma_post] = (region.pragma_post.clone(), pragma_post)

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

    Collect data in :any:`Item.trafo_data` for :any:`ProcedureItem` and
    :any:`ModuleItem` items and store analysis results under the
    provided :data:`key` (default: ``'GlobalVariableAnalysis'``) in the
    items' ``trafo_data``.

    For procedures, use the the Loki dataflow analysis functionality to compile
    a list of used and/or defined variables (i.e., read and/or written).
    Store these under the keys ``'uses_symbols'`` and ``'defines_symbols'``,
    respectively.

    For modules/:any:`ModuleItem`, store the list of variables declared in the
    module under the key ``'declares'`` and out of this the subset of variables that
    need offloading to device under the key ``'offload'``.

    Note that in every case, the full variable symbols are stored to allow access to
    type information in transformations using the analysis data.

    The generated trafo_data has the following schema::

        ModuleItem: {
            'declares': set(Variable, Variable, ...),
            'offload': set(Variable, ...)
        }

        ProcedureItem: {
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

    item_filter = (ProcedureItem, ModuleItem)
    """Process procedures and modules with global variable declarations."""

    def __init__(self, key=None):
        if key:
            self._key = key

    def transform_module(self, module, **kwargs):
        if 'item' not in kwargs:
            raise RuntimeError('Cannot apply GlobalVariableAnalysis without item to store analysis data')

        item = kwargs['item']

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
            uses_imported_symbols |= {
                var for var in routine.spec.uses_symbols
                if var.name in import_map or (var.parent and var.parents[0].name in import_map)
            }
            defines_imported_symbols = {
                var for var in routine.body.defines_symbols
                if var.name in import_map or (var.parent and var.parents[0].name in import_map)
            }

            # Filter out type and procedure imports by restricting to Scalar and Array symbols
            uses_imported_symbols = {var for var in uses_imported_symbols if isinstance(var, (Scalar, Array))}
            defines_imported_symbols = {var for var in defines_imported_symbols if isinstance(var, (Scalar, Array))}

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

        # Amend analysis data with data from successors
        # Note: This is a temporary workaround for the incomplete list of successor items
        # provided by the current scheduler implementation
        for successor in successors:
            if isinstance(successor, ProcedureItem):
                item.trafo_data[self._key]['uses_symbols'] |= successor.trafo_data[self._key]['uses_symbols']
                item.trafo_data[self._key]['defines_symbols'] |= successor.trafo_data[self._key]['defines_symbols']


class GlobalVarOffloadTransformation(Transformation):
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
    item_filter = (ProcedureItem, ModuleItem)

    def __init__(self, key=None):
        self._key = key or GlobalVariableAnalysis._key

    def transform_module(self, module, **kwargs):
        """
        Add device-side declarations for imported variables
        """
        if 'item' not in kwargs:
            raise RuntimeError('Cannot apply GlobalVarOffloadTransformation without trafo_data in item')

        item = kwargs['item']

        # Check for already declared offloads
        acc_pragmas = [pragma for pragma in FindNodes(Pragma).visit(module.spec) if pragma.keyword.lower() == 'acc']
        acc_pragma_parameters = get_pragma_parameters(acc_pragmas, starts_with='declare', only_loki_pragmas=False)
        declared_variables = set(flatten([
            v.replace(' ','').lower().split()
            for v in as_tuple(acc_pragma_parameters.get('create'))
        ]))

        # Build list of symbols to be offloaded (discard variables being parameter)
        offload_variables = {
            var.parents[0] if var.parent else var
            for var in item.trafo_data[self._key].get('offload', ()) if not var.type.parameter
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
        item = kwargs['item']

        if role == 'driver':
            self.process_driver(routine, successors)
        elif role == 'kernel':
            self.process_kernel(item, successors)

    def process_kernel(self, item, successors):
        """
        Propagate offload requirement to the items of the global variables
        """
        successors_map = CaseInsensitiveDict(
            (item.name, item) for item in successors if isinstance(item, ModuleItem)
        )
        for var, module in chain(
            item.trafo_data[self._key]['uses_symbols'],
            item.trafo_data[self._key]['defines_symbols']
        ):
            if var.type.parameter:
                continue
            if successor := successors_map.get(module):
                successor.trafo_data[self._key]['offload'].add(var)

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
            # discard variables being parameter
            parameters = {(var, module) for var, module in uses_symbols if var.type.parameter}
            uses_symbols ^= parameters

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
                symbols = tuple(var.clone(dimensions=None, scope=routine) for var in variables)
                routine.spec.prepend(Import(module=module, symbols=symbols))

            routine.spec.prepend(Comment(text=(
                '![Loki::GlobalVarOffloadTransformation] '
                '-------- Added global variable imports for offload directives -----------'
            )))


class GlobalVarHoistTransformation(Transformation):
    """
    Transformation to hoist module variables used in device routines

    This requires a prior analysis pass with :any:`GlobalVariableAnalysis` to collect
    the relevant global variable use information.

    Modules to be ignored can be specified. Further, it is possible to
    configure whether parameters/compile time constants are hoisted as well
    or not.

    .. note::
      Hoisted variables that could theoretically be ``intent(out)``
      are despite specified as ``intent(inout)``.

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

        call kernel()

        end subroutine driver

        subroutine kernel()
        use moduleB, only: var2,var3
        use moduleC, only: var4,var5
        implicit none

        var4 = var2
        var5 = var3

        end subroutine kernel

    is transformed to:

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
        use moduleB, only: var2,var3
        use moduleC, only: var4,var5
        implicit none

        call kernel(var2, var3, var4, var5)

        end subroutine driver

        subroutine kernel(var2, var3, var4, var5)
        implicit none
        real, intent(in) :: var2
        real, intent(in) :: var3
        real, intent(inout) :: var4
        real, intent(inout) :: var5

        var4 = var2
        var5 = var3

        end subroutine kernel

    Parameters
    ----------
    hoist_parameters : bool, optional
        Whether or not to hoist module variables being parameter/compile
        time constants (default: `False`).
    ignore_modules : (list, tuple) of str
        Modules to be ignored (default: `None`, thus no module to be ignored).
    key : str, optional
        Overwrite the key that is used to store analysis results in ``trafo_data``.
    """
    item_filter = ProcedureItem

    def __init__(self, hoist_parameters=False, ignore_modules=None, key=None):
        self._key = key or GlobalVariableAnalysis._key
        self.hoist_parameters = hoist_parameters
        self.ignore_modules = [module.lower() for module in as_tuple(ignore_modules)]

    def transform_subroutine(self, routine, **kwargs):
        """
        Hoist module variables.
        """
        role = kwargs.get('role')
        successors = kwargs.get('successors', ())
        item = kwargs.get('item', None)

        if role == 'driver':
            self.process_driver(routine, successors)
        elif role == 'kernel':
            self.process_kernel(routine, successors, item)

    def process_driver(self, routine, successors):
        """
        Hoist module variables for driver routines.

        This includes: appending the corresponding variables
        to calls within the driver and adding the relevant
        imports.
        """
        # get symbols per routine (successors)
        defines_symbols, uses_symbols = self._get_symbols(successors)

        # append symbols to calls (arguments)
        self._append_call_arguments(routine, uses_symbols, defines_symbols)

        # combine/collect symbols disregarding routine
        all_defines_symbols = set.union(*defines_symbols.values(), set())
        all_uses_symbols = set.union(*uses_symbols.values(), set())
        # add imports for symbols hoisted
        symbol_map = defaultdict(set)
        for var, module in chain(all_uses_symbols, all_defines_symbols):
            # filter modules that are supposed to be ignored
            if module.lower() in self.ignore_modules:
                continue
            symbol_map[module].add(var.parents[0] if var.parent else var)
        import_map = CaseInsensitiveDict()
        scope = routine
        while scope:
            import_map.update(scope.import_map)
            scope = scope.parent
        missing_imports_map = defaultdict(set)
        for module, variables in symbol_map.items():
            missing_imports_map[module] |= {var for var in variables if var.name not in import_map}
        if missing_imports_map:
            routine.spec.prepend(Comment(text=(
                '![Loki::GlobalVarHoistTransformation] ---------------------------------------'
            )))
            for module, variables in missing_imports_map.items():
                symbols = tuple(var.clone(dimensions=None, scope=routine) for var in variables)
                routine.spec.prepend(Import(module=module, symbols=symbols))

            routine.spec.prepend(Comment(text=(
                '![Loki::GlobalVarHoistTransformation] '
                '-------- Added global variable imports for offload directives -----------'
            )))

    def process_kernel(self, routine, successors, item):
        """
        Hoist mdule variables for kernel routines.

        This includes: appending the corresponding variables
        to the routine arguments as well as to calls within the kernel
        and removing the imports that became unused.
        """
        # get symbols per routine (successors)
        defines_symbols, uses_symbols = self._get_symbols(successors)

        # append symbols to routine (arguments)
        self._append_routine_arguments(routine, item)

        # append symbols to calls (arguments)
        self._append_call_arguments(routine, uses_symbols, defines_symbols)

        # get symbols for this routine/kernel
        kernel_defines_symbols = item.trafo_data.get(self._key, {}).get('defines_symbols', set())
        kernel_uses_symbols = item.trafo_data.get(self._key, {}).get('uses_symbols', set())
        # remove imports for symbols hoisted
        symbol_map = defaultdict(set)
        for var, module in chain(kernel_uses_symbols, kernel_defines_symbols):
            # filter modules that are supposed to be ignored
            if module.lower() in self.ignore_modules:
                continue
            symbol_map[module].add(var.parents[0] if var.parent else var)
        import_map = CaseInsensitiveDict(
            (s.name, imprt) for imprt in routine.all_imports[::-1] for s in imprt.symbols
        )
        redundant_imports_map = defaultdict(set)
        for module, variables in symbol_map.items():
            redundant = [var.parent[0] if var.parent else var for var in variables]
            redundant = {var.clone(dimensions=None) for var in redundant if var.name in import_map}
            redundant_imports_map[module] |= redundant
        import_map = {}
        imports = FindNodes(Import).visit(routine.spec)
        for _import in imports:
            new_symbols = tuple(
                var.clone(dimensions=None, scope=routine)
                for var in set(_import.symbols)-redundant_imports_map[_import.module.lower()]
            )
            if new_symbols:
                import_map[_import] = _import.clone(symbols=new_symbols)
            else:
                import_map[_import] = None
        routine.spec = Transformer(import_map).visit(routine.spec)

    def _get_symbols(self, successors):
        """
        Get module variables/symbols (grouped by routine/successor).
        """
        defines_symbols = CaseInsensitiveDict()
        uses_symbols = CaseInsensitiveDict()
        for item in successors:
            if not isinstance(item, ProcedureItem):
                continue
            defines_symbols[item.local_name] = set()
            uses_symbols[item.local_name] = set()
            defines_symbols[item.local_name] = item.trafo_data.get(self._key, {}).get('defines_symbols', set())
            uses_symbols[item.local_name] = item.trafo_data.get(self._key, {}).get('uses_symbols', set())
            # remove parameters if hoist_parameters is False
            if not self.hoist_parameters:
                parameters = {(var, module) for var, module in uses_symbols[item.local_name] if var.type.parameter}
                uses_symbols[item.local_name] ^= parameters
        return defines_symbols, uses_symbols

    def _append_call_arguments(self, routine, uses_symbols, defines_symbols):
        """
        Helper to append variables to the call(s) (arguments).
        """
        symbol_map = CaseInsensitiveDefaultDict(set)
        for key, _ in uses_symbols.items():
            all_symbols = uses_symbols[key]|defines_symbols[key]
            for var, module in all_symbols:
                # filter modules that are supposed to be ignored
                if module.lower() in self.ignore_modules:
                    continue
                symbol_map[key].add(var.parents[0] if var.parent else var)
        call_map = {}
        calls = FindNodes(CallStatement).visit(routine.body)
        for call in calls:
            if call.routine.name in uses_symbols:
                arguments = call.arguments
                new_args = sorted(
                    [var.clone(dimensions=None) for var in symbol_map[call.routine.name]],
                    key=lambda symbol: symbol.name
                )
                call_map[call] = call.clone(arguments=arguments + tuple(new_args))
        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)
        inline_calls = FindInlineCalls().visit(routine.body)
        inline_call_map = {}
        for call in inline_calls:
            if call.routine.name in uses_symbols:
                arguments = call.parameters
                new_args = sorted([var.clone(dimensions=None) for var in symbol_map[call.routine.name]],
                        key=lambda symbol: symbol.name)
                inline_call_map[call] = call.clone(parameters=arguments + tuple(new_args))
        if inline_call_map:
            routine.body = SubstituteExpressions(inline_call_map).visit(routine.body)

    def _append_routine_arguments(self, routine, item):
        """
        Helper to append variables to the routine (arguments).
        """
        all_defines_symbols = item.trafo_data.get(self._key, {}).get('defines_symbols', set())
        all_defines_vars = [var.parents[0] if var.parent else var for var, _ in all_defines_symbols]
        all_uses_symbols = item.trafo_data.get(self._key, {}).get('uses_symbols', set())
        # remove parameters if hoist_parameters is False
        if not self.hoist_parameters:
            parameters = {(var, module) for var, module in all_uses_symbols if var.type.parameter}
            all_uses_symbols ^= parameters
        all_symbols = all_uses_symbols|all_defines_symbols
        new_arguments = []
        for var, module in all_symbols:
            # filter modules that are supposed to be ignored
            if module.lower() in self.ignore_modules:
                continue
            new_arguments.append(var.parents[0] if var.parent else var)
        new_arguments = set(new_arguments) # remove duplicates
        new_arguments = [
            arg.clone(scope=routine, type=arg.type.clone(
                intent='inout' if arg in all_defines_vars else 'in',
                parameter=False, initial=None
            )) for arg in new_arguments
        ]
        routine.arguments += tuple(sorted(new_arguments, key=lambda symbol: symbol.name))
