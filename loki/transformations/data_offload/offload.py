# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import Array
from loki.ir import (
    FindNodes, PragmaRegion, CallStatement, Pragma, Transformer,
    pragma_regions_attached,
)
from loki.logging import warning, error
from loki.tools import as_tuple
from loki.types import BasicType


__all__ = ['DataOffloadTransformation']


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
    present_on_device : bool
        Assume arrays are already offloaded and present on device"
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
        self.present_on_device = kwargs.get('present_on_device', False)

        if self.assume_deviceptr and not self.present_on_device:
            error("[Loki] Data offload: Can't assume device pointer arrays without arrays being marked" +
                    "present on device.")
            raise RuntimeError

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

                # Now generate the pre- and post pragmas (OpenACC)
                if self.present_on_device:
                    if self.assume_deviceptr:
                        offload_args = inargs + outargs + inoutargs
                        if offload_args:
                            deviceptr = f' deviceptr({", ".join(offload_args)})'
                        else:
                            deviceptr = ''
                        pragma = Pragma(keyword='acc', content=f'data{deviceptr}')
                    else:
                        offload_args = inargs + outargs + inoutargs
                        if offload_args:
                            present = f' present({", ".join(offload_args)})'
                        else:
                            present = ''
                        pragma = Pragma(keyword='acc', content=f'data{present}')
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
