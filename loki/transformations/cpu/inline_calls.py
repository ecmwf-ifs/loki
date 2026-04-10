# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to inline small subroutine calls that appear inside
horizontal vector loops and block vectorisation.

This addresses Intel optrpt remark ``#15543`` (loop was not vectorised:
existence of vector dependence across call boundary), affecting ~3-4
loops in CLOUDSC (``CUADJTQ_LOKI``, ``CLOUD_SUPERSATCHECK_LOKI``).

The transformation wraps Loki's existing inlining utilities with
vectorisation-specific targeting: only calls *inside* horizontal loops
whose callees are small and free of I/O or nested calls are inlined.

**Numerical impact**: None -- inlining is semantically equivalent.
"""

from collections import defaultdict

from loki.batch import Transformation
from loki.ir import nodes as ir, FindNodes
from loki.types import BasicType

from loki.transformations.inline.procedures import inline_subroutine_calls
from loki.transformations.utilities import check_routine_sequential


__all__ = ['InlineCallSiteForVectorisation']


class InlineCallSiteForVectorisation(Transformation):
    """
    Inline subroutine calls that appear inside horizontal loops and
    block vectorisation.

    Only "small" callees (member procedures or resolved external
    procedures) that do not contain I/O or further calls are inlined.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        The dimension object describing the horizontal iteration space.
    max_callee_lines : int, optional
        Maximum number of IR body nodes in the callee for it to be
        eligible for inlining.  Default is ``100``.
    """

    # Process bottom-up so that inlining exposes callee bodies to
    # subsequent transformations in the pipeline.
    reverse_traversal = True

    def __init__(self, horizontal, max_callee_lines=100):
        self.horizontal = horizontal
        self.max_callee_lines = max_callee_lines

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply targeted inlining to a single subroutine.

        Skips driver-role routines and sequential routines.
        """
        role = kwargs.get('role', 'kernel')
        if role == 'driver':
            return
        if check_routine_sequential(routine):
            return

        # Collect calls inside horizontal loops that are eligible for
        # inlining.  Group by callee for batch processing.
        calls_by_callee = defaultdict(list)

        for loop in FindNodes(ir.Loop).visit(routine.body):
            if not self._is_horizontal_loop(loop):
                continue

            for call in FindNodes(ir.CallStatement).visit(loop.body):
                callee = self._resolve_callee(call, routine)
                if callee is None:
                    continue
                if self._callee_too_large(callee):
                    continue
                if self._has_io_or_nested_calls(callee):
                    continue
                calls_by_callee[callee].append(call)

        # Inline each group of calls
        for callee, calls in calls_by_callee.items():
            inline_subroutine_calls(routine, calls, callee)

    # -----------------------------------------------------------------
    # Detection helpers
    # -----------------------------------------------------------------

    def _is_horizontal_loop(self, loop):
        """Return ``True`` if *loop* iterates over the horizontal dimension."""
        return loop.variable == self.horizontal.index

    @staticmethod
    def _resolve_callee(call, routine):
        """
        Try to resolve the callee :any:`Subroutine` object.

        First checks member (contained) procedures; then falls back to
        the call's resolved ``routine`` attribute (which is set when
        the :any:`Scheduler` has linked up the call graph).

        Returns ``None`` if the callee cannot be resolved.
        """
        call_name = str(call.name).upper()

        # Check member / contained procedures
        if routine.members:
            for member in routine.members:
                if member.name.upper() == call_name:
                    return member

        # Check if the Scheduler has resolved the call
        resolved = call.routine
        if resolved is not None and resolved is not BasicType.DEFERRED:
            return resolved

        return None

    def _callee_too_large(self, callee):
        """
        Return ``True`` if *callee* exceeds the size threshold.
        """
        # Count top-level body nodes as a proxy for "lines"
        body_nodes = FindNodes(ir.Node).visit(callee.body)
        return len(body_nodes) > self.max_callee_lines

    @staticmethod
    def _has_io_or_nested_calls(callee):
        """
        Return ``True`` if *callee* contains I/O intrinsics (WRITE/PRINT)
        or further subroutine calls (which would remain opaque).
        """
        for intr in FindNodes(ir.Intrinsic).visit(callee.body):
            text = intr.text.upper().lstrip()
            if text.startswith('WRITE') or text.startswith('PRINT'):
                return True

        if FindNodes(ir.CallStatement).visit(callee.body):
            return True

        return False
