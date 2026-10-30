# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.analyse import dataflow_analysis_attached
from loki.batch import Transformation
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, FindInlineCalls,
    get_pragma_command_and_parameters, pragmas_attached
)
from loki.logging import info
from loki.tools import as_tuple


__all__ = ['SCCCollapseTransformation']


class SCCCollapseTransformation(Transformation):
    """
    Combine an already perfectly nested gang/vector pair in an SCC driver.

    This pass only changes generic loop annotations. It relies on the existing
    SCC parallelism contract and requires rectangular, invariant iteration
    spaces with unit steps and no private or reduction clauses. No statements
    are moved. Run it after temporary-storage synthesis and before
    :any:`PragmaModelTransformation`, so eligibility reflects the final nest.

    Parameters
    ----------
    collapse_gang_vector : bool, optional
        Enable collapsing eligible pairs, by default False.
    directive : str or False, optional
        Target programming model. ``'openacc'``, ``'omp-gpu'`` and generic
        Loki output (``False``) support this transformation.
    """

    def __init__(self, collapse_gang_vector=False, directive=False):
        self.collapse_gang_vector = collapse_gang_vector
        self.directive = directive

    @staticmethod
    def loop_parameters(pragmas, command, level, allowed):
        """
        Return parameters for a single supported generic loop annotation.

        Return None for missing, conflicting or unsupported annotations.
        """
        pragmas = as_tuple(pragmas)
        if len(pragmas) != 1 or pragmas[0].keyword.lower() != 'loki':
            return None
        pragma_command, params = get_pragma_command_and_parameters(pragmas[0])
        if pragma_command != command or level not in params:
            return None
        if params[level] is not None or not params.keys() <= allowed:
            return None
        if any(isinstance(v, list) for v in params.values()):
            return None
        return params

    @classmethod
    def check_candidate(cls, outer):
        """
        Return the inner vector loop and a rejection reason for a gang loop.

        The reason is None on success. This check requires attached loop
        pragmas and dataflow analysis and does not modify the loop nest.
        """
        if cls.loop_parameters(outer.pragma, 'loop', 'gang', {'gang', 'vlength', 'async'}) is None:
            return None, 'unsupported gang-loop annotations or clauses'
        if outer.pragma_post and cls.loop_parameters(outer.pragma_post, 'end-loop', 'gang', {'gang'}) is None:
            return None, 'unsupported gang-loop end annotations'

        body = tuple(n for n in outer.body if not isinstance(n, (ir.Comment, ir.CommentBlock)))
        if len(body) != 1 or not isinstance(body[0], ir.Loop):
            return None, 'gang and vector loops are not perfectly nested'
        inner = body[0]
        if cls.loop_parameters(inner.pragma, 'loop', 'vector', {'vector'}) is None:
            return None, 'unsupported vector-loop annotations or clauses'
        if inner.pragma_post and cls.loop_parameters(inner.pragma_post, 'end-loop', 'vector', {'vector'}) is None:
            return None, 'unsupported vector-loop end annotations'

        if outer.variable == inner.variable:
            return None, 'associated loops use the same induction variable'
        for loop in (outer, inner):
            if loop.loop_label or not loop.has_end_do or loop.bounds.step not in (None, 1):
                return None, 'associated loops must be structured unit-step loops'
            if FindInlineCalls().visit(loop.bounds):
                return None, 'calls in associated loop bounds are unsupported'
            variables = FindVariables().visit(loop.bounds)
            if any(v == outer.variable or v == inner.variable or v.name in outer.defines_symbols for v in variables):
                return None, 'associated loop bounds are dependent or modified within the nest'

        if FindNodes((ir.GenericStmt, ir.RawSource, ir.PreprocessorDirective)).visit(inner.body):
            return None, 'unsupported statements or control flow in the vector loop'
        return inner, None

    @staticmethod
    def collapse_loop(outer, inner):
        """Combine generic annotations for an eligible pair in place."""
        pragma = outer.pragma[0]
        _, params = get_pragma_command_and_parameters(pragma)
        clauses = ''.join(f' {key}({value})' if value is not None else f' {key}'
                          for key, value in params.items() if key != 'gang')
        content = f'loop gang vector collapse(2){clauses}'
        pragma_post = tuple(p.clone(content='end loop gang vector') for p in as_tuple(outer.pragma_post))
        outer._update(pragma=(pragma.clone(content=content),), pragma_post=pragma_post)
        inner._update(pragma=None, pragma_post=None)

    def transform_subroutine(self, routine, **kwargs):
        """Collapse eligible annotated pairs in driver routines when enabled."""
        if not self.collapse_gang_vector or kwargs.get('role') != 'driver':
            return
        if self.directive not in (False, 'openacc', 'omp-gpu'):
            info(f'[Loki-SCC::Collapse] Skipping {routine.name}: unsupported directive {self.directive}')
            return

        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            with dataflow_analysis_attached(routine):
                for outer in FindNodes(ir.Loop).visit(routine.body):
                    params = get_pragma_command_and_parameters(outer.pragma[0]) if outer.pragma else (None, {})
                    if params[0] != 'loop' or 'gang' not in params[1] or 'collapse' in params[1]:
                        continue
                    inner, reason = self.check_candidate(outer)
                    if reason:
                        info(f'[Loki-SCC::Collapse] Skipping {routine.name}/{outer.variable}: {reason}')
                    else:
                        self.collapse_loop(outer, inner)
