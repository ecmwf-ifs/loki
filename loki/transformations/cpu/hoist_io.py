# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to hoist ``WRITE`` and ``PRINT`` statements out of
horizontal vector loops by replacing them with boolean flags.

This addresses Intel optrpt remark ``#15344`` (vector dependence caused
by I/O operations), which prevents vectorisation even when the I/O is
conditional and rarely executed.

The transformation replaces each I/O statement inside a horizontal loop
with a boolean flag assignment (``LLHOIST_WARN = .TRUE.``), initialises
the flag to ``.FALSE.`` before the loop, and emits a diagnostic ``WRITE``
after the loop guarded by the flag.  This removes the output dependence
from the loop body.

**Numerical impact**: None -- only diagnostic output is affected.
"""

from loki.batch import Transformation
from loki.ir import nodes as ir, FindNodes, Transformer
from loki.expression import symbols as sym
from loki.types import BasicType, SymbolAttributes
from loki.tools import as_tuple

from loki.transformations.utilities import check_routine_sequential


__all__ = ['HoistWriteFromLoop']


class HoistWriteFromLoop(Transformation):
    """
    Hoist ``WRITE`` / ``PRINT`` statements out of horizontal vector
    loops and replace them with boolean-flag diagnostics.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        The dimension object describing the horizontal iteration space.
    """

    def __init__(self, horizontal):
        self.horizontal = horizontal

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply I/O hoisting to a single subroutine.

        Skips driver-role routines and sequential routines.
        """
        role = kwargs.get('role', 'kernel')
        if role == 'driver':
            return
        if check_routine_sequential(routine):
            return

        # Collect all loops that need transformation.  We process each
        # horizontal loop independently.
        loop_replacements = {}   # {original_loop: (pre_stmts, new_loop, post_stmts)}
        flag_counter = 0

        for loop in FindNodes(ir.Loop).visit(routine.body):
            if not self._is_horizontal_loop(loop):
                continue

            # Find all I/O intrinsics in this loop's body
            io_nodes = [
                n for n in FindNodes(ir.Intrinsic).visit(loop.body)
                if self._is_io_intrinsic(n)
            ]
            if not io_nodes:
                continue

            flag_counter += 1
            flag_name = f'LLHOIST_WARN_{flag_counter}'

            # Create a boolean flag variable and add it to the routine
            flag_type = SymbolAttributes(BasicType.LOGICAL)
            flag_var = sym.Variable(name=flag_name, type=flag_type,
                                    scope=routine)
            routine.variables += (flag_var,)

            # Build replacement map: each I/O node -> flag = .TRUE.
            io_map = {}
            original_texts = []
            for io_node in io_nodes:
                original_texts.append(io_node.text)
                flag_assign = ir.Assignment(
                    lhs=flag_var,
                    rhs=sym.LogicLiteral('.TRUE.')
                )
                io_map[io_node] = flag_assign

            # Rebuild the loop body with I/O replaced by flag assignments
            new_body = Transformer(io_map).visit(loop.body)
            new_loop = loop.clone(body=new_body)

            # Pre-loop: flag = .FALSE.
            pre_init = ir.Assignment(
                lhs=flag_var,
                rhs=sym.LogicLiteral('.FALSE.')
            )

            # Post-loop: IF (flag) THEN WRITE(...) END IF
            # Build a simplified diagnostic message
            diag_text = self._build_diagnostic_text(original_texts)
            diag_write = ir.Intrinsic(text=diag_text)
            post_cond = ir.Conditional(
                condition=flag_var,
                body=(diag_write,),
                else_body=None
            )

            loop_replacements[loop] = (pre_init, new_loop, post_cond)

        if not loop_replacements:
            return

        # Apply all replacements.  Each original loop is replaced by
        # the sequence (pre_init, new_loop, post_cond).
        loop_map = {}
        for orig_loop, (pre, new_loop, post) in loop_replacements.items():
            loop_map[orig_loop] = (pre, new_loop, post)

        routine.body = Transformer(loop_map).visit(routine.body)

    # -----------------------------------------------------------------
    # Detection helpers
    # -----------------------------------------------------------------

    def _is_horizontal_loop(self, loop):
        """Return ``True`` if *loop* iterates over the horizontal dimension."""
        return loop.variable == self.horizontal.index

    @staticmethod
    def _is_io_intrinsic(node):
        """
        Return ``True`` if *node* is a ``WRITE`` or ``PRINT`` intrinsic.
        """
        text = node.text.upper().lstrip()
        return text.startswith('WRITE') or text.startswith('PRINT')

    # -----------------------------------------------------------------
    # Code generation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _build_diagnostic_text(original_texts):
        """
        Build a single diagnostic ``WRITE`` statement summarising
        the hoisted I/O operations.

        If there was a single original WRITE/PRINT, we reproduce it
        with a "(hoisted from loop)" annotation.  For multiple I/O
        statements we produce a generic summary.
        """
        if len(original_texts) == 1:
            orig = original_texts[0]
            # Try to insert a hoisting note into the message
            upper = orig.upper()
            if upper.startswith('WRITE'):
                # Preserve the WRITE(...) prefix and append a note
                return f"WRITE(*, *) '[HOISTED] originally: {_escape_quotes(orig)}'"
            elif upper.startswith('PRINT'):
                return f"PRINT *, '[HOISTED] originally: {_escape_quotes(orig)}'"
        # Multiple statements: generic summary
        return (
            f"WRITE(*, *) '[HOISTED] {len(original_texts)} "
            f"diagnostic I/O statement(s) hoisted from loop'"
        )


def _escape_quotes(text):
    """Escape single quotes in *text* for embedding in a Fortran string."""
    return text.replace("'", "''")
