# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

from loki.backend import pygen, dacegen
from loki.batch import Transformation
from loki.expression import (
    symbols as sym, FindInlineCalls, SubstituteExpressions
)
from loki.ir import nodes as ir, FindNodes, Transformer, pragmas_attached
from loki.sourcefile import Sourcefile

from loki.transformations.array_indexing import (
    shift_to_zero_indexing, invert_array_indices
)
from loki.transformations.sanitise import resolve_associates
from loki.transformations.utilities import (
    convert_to_lower_case, replace_intrinsics
)


__all__ = ['FortranPythonTransformation']


class FortranPythonTransformation(Transformation):
    """
    A transformer class to convert Fortran to Python or DaCe.

    This :any:`Transformation` will generate Python code from a
    given Fortran routine, and if configured, annotate it with DaCe
    dataflow pragmas.

    Parameters
    ----------
    with_dace : bool
        Generate DaCe-specific Python code via :any:`dacegen` backend.
        This option implies inverted array indexing; default: ``False``
    invert_indices : bool
        Switch to C-style indexing (row-major) with fastest loop
        indices being used rightmost; default: ``False``
    suffix : str
        Optional suffix to append to converted routine names.
    """

    def __init__(self, **kwargs):
        self.with_dace = kwargs.pop('with_dace', False)
        self.invert_indices = kwargs.pop('invert_indices', False)
        self.suffix = kwargs.pop('suffix', '')

    def transform_subroutine(self, routine, **kwargs):
        path = Path(kwargs.get('path'))

        # Rename subroutine to generate Python kernel
        routine.name = f'{routine.name}{self.suffix}'.lower()

        # Remove all "IMPLICT" intrinsic statements
        mapper = {
            i: None for i in FindNodes(ir.Intrinsic).visit(routine.spec)
            if 'implicit' in i.text.lower()
        }
        routine.spec = Transformer(mapper).visit(routine.spec)

        # Force all variables to lower-caps, as Python is case-sensitive
        convert_to_lower_case(routine)

        # Resolve implicit struct mappings through "associates"
        resolve_associates(routine)

        # Do some vector and indexing transformations
        if self.with_dace or self.invert_indices:
            invert_array_indices(routine)
        shift_to_zero_indexing(routine)

        # We replace calls to intrinsic functions with their Python counterparts
        # Note that this substitution is case-insensitive, and therefore we have
        # this seemingly identity mapping to make sure Python function names are
        # lower-case
        intrinsic_map = {
            'min': 'min', 'max': 'max', 'abs': 'abs',
            'exp': 'np.exp', 'sqrt': 'np.sqrt',
        }
        replace_intrinsics(routine, function_map=intrinsic_map)

        # Sign intrinsic function takes a little more thought
        sign_map = {}
        for c in FindInlineCalls(unique=False).visit(routine.ir):
            if c.function == 'sign':
                assert len(c.parameters) == 2
                sign = sym.InlineCall(
                    function=sym.ProcedureSymbol(name='np.sign', scope=routine),
                    parameters=(c.parameters[1],)
                )
                sign_map[c] = sym.Product((c.parameters[0], sign))

        routine.spec = SubstituteExpressions(sign_map).visit(routine.spec)
        routine.body = SubstituteExpressions(sign_map).visit(routine.body)

        # Rename subroutine to generate Python kernel
        self.py_path = (path/routine.name.lower()).with_suffix('.py')
        self.mod_name = routine.name.lower()
        # Need to attach Loop pragmas to honour dataflow pragmas for loops
        with pragmas_attached(routine, ir.Loop):
            source = dacegen(routine) if self.with_dace else pygen(routine)
        Sourcefile.to_file(source=source, path=self.py_path)
