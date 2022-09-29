from pathlib import Path

from loki.backend import pygen, dacegen
from loki import ir
from loki.pragma_utils import pragmas_attached
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine
from loki.transform.transformation import Transformation
from loki.transform.transform_array_indexing import (
    shift_to_zero_indexing, invert_array_indices, normalize_range_indexing
)
from loki.transform.transform_utilities import (
    convert_to_lower_case, replace_intrinsics, resolve_associates
)
from loki.visitors import Transformer


__all__ = ['FortranPythonTransformation']


class FortranPythonTransformation(Transformation):
    """
    A transformer class to convert Fortran to Python.
    """

    def transform_subroutine(self, routine, **kwargs):
        path = Path(kwargs.get('path'))

        # Generate Python kernel
        kernel = self.generate_kernel(routine, **kwargs)
        self.py_path = (path/kernel.name.lower()).with_suffix('.py')
        self.mod_name = kernel.name.lower()
        # Need to attach Loop pragmas to honour dataflow pragmas for loops
        with pragmas_attached(kernel, ir.Loop):
            source = dacegen(kernel) if kwargs.get('with_dace', False) is True else pygen(kernel)
        Sourcefile.to_file(source=source, path=self.py_path)

    @classmethod
    def generate_kernel(cls, routine, **kwargs):
        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = ir.Section(body=())
        body = ir.Section(body=Transformer({}).visit(routine.body))
        kernel = Subroutine(name=f'{routine.name}_py', spec=spec, body=body)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        # Force all variables to lower-caps, as Python is case-sensitive
        convert_to_lower_case(kernel)

        # Resolve implicit struct mappings through "associates"
        resolve_associates(kernel)

        # Do some vector and indexing transformations
        normalize_range_indexing(kernel)
        if kwargs.get('with_dace', False) is True:
            invert_array_indices(kernel)
        shift_to_zero_indexing(kernel)

        intrinsic_map = {'min': 'min', 'max': 'max', 'abs': 'abs'}
        replace_intrinsics(kernel, function_map=intrinsic_map)

        return kernel
