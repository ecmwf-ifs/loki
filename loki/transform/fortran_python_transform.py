from pathlib import Path

from loki.visitors import Transformer, FindNodes
from loki.transform import Transformation, FortranCTransformation
from loki.expression import symbol_types as sym, FindVariables, SubstituteExpressions
from loki.backend import pygen, dacegen
from loki import ir, Subroutine, SourceFile


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
        source = dacegen(kernel) if kwargs.get('with_dace', False) is True else pygen(kernel)
        SourceFile.to_file(source=source, path=self.py_path)

    @staticmethod
    def generate_kernel(routine, **kwargs):
        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = ir.Section(body=())
        body = ir.Section(body=Transformer({}).visit(routine.body))
        kernel = Subroutine(name='{}_py'.format(routine.name), spec=spec, body=body)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        # Force all variables to lower-caps, as Python is case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(kernel.body)
                if isinstance(v, (sym.Scalar, sym.Array)) and not v.name.islower()}
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        # Resolve implicit struct mappings through "associates"
        assoc_map = {}
        vmap = {}
        for assoc in FindNodes(ir.Scope).visit(kernel.body):
            invert_assoc = {v.name: k for k, v in assoc.associations.items()}
            for v in FindVariables(unique=False).visit(kernel.body):
                if v.name in invert_assoc:
                    vmap[v] = invert_assoc[v.name]
            assoc_map[assoc] = assoc.body
        kernel.body = Transformer(assoc_map).visit(kernel.body)
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        # Do some vector and indexing transformations
        FortranCTransformation._resolve_vector_notation(kernel, **kwargs)
        FortranCTransformation._resolve_omni_size_indexing(kernel, **kwargs)
        if kwargs.get('with_dace', False) is True:
            FortranCTransformation._invert_array_indices(kernel, **kwargs)
        FortranCTransformation._shift_to_zero_indexing(kernel, **kwargs)
        # self._replace_intrinsics(kernel, **kwargs)

        return kernel
