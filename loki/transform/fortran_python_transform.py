from pathlib import Path

from loki.visitors import Transformer, FindNodes
from loki.transform import Transformation, FortranCTransformation
from loki.expression import (
    symbol_types as sym, FindVariables, SubstituteExpressions, FindInlineCalls)
from loki.backend import pygen, dacegen
from loki import ir, Subroutine, SourceFile, as_tuple


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

    @classmethod
    def generate_kernel(cls, routine, **kwargs):
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
        # FortranCTransformation._resolve_vector_notation(kernel, **kwargs)
        FortranCTransformation._resolve_omni_size_indexing(kernel, **kwargs)
        if kwargs.get('with_dace', False) is True:
            FortranCTransformation._invert_array_indices(kernel, **kwargs)
        cls._shift_to_zero_indexing(kernel, **kwargs)
        cls._replace_intrinsics(kernel, **kwargs)

        return kernel

    @staticmethod
    def _shift_to_zero_indexing(kernel, **kwargs):  # pylint: disable=unused-argument
        """
        Shift all array indices to adjust to Python indexing conventions
        """
        vmap = {}
        for v in FindVariables(unique=False).visit(kernel.body):
            if isinstance(v, sym.Array):
                new_dims = []
                for d in v.dimensions.index_tuple:
                    if isinstance(d, sym.RangeIndex):
                        start = d.start - 1 if d.start is not None else None
                        # no shift for stop because Python ranges are [start, stop)
                        new_dims += [sym.RangeIndex((start, d.stop, d.step))]
                    else:
                        new_dims += [d - 1]
                vmap[v] = v.clone(dimensions=sym.ArraySubscript(as_tuple(new_dims)))
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

    @staticmethod
    def _replace_intrinsics(kernel, **kwargs):  # pylint: disable=unused-argument
        """
        Replace known numerical intrinsic functions.
        """
        _intrinsic_map = {
            'min': 'min', 'max': 'max',
            'abs': 'abs',
        }

        callmap = {}
        for c in FindInlineCalls(unique=False).visit(kernel.body):
            cname = c.name.lower()
            if cname in _intrinsic_map:
                callmap[c] = sym.InlineCall(_intrinsic_map[cname], parameters=c.parameters,
                                            kw_parameters=c.kw_parameters)
        kernel.body = SubstituteExpressions(callmap).visit(kernel.body)
