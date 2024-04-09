# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import Array
from loki.types import DerivedType
from loki.backend.cppgen import CppCodegen, CppCodeMapper, IntrinsicTypeCpp

__all__ = ['cudagen', 'CudaCodegen', 'CudaCodeMapper']


class IntrinsicTypeCuda(IntrinsicTypeCpp):
    pass

cuda_intrinsic_type = IntrinsicTypeCuda()


class CudaCodeMapper(CppCodeMapper):
    # pylint: disable=abstract-method, unused-argument
    pass


class CudaCodegen(CppCodegen):
    """
    ...
    """

    standard_imports = ['stdio.h', 'stdbool.h', 'float.h',
            'math.h', 'cuda.h', 'cuda_runtime.h']

    def __init__(self, depth=0, indent='  ', linewidth=90, **kwargs):
        symgen = kwargs.get('symgen', CudaCodeMapper(cuda_intrinsic_type))
        line_cont = kwargs.get('line_cont', '\n{}  '.format)

        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont=line_cont, symgen=symgen)


    def _subroutine_arguments(self, o, **kwargs):
        var_keywords = []
        pass_by = []
        for a in o.arguments:
            if isinstance(a, Array) > 0 and a.type.intent.lower() == "in":
                var_keywords += ['const ']
            else:
                var_keywords += ['']
            if isinstance(a, Array) > 0:
                pass_by += ['* __restrict__ ']
            elif isinstance(a.type.dtype, DerivedType):
                pass_by += ['*']
            elif a.type.pointer:
                pass_by += ['*']
            else:
                pass_by += ['']
        return pass_by, var_keywords

    def visit_CallStatement(self, o, **kwargs):
        args = self.visit_all(o.arguments, **kwargs)
        assert not o.kwarguments
        chevron = f'<<<{",".join([str(elem) for elem in o.chevron])}>>>' if o.chevron is not None else ''
        return self.format_line(str(o.name).lower(), chevron, '(', self.join_items(args), ');')


def cudagen(ir, **kwargs):
    """
    ...
    """
    return CudaCodegen().visit(ir, **kwargs)
