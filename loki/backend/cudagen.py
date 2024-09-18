# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.types import DerivedType
from loki.backend.cppgen import CppCodegen, CppCodeMapper, IntrinsicTypeCpp
from loki.ir import Import, FindNodes
from loki.expression import Array

__all__ = ['cudagen', 'CudaCodegen', 'CudaCodeMapper']


class IntrinsicTypeCuda(IntrinsicTypeCpp):
    """
    Mapping Fortran type to corresponding CUDA type.
    """
    # pylint: disable=unnecessary-pass
    pass

cuda_intrinsic_type = IntrinsicTypeCuda()


class CudaCodeMapper(CppCodeMapper):
    """
    A :class:`StringifyMapper`-derived visitor for Pymbolic expression trees that converts an
    expression to a string adhering to standardized CUDA.
    """
    # pylint: disable=abstract-method, unused-argument, unnecessary-pass
    pass


class CudaCodegen(CppCodegen):
    """
    Tree visitor to generate standardized CUDA code from IR.
    """

    standard_imports = ['stdio.h', 'stdbool.h', 'float.h',
            'math.h', 'cuda.h', 'cuda_runtime.h']

    def __init__(self, depth=0, indent='  ', linewidth=90, **kwargs):
        symgen = kwargs.get('symgen', CudaCodeMapper(cuda_intrinsic_type))
        line_cont = kwargs.get('line_cont', '\n{}  '.format)

        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont=line_cont, symgen=symgen)

    def _subroutine_header(self, o, **kwargs):
        opt_header = kwargs.get('header', False)
        opt_extern = kwargs.get('extern', False)
        if opt_header or opt_extern:
            header = []
        else:
            # Some boilerplate imports...
            header = [self.format_line('#include <', name, '>') for name in self.standard_imports]
            # ...and imports from the spec
            spec_imports = FindNodes(Import).visit(o.spec)
            header += [self.visit(spec_imports, **kwargs)]
        if o.prefix and "global" in o.prefix[0].lower():
            # include launcher and header file
            header += [self.format_line('')]
            if not opt_header:
                header += [self.format_line('#include "', o.name, '.h', '"')]
                header += [self.format_line('#include "', o.name, '_launch.h', '"')]
        return header

    def _subroutine_arguments(self, o, **kwargs):
        var_keywords = []
        pass_by = []
        for a in o.arguments:
            if a.type.intent is None:
                print(f"WHY THE FUCK is a {a}.type.intent None? {o.name}")
            if isinstance(a, Array) > 0 and a.type.intent is not None and a.type.intent.lower() == "in":
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

    def _subroutine_declaration(self, o, **kwargs):
        pass_by, var_keywords = self._subroutine_arguments(o, **kwargs)
        arguments = [f'{k}{self.visit(a.type, **kwargs)} {p}{a.name}'
                     for a, p, k in zip(o.arguments, pass_by, var_keywords)]
        opt_header = kwargs.get('header', False)
        end = ' {' if not opt_header else ';'
        prefix = ''
        if o.prefix and "global" in o.prefix[0].lower():
            prefix = '__global__ '
        if o.prefix and "device" in o.prefix[0].lower():
            prefix = '__device__ '
        if o.is_function:
            return_type = cuda_intrinsic_type(o.return_type)
        else:
            return_type = 'void'
        opt_extern = kwargs.get('extern', False)
        declaration = [self.format_line('extern "C" {\n')] if opt_extern else []
        declaration += [self.format_line(prefix, f'{return_type} ', o.name, '(', self.join_items(arguments), ')', end)]
        return declaration

    def _subroutine_body(self, o, **kwargs):
        self.depth += 1
        # ...and generate the spec without imports and argument declarations
        skip_imports = kwargs.pop('skip_imports', None)
        skip_argument_declarations = kwargs.pop('skip_argument_declarations', None)
        body = [self.visit(o.spec, skip_imports=skip_imports or True, skip_argument_declarations=skip_argument_declarations or True, **kwargs)]
        # Fill the body
        body += [self.visit(o.body, **kwargs)]
        opt_extern = kwargs.get('extern', False)
        if opt_extern:
            body += [self.format_line('cudaDeviceSynchronize();')]
        # if something to be returned, add 'return <var>' statement
        if o.result_name is not None:
            body += [self.format_line(f'return {o.result_name.lower()};')]
        # Close everything off
        self.depth -= 1
        # footer = [self.format_line('}')]
        return body

    def _subroutine_footer(self, o, **kwargs):
        postfix = ''
        opt_extern = kwargs.get('extern', False)
        footer = [self.format_line('}'), self.format_line(postfix)]
        footer += [self.format_line('\n} // extern')] if opt_extern else []
        return footer

    def visit_CallStatement(self, o, **kwargs):
        args = self.visit_all(o.arguments, **kwargs)
        assert not o.kwarguments
        chevron = f'<<<{",".join([str(elem) for elem in o.chevron])}>>>' if o.chevron is not None else ''
        return self.format_line(str(o.name), chevron, '(', self.join_items(args), ');')


def cudagen(ir, **kwargs):
    """
    Generate standardized CUDA code from one or many IR objects/trees.
    """
    return CudaCodegen().visit(ir, **kwargs)
