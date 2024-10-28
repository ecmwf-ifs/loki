# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
        symgen = kwargs.pop('symgen', CudaCodeMapper(cuda_intrinsic_type))
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         symgen=symgen, **kwargs)

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

    def _subroutine_argument_pass_by(self, a):
        if isinstance(a, Array):
            return '* __restrict__ '
        return super()._subroutine_argument_pass_by(a)

    def _subroutine_declaration(self, o, **kwargs):
        arguments = [
            (f'{self._subroutine_argument_keyword(a)}{self.visit(a.type, **kwargs)} '
            f'{self._subroutine_argument_pass_by(a)}{a.name}')
            for a in o.arguments
        ]
        opt_header = kwargs.get('header', False)
        end = ' {' if not opt_header else ';'
        prefix = ''
        if o.prefix and "global" in o.prefix[0].lower():
            prefix = '__global__ '
        if o.prefix and "device" in o.prefix[0].lower():
            prefix = '__device__ '
        if o.is_function:
            return_type = self.symgen.intrinsic_type_mapper(o.return_type)
        else:
            return_type = 'void'
        opt_extern = kwargs.get('extern', False)
        declaration = [self.format_line('extern "C" {\n')] if opt_extern else []
        declaration += [self.format_line(prefix, f'{return_type} ', o.name, '(', self.join_items(arguments), ')', end)]
        return declaration

    def _subroutine_body(self, o, **kwargs):
        self.depth += 1
        skip_imports_kwargs = kwargs.pop('skip_imports', True)
        skip_argument_declarations_kwargs = kwargs.pop('skip_argument_declarations', True)
        # ...and generate the spec without imports and argument declarations
        body = [self.visit(o.spec, skip_imports=True, skip_argument_declarations=True, **kwargs)]
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
        return body

    def _subroutine_footer(self, o, **kwargs):
        postfix = ''
        opt_extern = kwargs.get('extern', False)
        footer = [self.format_line('}'), self.format_line(postfix)]
        footer += [self.format_line('\n} // extern')] if opt_extern else []
        return footer

    def visit_CallStatement(self, o, **kwargs):
        args = self.visit_all(o.arguments, **kwargs)
        if o.kwarguments:
            raise RuntimeError(f'Keyword arguments in call to {o.name} not supported in CUDA code.')
        chevron = f'<<<{",".join([str(elem) for elem in o.chevron])}>>>' if o.chevron is not None else ''
        return self.format_line(str(o.name), chevron, '(', self.join_items(args), ');')


def cudagen(ir, **kwargs):
    """
    Generate standardized CUDA code from one or many IR objects/trees.
    """
    return CudaCodegen().visit(ir, **kwargs)
