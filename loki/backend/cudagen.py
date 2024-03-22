# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from operator import gt
from pymbolic.mapper.stringifier import (
    PREC_UNARY, PREC_LOGICAL_OR, PREC_LOGICAL_AND, PREC_NONE, PREC_CALL
)

from loki.visitors import Stringifier, FindNodes
from loki.ir import Import, Pragma
from loki.expression import LokiStringifyMapper, Array, symbolic_op, Literal
from loki.types import BasicType, SymbolAttributes, DerivedType
from loki.backend.cppgen import CppCodegen, CppCodeMapper

__all__ = ['cudagen', 'CudaCodegen', 'CudaCodeMapper']

def c_intrinsic_type(_type):
    if _type.dtype == BasicType.LOGICAL:
        return 'int'
    if _type.dtype == BasicType.INTEGER:
        if _type.parameter:
            return 'const int'
        return 'int'
    if _type.dtype == BasicType.REAL:
        if str(_type.kind) in ['real32']:
            return 'float'
        return 'double'
    raise ValueError(str(_type))

class CudaCodeMapper(CppCodeMapper): # LokiStringifyMapper):

    pass

class CudaCodegen(CppCodegen): # Stringifier):
    """
    ...
    """
    
    standard_imports = ['stdio.h', 'stdbool.h', 'float.h', 'math.h', 'cuda.h', 'cuda_runtime.h']

    def __init__(self, depth=0, indent='  ', linewidth=90, **kwargs):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont='\n{}  '.format, symgen=CudaCodeMapper())

    def visit_Subroutine(self, o, **kwargs):
        """
        Format as:

          ...imports...
          int <name>(<args>) {
            ...spec without imports and argument declarations...
            ...body...
          }
        """
        # Some boilerplate imports...
        # standard_imports = ['stdio.h', 'stdbool.h', 'float.h', 'math.h']
        header = [self.format_line('#include <', name, '>') for name in self.standard_imports]

        # ...and imports from the spec
        spec_imports = FindNodes(Import).visit(o.spec)
        header += [self.visit(spec_imports, **kwargs)]

        ##
        # ftype = 'FUNCTION' if o.is_function else 'SUBROUTINE'
        # prefix = self.join_items(o.prefix, sep=' ')
        # if o.prefix:
        #     prefix += ' '
        # arguments = self.join_items(o.argnames)
        # result = f' RESULT({o.result_name})' if o.result_name else ''
        return_var = None
        if o.is_function:
            return_var_name = o.name.replace("_c", "")
            if return_var_name in o.variable_map:
                return_var = o.variable_map[return_var_name]

        subroutine_prefix = o.prefix[0].lower() if o.prefix else ''
        if 'global' in subroutine_prefix or 'device' in subroutine_prefix:
            is_device_function = True
        else:
            is_device_function = False
        # Generate header with argument signature
        aptr = []
        bptr = []
        for a in o.arguments:
            if isinstance(a, Array) > 0:
                if a.type.intent.lower() == "in":
                    bptr += ['const ']
                else:
                    bptr += ['']
                if is_device_function: # "global" in o.prefix[0].lower():
                    aptr += ['* '] # ['* __restrict__ '] # ['* restrict '] # v_
                else:
                    aptr += ['* ']
            elif isinstance(a.type.dtype, DerivedType):
                aptr += ['*']
                bptr += ['']
            elif a.type.pointer:
                aptr += ['*']
                bptr += ['']
            else:
                aptr += ['']
                bptr += ['']
        arguments = [f'{b}{self.visit(a.type, **kwargs)} {p}{a.name.lower()}'
                     for b, a, p in zip(bptr, o.arguments, aptr)]
        # header += [self.format_line('void ', o.name, '(', self.join_items(arguments), ') {')]
        # if is_device_function:
        
        prefix = ''
        extern = ''
        postfix = ''
        skip_decls = False
        # global_whatever = False
        return_type_specifier = c_intrinsic_type(return_var.type) + ' ' if return_var is not None else 'void '
        if o.prefix:
            if "global" in o.prefix[0].lower():
                prefix = '__global__ '
                header += [self.format_line(prefix, 'void ', '__launch_bounds__(128, 1) ', o.name, '(', self.join_items(arguments), ');')]
                header += [self.format_line('#include "', o.name, '_launch.h', '"')]
            elif "header_only" in o.prefix[0].lower():
                if "device" in o.prefix[0].lower():
                    prefix = "__device__ "
                # header += [self.format_line(extern), self.format_line(prefix, 'void ', o.name, '(', self.join_items(arguments), ');')]
                header += [self.format_line(extern), self.format_line(prefix, return_type_specifier, o.name, '(', self.join_items(arguments), ');')]
                return self.join_lines(*header)
            elif "device" in o.prefix[0].lower():
                prefix = "__device__ "
            elif "extern_c" in o.prefix[0].lower():
                extern = 'extern "C" {'
                postfix = '}'
                skip_decls = True
        
        # header += [self.format_line(extern), self.format_line(prefix, 'void ', o.name, '(', self.join_items(arguments), ') {')]
        header += [self.format_line(extern), self.format_line(prefix, return_type_specifier, o.name, '(', self.join_items(arguments), ') {')]

        self.depth += 1

        # ...and generate the spec without imports and argument declarations
        skip_imports = kwargs.pop('skip_imports', True)
        skip_decls = kwargs.pop('skip_decls', skip_decls)
        skip_argument_declarations = kwargs.pop('skip_argument_declarations', True)
        body = [self.visit(o.spec, skip_imports=skip_imports, skip_decls=skip_decls, skip_argument_declarations=True, **kwargs)]
        # body = [self.visit(o.spec, skip_imports=True, skip_decls=skip_decls, skip_argument_declarations=True, **kwargs)]

        # if skip_decls:
        #     body += [self.format_line('printf("executing c launch ...\\n");')]
        #     body += [self.format_line('printf("ngptot: %i,  nproma: %i\\n", ngptot, nproma);')]
        #     # pragmas = FindNodes(Pragma).visit(o.spec)
        #     # for pragma in pragmas:
        #     #     if pragma.keyword == "loki" and "griddim" in pragma.content:
        #     #         body += [self.format_line(f'{pragma.content.replace("griddim", "", 1)}')]
        #     #     if pragma.keyword == "loki" and "blockdim" in pragma.content:
        #     #         body += [self.format_line(f'{pragma.content.replace("blockdim", "", 1)}')]

        # Fill the body
        body += [self.visit(o.body, **kwargs)]
        
        if return_var is not None:
            body += [self.format_line(f'return {return_var.name.lower()};')]

        if skip_decls:
            body += [self.format_line('cudaDeviceSynchronize();')]
        # body += [self.format_line('return 0;')]

        # Close everything off
        self.depth -= 1
        footer = [self.format_line('}'), self.format_line(postfix)]

        return self.join_lines(*header, *body, *footer)

    def visit_CallStatement(self, o, **kwargs):
        args = self.visit_all(o.arguments, **kwargs)
        assert not o.kwarguments
        if o.chevron is not None:
            chevron = f"<<<{','.join([str(elem) for elem in o.chevron])}>>>"
            # args = list(map(lambda x: x.replace('yrecldp', 'd_yrecldp'), list(args)))
        else:
            chevron = ""
        try:
            return self.format_line(str(o.name).lower(), chevron, '(', self.join_items(args), ');')
        except:
            return self.format_line(o.name, chevron, '(', self.join_items(args), ');')


def cudagen(ir):
    """
    ...
    """
    return CudaCodegen().visit(ir)
