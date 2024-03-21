# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
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
from loki.backend.cgen import CCodegen, CCodeMapper

__all__ = ['cppgen', 'CppCodegen', 'CppCodeMapper']


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


class CppCodeMapper(CCodeMapper): # LokiStringifyMapper):

    # TODO: only due to c_intrinsic_type ...
    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            _type = SymbolAttributes(BasicType.REAL, kind=expr.kind)
            return f'({c_intrinsic_type(_type)}) {str(expr.value)}'
        return str(expr.value)

    # TODO: only due to c_intrinsic_type ...
    def map_int_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            _type = SymbolAttributes(BasicType.INTEGER, kind=expr.kind)
            return f'({c_intrinsic_type(_type)}) {str(expr.value)}'
        return str(expr.value)

    # TODO: only due to c_intrinsic_type ...
    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        _type = SymbolAttributes(BasicType.from_fortran_type(expr.name), kind=expr.kind)
        expression = self.parenthesize_if_needed(
            self.join_rec('', expr.parameters, PREC_NONE, *args, **kwargs),
            PREC_CALL, PREC_NONE)
        return self.parenthesize_if_needed(
            self.format('(%s) %s', c_intrinsic_type(_type), expression), enclosing_prec, PREC_CALL)

    def map_variable_symbol(self, expr, enclosing_prec, *args, **kwargs):
        # ptr = '*' if expr.type and expr.type.pointer else ''
        ptr = ''
        if expr.parent in ["threadIdx", "blockIdx"]:
            parent = self.rec(expr.parent, PREC_NONE, *args, **kwargs)
            return self.format('%s%s.%s', ptr, parent, expr.basename)
        elif expr.parent is not None:
            parent = self.parenthesize(self.rec(expr.parent, PREC_NONE, *args, **kwargs))
            return self.format('%s%s.%s', ptr, parent, expr.basename)
        return self.format('%s%s', ptr, expr.name)
    
    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        name_str = self.rec(expr.aggregate, PREC_NONE, *args, **kwargs)
        if expr.aggregate.type is not None:
            index_str = ''
            for index in expr.index_tuple:
                d = self.format(self.rec(index, PREC_NONE, *args, **kwargs))
                if d:
                    index_str += self.format('[%s]', d)
            return self.format('%s%s', name_str, index_str)
        else:
            return self.format('%s', name_str)
        # TODO: used to be like that: ...
        # try:
        #     name_str = self.rec(expr.aggregate, PREC_NONE, *args, **kwargs)
        #     if expr.aggregate.type.pointer and name_str.startswith('*'):
        #         # Strip the pointer '*' because subscript dereference
        #         name_str = name_str[1:]
        #     index_str = ''
        #     for index in expr.index_tuple:
        #         d = self.format(self.rec(index, PREC_NONE, *args, **kwargs))
        #         if d:
        #             index_str += self.format('[%s]', d)
        #     return self.format('%s%s', name_str, index_str)
        # except:
        #     return self.format('%s', name_str)

    map_deferred_type_symbol = map_variable_symbol

class CppCodegen(CCodegen): # Stringifier):
    """
    ...
    """
    
    standard_imports = ['stdio.h', 'stdbool.h', 'float.h', 'math.h']

    def __init__(self, depth=0, indent='  ', linewidth=90, **kwargs):
        symgen = kwargs.get('symgen', CppCodeMapper())
        line_cont = kwargs.get('line_cont', '\n{}  '.format)
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont=line_cont, symgen=symgen)
        # super().__init__(depth=depth, indent=indent, linewidth=linewidth,
        #                  line_cont='\n{}  '.format, symgen=CppCodeMapper())

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

        if o.prefix:
            if "global" in o.prefix[0].lower():
                prefix = '__global__ '
                header += [self.format_line(prefix, 'void ', '__launch_bounds__(128, 1) ', o.name, '(', self.join_items(arguments), ');')]
                header += [self.format_line('#include "', o.name, '_launch.h', '"')]
            elif "header_only" in o.prefix[0].lower():
                if "device" in o.prefix[0].lower():
                    prefix = "__device__ "
                header += [self.format_line(extern), self.format_line(prefix, 'void ', o.name, '(', self.join_items(arguments), ');')]
                return self.join_lines(*header)
            elif "device" in o.prefix[0].lower():
                prefix = "__device__ "
            elif "extern_c" in o.prefix[0].lower():
                extern = 'extern "C" {'
                postfix = '}'
                skip_decls = True
        
        header += [self.format_line(extern), self.format_line(prefix, 'void ', o.name, '(', self.join_items(arguments), ') {')]

        self.depth += 1

        # ...and generate the spec without imports and argument declarations
        body = [self.visit(o.spec, skip_imports=True, skip_decls=skip_decls, skip_argument_declarations=True, **kwargs)]

        if skip_decls:
            # body += [self.format_line('printf("executing c launch ...\\n");')]
            # body += [self.format_line('printf("ngptot: %i,  nproma: %i\\n", ngptot, nproma);')]
            # TODO: still necessary?
            pragmas = FindNodes(Pragma).visit(o.spec)
            for pragma in pragmas:
                if pragma.keyword == "loki" and "griddim" in pragma.content:
                    body += [self.format_line(f'{pragma.content.replace("griddim", "", 1)}')]
                if pragma.keyword == "loki" and "blockdim" in pragma.content:
                    body += [self.format_line(f'{pragma.content.replace("blockdim", "", 1)}')]

        # Fill the body
        body += [self.visit(o.body, **kwargs)]
        if skip_decls:
            body += [self.format_line('cudaDeviceSynchronize();')]
        # body += [self.format_line('return 0;')]

        # Close everything off
        self.depth -= 1
        footer = [self.format_line('}'), self.format_line(postfix)]

        return self.join_lines(*header, *body, *footer)

    def visit_CommentBlock(self, o, **kwargs):
        if kwargs.pop('skip_decls', False):
            return None
        super().visit_CommentBlock(o, **kwargs)

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

    # TODO: only due to c_intrinsic_type ...
    def visit_SymbolAttributes(self, o, **kwargs):  # pylint: disable=unused-argument
        if isinstance(o.dtype, DerivedType):
            # return f'struct {o.dtype.name}'
            return f'{o.dtype.name}'
        return c_intrinsic_type(o)


def cppgen(ir):
    """
    ...
    """
    return CppCodegen().visit(ir)

