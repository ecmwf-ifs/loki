# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import Array
from loki.types import BasicType, DerivedType
from loki.backend.cgen import CCodegen, CCodeMapper, IntrinsicTypeC

__all__ = ['cppgen', 'CppCodegen', 'CppCodeMapper', 'IntrinsicTypeCpp']


class IntrinsicTypeCpp(IntrinsicTypeC):

    def c_intrinsic_type(self, _type, *args, **kwargs):
        if _type.dtype == BasicType.INTEGER:
            if _type.parameter:
                return 'const int'
            return 'int'
        return super().c_intrinsic_type(_type, *args, **kwargs)

cpp_intrinsic_type = IntrinsicTypeCpp()


class CppCodeMapper(CCodeMapper):
    # pylint: disable=abstract-method, unused-argument
    pass


class CppCodegen(CCodegen):
    """
    Tree visitor to generate standardized C++ code from IR.
    """
    standard_imports = ['stdio.h', 'stdbool.h', 'float.h', 'math.h']

    def __init__(self, depth=0, indent='  ', linewidth=90, **kwargs):
        symgen = kwargs.get('symgen', CppCodeMapper(cpp_intrinsic_type))
        line_cont = kwargs.get('line_cont', '\n{}  '.format)

        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont=line_cont, symgen=symgen)

    def _subroutine_header(self, o, **kwargs):
        header = super()._subroutine_header(o, **kwargs)
        return header

    def _subroutine_arguments(self, o, **kwargs):
        # opt_extern = kwargs.get('extern', False)
        # if opt_extern:
        #     return super()._subroutine_arguments(o, **kwargs)
        var_keywords = []
        pass_by = []
        for a in o.arguments:
            if isinstance(a, Array) > 0 and a.type.intent.lower() == "in":
                var_keywords += ['const ']
            else:
                var_keywords += ['']
            if isinstance(a, Array) > 0:
                pass_by += ['* restrict ']
            elif isinstance(a.type.dtype, DerivedType):
                pass_by += ['*']
            elif a.type.pointer:
                pass_by += ['*']
            else:
                pass_by += ['']
        return pass_by, var_keywords

    def _subroutine_declaration(self, o, **kwargs):
        opt_extern = kwargs.get('extern', False)
        declaration = [self.format_line('extern "C" {\n')] if opt_extern else []
        declaration += super()._subroutine_declaration(o, **kwargs)
        return declaration

    def _subroutine_body(self, o, **kwargs):
        body = super()._subroutine_body(o, **kwargs)
        return body

    def _subroutine_footer(self, o, **kwargs):
        opt_extern = kwargs.get('extern', False)
        footer = super()._subroutine_footer(o, **kwargs)
        footer += [self.format_line('\n} // extern')] if opt_extern else []
        return footer


def cppgen(ir, **kwargs):
    """
    Generate standardized C++ code from one or many IR objects/trees.
    """
    return CppCodegen().visit(ir, **kwargs)
