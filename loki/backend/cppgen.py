# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import Array
from loki.types import BasicType
from loki.backend.cgen import CCodegen, CCodeMapper, IntrinsicTypeC

__all__ = ['cppgen', 'CppCodegen', 'CppCodeMapper', 'IntrinsicTypeCpp']


class IntrinsicTypeCpp(IntrinsicTypeC):
    """
    Mapping Fortran type to corresponding C++ type.
    """

    def get_str_from_symbol_attr(self, _type, *args, **kwargs):
        if _type.dtype == BasicType.INTEGER:
            if _type.parameter:
                return 'const int'
            return 'int'
        return super().get_str_from_symbol_attr(_type, *args, **kwargs)

cpp_intrinsic_type = IntrinsicTypeCpp()


class CppCodeMapper(CCodeMapper):
    """
    A :class:`StringifyMapper`-derived visitor for Pymbolic expression trees that converts an
    expression to a string adhering to standardized C++.
    """
    # pylint: disable=abstract-method, unused-argument

    def map_inline_call(self, expr, enclosing_prec, *args, **kwargs):
        if expr.function.name.lower() == 'present':
            return self.format('%s', expr.parameters[0].name)
        return super().map_inline_call(expr, enclosing_prec, *args, **kwargs)


class CppCodegen(CCodegen):
    """
    Tree visitor to generate standardized C++ code from IR.
    """

    def __init__(self, depth=0, indent='  ', linewidth=90, **kwargs):
        symgen = kwargs.pop('symgen', CppCodeMapper(cpp_intrinsic_type))

        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         symgen=symgen, **kwargs)

    def _subroutine_argument_keyword(self, a):
        if isinstance(a, Array) and a.type.intent.lower() == "in":
            return 'const '
        return ''

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

    def _subroutine_optional_args(self, a):
        if a.type.optional:
            return ' = nullptr'
        return ''

def cppgen(ir, **kwargs):
    """
    Generate standardized C++ code from one or many IR objects/trees.
    """
    return CppCodegen().visit(ir, **kwargs)
