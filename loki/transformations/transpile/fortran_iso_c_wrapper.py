# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki.ir import nodes as ir
from loki.types import BasicType
from loki.tools import as_tuple


__all__ = [
    'c_intrinsic_kind', 'iso_c_intrinsic_import',
    'iso_c_intrinsic_kind', 'c_struct_typedef'
]


def c_intrinsic_kind(_type, scope):
    if _type.dtype == BasicType.LOGICAL:
        return sym.Variable(name='int', scope=scope)
    if _type.dtype == BasicType.INTEGER:
        return sym.Variable(name='int', scope=scope)
    if _type.dtype == BasicType.REAL:
        kind = str(_type.kind)
        if kind.lower() in ('real32', 'c_float'):
            return sym.Variable(name='float', scope=scope)
        if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
            return sym.Variable(name='double', scope=scope)
    return None


def iso_c_intrinsic_import(scope, use_c_ptr=False):
    import_symbols = ['c_int', 'c_double', 'c_float']
    if use_c_ptr:
        import_symbols += ['c_ptr', 'c_loc']
    symbols = as_tuple(sym.Variable(name=name, scope=scope) for name in import_symbols)
    isoc_import = ir.Import(module='iso_c_binding', symbols=symbols)
    return isoc_import


def iso_c_intrinsic_kind(_type, scope, is_array=False, use_c_ptr=False):
    if _type.dtype == BasicType.INTEGER:
        return sym.Variable(name='c_int', scope=scope)

    if _type.dtype == BasicType.REAL:
        kind = str(_type.kind)
        if kind.lower() in ('real32', 'c_float'):
            return sym.Variable(name='c_float', scope=scope)
        if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double', 'c_ptr'):
            if use_c_ptr and is_array:
                return sym.Variable(name='c_ptr', scope=scope)
            return sym.Variable(name='c_double', scope=scope)

    return None


def c_struct_typedef(derived, use_c_ptr=False):
    """
    Create the :class:`TypeDef` for the C-wrapped struct definition.
    """
    typename = f'{derived.name if isinstance(derived, ir.TypeDef) else derived.dtype.name}_c'
    typedef = ir.TypeDef(name=typename.lower(), body=(), bind_c=True)  # pylint: disable=unexpected-keyword-arg
    if isinstance(derived, ir.TypeDef):
        variables = derived.variables
    else:
        variables = derived.dtype.typedef.variables
    declarations = []
    for v in variables:
        ctype = v.type.clone(kind=iso_c_intrinsic_kind(v.type, typedef, use_c_ptr=use_c_ptr))
        vnew = v.clone(name=v.basename.lower(), scope=typedef, type=ctype)
        declarations += (ir.VariableDeclaration(symbols=(vnew,)),)
    typedef._update(body=as_tuple(declarations))
    return typedef
