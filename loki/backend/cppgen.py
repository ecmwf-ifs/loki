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
from loki.ir import Import
from loki.expression import LokiStringifyMapper, Array, symbolic_op, Literal
from loki.types import BasicType, SymbolAttributes, DerivedType

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
    
    pass

class CppCodegen(CCodegen): # Stringifier):
    """
    ...
    """
    
    standard_imports = ['stdio.h', 'stdbool.h', 'float.h', 'math.h', 'cuda.h', 'cuda_runtime.h']

    def __init__(self, depth=0, indent='  ', linewidth=90):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont='\n{}  '.format, symgen=CppCodeMapper())

def cppgen(ir):
    """
    ...
    """
    return CppCodegen().visit(ir)

