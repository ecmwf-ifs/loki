# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation utilities to manage and inject FIELD-API boilerplate code.
"""

from enum import Enum
from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer
)
from loki.scope import Scope
from loki.logging import warning
from loki.tools import as_tuple

__all__ = [
    'remove_field_api_view_updates', 'add_field_api_view_updates', 'get_field_type',
    'field_get_device_data', 'field_sync_host', 'FieldAPITransferType'
]


def remove_field_api_view_updates(routine, field_group_types, dim_object=None):
    """
    Remove FIELD API boilerplate calls for view updates of derived types.

    This utility is intended to remove the IFS-specific group type
    objects that provide block-scope view pointers to deep kernel
    trees. It will remove all calls to ``UPDATE_VIEW`` on derive-type
    objects with the respective types.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The routine from which to remove FIELD API update calls
    field_group_types : tuple of str
        List of names of the derived types of "field group" objects to remove
    dim_object : str, optional
        Optional name of the "dimension" object; if provided it will remove the
        call to ``<dim>%UPDATE(...)`` accordingly.
    """
    field_group_types = as_tuple(str(fgt).lower() for fgt in field_group_types)

    class RemoveFieldAPITransformer(Transformer):

        def visit_CallStatement(self, call, **kwargs):  # pylint: disable=unused-argument

            if '%update_view' in str(call.name).lower():
                if not str(call.name.parent.type.dtype).lower() in field_group_types:
                    warning(f'[Loki::ControlFlow] Removing {call.name} call, but not in field group types!')

                return None

            if dim_object and f'{dim_object}%update'.lower() in str(call.name).lower():
                return None

            return call

        def visit_Assignment(self, assign, **kwargs):  # pylint: disable=unused-argument
            if str(assign.lhs.type.dtype).lower() in field_group_types:
                warning(f'[Loki::ControlFlow] Found LHS field group assign: {assign}')
            return assign

        def visit_Loop(self, loop, **kwargs):
            loop = self.visit_Node(loop, **kwargs)
            return loop if loop.body else None

        def visit_Conditional(self, cond, **kwargs):
            cond = super().visit_Node(cond, **kwargs)
            return cond if cond.body else None

    routine.body = RemoveFieldAPITransformer().visit(routine.body)


def add_field_api_view_updates(routine, dimension, field_group_types, dim_object=None):
    """
    Adds FIELD API boilerplate calls for view updates.

    The provided :any:`Dimension` object describes the local loop variables to
    pass to the respective update calls. In particular, ``dimension.indices[1]``
    is used to denote the block loop index that is passed to ``UPDATE_VIEW()``
    calls on field group object. The list of type names ``field_group_types``
    is used to identify for which objcets the view update calls get added.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The routine from which to remove FIELD API update calls
    dimension : :any:`Dimension`
        The dimension object describing the block loop variables.
    field_group_types : tuple of str
        List of names of the derived types of "field group" objects to remove
    dim_object : str, optional
        Optional name of the "dimension" object; if provided it will remove the
        call to ``<dim>%UPDATE(...)`` accordingly.
    """

    def _create_dim_update(scope, dim_object):
        index = scope.parse_expr(dimension.index)
        upper = scope.parse_expr(dimension.upper[1])
        bindex = scope.parse_expr(dimension.indices[1])
        idims = scope.get_symbol(dim_object)
        csym = sym.ProcedureSymbol(name='UPDATE', parent=idims, scope=idims.scope)
        return ir.CallStatement(name=csym, arguments=(bindex, upper, index), kwarguments=())

    def _create_view_updates(section, scope):
        bindex = scope.parse_expr(dimension.indices[1])

        fgroup_vars = sorted(tuple(
            v for v in FindVariables(unique=True).visit(section)
            if str(v.type.dtype) in field_group_types
        ), key=str)
        calls = ()
        for fgvar in fgroup_vars:
            fgsym = scope.get_symbol(fgvar.name)
            csym = sym.ProcedureSymbol(name='UPDATE_VIEW', parent=fgsym, scope=fgsym.scope)
            calls += (ir.CallStatement(name=csym, arguments=(bindex,), kwarguments=()),)

        return calls

    class InsertFieldAPIViewsTransformer(Transformer):
        """ Injects FIELD-API view updates into block loops """

        def visit_Loop(self, loop, **kwargs):  # pylint: disable=unused-argument
            if not loop.variable == 'JKGLO':
                return loop

            scope = kwargs.get('scope')

            # Find the loop-setup assignments
            _loop_symbols = dimension.indices
            _loop_symbols += as_tuple(dimension.lower) + as_tuple(dimension.upper)
            loop_setup = tuple(
                a for a in FindNodes(ir.Assignment).visit(loop.body)
                if a.lhs in _loop_symbols
            )
            idx = max(loop.body.index(a) for a in loop_setup) + 1

            # Prepend FIELD API boilerplate
            preamble = (
                ir.Comment(''), ir.Comment('! Set up thread-local view pointers')
            )
            if dim_object:
                preamble += (_create_dim_update(scope, dim_object=dim_object),)
            preamble += _create_view_updates(loop.body, scope)

            loop._update(body=loop.body[:idx] + preamble + loop.body[idx:])
            return loop

    routine.body = InsertFieldAPIViewsTransformer().visit(routine.body, scope=routine)


def get_field_type(a: sym.Array) -> sym.DerivedType:
    """
    Returns the corresponding FIELD API type for an array.

    This function is IFS specific and assumes that the
    type is an array declared with one of the IFS type specifiers, e.g. KIND=JPRB
    """
    type_map = ["jprb",
                "jpit",
                "jpis",
                "jpim",
                "jpib",
                "jpia",
                "jprt",
                "jprs",
                "jprm",
                "jprd",
                "jplm"]
    type_name = a.type.kind.name

    assert type_name.lower() in type_map, ('Error array type kind is: '
                                           f'"{type_name}" which is not a valid IFS type specifier')
    rank = len(a.shape)
    field_type = sym.DerivedType(name="field_" + str(rank) + type_name[2:4].lower())
    return field_type



class FieldAPITransferType(Enum):
    READ_ONLY = 1
    READ_WRITE = 2
    WRITE_ONLY = 3


def field_get_device_data(field_ptr, dev_ptr, transfer_type: FieldAPITransferType, scope: Scope):
    """
    Utility function to generate a :any:`CallStatement` corresponding to a Field API
    ``GET_DEVICE_DATA`` call.
    
    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``GET_DEVICE_DATA`` from.
    dev_ptr: :any:`Array`
        Device pointer array
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which ``GET_DEVICE_DATA`` method to call.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    """
    if not isinstance(transfer_type, FieldAPITransferType):
        raise TypeError(f"transfer_type must be of type FieldAPITransferType, but is of type {type(transfer_type)}")
    if transfer_type == FieldAPITransferType.READ_ONLY:
        suffix = 'RDONLY'
    elif transfer_type == FieldAPITransferType.READ_WRITE:
        suffix = 'RDWR'
    elif transfer_type == FieldAPITransferType.WRITE_ONLY:
        suffix = 'WRONLY'
    else:
        suffix = ''
    procedure_name = 'GET_DEVICE_DATA_' + suffix
    return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope),
                            arguments=(dev_ptr.clone(dimensions=None),), )


def field_sync_host(field_ptr, scope):
    """
    Utility function to generate a :any:`CallStatement` corresponding to a Field API
    ``SYNC_HOST`` call.
    
    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``SYNC_HOST`` from.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    """

    procedure_name = 'SYNC_HOST_RDWR'
    return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope), arguments=())
