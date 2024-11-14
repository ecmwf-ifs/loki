# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation utilities to manage and inject FIELD-API boilerplate code.
"""

from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer
)
from loki.logging import warning
from loki.tools import as_tuple


__all__ = [
    'do_remove_field_api_view_updates', 'do_add_field_api_view_updates',
    'InsertFieldAPIViewsTransformer'
]


def do_remove_field_api_view_updates(routine, field_group_types, dim_object=None):
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


class InsertFieldAPIViewsTransformer(Transformer):
    """ Injects FIELD-API view updates into block loops """

    def __init__(self, dimension, field_group_types, dim_object=None, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.field_group_types = field_group_types
        self.dim_object = dim_object

    def _create_dim_update(self, scope):
        index = scope.parse_expr(self.dimension.index)
        upper = scope.parse_expr(self.dimension.upper[1])
        bindex = scope.parse_expr(self.dimension.indices[1])
        idims = scope.get_symbol(self.dim_object)
        csym = sym.ProcedureSymbol(name='UPDATE', parent=idims, scope=idims.scope)
        return ir.CallStatement(name=csym, arguments=(bindex, upper, index), kwarguments=())

    def _create_view_updates(self, section, scope):
        bindex = scope.parse_expr(self.dimension.indices[1])

        fgroup_vars = sorted(tuple(
            v for v in FindVariables(unique=True).visit(section)
            if str(v.type.dtype) in self.field_group_types
        ), key=str)
        calls = ()
        for fgvar in fgroup_vars:
            fgsym = scope.get_symbol(fgvar.name)
            csym = sym.ProcedureSymbol(name='UPDATE_VIEW', parent=fgsym, scope=fgsym.scope)
            calls += (ir.CallStatement(name=csym, arguments=(bindex,), kwarguments=()),)

        return calls

    def visit_Loop(self, loop, **kwargs):  # pylint: disable=unused-argument
        if not loop.variable == 'JKGLO':
            return loop

        scope = kwargs.get('scope')

        # Find the loop-setup assignments
        loop_symbols = self.dimension.indices
        loop_symbols += as_tuple(self.dimension.lower)
        loop_symbols += as_tuple(self.dimension.upper)
        loop_setup = tuple(
            a for a in FindNodes(ir.Assignment).visit(loop.body)
            if a.lhs in loop_symbols
        )
        idx = max(loop.body.index(a) for a in loop_setup) + 1

        # Prepend FIELD API boilerplate
        preamble = (
            ir.Comment(''), ir.Comment('! Set up thread-local view pointers')
        )
        if self.dim_object:
            preamble += (self._create_dim_update(scope),)
        preamble += self._create_view_updates(loop.body, scope)

        loop._update(body=loop.body[:idx] + preamble + loop.body[idx:])
        return loop


def do_add_field_api_view_updates(routine, dimension, field_group_types, dim_object=None):
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

    routine.body = InsertFieldAPIViewsTransformer(
        dimension=dimension, field_group_types=field_group_types,
        dim_object=dim_object
    ).visit(routine.body, scope=routine)
