# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Utilities to change indices and indexing in array expressions. """

from loki.batch import Transformation, ProcedureItem
from loki.expression import symbols as sym, simplify, is_constant
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, SubstituteExpressions
)
from loki.tools import as_tuple
from loki.transformations.inline import inline_constant_parameters


__all__ = [
    'shift_to_zero_indexing', 'invert_array_indices',
    'normalize_range_indexing', 'flatten_arrays',
    'normalize_array_shape_and_access', 'LowerConstantArrayIndices',
]


def shift_to_zero_indexing(routine, ignore=None):
    """
    Shift all array indices to adjust to 0-based indexing conventions (eg. for C or Python)

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which the array dimensions should be shifted
    ignore : list of str
        List of variable names for which, if found in the dimension expression
        of an array subscript, that dimension is not shifted to zero.
    """
    ignore = as_tuple(ignore)
    vmap = {}
    for v in FindVariables(unique=False).visit(routine.body):
        if isinstance(v, sym.Array):
            new_dims = []
            for d in v.dimensions:
                if isinstance(d, sym.RangeIndex):
                    start = d.start - sym.Literal(1) if d.start is not None else None
                    # no shift for stop because Python ranges are [start, stop)
                    new_dims += [sym.RangeIndex((start, d.stop, d.step))]
                else:
                    if ignore and any(var in ignore for var in FindVariables().visit(d)):
                        new_dims += [d]
                    else:
                        new_dims += [d - sym.Literal(1)]
            vmap[v] = v.clone(dimensions=as_tuple(new_dims))
    routine.body = SubstituteExpressions(vmap).visit(routine.body)


def invert_array_indices(routine):
    """
    Invert data/loop accesses from column to row-major

    TODO: Take care of the indexing shift between C and Fortran.
    Basically, we are relying on the CGen to shift the iteration
    indices and dearly hope that nobody uses the index's value.
    """
    # Invert array indices in the routine body
    vmap = {}
    for v in FindVariables(unique=True).visit(routine.body):
        if isinstance(v, sym.Array):
            rdim = as_tuple(reversed(v.dimensions))
            vmap[v] = v.clone(dimensions=rdim)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Invert variable and argument dimensions for the automatic cast generation
    for v in routine.variables:
        if isinstance(v, sym.Array):
            rdim = as_tuple(reversed(v.dimensions))
            if v.shape:
                rshape = as_tuple(reversed(v.shape))
                vmap[v] = v.clone(dimensions=rdim, type=v.type.clone(shape=rshape))
            else:
                vmap[v] = v.clone(dimensions=rdim)
    routine.variables = [vmap.get(v, v) for v in routine.variables]


def normalize_range_indexing(routine):
    """
    Replace the ``(1:size)`` indexing in array sizes that OMNI introduces.
    """
    def is_one_index(dim):
        return isinstance(dim, sym.RangeIndex) and dim.lower == 1 and dim.step is None

    vmap = {}
    for v in routine.variables:
        if isinstance(v, sym.Array):
            new_dims = [d.upper if is_one_index(d) else d for d in v.dimensions]
            new_shape = [d.upper if is_one_index(d) else d for d in v.shape]
            new_type = v.type.clone(shape=as_tuple(new_shape))
            vmap[v] = v.clone(dimensions=as_tuple(new_dims), type=new_type)
    routine.variables = [vmap.get(v, v) for v in routine.variables]


def flatten_arrays(routine, order='F', start_index=1):
    """
    Flatten arrays, converting multi-dimensional arrays to
    one-dimensional arrays.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which the variables should be promoted.
    order : str
        Assume Fortran (F) vs. C memory/array order.
    start_index : int
        Assume array indexing starts with `start_index`.
    """
    def new_dims(dim, shape):
        if all(_dim == sym.RangeIndex((None, None)) for _dim in dim):
            return None
        if len(dim) > 1:
            if isinstance(shape[-2], sym.RangeIndex):
                raise TypeError(f'Resolve shapes being of type RangeIndex, e.g., "{shape[-2]}" before flattening!')
            _dim = (sym.Sum((dim[-2], sym.Product((shape[-2], dim[-1] - start_index)))),)
            new_dim = dim[:-2]
            new_dim += _dim
            return new_dims(new_dim, shape[:-1])
        return dim

    if order == 'C':
        array_map = {
            var: var.clone(dimensions=new_dims(var.dimensions[::-1], var.shape[::-1]))
            for var in FindVariables().visit(routine.body)
            if isinstance(var, sym.Array) and var.shape and len(var.shape)
        }
    elif order == 'F':
        array_map = {
            var: var.clone(dimensions=new_dims(var.dimensions, var.shape))
            for var in FindVariables().visit(routine.body)
            if isinstance(var, sym.Array) and var.shape and len(var.shape)
        }
    else:
        raise ValueError(f'Unsupported array order "{order}"')

    routine.body = SubstituteExpressions(array_map).visit(routine.body)

    routine.variables = [v.clone(dimensions=as_tuple(sym.Product(v.shape)),
                                 type=v.type.clone(shape=as_tuple(sym.Product(v.shape))))
                         if isinstance(v, sym.Array) else v for v in routine.variables]


def normalize_array_shape_and_access(routine):
    """
    Shift all arrays to start counting at "1"
    """
    def is_explicit_range_index(dim):
        # return False if assumed sized array or lower dimension equals to 1
        # return (isinstance(dim, sym.RangeIndex) and not dim.lower == 1 and not dim is None
        #             and not dim.lower is None and not dim.upper is None)
        return (isinstance(dim, sym.RangeIndex)
                and not (dim.lower == 1 or dim.lower is None or dim.upper is None))

    vmap = {}
    for v in FindVariables(unique=False).visit(routine.body):
        if isinstance(v, sym.Array):
            # skip if e.g., `array(len)`, passed as `call routine(array)`
            if not v.dimensions:
                continue
            new_dims = []
            for i, d in enumerate(v.shape):
                if is_explicit_range_index(d):
                    if isinstance(v.dimensions[i], sym.RangeIndex):
                        start = simplify(v.dimensions[i].start - d.start + 1) if d.start is not None else None
                        stop = simplify(v.dimensions[i].stop - d.start + 1) if d.stop is not None else None
                        new_dims += [sym.RangeIndex((start, stop, d.step))]
                    else:
                        start = simplify(v.dimensions[i] - d.start + 1) if d.start is not None else None
                        new_dims += [start]
                else:
                    new_dims += [v.dimensions[i]]
            if new_dims:
                vmap[v] = v.clone(dimensions=as_tuple(new_dims))
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    vmap = {}
    for v in routine.variables:
        if isinstance(v, sym.Array):
            new_dims = [sym.RangeIndex((1, simplify(d.upper - d.lower + 1)))
                if is_explicit_range_index(d) else d for d in v.dimensions]
            new_shape = [sym.RangeIndex((1, simplify(d.upper - d.lower + 1)))
                if is_explicit_range_index(d) else d for d in v.shape]
            new_type = v.type.clone(shape=as_tuple(new_shape))
            vmap[v] = v.clone(dimensions=as_tuple(new_dims), type=new_type)
    routine.variables = [vmap.get(v, v) for v in routine.variables]
    normalize_range_indexing(routine)


class LowerConstantArrayIndices(Transformation):
    """
    A transformation to pass/lower constant array indices down the call tree.

    For example, the following code:

    .. code-block:: fortran

      subroutine driver(...)
        real, intent(inout) :: var(nlon,nlev,5,nb)
        do ibl=1,10
          call kernel(var(:, :, 1, ibl), var(:, :, 2:5, ibl))
        end do
      end subroutine driver

      subroutine kernel(var1, var2)
        real, intent(inout) :: var1(nlon, nlev)
        real, intent(inout) :: var2(nlon, nlev, 4)
        var1(:, :) = ...
        do jk=1,nlev
          do jl=1,nlon
            var1(jl, jk) = ...
            do jt=1,4
              var2(jl, jk, jt) = ...
            enddo
          enddo
        enddo
      end subroutine kernel

    is transformed to:

    .. code-block:: fortran

      subroutine driver(...)
        real, intent(inout) :: var(nlon,nlev,5,nb)
        do ibl=1,10
          call kernel(var(:, :, :, ibl), var(:, :, :, ibl))
        end do
      end subroutine driver

      subroutine kernel(var1, var2)
        real, intent(inout) :: var1(nlon, nlev, 5)
        real, intent(inout) :: var2(nlon, nlev, 5)
        var1(:, :, 1) = ...
        do jk=1,nlev
          do jl=1,nlon
            var1(jl, jk, 1) = ...
            do jt=1,4
              var2(jl, jk, jt + 2 + -1) = ...
            enddo
          enddo
        enddo
      end subroutine kernel

    Parameters
    ----------
    recurse_to_kernels: bool
        Recurse to kernels, thus lower constant array indices below the driver level for nested
        kernel calls (default: `True`).
    inline_external_only: bool
        Inline only external constant expressions or all of them (default: `False`)
    """

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, recurse_to_kernels=False, inline_external_only=True):
        self.recurse_to_kernels = recurse_to_kernels
        self.inline_external_only = inline_external_only

    @staticmethod
    def explicit_dimensions(routine):
        """
        Make dimensions of arrays explicit within :any:`Subroutine` ``routine``.
        E.g., convert two-dimensional array ``arr2d`` to ``arr2d(:,:)`` or
        ``arr3d`` to ``arr3d(:,:,:)``.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to check
        """
        arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
        array_map = {}
        for array in arrays:
            if not array.dimensions:
                new_dimensions = (sym.RangeIndex((None, None)),) * len(array.shape)
                array_map[array] = array.clone(dimensions=new_dimensions)
        routine.body = SubstituteExpressions(array_map).visit(routine.body)

    @staticmethod
    def is_constant_dim(dim):
        """
        Check whether dimension dim is constant, thus, either a constant
        value or a constant range index.

        Parameters
        ----------
        dim: :py:class:`pymbolic.primitives.Expression`
        """
        if is_constant(dim):
            return True
        if isinstance(dim, sym.RangeIndex)\
                and all(child is not None and is_constant(child) for child in dim.children[:-1]):
            return True
        return False

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))
        if role == 'driver' or self.recurse_to_kernels:
            inline_constant_parameters(routine, external_only=self.inline_external_only)
            self.process(routine, targets)

    def process(self, routine, targets):
        """
        Process the driver and possibly kernels
        """
        dispatched_routines = ()
        offset_map = {}
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if str(call.name).lower() not in targets:
                continue
            # skip already dispatched routines but still update the call signature
            if call.routine in dispatched_routines:
                self.update_call_signature(call)
                continue
            # explicit array dimensions for the callee
            self.explicit_dimensions(call.routine)
            dispatched_routines += (call.routine,)
            # create the offset map and apply to call and callee
            offset_map[call.routine.name.lower()] = self.create_offset_map(call)
            self.process_callee(call.routine, offset_map[call.routine.name.lower()])
            self.update_call_signature(call)

    def update_call_signature(self, call):
        """
        Replace constant indices for call arguments being arrays with ':' and update the call.
        """
        new_args = [arg.clone(dimensions=\
                tuple(sym.RangeIndex((None, None)) if self.is_constant_dim(d) else d for d in arg.dimensions))\
                if isinstance(arg, sym.Array) else arg for arg in call.arguments]
        new_kwargs = [(kw[0], kw[1].clone(dimensions=\
                tuple(sym.RangeIndex((None, None)) if self.is_constant_dim(d) else d for d in kw[1].dimensions)))\
                if isinstance(kw[1], sym.Array) else kw for kw in call.kwarguments]
        call._update(arguments=as_tuple(new_args), kwarguments=as_tuple(new_kwargs))

    def create_offset_map(self, call):
        """
        Create map/dictionary for arguments with constant array indices.
        
        For, e.g., 

        integer :: arg(len1, len2, len3, len4)
        call kernel(..., arg(:, 2, 4:6, i), ...)

        offset_map[arg] = {
            0: (0, None, None),  # same index as before, no offset
            1: (None, 1, len2),  # New index, offset 1, size of the dimension is len2
            2: (1, 4, len3),     # Used to be position 1, offset 4, size of the dimension is len3
            3: (-1, None, None), # disregard as this is neither constant nor passed to callee
        }
        """
        offset_map = {}
        for routine_arg, call_arg in call.arg_iter():
            if not isinstance(routine_arg, sym.Array):
                continue
            offset_map[routine_arg.name] = {}
            current_index = 0
            for i, dim in enumerate(call_arg.dimensions):
                if self.is_constant_dim(dim):
                    if isinstance(dim, sym.RangeIndex):
                        # constant array index is e.g. '1:3' or '5:10'
                        offset_map[routine_arg.name][i] = (current_index, dim.children[0], call_arg.shape[i])
                    else:
                        # constant array index is e.g., '1' or '42'
                        offset_map[routine_arg.name][i] = (None, dim, call_arg.shape[i])
                        current_index -= 1
                else:
                    if not isinstance(dim, sym.RangeIndex):
                        # non constant array index is a variable e.g. 'jl'
                        offset_map[routine_arg.name][i] = (-1, None, None)
                        current_index -= 1
                    else:
                        # non constant array index is ':'
                        offset_map[routine_arg.name][i] = (current_index, None, None)
                current_index += 1
        return offset_map

    def process_callee(self, routine, offset_map):
        """
        Process/adapt the callee according to information in `offset_map`.

        Adapt the variable declarations and usage/indexing.
        """
        # adapt variable declarations, thus adapt the dimension and shape of the corresponding arguments
        vmap = {}
        variable_map = routine.variable_map
        for var_name in offset_map:
            var = variable_map[var_name]
            new_dims = ()
            for i in range(max(k for k, v in offset_map[var.name].items() if v != 0) + 1):
                original_index = offset_map[var_name][i][0]
                offset = offset_map[var_name][i][1]
                size = offset_map[var_name][i][2]
                if not (original_index is None or 0 <= original_index < len(var.dimensions)):
                    continue
                if offset is not None:
                    new_dims += (size,)
                else:
                    new_dims += (var.shape[original_index],)
            vmap[var] = var.clone(dimensions=new_dims, type=var.type.clone(shape=new_dims))
        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
        # adapt the variable usage, thus the indexing/dimension
        vmap = {}
        for var in FindVariables(unique=False).visit(routine.body):
            if var.name in offset_map and var.dimensions is not None and var.dimensions:
                new_dims = ()
                for i in range(max(k for k, v in offset_map[var.name].items() if v != 0) + 1):
                    original_index = offset_map[var.name][i][0]
                    offset = offset_map[var.name][i][1]
                    if not (original_index is None or 0 <= original_index < len(var.dimensions)):
                        continue
                    if offset is not None:
                        if original_index is None:
                            new_dims += (offset,)
                        else:
                            new_dims += (var.dimensions[original_index] + offset - 1,)
                    else:
                        new_dims += (var.dimensions[original_index],)
                vmap[var] = var.clone(dimensions=new_dims)
        routine.body = SubstituteExpressions(vmap).visit(routine.body)
