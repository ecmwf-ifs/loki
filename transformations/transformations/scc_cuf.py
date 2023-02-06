# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Single-Column-Coalesced CUDA Fortran (SCC-CUF) transformation.
"""

from loki.expression import symbols as sym
from loki.transform import resolve_associates, single_variable_declaration, HoistVariablesTransformation
from loki import ir
from loki import (
    Transformation, FindNodes, FindVariables, Transformer,
    SubstituteExpressions, SymbolAttributes,
    CaseInsensitiveDict, as_tuple, flatten, types
)

from transformations.single_column_coalesced import (
    resolve_masked_stmts, get_integer_variable, kernel_remove_vector_loops,
    resolve_vector_dimension
)

__all__ = ['SccCufTransformation', 'HoistTemporaryArraysDeviceAllocatableTransformation']


class HoistTemporaryArraysDeviceAllocatableTransformation(HoistVariablesTransformation):
    """
    Synthesis part for variable/array hoisting for CUDA Fortran (CUF) (transformation).
    """

    def driver_variable_declaration(self, routine, var):
        """
        CUDA Fortran (CUF) Variable/Array device declaration including
        allocation and de-allocation.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to add the variable declaration
        var: :any:`Variable`
            The variable to be declared
        """
        vtype = var.type.clone(device=True, allocatable=True)
        routine.variables += tuple([var.clone(scope=routine, dimensions=as_tuple(
            [sym.RangeIndex((None, None))] * (len(var.dimensions))), type=vtype)])

        allocations = FindNodes(ir.Allocation).visit(routine.body)
        if allocations:
            insert_index = routine.body.body.index(allocations[-1])
            routine.body.insert(insert_index + 1, ir.Allocation((var.clone(),)))
        else:
            routine.body.prepend(ir.Allocation((var.clone(),)))
        de_allocations = FindNodes(ir.Deallocation).visit(routine.body)
        if de_allocations:
            insert_index = routine.body.body.index(de_allocations[-1])
            routine.body.insert(insert_index + 1, ir.Deallocation((var.clone(dimensions=None),)))
        else:
            routine.body.append(ir.Deallocation((var.clone(dimensions=None),)))


def dynamic_local_arrays(routine, vertical):
    """
    Declaring local arrays with the ``vertical`` :any:`Dimension` to be
    dynamically allocated.

    .. warning :: depends on single/unique variable declarations

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to dynamically allocate the local arrays
    vertical: :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """
    local_arrays = []
    argnames = [name.lower() for name in routine.argnames]
    decl_map = {}
    for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
        if any(isinstance(smbl, sym.Array) for smbl in decl.symbols) and not \
                any(smbl in argnames for smbl in decl.symbols) and \
                any(vertical.size in list(FindVariables().visit(smbl.shape)) for smbl in decl.symbols):
            local_arrays.extend(decl.symbols)
            dimensions = [sym.RangeIndex((None, None))] * len(decl.symbols[0].dimensions)
            symbols = [smbl.clone(type=smbl.type.clone(device=True, allocatable=True),
                                  dimensions=as_tuple(dimensions)) for smbl in decl.symbols]
            decl_map[decl] = decl.clone(symbols=as_tuple(symbols))
    routine.spec = Transformer(decl_map).visit(routine.spec)

    allocations = FindNodes(ir.Allocation).visit(routine.body)
    if allocations:
        insert_index = routine.body.body.index(allocations[-1]) + 1
        for local_array in local_arrays:
            routine.body.insert(insert_index, ir.Allocation((local_array,)))
    else:
        for local_array in reversed(local_arrays):
            routine.body.prepend(ir.Allocation((local_array,)))

    for local_array in local_arrays:
        routine.body.append(ir.Deallocation((local_array.clone(dimensions=None),)))


def increase_heap_size(routine):
    """
    Increase the heap size via call to `cudaDeviceSetLimit` needed for version with dynamic
    memory allocation on the device.

    .. note :: `cudaDeviceSetLimit` need to be called before the first kernel call!

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine (e.g. the driver) to increase the heap size
    """
    vtype = SymbolAttributes(types.BasicType.INTEGER, kind=sym.Variable(name="cuda_count_kind"))
    routine.spec.append(ir.VariableDeclaration((sym.Variable(name="cudaHeapSize", type=vtype),)))

    assignment_lhs = routine.variable_map["istat"]
    assignment_rhs = sym.InlineCall(function=sym.ProcedureSymbol(name="cudaDeviceSetLimit", scope=routine),
                                    parameters=(sym.Variable(name="cudaLimitMallocHeapSize"),
                                                routine.variable_map["cudaHeapSize"]))

    routine.body.prepend(ir.Assignment(lhs=assignment_lhs, rhs=assignment_rhs))
    routine.body.prepend(ir.Comment(''))

    # TODO: heap size, to be calculated?
    routine.body.prepend(
        ir.Assignment(lhs=routine.variable_map["cudaHeapSize"], rhs=sym.Product((10, 1024, 1024, 1024))))


def remove_pragmas(routine):
    """
    Remove all pragmas.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine in which to remove all pragmas
    """
    pragma_map = {p: None for p in FindNodes(ir.Pragma).visit(routine.body)}
    routine.body = Transformer(pragma_map).visit(routine.body)


# TODO: correct "definition" of elemental/pure routines ...
def is_elemental(routine):
    """
    Check whether :any:`Subroutine` ``routine`` is an elemental routine.

    Need for distinguishing elemental and non-elemental function to transform
    those in a different way.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to check whether elemental
    """
    for prefix in routine.prefix:
        if prefix.lower() == 'elemental':
            return True
    return False


def kernel_cuf(routine, horizontal, vertical, block_dim, disable, transformation_type, depth,
               derived_type_variables):
    """
    For CUDA Fortran (CUF) kernels and device functions: thread mapping, array dimension transformation,
    transforming (call) arguments, ...

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine (kernel/device subroutine)
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    vertical: :any:`Dimension`
        The dimension object specifying the vertical loop dimension
    block_dim: :any:`Dimension`
        The dimension object specifying the block loop dimension
    disable: tuple
        Tuple of routines not to be processed
    transformation_type: int
        Type of SCC-CUF transformation
    depth: int
        Depth of routine (within the call graph) to distinguish between kernels (`global` subroutines)
        and device functions (`device` subroutines)
    derived_type_variables: tuple
        Tuple of derived types within the routine
    """

    if is_elemental(routine):
        # TODO: correct "definition" of elemental/pure routines and corresponding removing
        #  of subroutine prefix(es)/specifier(s)
        routine.prefix = as_tuple([prefix for prefix in routine.prefix if prefix not in ["ELEMENTAL"]])
        return

    kernel_demote_private_locals(routine, horizontal, vertical)

    if depth > 1:
        single_variable_declaration(routine, variables=(horizontal.index, block_dim.index))

    # This adds argument and variable declaration !
    vtype = routine.variable_map[horizontal.size].type.clone(intent='in', value=True)
    new_argument = routine.variable_map[horizontal.size].clone(name=block_dim.size, type=vtype)
    routine.arguments = list(routine.arguments) + [new_argument]

    vtype = routine.variable_map[horizontal.index].type.clone()
    jblk_var = routine.variable_map[horizontal.index].clone(name=block_dim.index, type=vtype)
    routine.spec.append(ir.VariableDeclaration((jblk_var,)))

    if depth == 1:
        # CUDA thread mapping
        var_thread_idx = sym.Variable(name="THREADIDX")
        var_x = sym.Variable(name="X", parent=var_thread_idx)
        jl_assignment = ir.Assignment(lhs=routine.variable_map[horizontal.index], rhs=var_x)

        var_thread_idx = sym.Variable(name="BLOCKIDX")
        var_x = sym.Variable(name="Z", parent=var_thread_idx)
        jblk_assignment = ir.Assignment(lhs=routine.variable_map[block_dim.index], rhs=var_x)

        condition = sym.LogicalAnd((sym.Comparison(routine.variable_map[block_dim.index], '<=',
                                                   routine.variable_map[block_dim.size]),
                                    sym.Comparison(routine.variable_map[horizontal.index], '<=',
                                                   routine.variable_map[horizontal.size])))

        routine.body = ir.Section((jl_assignment, jblk_assignment, ir.Comment(''),
                        ir.Conditional(condition=condition, body=routine.body, else_body=None)))

    elif depth > 1:
        vtype = routine.variable_map[horizontal.size].type.clone(intent='in', value=True)
        new_arguments = [routine.variable_map[horizontal.index].clone(type=vtype), jblk_var.clone(type=vtype)]
        routine.arguments = list(routine.arguments) + new_arguments

    calls = [call for call in FindNodes(ir.CallStatement).visit(routine.body) if str(call.name).upper() not in disable]
    for call in calls:  # FindNodes(ir.CallStatement).visit(routine.body):
        if not is_elemental(call.routine):
            call.arguments += (routine.variable_map[block_dim.size], routine.variable_map[horizontal.index], jblk_var)

    variables = routine.variables
    arguments = routine.arguments

    relevant_local_arrays = []

    var_map = {}
    for var in variables:
        if var in arguments:
            if isinstance(var, sym.Scalar) and var.name != block_dim.size and var not in derived_type_variables:
                var_map[var] = var.clone(type=var.type.clone(value=True))
            elif isinstance(var, sym.Array):
                dimensions = list(var.dimensions) + [routine.variable_map[block_dim.size]]
                shape = list(var.shape) + [routine.variable_map[block_dim.size]]
                vtype = var.type.clone(shape=as_tuple(shape))
                var_map[var] = var.clone(dimensions=as_tuple(dimensions), type=vtype)
        else:
            if isinstance(var, sym.Array):
                dimensions = list(var.dimensions)
                if horizontal.size in list(FindVariables().visit(var.dimensions)):
                    if transformation_type == 'hoist':
                        dimensions += [routine.variable_map[block_dim.size]]
                        shape = list(var.shape) + [routine.variable_map[block_dim.size]]
                        vtype = var.type.clone(shape=as_tuple(shape))
                        relevant_local_arrays.append(var.name)
                    else:
                        dimensions.remove(horizontal.size)
                        relevant_local_arrays.append(var.name)
                        vtype = var.type.clone(device=True)
                    var_map[var] = var.clone(dimensions=as_tuple(dimensions), type=vtype)

    routine.spec = SubstituteExpressions(var_map).visit(routine.spec)

    var_map = {}
    arguments_name = [var.name for var in arguments]
    for var in FindVariables().visit(routine.body):
        if var.name in arguments_name:
            if isinstance(var, sym.Array):
                dimensions = list(var.dimensions)
                dimensions.append(routine.variable_map[block_dim.index])
                var_map[var] = var.clone(dimensions=as_tuple(dimensions),
                                         type=var.type.clone(shape=as_tuple(dimensions)))
        else:
            if transformation_type == 'hoist':
                if var in relevant_local_arrays:
                    var_map[var] = var.clone(dimensions=var.dimensions + (routine.variable_map[block_dim.index],))
            else:
                if var in relevant_local_arrays:
                    dimensions = list(var.dimensions)
                    var_map[var] = var.clone(dimensions=as_tuple(dimensions[1:]))

    routine.body = SubstituteExpressions(var_map).visit(routine.body)

    calls = [call for call in FindNodes(ir.CallStatement).visit(routine.body) if str(call.name).upper() not in disable]
    for call in calls:
        if not is_elemental(call.routine):
            arguments = []
            for arg in call.arguments:
                if isinstance(arg, sym.Array):
                    arguments.append(arg.clone(dimensions=None))
                else:
                    arguments.append(arg)
            call.arguments = arguments


def kernel_demote_private_locals(routine, horizontal, vertical):
    """
    Demotes all local variables.

    Array variables whose dimensions include only the vector dimension
    or known (short) constant dimensions (eg. local vector or matrix arrays)
    can be privatized without requiring shared GPU memory. Array variables
    with unknown (at compile time) dimensions (eg. the vertical dimension)
    cannot be privatized at the vector loop level and should therefore not
    be demoted here.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to demote the private locals
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    vertical: :any:`Dimension`
        The dimension object specifying the vertical loop dimension
    """

    # Establish the new dimensions and shapes first, before cloning the variables
    # The reason for this is that shapes of all variable instances are linked
    # via caching, meaning we can easily void the shape of an unprocessed variable.
    variables = list(routine.variables)
    variables += list(FindVariables(unique=False).visit(routine.body))

    # Filter out purely local array variables
    argument_map = CaseInsensitiveDict({a.name: a for a in routine.arguments})
    variables = [v for v in variables if not v.name in argument_map]
    variables = [v for v in variables if isinstance(v, sym.Array)]

    # Find all arrays with shapes that do not include the vertical
    # dimension and can thus be privatized.
    variables = [v for v in variables if v.shape is not None]
    variables = [v for v in variables if not any(vertical.size in d for d in v.shape)]

    # Filter out variables that we will pass down the call tree
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    call_args = flatten(call.arguments for call in calls)
    call_args += flatten(list(dict(call.kwarguments).values()) for call in calls)
    variables = [v for v in variables if v.name not in call_args]

    shape_map = CaseInsensitiveDict({v.name: v.shape for v in variables})
    vmap = {}
    for v in variables:
        old_shape = shape_map[v.name]
        # TODO: "s for s in old_shape if s not in expressions" sufficient?
        new_shape = as_tuple(s for s in old_shape if s not in horizontal.size_expressions)

        if old_shape and old_shape[0] in horizontal.size_expressions:
            new_type = v.type.clone(shape=new_shape or None)
            new_dims = v.dimensions[1:] or None
            vmap[v] = v.clone(dimensions=new_dims, type=new_type)

    routine.body = SubstituteExpressions(vmap).visit(routine.body)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)


def driver_device_variables(routine, disable):
    """
    Driver device variable versions including

    * variable declaration
    * allocation
    * host-device synchronisation
    * de-allocation

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine (driver) to handle the device variables
    disable: tuple
        Tuple of routines not to be processed
    """

    # istat: status of CUDA runtime function (e.g. for cudaDeviceSynchronize(), cudaMalloc(), cudaFree(), ...)
    i_type = SymbolAttributes(types.BasicType.INTEGER)
    routine.spec.append(ir.VariableDeclaration(symbols=(sym.Variable(name="istat", type=i_type),)))

    relevant_arrays = []
    calls = [call for call in FindNodes(ir.CallStatement).visit(routine.body) if str(call.name).upper() not in disable]
    for call in calls:
        relevant_arrays.extend([arg for arg in call.arguments if isinstance(arg, sym.Array)])

    relevant_arrays = list(dict.fromkeys(relevant_arrays))

    # Declaration
    routine.spec.append(ir.Comment(''))
    routine.spec.append(ir.Comment('! Device arrays'))
    for array in relevant_arrays:
        vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
        vdimensions = [sym.RangeIndex((None, None))] * len(array.dimensions)
        var = array.clone(name=f"{array.name}_d", type=vtype, dimensions=as_tuple(vdimensions))
        routine.spec.append(ir.VariableDeclaration(symbols=as_tuple(var)))

    # Allocation
    for array in reversed(relevant_arrays):
        vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
        routine.body.prepend(ir.Allocation((array.clone(name=f"{array.name}_d", type=vtype,
                                                        dimensions=routine.variable_map[array.name].dimensions),)))
    routine.body.prepend(ir.Comment('! Device array allocation'))
    routine.body.prepend(ir.Comment(''))

    allocations = FindNodes(ir.Allocation).visit(routine.body)
    if allocations:
        insert_index = routine.body.body.index(allocations[-1]) + 1
    else:
        insert_index = None
    # or: insert_index = routine.body.body.index(calls[0])
    # Copy host to device
    for array in reversed(relevant_arrays):
        vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
        lhs = array.clone(name=f"{array.name}_d", type=vtype, dimensions=None)
        rhs = array.clone(dimensions=None)
        if insert_index is not None:
            routine.body.insert(insert_index, ir.Assignment(lhs=lhs, rhs=rhs))
        else:
            routine.body.prepend(ir.Assignment(lhs=lhs, rhs=rhs))
    routine.body.insert(insert_index, ir.Comment('! Copy host to device'))
    routine.body.insert(insert_index, ir.Comment(''))

    # TODO: this just assumes that host-device-synchronisation is only needed at the beginning and end
    # Copy device to host
    insert_index = None
    for call in FindNodes(ir.CallStatement).visit(routine.body):
        if "THREAD_END" in str(call.name):  # TODO: fix/check: very specific to CLOUDSC
            insert_index = routine.body.body.index(call) + 1

    if insert_index is None:
        routine.body.append(ir.Comment(''))
        routine.body.append(ir.Comment('! Copy device to host'))
    for v in reversed(relevant_arrays):
        if v.type.intent != "in":
            lhs = v.clone(dimensions=None)
            vtype = v.type.clone(device=True, allocatable=True, intent=None, shape=None)
            rhs = v.clone(name=f"{v.name}_d", type=vtype, dimensions=None)
            if insert_index is None:
                routine.body.append(ir.Assignment(lhs=lhs, rhs=rhs))
            else:
                routine.body.insert(insert_index, ir.Assignment(lhs=lhs, rhs=rhs))
    if insert_index is not None:
        routine.body.insert(insert_index, ir.Comment('! Copy device to host'))

    # De-allocation
    routine.body.append(ir.Comment(''))
    routine.body.append(ir.Comment('! De-allocation'))
    for array in relevant_arrays:
        routine.body.append(ir.Deallocation((array.clone(name=f"{array.name}_d", dimensions=None),)))

    call_map = {}
    for call in calls:
        arguments = []
        for arg in call.arguments:
            if arg in relevant_arrays:
                vtype = arg.type.clone(device=True, allocatable=True, shape=None, intent=None)
                arguments.append(arg.clone(name=f"{arg.name}_d", type=vtype, dimensions=None))
            else:
                arguments.append(arg)
        call_map[call] = call.clone(arguments=arguments)
    routine.body = Transformer(call_map).visit(routine.body)


def driver_launch_configuration(routine, block_dim, disable):
    """
    Launch configuration for kernel calls within the driver with the
    CUDA Fortran (CUF) specific chevron syntax `<<<griddim, blockdim>>>`.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to specify the launch configurations for kernel calls.
    block_dim: :any:`Dimension`
        The dimension object specifying the block loop dimension
    disable: tuple
        Tuple of routines not to be processed
    """

    d_type = SymbolAttributes(types.DerivedType("DIM3"))
    routine.spec.append(ir.VariableDeclaration(symbols=(sym.Variable(name="GRIDDIM", type=d_type),
                                                        sym.Variable(name="BLOCKDIM", type=d_type))))

    mapper = {}
    for loop in FindNodes(ir.Loop).visit(routine.body):
        # TODO: fix/check: do not use _aliases
        if loop.variable == block_dim.index or loop.variable in block_dim._aliases:
            mapper[loop] = loop.body
            kernel_within = False
            for call in FindNodes(ir.CallStatement).visit(loop.body):
                if str(call.name).upper() not in disable:
                    kernel_within = True

                    assignment_lhs = routine.variable_map["istat"]
                    assignment_rhs = sym.InlineCall(
                        function=sym.ProcedureSymbol(name="cudaDeviceSynchronize", scope=routine),
                        parameters=())

                    mapper[call] = (call.clone(chevron=(routine.variable_map["GRIDDIM"],
                                                        routine.variable_map["BLOCKDIM"]),
                                               arguments=call.arguments + (routine.variable_map[block_dim.size],)),
                                    ir.Assignment(lhs=assignment_lhs, rhs=assignment_rhs))

            if kernel_within:
                upper = routine.variable_map[loop.bounds.children[1].name]
                if loop.bounds.children[2]:
                    step = routine.variable_map[loop.bounds.children[2].name]
                else:
                    step = sym.IntLiteral(1)

                func_dim3 = sym.ProcedureSymbol(name="DIM3", scope=routine)
                func_ceiling = sym.ProcedureSymbol(name="CEILING", scope=routine)

                # BLOCKDIM
                lhs = routine.variable_map["blockdim"]
                rhs = sym.InlineCall(function=func_dim3, parameters=(step, sym.IntLiteral(1), sym.IntLiteral(1)))
                blockdim_assignment = ir.Assignment(lhs=lhs, rhs=rhs)

                # GRIDDIM
                lhs = routine.variable_map["griddim"]
                rhs = sym.InlineCall(function=func_dim3, parameters=(sym.IntLiteral(1), sym.IntLiteral(1),
                                                                     sym.InlineCall(function=func_ceiling,
                                                                                    parameters=as_tuple(
                                                                                        sym.Cast(name="REAL",
                                                                                                 expression=upper) /
                                                                                        sym.Cast(name="REAL",
                                                                                                 expression=step)))))
                griddim_assignment = ir.Assignment(lhs=lhs, rhs=rhs)
                mapper[loop] = (blockdim_assignment, griddim_assignment, loop.body)
            else:
                mapper[loop] = loop.body

    routine.body = Transformer(mapper=mapper).visit(routine.body)


def device_derived_types(routine, disable, derived_types):
    """
    Create device versions of variables of specific derived types including
    host-device-synchronisation.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to create device versions of the specified derived type variables.
    disable: tuple
        Tuple of routines not to be processed
    derived_types: tuple
        Tuple of derived types within the routine
    """
    used_members = [v for v in FindVariables().visit(routine.ir) if v.parent]
    variables = [v for v in used_members if v.parent.type.dtype.name.lower() in derived_types]

    var_map = {}
    for var in variables:
        new_var = var.clone(name=f"{var.name}_d", type=var.type.clone(intent=None, imported=None,
                                                                      allocatable=None, device=True,
                                                                      module=None))
        var_map[var] = new_var
        routine.spec.append(ir.VariableDeclaration((new_var,)))
        routine.body.prepend(ir.Assignment(lhs=new_var, rhs=var))

    calls = [call for call in FindNodes(ir.CallStatement).visit(routine.body) if
             str(call.name).upper() not in disable]
    for call in calls:
        arguments = [var_map.get(arg, arg) for arg in call.arguments]
        call.arguments = arguments
    return variables


def device_subroutine_prefix(routine, depth):
    """
    Add prefix/specifier `ATTRIBUTES(GLOBAL)` for kernel subroutines and
    `ATTRIBUTES(DEVICE)` for device subroutines.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine (kernel/device subroutine) to add a prefix/specifier
    depth: int
        The subroutines depth
    """
    if depth == 1:
        routine.prefix += ("ATTRIBUTES(GLOBAL)",)
    elif depth > 1:
        routine.prefix += ("ATTRIBUTES(DEVICE)",)


class SccCufTransformation(Transformation):
    """
    Single Column Coalesced CUDA Fortran - SCC-CUF: Direct CPU-to-GPU
    transformation for block-indexed gridpoint routines.

    This transformation will remove individiual CPU-style
    vectorization loops from "kernel" routines and distributes the
    work for GPU threads according to the CUDA programming model using
    CUDA Fortran (CUF) syntax.

    .. note:: this requires preprocessing with the :any:`DerivedTypeArgumentsTransformation`.

    .. note:: in dependence of the transformation type ``transformation_type`` further
     transformation are necessary:

     * ``transformation_type = 'parametrise'`` requires a subsequent :any:`ParametriseTransformation`
      transformation with the necessary information to parametrise (at least) the ``vertical`` `size`
     * ``transformation_type = 'hoist'`` requires subsequent :any:`HoistVariablesAnalysis` and
      :class:`HoistVariablesTransformation` transformations (e.g. :any:`HoistTemporaryArraysAnalysis`
      for analysis and :any:`HoistTemporaryArraysTransformationDeviceAllocatable` for synthesis)
     * ``transformation_type = 'dynamic'`` does not require a subsequent transformation

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    vertical : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the vertical dimension, as needed to decide array privatization.
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension
        to use for hoisted column arrays if hoisting is enabled.
    disable: tuple
        Optional, tuple of subroutines not to be processed.
    transformation_type : str
        Kind of SCC-CUF transformation, as automatic arrays currently not supported. Thus
        automatic arrays need to transformed by either

        - `parametrise`: parametrising the array dimensions to make the vertical dimension
          a compile-time constant
        - `hoist`: host side hoisting of (relevant) arrays
        - `dynamic`: dynamic memory allocation on the device (not recommended for performance reasons)

    """

    def __init__(self, horizontal, vertical, block_dim, disable=None, transformation_type='parametrise',
                 derived_types=None):
        self.horizontal = horizontal
        self.vertical = vertical
        self.block_dim = block_dim

        self.transformation_type = transformation_type
        # `parametrise` : parametrising the array dimensions
        # `hoist`: host side hoisting
        # `dynamic`: dynamic memory allocation on the device
        assert self.transformation_type in ['parametrise', 'hoist', 'dynamic']
        self.transformation_description = {'parametrise': 'parametrised array dimensions of local arrays',
                                           'hoist': 'host side hoisted local arrays',
                                           'dynamic': 'dynamic memory allocation on the device'}

        if disable is None:
            self.disable = ()
        else:
            self.disable = [_.upper() for _ in disable]
        if derived_types is None:
            self.derived_types = ()
        else:
            self.derived_types = [_.lower() for _ in derived_types]
        self.derived_type_variables = ()

    def transform_module(self, module, **kwargs):

        role = kwargs.get('role')

        if role == 'driver':
            module.spec.prepend(ir.Import(module="cudafor"))

    def transform_subroutine(self, routine, **kwargs):

        item = kwargs.get('item', None)
        if item and not item.local_name == routine.name.lower():
            return

        role = kwargs.get('role')
        depths = kwargs.get('depths', None)
        if depths is None:
            if role == 'driver':
                depth = 0
            elif role == 'kernel':
                depth = 1
        else:
            depth = depths[item]

        remove_pragmas(routine)
        single_variable_declaration(routine=routine, group_by_shape=True)
        device_subroutine_prefix(routine, depth)

        if depth > 0:
            routine.spec.prepend(ir.Import(module="cudafor"))

        if role == 'driver':
            self.process_routine_driver(routine)

        if role == 'kernel':
            self.process_routine_kernel(routine, depth=depth)

    def process_routine_kernel(self, routine, depth=1):
        """
        Kernel/Device subroutine specific changes/transformations.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine (kernel/device subroutine) to process
        depth: int
            The subroutines depth
        """

        v_index = get_integer_variable(routine, name=self.horizontal.index)
        resolve_associates(routine)
        resolve_masked_stmts(routine, loop_variable=v_index)
        resolve_vector_dimension(routine, loop_variable=v_index, bounds=self.horizontal.bounds)
        kernel_remove_vector_loops(routine, self.horizontal)

        kernel_cuf(routine, self.horizontal, self.vertical, self.block_dim, self.disable,
                   self.transformation_type, depth=depth, derived_type_variables=self.derived_type_variables)

        # dynamic memory allocation of local arrays (only for version with dynamic memory allocation on device)
        if self.transformation_type == 'dynamic':
            dynamic_local_arrays(routine, self.vertical)

    def process_routine_driver(self, routine):
        """
        Driver subroutine specific changes/transformations.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine (driver) to process
        """

        self.derived_type_variables = device_derived_types(routine=routine, disable=self.disable,
                                                           derived_types=self.derived_types)

        # create variables needed for the device execution, especially generate device versions of arrays
        driver_device_variables(routine=routine, disable=self.disable)
        # remove block loop and generate launch configuration for CUF kernels
        driver_launch_configuration(routine=routine, block_dim=self.block_dim, disable=self.disable)

        # increase heap size (only for version with dynamic memory allocation on device)
        if self.transformation_type == 'dynamic':
            increase_heap_size(routine)

        routine.body.prepend(ir.Comment(f"!@cuf print *, 'executing SCC-CUF type: {self.transformation_type} - "
                                        f"{self.transformation_description[self.transformation_type]}'"))
        routine.body.prepend(ir.Comment(""))
