# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from more_itertools import pairwise

from loki.expression import symbols as sym
from loki.transform import (resolve_associates, single_variable_declarations, single_variable_declaration,
                            HoistVariablesTransformation)
from loki import ir
from loki import (
    Transformation, FindNodes, FindScopes, FindVariables,
    FindExpressions, Transformer, NestedTransformer, NestedMaskedTransformer,
    SubstituteExpressions, SymbolAttributes, BasicType, DerivedType,
    pragmas_attached, CaseInsensitiveDict, as_tuple, flatten, types, fgen
)

__all__ = ['SccCuf', 'HoistTemporaryArraysTransformationDeviceAllocatable']

########################################################################################################################
# For host side hoisted (memory allocation of) local arrays
########################################################################################################################
class HoistTemporaryArraysTransformationDeviceAllocatable(HoistVariablesTransformation):

    def __init__(self, key=None, **kwargs):
        super().__init__(key=key, **kwargs)

    def driver_variable_declaration(self, routine, var):
        type = var.type.clone(device=True, allocatable=True)
        routine.variables += tuple([var.clone(scope=routine, dimensions=as_tuple(
            [sym.RangeIndex((None, None))] * (len(var.dimensions))), type=type)])

        # EITHER
        # routine.body.prepend(Allocation((var.clone(),)))
        # routine.body.append(Deallocation((var.clone(dimensions=None),)))

        # OR: just for better formatting ...
        allocations = FindNodes(ir.Allocation).visit(routine.body)
        if allocations:
            insert_index = routine.body.body.index(allocations[-1])
            routine.body.insert(insert_index + 1, ir.Allocation((var.clone(),)))
        else:
            routine.body.prepend(ir.Allocation((var.clone(),)))
        de_allocations = FindNodes(ir.Deallocation).visit(routine.body)
        if allocations:
            insert_index = routine.body.body.index(de_allocations[-1])
            routine.body.insert(insert_index + 1, ir.Deallocation((var.clone(dimensions=None),)))
        else:
            routine.body.append(ir.Deallocation((var.clone(dimensions=None),)))
########################################################################################################################


########################################################################################################################
# For dynamic memory on device (transformation_type = 2)
########################################################################################################################
def dynamic_local_arrays(routine, vertical):
    local_arrays = []
    arguments = [arg.name for arg in routine.arguments]
    decl_map = {}
    for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
        if any(isinstance(smbl, sym.Array) for smbl in decl.symbols) and not \
                any(smbl.name in arguments for smbl in decl.symbols) and \
                any(vertical.size in list(FindVariables().visit(smbl.shape)) for smbl in decl.symbols):
            local_arrays.extend(decl.symbols)
            dimensions = [sym.RangeIndex((None, None))] * len(decl.symbols[0].dimensions)
            symbols = [smbl.clone(type=smbl.type.clone(device=True, allocatable=True),
                                  dimensions=as_tuple(dimensions)) for smbl in decl.symbols]
            decl_map[decl] = decl.clone(symbols=as_tuple(symbols))
    routine.spec = Transformer(decl_map).visit(routine.spec)

    for local_array in reversed(local_arrays):
        routine.body.prepend(ir.Allocation((local_array,)))

    for local_array in local_arrays:
        routine.body.append(ir.Deallocation((local_array.clone(dimensions=None),)))


def increase_heap_size(routine):
    vtype = SymbolAttributes(types.BasicType.INTEGER, kind=sym.IntrinsicLiteral("cuda_count_kind"))
    routine.spec.append(ir.VariableDeclaration((sym.Scalar("cudaHeapSize", type=vtype),)))

    assignment_lhs = routine.variable_map["istat"]
    assignment_rhs = sym.InlineCall(function=sym.ProcedureSymbol(name="cudaDeviceSetLimit", scope=routine),
                                    parameters=(sym.IntrinsicLiteral("cudaLimitMallocHeapSize"),
                                                routine.variable_map["cudaHeapSize"]))

    routine.body.prepend(ir.Assignment(lhs=assignment_lhs, rhs=assignment_rhs))
    routine.body.prepend(ir.Comment(''))

    # TODO: heap size, to be calculated?
    routine.body.prepend(
        ir.Assignment(lhs=routine.variable_map["cudaHeapSize"], rhs=sym.Product((10, 1024, 1024, 1024))))
########################################################################################################################


def remove_pragmas(routine):
    """

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine in which to remove all pragmas
    """
    pragma_map = {}
    for pragma in FindNodes(ir.Pragma).visit(routine.body):
        pragma_map[pragma] = None
    routine.body = Transformer(pragma_map).visit(routine.body)


def get_integer_variable(routine, name):
    """
    Find a local variable in the routine, or create an integer-typed one.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to find the variable
    name : string
        Name of the variable to find the in the routine.
    """
    if name in routine.variable_map:
        v_index = routine.variable_map[name]
    else:
        dtype = SymbolAttributes(BasicType.INTEGER)
        v_index = sym.Variable(name=name, type=dtype, scope=routine)
    return v_index


def kernel_remove_vector_loops(routine, horizontal):
    """
    Remove all vector loops over the specified dimension.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal : :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """
    loop_map = {}
    for loop in FindNodes(ir.Loop).visit(routine.body):
        if loop.variable == horizontal.index:
            loop_map[loop] = loop.body
    routine.body = Transformer(loop_map).visit(routine.body)


def is_elemental(routine):
    if "ELEMENTAL" in routine.prefix:
        return True
    return False


def kernel_block_size_argument(routine, horizontal, vertical, block_dim, disable, transformation_type, depth):

    if is_elemental(routine):
        routine.prefix = as_tuple([prefix for prefix in routine.prefix if prefix not in ["ELEMENTAL"]])
        return

    kernel_demote_private_locals(routine, horizontal, vertical)

    if depth > 1:
        single_variable_declaration(routine, variables=(horizontal.index, block_dim.index))

    # This adds argument and variable declaration !
    type = routine.variable_map[horizontal.size].type.clone(intent='in', value=True)
    new_argument = routine.variable_map[horizontal.size].clone(name=block_dim.size, type=type)
    routine.arguments = list(routine.arguments) + [new_argument]

    type = routine.variable_map[horizontal.index].type.clone()
    jblk_var = routine.variable_map[horizontal.index].clone(name=block_dim.index, type=type)
    routine.spec.append(ir.VariableDeclaration((jblk_var,)))

    if depth == 1:
        print(f"routine {routine.name}: thread assignment, depth: {depth}")
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
        # routine.arguments += jblk_var.clone(type=jblk_var.type.clone(intent="in"))
        type = routine.variable_map[horizontal.size].type.clone(intent='in', value=True)
        new_arguments = [routine.variable_map[horizontal.index].clone(type=type), jblk_var.clone(type=type)]
        routine.arguments = list(routine.arguments) + new_arguments
        # new_arguments = [routine.variable_map[horizontal.index].clone(type=type)] #.clone(type=type)]
        # routine.arguments = list(routine.arguments) + new_arguments

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
            if isinstance(var, sym.Scalar) and var.name != block_dim.size and var.name != "YRECLDP":
                var_map[var] = var.clone(type=var.type.clone(value=True))
            elif isinstance(var, sym.Array):
                dimensions = list(var.dimensions) + [routine.variable_map[block_dim.size]]
                shape = list(var.shape) + [routine.variable_map[block_dim.size]]
                type = var.type.clone(shape=as_tuple(shape))
                var_map[var] = var.clone(dimensions=as_tuple(dimensions), type=type)
        else:
            if isinstance(var, sym.Array):
                dimensions = list(var.dimensions)
                if horizontal.size in dimensions:
                    if transformation_type == 1:
                        dimensions += [routine.variable_map[block_dim.size]]
                        shape = list(var.shape) + [routine.variable_map[block_dim.size]]
                        type = var.type.clone(shape=as_tuple(shape))
                    else:
                        dimensions.remove(horizontal.size)
                        relevant_local_arrays.append(var.name)
                        type = var.type.clone(device=True)
                var_map[var] = var.clone(dimensions=as_tuple(dimensions), type=type)

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
            if transformation_type == 1:
                if isinstance(var, sym.Array):
                    dimensions = list(var.dimensions)
                    dimensions.append(routine.variable_map[block_dim.index])
                    var_map[var] = var.clone(type=var.type.clone(shape=as_tuple(dimensions)))
            else:
                if var.name in relevant_local_arrays:
                    dimensions = list(var.dimensions)
                    dimensions.pop(0)
                    var_map[var] = var.clone(dimensions=as_tuple(dimensions))

    routine.body = SubstituteExpressions(var_map).visit(routine.body)

    calls = [call for call in FindNodes(ir.CallStatement).visit(routine.body) if str(call.name).upper() not in disable]
    for call in calls:
        if not is_elemental(call.routine):
            arguments = []
            for arg in call.arguments:
                if isinstance(arg, sym.Array):
                    arguments.append(arg.clone(dimensions=None))  # , type=arg.type.clone(shape=None)))
                else:
                    arguments.append(arg)
            call.arguments = arguments


def demote_variables(routine, variables, expressions):
    shape_map = CaseInsensitiveDict({v.name: v.shape for v in variables})
    vmap = {}
    for v in variables:
        old_shape = shape_map[v.name]
        # TODO: "s for s in old_shape if s not in expressions" sufficient?
        new_shape = as_tuple(s for s in old_shape if s not in expressions)

        if old_shape and old_shape[0] in expressions:
            new_type = v.type.clone(shape=new_shape or None)
            new_dims = v.dimensions[1:] or None
            vmap[v] = v.clone(dimensions=new_dims, type=new_type)

    routine.body = SubstituteExpressions(vmap).visit(routine.body)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)


def kernel_demote_private_locals(routine, horizontal, vertical):
    """
    Demotes all local variables that can be privatized at the `acc loop vector`
    level.

    Array variables whose dimensions include only the vector dimension
    or known (short) constant dimensions (eg. local vector or matrix arrays)
    can be privatized without requiring shared GPU memory. Array variables
    with unknown (at compile time) dimensions (eg. the vertical dimension)
    cannot be privatized at the vector loop level and should therefore not
    be demoted here.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    vertical: :any:`Dimension`
        The dimension object specifying the vertical loop dimension
    """

    # Establish the new dimensions and shapes first, before cloning the variables
    # The reason for this is that shapes of all variable instances are linked
    # via caching, meaning we can easily void the shape of an unprocessed variable.
    print(f"routine: {routine.name}")
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

    demote_variables(routine, variables, horizontal.size_expressions)


def driver_device_variables(routine, disable):

    # istat: status of CUDA runtime function (e.g. for cudaDeviceSynchronize(), cudaMalloc(), cudaFree(), ...)
    i_type = SymbolAttributes(types.BasicType.INTEGER)
    routine.spec.append(ir.VariableDeclaration(symbols=(sym.Scalar(name="istat", type=i_type),)))

    relevant_arrays = []
    calls = [call for call in FindNodes(ir.CallStatement).visit(routine.body) if str(call.name).upper() not in disable]
    for call in calls:
        relevant_arrays.extend([arg for arg in call.arguments if isinstance(arg, sym.Array)])

    relevant_arrays = list(dict.fromkeys(relevant_arrays))  # list(set(relevant_arrays))

    # Declaration
    routine.spec.append(ir.Comment(''))
    routine.spec.append(ir.Comment('! Device arrays'))
    for array in relevant_arrays:
        vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
        vdimensions = [sym.RangeIndex((None, None))] * len(array.dimensions)
        var = array.clone(name="{}_d".format(array.name), type=vtype, dimensions=as_tuple(vdimensions))
        routine.spec.append(ir.VariableDeclaration(symbols=as_tuple(var)))

    # Allocation
    for array in reversed(relevant_arrays):
        vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
        routine.body.prepend(ir.Allocation((array.clone(name="{}_d".format(array.name), type=vtype,
                                                        dimensions=routine.variable_map[array.name].dimensions),)))
    routine.body.prepend(ir.Comment('! Device array allocation'))
    routine.body.prepend(ir.Comment(''))

    allocations = FindNodes(ir.Allocation).visit(routine.body)
    insert_index = routine.body.body.index(allocations[-1]) + 1
    # or: insert_index = routine.body.body.index(calls[0])
    # Copy host to device
    for array in reversed(relevant_arrays):
        vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
        lhs = array.clone(name="{}_d".format(array.name), type=vtype, dimensions=None)
        rhs = array.clone(dimensions=None)
        routine.body.insert(insert_index, ir.Assignment(lhs=lhs, rhs=rhs))
    routine.body.insert(insert_index, ir.Comment('! Copy host to device'))
    routine.body.insert(insert_index, ir.Comment(''))

    # De-allocation
    routine.body.append(ir.Comment(''))
    routine.body.append(ir.Comment('! De-allocation'))
    for array in relevant_arrays:
        routine.body.append(ir.Deallocation((array.clone(name="{}_d".format(array.name), dimensions=None),)))

    # TODO: this just assumes that host-device-synchronisation is only needed at the beginning and end
    # Copy device to host
    insert_index = None
    for call in FindNodes(ir.CallStatement).visit(routine.body):
        if "THREAD_END" in str(call.name):
            insert_index = routine.body.body.index(call) + 1

    if insert_index is None:
        routine.body.append(ir.Comment(''))
        routine.body.append(ir.Comment('! Copy device to host'))
    for v in reversed(relevant_arrays):
        if v.type.intent != "in":
            lhs = v.clone(dimensions=None)
            vtype = v.type.clone(device=True, allocatable=True, intent=None, shape=None)
            rhs = v.clone(name="{}_d".format(v.name), type=vtype, dimensions=None)
            if insert_index is None:
                routine.body.append(ir.Assignment(lhs=lhs, rhs=rhs))
            else:
                routine.body.insert(insert_index, ir.Assignment(lhs=lhs, rhs=rhs))
    if insert_index is not None:
        routine.body.insert(insert_index, ir.Comment('! Copy device to host'))

    call_map = {}
    for call in calls:
        arguments = []
        for arg in call.arguments:
            if arg in relevant_arrays:  # if isinstance(arg, sym.Array):
                vtype = arg.type.clone(device=True, allocatable=True, shape=None, intent=None)
                arguments.append(arg.clone(name="{}_d".format(arg.name), type=vtype, dimensions=None))
            else:
                arguments.append(arg)
        call_map[call] = call.clone(arguments=arguments)
    routine.body = Transformer(call_map).visit(routine.body)

    # TODO: is this necessary? add global "if"
    #  jl <= end
    #  ...
    # upper = routine.variable_map[loop.bounds.children[1].name]
    # if loop.bounds.children[2]:
    #     step = routine.variable_map[loop.bounds.children[2].name]
    # else:
    #     step = sym.IntLiteral(1)
    # rhs = (upper / step) + \
    #       sym.InlineCall(function=sym.ProcedureSymbol(name="MIN", scope=routine),
    #                      parameters=(sym.InlineCall(function=sym.ProcedureSymbol(name="MOD", scope=routine),
    #                                                 parameters=(upper,
    #                                                             step)), sym.IntLiteral(1)))
    # routine.body.prepend(ir.Assignment(lhs=routine.variable_map[block_dim.size], rhs=rhs))


def driver_launch_configuration(routine, block_dim, disable):

    d_type = SymbolAttributes(types.DerivedType("DIM3"))
    routine.spec.append(ir.VariableDeclaration(symbols=(sym.Variable(name="GRIDDIM", type=d_type),
                                                        sym.Variable(name="BLOCKDIM", type=d_type))))

    loop_map = {}
    call_map = {}
    for loop in FindNodes(ir.Loop).visit(routine.body):
        if loop.variable == block_dim.index or loop.variable in block_dim.aliases:
            loop_map[loop] = loop.body
            kernel_within = False
            for call in FindNodes(ir.CallStatement).visit(loop.body):
                nodes = []
                if str(call.name).upper() not in disable:
                    kernel_within = True
                    nodes.append(call.clone(chevron=(routine.variable_map["GRIDDIM"],
                                                     routine.variable_map["BLOCKDIM"])))

                    assignment_lhs = routine.variable_map["istat"]
                    assignment_rhs = sym.InlineCall(
                        function=sym.ProcedureSymbol(name="cudaDeviceSynchronize", scope=routine),
                        parameters=())

                    nodes.append(ir.Assignment(lhs=assignment_lhs, rhs=assignment_rhs))

                    call_map[call] = as_tuple(nodes)

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
                loop_map[loop] = (blockdim_assignment, griddim_assignment, loop.body)
            else:
                loop_map[loop] = loop.body

    routine.body = Transformer(loop_map).visit(routine.body)
    routine.body = Transformer(call_map).visit(routine.body)

    for call in FindNodes(ir.CallStatement).visit(routine.body):
        if str(call.name).upper() not in disable:
            call.arguments += (routine.variable_map[block_dim.size],)


def resolve_masked_stmts(routine, loop_variable):
    """
    Resolve :any:`MaskedStatement` (WHERE statement) objects to an
    explicit combination of :any:`Loop` and :any:`Conditional` combination.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to resolve masked statements
    loop_variable : :any:`Scalar`
        The induction variable for the created loops.
    """
    mapper = {}
    for masked in FindNodes(ir.MaskedStatement).visit(routine.body):
        ranges = [e for e in FindExpressions().visit(masked.condition) if isinstance(e, sym.RangeIndex)]
        exprmap = {r: loop_variable for r in ranges}
        assert len(ranges) > 0
        assert all(r == ranges[0] for r in ranges)
        bounds = sym.LoopRange((ranges[0].start, ranges[0].stop, ranges[0].step))
        cond = ir.Conditional(condition=masked.condition, body=masked.body, else_body=masked.default)
        loop = ir.Loop(variable=loop_variable, bounds=bounds, body=cond)
        # Substitute the loop ranges with the loop index and add to mapper
        mapper[masked] = SubstituteExpressions(exprmap).visit(loop)

    routine.body = Transformer(mapper).visit(routine.body)


def resolve_vector_dimension(routine, loop_variable, bounds):
    """
    Resolve vector notation for a given dimension only. The dimension
    is defined by a loop variable and the bounds of the given range.

    TODO: Consolidate this with the internal
    `loki.transform.transform_array_indexing.resolve_vector_notation`.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to resolve vector notation usage.
    loop_variable : :any:`Scalar`
        The induction variable for the created loops.
    bounds : tuple of :any:`Scalar`
        Tuple defining the iteration space of the inserted loops.
    """
    bounds_str = f'{bounds[0]}:{bounds[1]}'

    mapper = {}
    for stmt in FindNodes(ir.Assignment).visit(routine.body):
        ranges = [e for e in FindExpressions().visit(stmt)
                  if isinstance(e, sym.RangeIndex) and e == bounds_str]
        if ranges:
            exprmap = {r: loop_variable for r in ranges}
            loop = ir.Loop(variable=loop_variable, bounds=sym.LoopRange(bounds),
                           body=SubstituteExpressions(exprmap).visit(stmt))
            mapper[stmt] = loop

    routine.body = Transformer(mapper).visit(routine.body)


def device_subroutine_prefix(routine, depth):
    if depth == 1:
        routine.prefix += ("ATTRIBUTES(GLOBAL)",)
    elif depth > 1:
        routine.prefix += ("ATTRIBUTES(DEVICE)",)


class SccCuf(Transformation):

    def __init__(self, horizontal, vertical=None, block_dim=None, disable=None,
                 transformation_type=0):

        self.horizontal = horizontal
        self.vertical = vertical
        self.block_dim = block_dim

        self.transformation_type = transformation_type
        assert self.transformation_type in [0, 1, 2]

        if disable is None:
            self.disable = ()
        else:
            self.disable = [_.upper() for _ in disable]

    def transform_module(self, module, **kwargs):
        role = kwargs.get('role')
        targets = kwargs.get('targets', None)

        if role == 'driver':
            module.spec.prepend(ir.Import(module="cudafor"))

    def transform_subroutine(self, routine, **kwargs):

        item = kwargs.get('item', None)

        if item and not item.local_name == routine.name.lower():
            return

        role = kwargs.get('role')
        targets = kwargs.get('targets', None)
        depths = kwargs.get('depths', None)

        remove_pragmas(routine)
        single_variable_declarations(routine=routine, strict=False)
        device_subroutine_prefix(routine, depths[item])

        # TODO: needed for every subroutine or only those with THREADIDX...
        if depths[item] > 0:
            routine.spec.prepend(ir.Import(module="cudafor"))

        if role == 'driver':
            self.process_routine_driver(routine, depth=depths[item], targets=targets)

        if role == 'kernel':
            self.process_routine_kernel(routine, depth=depths[item], targets=targets)

    def process_routine_kernel(self, routine, depth=1, targets=None):

        v_index = get_integer_variable(routine, name=self.horizontal.index)
        resolve_associates(routine)
        resolve_masked_stmts(routine, loop_variable=v_index)
        resolve_vector_dimension(routine, loop_variable=v_index, bounds=self.horizontal.bounds)
        kernel_remove_vector_loops(routine, self.horizontal)

        kernel_block_size_argument(routine, self.horizontal, self.vertical, self.block_dim, self.disable,
                                   self.transformation_type, depth=depth)

        # dynamic memory allocation of local arrays (only for version with dynamic memory allocation on device)
        if self.transformation_type == 2:
            dynamic_local_arrays(routine, self.vertical)

    def process_routine_driver(self, routine, depth=0, targets=None):

        # create variables needed for the device execution, especially generate device versions of arrays
        driver_device_variables(routine=routine, disable=self.disable)
        # remove block loop and generate launch configuration for CUF kernels
        driver_launch_configuration(routine=routine, block_dim=self.block_dim, disable=self.disable)

        # increase heap size (only for version with dynamic memory allocation on device)
        if self.transformation_type == 2:
            increase_heap_size(routine)
