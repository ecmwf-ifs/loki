# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Single-Column-Coalesced CUDA Fortran (SCC-CUF) transformation.
"""

from loki.logging import info
from loki.batch import Transformation
from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, Transformer, FindVariables,
    SubstituteExpressions
)
from loki.tools import CaseInsensitiveDict, as_tuple, flatten
from loki.types import BasicType, DerivedType
from loki.scope import SymbolAttributes

from loki.transformations.hoist_variables import HoistVariablesTransformation
from loki.transformations.sanitise import resolve_associates
from loki.transformations.single_column.base import SCCBaseTransformation
from loki.transformations.single_column.vector import SCCDevectorTransformation
from loki.transformations.utilities import single_variable_declaration
from loki.ir.pragma_utils import get_pragma_parameters
from loki.transformations.utilities import get_integer_variable

__all__ = [
    'HoistTemporaryArraysDeviceAllocatableTransformation',
    'HoistTemporaryArraysPragmaOffloadTransformation',
    'SccLowLevelLaunchConfiguration',
    'SccLowLevelDataOffload',
]


class HoistTemporaryArraysDeviceAllocatableTransformation(HoistVariablesTransformation):
    """
    Synthesis part for variable/array hoisting for CUDA Fortran (CUF) (transformation).
    """

    def driver_variable_declaration(self, routine, variables):
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
        for var in variables:
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


class HoistTemporaryArraysPragmaOffloadTransformation(HoistVariablesTransformation):
    """
    Synthesis part for variable/array hoisting, offload via pragmas e.g., OpenACC.
    """

    def driver_variable_declaration(self, routine, variables):
        """
        Standard Variable/Array declaration including
        device offload via pragmas.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to add the variable declaration
        var: :any:`Variable`
            The variable to be declared
        """
        routine.variables += tuple(var.clone(scope=routine) for var in variables)

        vnames = ', '.join(v.name for v in variables)
        pragma = ir.Pragma(keyword='acc', content=f'enter data create({vnames})')
        pragma_post = ir.Pragma(keyword='acc', content=f'exit data delete({vnames})')

        # Add comments around standalone pragmas to avoid false attachment
        routine.body.prepend((ir.Comment(''), pragma, ir.Comment('')))
        routine.body.append((ir.Comment(''), pragma_post, ir.Comment('')))


def remove_non_loki_pragmas(routine):
    """
    Remove all pragmas.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine in which to remove all pragmas
    """
    pragma_map = {p: None for p in FindNodes(ir.Pragma).visit(routine.body) if p.keyword.lower()!="loki"}
    routine.body = Transformer(pragma_map).visit(routine.body)

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

class SccLowLevelLaunchConfiguration(Transformation):
    """
    Part of the pipeline for generating Single Column Coalesced
    Low Level GPU (CUDA Fortran, CUDA C, HIP, ...) for block-indexed gridpoint/single-column
    routines (responsible for the launch configuration including the chevron notation).
    """

    def __init__(self, horizontal, vertical, block_dim, transformation_type='parametrise', mode="CUF"):
        """
        Part of the pipeline for generating Single Column Coalesced
        Low Level GPU (CUDA Fortran, CUDA C, HIP, ...) for block-indexed gridpoint/single-column
        routines responsible for the launch configuration including the chevron notation.

        .. note::
            In dependence of the transformation type ``transformation_type``, further
            transformations are necessary:

            * ``transformation_type = 'parametrise'`` requires a subsequent
              :any:`ParametriseTransformation` transformation with the necessary information
              to parametrise (at least) the ``vertical`` `size`
            * ``transformation_type = 'hoist'`` requires subsequent :any:`HoistVariablesAnalysis`
              and :class:`HoistVariablesTransformation` transformations (e.g.
              :any:`HoistTemporaryArraysAnalysis` for analysis and
              :any:`HoistTemporaryArraysTransformationDeviceAllocatable` or
              :any:`HoistTemporaryArraysPragmaOffloadTransformation` for synthesis)

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
        transformation_type : str
            Kind of transformation/Handling of temporaries/local arrays

            - `parametrise`: parametrising the array dimensions to make the vertical dimension
              a compile-time constant
            - `hoist`: host side hoisting of (relevant) arrays
        mode: str
            Mode/language to target

            - `CUF` - CUDA Fortran
            - `CUDA` - CUDA C
            - `HIP` - HIP
        """
        self.horizontal = horizontal
        self.vertical = vertical
        self.block_dim = block_dim
        self.mode = mode.lower()
        assert self.mode in ['cuf', 'cuda', 'hip']

        self.transformation_type = transformation_type
        # `parametrise` : parametrising the array dimensions
        # `hoist`: host side hoisting
        info(f"[SccLowLevelLaunchConfiguration] Applying transformation type: '{self.transformation_type}'")
        assert self.transformation_type in ['parametrise', 'hoist']
        self.transformation_description = {'parametrise': 'parametrised array dimensions of local arrays',
                                           'hoist': 'host side hoisted local arrays'}

    def transform_subroutine(self, routine, **kwargs):

        item = kwargs.get('item', None)
        role = kwargs.get('role')
        depths = kwargs.get('depths', None)
        targets = kwargs.get('targets', None)
        depth = 0
        if depths is None:
            if role == 'driver':
                depth = 0
            elif role == 'kernel':
                depth = 1
        else:
            depth = depths[item]

        remove_non_loki_pragmas(routine)
        single_variable_declaration(routine=routine)
        device_subroutine_prefix(routine, depth)

        if self.mode == 'cuf':
            routine.spec.prepend(ir.Import(module="cudafor"))

        if role == 'driver':
            self.process_driver(routine, targets=targets)
        if role == 'kernel':
            self.process_kernel(routine, depth=depth, targets=targets)

        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if call.name in as_tuple(targets):
                # call.sort_kwarguments()
                call.convert_kwargs_to_args()

    def process_kernel(self, routine, depth=1, targets=None):
        """
        Kernel/Device subroutine specific changes/transformations.
        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine (kernel/device subroutine) to process
        depth: int
            The subroutines depth
        """

        self.kernel_cuf(
            routine, self.horizontal, self.vertical, self.block_dim, depth=depth,
            targets=targets
        )

    def process_driver(self, routine, targets=None):
        """
        Driver subroutine specific changes/transformations.
        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine (driver) to process
        """

        upper, step, _, blockdim_var, griddim_var, blockdim_assignment, griddim_assignment =\
                self.driver_launch_configuration(routine=routine, block_dim=self.block_dim, targets=targets)

        if self.mode in ['cuda', 'hip']:
            call_map = {}
            for call in FindNodes(ir.CallStatement).visit(routine.body):
                if str(call.name).lower() in as_tuple(targets):
                    new_args = ()
                    if upper.name not in call.routine.arguments:
                        new_args += (upper.clone(type=upper.type.clone(intent='in'), scope=call.routine),)
                    if step.name not in call.routine.arguments:
                        new_args += (step.clone(type=step.type.clone(intent='in'), scope=call.routine),)
                    new_kwargs = tuple((_.name, _) for _ in new_args)
                    if new_args:
                        call.routine.arguments = list(call.routine.arguments) + list(new_args)
                        call_map[call] = call.clone(kwarguments=as_tuple(list(call.kwarguments) + list(new_kwargs)))
                    call.routine.variables += (blockdim_var, griddim_var)
                    call.routine.body = (blockdim_assignment, griddim_assignment) + as_tuple(call.routine.body)
            routine.body = Transformer(call_map).visit(routine.body)
        elif self.mode == 'cuf':
            routine.body.prepend(ir.Comment(f"!@cuf print *, 'executing SCC-CUF type: {self.transformation_type} - "
                                            f"{self.transformation_description[self.transformation_type]}'"))
            routine.body.prepend(ir.Comment(""))

    def kernel_cuf(self, routine, horizontal, vertical, block_dim,
               depth, targets=None):

        if SCCBaseTransformation.is_elemental(routine):
            # TODO: correct "definition" of elemental/pure routines and corresponding removing
            #  of subroutine prefix(es)/specifier(s)
            routine.prefix = as_tuple([prefix for prefix in routine.prefix if prefix not in ["ELEMENTAL"]]) #,"PURE"]])
            return

        single_variable_declaration(routine, variables=(horizontal.index, block_dim.index))

        #  this does not make any difference ...
        self.kernel_demote_private_locals(routine, horizontal, vertical)

        # find vertical and block loops and replace with implicit "loops"
        loop_map = {}
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if loop.variable == self.block_dim.index or loop.variable.name.lower()\
                    in [_.lower() for _ in self.block_dim.sizes]:
                loop_map[loop] = loop.body
            if loop.variable == self.horizontal.index or loop.variable.name.lower()\
                    in [_.lower() for _ in self.horizontal.sizes]:
                loop_map[loop] = loop.body
        routine.body = Transformer(loop_map).visit(routine.body)

        if depth == 1:

            ## bit hacky ...
            assignments = FindNodes(ir.Assignment).visit(routine.body)
            assignments2remove = [block_dim.index.lower()] + [_.lower() for _ in horizontal.bounds]
            assignment_map = {assign: None for assign in assignments if assign.lhs.name.lower() in assignments2remove}
            routine.body = Transformer(assignment_map).visit(routine.body)
            ##end: bit hacky

            # CUDA thread mapping
            if self.mode == 'cuf':
                var_thread_idx = sym.Variable(name="THREADIDX")
                var_x = sym.Variable(name="X", parent=var_thread_idx)
            else:
                ctype = SymbolAttributes(DerivedType(name="threadIdx"))
                var_thread_idx = sym.Variable(name="threadIdx", case_sensitive=True)
                var_x = sym.Variable(name="x", parent=var_thread_idx, case_sensitive=True, type=ctype)
            horizontal_assignment = ir.Assignment(lhs=routine.variable_map[horizontal.index], rhs=var_x)

            if self.mode == 'cuf':
                var_thread_idx = sym.Variable(name="BLOCKIDX")
                var_x = sym.Variable(name="Z", parent=var_thread_idx)
            else:
                ctype = SymbolAttributes(DerivedType(name="blockIdx"))
                var_thread_idx = sym.Variable(name="blockIdx", case_sensitive=True)
                var_x = sym.Variable(name="x", parent=var_thread_idx, case_sensitive=True, type=ctype)
            block_dim_assignment = ir.Assignment(lhs=routine.variable_map[block_dim.index], rhs=var_x)

            condition = sym.LogicalAnd((sym.Comparison(routine.variable_map[block_dim.index], '<=',
                                                       routine.variable_map[block_dim.size]),
                                        sym.Comparison(routine.variable_map[horizontal.index], '<=',
                                                       routine.variable_map[horizontal.size])))

            routine.body = ir.Section((horizontal_assignment, block_dim_assignment, ir.Comment(''),
                            ir.Conditional(condition=condition, body=as_tuple(routine.body), else_body=())))
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if call.routine.name.lower() in targets and not SCCBaseTransformation.is_elemental(call.routine):
                horizontal_index = routine.variable_map[horizontal.index]
                block_dim_index = routine.variable_map[block_dim.index]
                additional_args = ()
                additional_kwargs = ()
                if horizontal_index.name not in call.routine.arguments:
                    if horizontal_index.name in call.routine.variables:
                        call.routine.symbol_attrs.update({horizontal_index.name:\
                                call.routine.variable_map[horizontal_index.name].type.clone(intent='in')})
                    additional_args += (horizontal_index.clone(),)
                if horizontal_index.name not in call.arg_map:
                    additional_kwargs += ((horizontal_index.name, horizontal_index.clone()),)

                if block_dim_index.name not in call.routine.arguments:
                    additional_args += (block_dim_index.clone(type=block_dim_index.type.clone(intent='in',
                        scope=call.routine)),)
                    additional_kwargs += ((block_dim_index.name, block_dim_index.clone()),)
                if additional_kwargs:
                    call._update(kwarguments=call.kwarguments+additional_kwargs)
                if additional_args:
                    call.routine.arguments += additional_args

    @staticmethod
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

    def driver_launch_configuration(self, routine, block_dim, targets=None):
        """
        Launch configuration for kernel calls within the driver with the
        CUDA Fortran (CUF) specific chevron syntax `<<<griddim, blockdim>>>`.
        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to specify the launch configurations for kernel calls.
        block_dim: :any:`Dimension`
            The dimension object specifying the block loop dimension
        targets : tuple of str
            Tuple of subroutine call names that are processed in this traversal
        """

        d_type = SymbolAttributes(DerivedType("dim3"))
        blockdim_var = sym.Variable(name="BLOCKDIM", type=d_type)
        griddim_var = sym.Variable(name="GRIDDIM", type=d_type)
        if self.mode == 'cuf':
            routine.spec.append(ir.VariableDeclaration(symbols=(griddim_var, blockdim_var)))

        # istat: status of CUDA runtime function (e.g. for cudaDeviceSynchronize(), cudaMalloc(), cudaFree(), ...)
        i_type = SymbolAttributes(BasicType.INTEGER)
        routine.spec.append(ir.VariableDeclaration(symbols=(sym.Variable(name="istat", type=i_type),)))

        blockdim_assignment = None
        griddim_assignment = None
        mapper = {}

        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if call.name not in as_tuple(targets):
                continue

            if call.pragma:
                parameters = get_pragma_parameters(call.pragma, starts_with='removed_loop')
            else:
                parameters = ()
            assignment_lhs = routine.variable_map["istat"]
            assignment_rhs = sym.InlineCall(
                function=sym.ProcedureSymbol(name="cudaDeviceSynchronize", scope=routine),
                parameters=())

            upper = routine.variable_map[parameters['upper']]
            try:
                step = routine.variable_map[parameters['step']]
            except Exception as e:
                print(f"Exception: {e}")
                step = sym.IntLiteral(1)


            if self.mode == 'cuf':
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
                mapper[call] = (blockdim_assignment, griddim_assignment, ir.Comment(""),
                        call.clone(chevron=(routine.variable_map["GRIDDIM"], routine.variable_map["BLOCKDIM"]),),
                        ir.Assignment(lhs=assignment_lhs, rhs=assignment_rhs))
            else:
                func_dim3 = sym.ProcedureSymbol(name="dim3", scope=routine)
                func_ceiling = sym.ProcedureSymbol(name="ceil", scope=routine)

                # BLOCKDIM
                lhs = blockdim_var
                rhs = sym.InlineCall(function=func_dim3, parameters=(step, sym.IntLiteral(1), sym.IntLiteral(1)))
                blockdim_assignment = ir.Assignment(lhs=lhs, rhs=rhs)
                # GRIDDIM
                lhs = griddim_var
                rhs = sym.InlineCall(function=func_dim3, parameters=(sym.InlineCall(function=func_ceiling,
                    parameters=as_tuple(
                        sym.Cast(name="REAL", expression=upper) /
                        sym.Cast(name="REAL", expression=step))),
                    sym.IntLiteral(1), sym.IntLiteral(1)))
                griddim_assignment = ir.Assignment(lhs=lhs, rhs=rhs)

        routine.body = Transformer(mapper=mapper).visit(routine.body)
        return upper, step, routine.variable_map[block_dim.size], blockdim_var, griddim_var,\
                blockdim_assignment, griddim_assignment


class SccLowLevelDataOffload(Transformation):
    """
    Part of the pipeline for generating Single Column Coalesced
    Low Level GPU (CUDA Fortran, CUDA C, HIP, ...) for block-indexed gridpoint/single-column
    routines (responsible for the data offload).
    """

    def __init__(self, horizontal, vertical, block_dim, transformation_type='parametrise',
                 derived_types=None, mode="CUF"):
        """
        Part of the pipeline for generating Single Column Coalesced
        Low Level GPU (CUDA Fortran, CUDA C, HIP, ...) for block-indexed gridpoint/single-column
        routines responsible for the data offload..

        .. note::
            In dependence of the transformation type ``transformation_type``, further
            transformations are necessary:

            * ``transformation_type = 'parametrise'`` requires a subsequent
              :any:`ParametriseTransformation` transformation with the necessary information
              to parametrise (at least) the ``vertical`` `size`
            * ``transformation_type = 'hoist'`` requires subsequent :any:`HoistVariablesAnalysis`
              and :class:`HoistVariablesTransformation` transformations (e.g.
              :any:`HoistTemporaryArraysAnalysis` for analysis and
              :any:`HoistTemporaryArraysTransformationDeviceAllocatable` or
              :any:`HoistTemporaryArraysPragmaOffloadTransformation` for synthesis)

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
        derived_types: tuple
            Derived types that are relevant
        transformation_type : str
            Kind of transformation/Handling of temporaries/local arrays

            - `parametrise`: parametrising the array dimensions to make the vertical dimension
              a compile-time constant
            - `hoist`: host side hoisting of (relevant) arrays
        mode: str
            Mode/language to target

            - `CUF` - CUDA Fortran
            - `CUDA` - CUDA C
            - `HIP` - HIP
        """
        self.horizontal = horizontal
        self.vertical = vertical
        self.block_dim = block_dim
        self.mode = mode.lower()
        assert self.mode in ['cuf', 'cuda', 'hip']

        self.transformation_type = transformation_type
        # `parametrise` : parametrising the array dimensions
        # `hoist`: host side hoisting
        assert self.transformation_type in ['parametrise', 'hoist']
        self.transformation_description = {'parametrise': 'parametrised array dimensions of local arrays',
                                           'hoist': 'host side hoisted local arrays'}

        if derived_types is None:
            self.derived_types = ()
        else:
            self.derived_types = [_.upper() for _ in derived_types]
        self.derived_type_variables = ()

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs.get('role')
        targets = kwargs.get('targets', None)

        remove_non_loki_pragmas(routine)
        single_variable_declaration(routine=routine, group_by_shape=True)

        if self.mode == 'cuf':
            routine.spec.prepend(ir.Import(module="cudafor"))

        if role == 'driver':
            self.process_driver(routine, targets=targets)
        if role == 'kernel':
            self.process_kernel(routine) # , depth=depth, targets=targets)

        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if str(call.name).lower() in as_tuple(targets):
                call.convert_kwargs_to_args()

    def process_driver(self, routine, targets=None):
        """
        Driver subroutine specific changes/transformations.
        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine (driver) to process
        """

        self.derived_type_variables = self.device_derived_types(
            routine=routine, derived_types=self.derived_types, targets=targets
        )
        # create variables needed for the device execution, especially generate device versions of arrays
        self.driver_device_variables(routine=routine, targets=targets)

    def process_kernel(self, routine): # , depth=1, targets=None):
        """
        Kernel/Device subroutine specific changes/transformations.
        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine (kernel/device subroutine) to process
        """

        v_index = get_integer_variable(routine, name=self.horizontal.index)
        resolve_associates(routine)
        SCCBaseTransformation.resolve_masked_stmts(routine, loop_variable=v_index)
        SCCBaseTransformation.resolve_vector_dimension(routine, loop_variable=v_index, bounds=self.horizontal.bounds)
        SCCDevectorTransformation.kernel_remove_vector_loops(routine, self.horizontal)

        self.kernel_cuf(
            routine, self.horizontal, self.block_dim, self.transformation_type,
            derived_type_variables=self.derived_type_variables
        )

    def kernel_cuf(self, routine, horizontal, block_dim, transformation_type,
               derived_type_variables):

        relevant_local_arrays = []
        var_map = {}
        for var in routine.variables:
            if var in routine.arguments:
                if isinstance(var, sym.Scalar) and var not in derived_type_variables\
                        and var.type.intent.lower() == 'in':
                    var_map[var] = var.clone(type=var.type.clone(value=True))
            else:
                if isinstance(var, sym.Array):
                    dimensions = list(var.dimensions)
                    shape = list(var.shape)
                    if horizontal.size in list(FindVariables().visit(var.dimensions)):
                        if transformation_type == 'hoist':
                            dimensions += [routine.variable_map[block_dim.size]]
                            shape = list(var.shape) + [routine.variable_map[block_dim.size]]
                            vtype = var.type.clone(shape=as_tuple(shape))
                            relevant_local_arrays.append(var.name)
                        else:
                            dimensions.remove(horizontal.size)
                            shape.remove(horizontal.size)
                            relevant_local_arrays.append(var.name)
                            vtype = var.type.clone(device=True, shape=shape)
                        var_map[var] = var.clone(dimensions=as_tuple(dimensions), type=vtype)

        routine.spec = SubstituteExpressions(var_map).visit(routine.spec)

        var_map = {}
        arguments_name = [var.name for var in routine.arguments]
        for var in FindVariables().visit(routine.body):
            if var.name not in arguments_name:
                if transformation_type == 'hoist':
                    if var.name in relevant_local_arrays:
                        var_map[var] = var.clone(dimensions=var.dimensions + (routine.variable_map[block_dim.index],))
                else:
                    if var.name in relevant_local_arrays:
                        dimensions = list(var.dimensions)
                        var_map[var] = var.clone(dimensions=as_tuple(dimensions[1:]))

        routine.body = SubstituteExpressions(var_map).visit(routine.body)

    def device_derived_types(self, routine, derived_types, targets=None):
        """
        Create device versions of variables of specific derived types including
        host-device-synchronisation.
        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to create device versions of the specified derived type variables.
        derived_types: tuple
            Tuple of derived types within the routine
        targets : tuple of str
            Tuple of subroutine call names that are processed in this traversal
        """
        _variables = list(FindVariables().visit(routine.ir))
        variables = []
        for var in _variables:
            for derived_type in derived_types:
                if derived_type in str(var.type):
                    variables.append(var)

        var_map = {}
        for var in variables:
            new_var = var.clone(name=f"{var.name}_d", type=var.type.clone(intent=None, imported=None,
                                                                          allocatable=None, device=True,
                                                                          module=None))
            var_map[var] = new_var
            routine.spec.append(ir.VariableDeclaration((new_var,)))
            routine.body.prepend(ir.Assignment(lhs=new_var, rhs=var))

        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if call.name not in as_tuple(targets):
                continue
            arguments = tuple(var_map.get(arg, arg) for arg in call.arguments)
            call._update(arguments=arguments)
        return variables

    def driver_device_variables(self, routine, targets=None):
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
        targets : tuple of str
            Tuple of subroutine call names that are processed in this traversal
        """

        # # istat: status of CUDA runtime function (e.g. for cudaDeviceSynchronize(), cudaMalloc(), cudaFree(), ...)
        # i_type = SymbolAttributes(types.BasicType.INTEGER)
        # routine.spec.append(ir.VariableDeclaration(symbols=(sym.Variable(name="istat", type=i_type),)))

        relevant_arrays = []
        calls = tuple(
            call for call in FindNodes(ir.CallStatement).visit(routine.body)
            if call.name in as_tuple(targets)
        )
        for call in calls:
            relevant_arrays.extend([arg for arg in call.arguments if isinstance(arg, sym.Array)])

        relevant_arrays = list(dict.fromkeys(relevant_arrays))

        if self.mode in ['cuda', 'hip']:
            # Collect the three types of device data accesses from calls
            inargs = ()
            inoutargs = ()
            outargs = ()

            # insert_index = routine.body.body.index(calls[-1])
            # insert_index = None
            for call in calls:
                if call.routine is BasicType.DEFERRED:
                    # warning(f'[Loki] Data offload: Routine {routine.name} has not been enriched with ' +
                    #     f'in {str(call.name).lower()}')
                    continue
                for param, arg in call.arg_iter():
                    if isinstance(param, sym.Array) and param.type.intent.lower() == 'in':
                        inargs += (str(arg.name).lower(),)
                    if isinstance(param, sym.Array) and param.type.intent.lower() == 'inout':
                        inoutargs += (str(arg.name).lower(),)
                    if isinstance(param, sym.Array) and param.type.intent.lower() == 'out':
                        outargs += (str(arg.name).lower(),)

            # Sanitize data access categories to avoid double-counting variables
            inoutargs += tuple(v for v in inargs if v in outargs)
            inargs = tuple(v for v in inargs if v not in inoutargs)
            outargs = tuple(v for v in outargs if v not in inoutargs)

            # Filter for duplicates
            inargs = tuple(dict.fromkeys(inargs))
            outargs = tuple(dict.fromkeys(outargs))
            inoutargs = tuple(dict.fromkeys(inoutargs))

            copy_pragmas = []
            copy_end_pragmas = []
            if outargs:
                copy_pragmas += [ir.Pragma(keyword='acc', content=f'data copyout({", ".join(outargs)})')]
                copy_end_pragmas += [ir.Pragma(keyword='acc', content='end data')]
            if inoutargs:
                copy_pragmas += [ir.Pragma(keyword='acc', content=f'data copy({", ".join(inoutargs)})')]
                copy_end_pragmas += [ir.Pragma(keyword='acc', content='end data')]
            if inargs:
                copy_pragmas += [ir.Pragma(keyword='acc', content=f'data copyin({", ".join(inargs)})')]
                copy_end_pragmas += [ir.Pragma(keyword='acc', content='end data')]

            if copy_pragmas:
                pragma_map = {}
                for pragma in FindNodes(ir.Pragma).visit(routine.body):
                    if pragma.content == 'data' and 'loki' == pragma.keyword:
                        pragma_map[pragma] = as_tuple(copy_pragmas)
                if pragma_map:
                    routine.body = Transformer(pragma_map).visit(routine.body)
            if copy_end_pragmas:
                pragma_map = {}
                for pragma in FindNodes(ir.Pragma).visit(routine.body):
                    if pragma.content == 'end data' and 'loki' == pragma.keyword:
                        pragma_map[pragma] = as_tuple(copy_end_pragmas)
                if pragma_map:
                    routine.body = Transformer(pragma_map).visit(routine.body)
        else:
            # Declaration
            routine.spec.append(ir.Comment(''))
            routine.spec.append(ir.Comment('! Device arrays'))
            for array in relevant_arrays:
                vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
                vdimensions = [sym.RangeIndex((None, None))] * len(array.shape)
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
                lhs = array.clone(name=f"{array.name}_d", type=vtype, dimensions=())
                rhs = array.clone(dimensions=())
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
                    lhs = v.clone(dimensions=())
                    vtype = v.type.clone(device=True, allocatable=True, intent=None, shape=None)
                    rhs = v.clone(name=f"{v.name}_d", type=vtype, dimensions=())
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
                routine.body.append(ir.Deallocation((array.clone(name=f"{array.name}_d", dimensions=()),)))

            call_map = {}
            for call in calls:
                arguments = []
                for arg in call.arguments:
                    if arg in relevant_arrays:
                        vtype = arg.type.clone(device=True, allocatable=True, intent=None)
                        arguments.append(arg.clone(name=f"{arg.name}_d", type=vtype, dimensions=()))
                    else:
                        arguments.append(arg)
                call_map[call] = call.clone(arguments=as_tuple(arguments))
            routine.body = Transformer(call_map).visit(routine.body)
