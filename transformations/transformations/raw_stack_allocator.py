# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict

from loki.transform.transformation import Transformation
from loki.transform.transform_utilities import recursive_expression_map_update
from loki.expression import Array, Scalar
from loki.types import DerivedType, BasicType
from loki.analyse import dataflow_analysis_attached
from loki.expression.symbolic import is_dimension_constant, simplify
from loki.expression.symbols import Variable, Literal, Product, Sum, InlineCall, Cast, IntLiteral, LogicLiteral, RangeIndex
from loki.expression.mappers import DetachScopesMapper
from loki.expression.expr_visitors import FindVariables, SubstituteExpressions
from loki.types import SymbolAttributes
from loki.ir import Assignment, Intrinsic, CallStatement
from loki.tools import as_tuple
from loki.visitors import FindNodes
from loki.bulk import SubroutineItem

from loki import fgen

__all__ = ['TemporariesRawStackTransformation']


class TemporariesRawStackTransformation(Transformation):
    """
    Transformation to inject a pool allocator that allocates a large scratch space per block
    on the driver and maps temporary arrays in kernels to this scratch space

    It is built on top of a derived type declared in a separate Fortran module (by default
    called ``stack_mod``), which should simply be commited to the target code base and included
    into the list of source files for transformed targets. It should look similar to this:

    .. code-block:: Fortran

        MODULE STACK_MOD
            IMPLICIT NONE
            TYPE STACK
                INTEGER*8 :: L, U
            END TYPE
            PRIVATE
            PUBLIC :: STACK
        END MODULE

    It provides two integer variables, ``L`` and ``U``, which are used as a stack pointer and
    stack end pointer, respectively. Naming is flexible and can be changed via options to the transformation.

    The transformation needs to be applied in reverse order, which will do the following for each **kernel**:

    * Import the ``STACK`` derived type
    * Add an argument to the kernel call signature to pass the stack derived type
    * Create a local copy of the stack derived type inside the kernel
    * Determine the combined size of all local arrays that are to be allocated by the pool allocator,
      taking into account calls to nested kernels. This is reported in :any:`Item`'s ``trafo_data``.
    * Inject Cray pointer assignments and stack pointer increments for all temporaries
    * Pass the local copy of the stack derived type as argument to any nested kernel calls

    By default, all local array arguments are allocated by the pool allocator, but this can be restricted
    to include only those that have at least one dimension matching one of those provided in :data:`allocation_dims`.

    In a **driver** routine, the transformation will:

    * Determine the required scratch space from ``trafo_data``
    * Allocate the scratch space to that size
    * Insert data transfers (for OpenACC offloading)
    * Insert data sharing clauses into OpenMP or OpenACC pragmas
    * Assign stack base pointer and end pointer for each block (identified via :data:`block_dim`)
    * Pass the stack argument to kernel calls

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension
        to use for hoisted column arrays if hoisting is enabled.
    stack_type_name : str, optional
        Name of the derived type for the stack definition (default: ``'STACK'``)
    stack_ptr_name : str, optional
        Name of the stack pointer variable inside the derived type (default: ``'L'``)
    stack_end_name : str, optional
        Name of the stack end pointer variable inside the derived type (default: ``'U'``)
    stack_size_name : str, optional
        Name of the variable that holds the size of the scratch space in the
        driver (default: ``'ISTSZ'``)
    stack_storage_name : str, optional
        Name of the scratch space variable that is allocated in the
        driver (default: ``'ZSTACK'``)
    stack_argument_name : str, optional
        Name of the stack argument that is added to kernels (default: ``'YDSTACK'``)
    stack_local_var_name : str, optional
        Name of the local copy of the stack argument (default: ``'YLSTACK'``)
    local_ptr_var_name_pattern : str, optional
        Python format string pattern for the name of the Cray pointer variable
        for each temporary (default: ``'IP_{name}'``)
    directive : str, optional
        Can be ``'openmp'`` or ``'openacc'``. If given, insert data sharing clauses for
        the stack derived type, and insert data transfer statements (for OpenACC only).
    check_bounds : bool, optional
        Insert bounds-checks in the kernel to make sure the allocated stack size is not
        exceeded (default: `True`)
    key : str, optional
        Overwrite the key that is used to store analysis results in ``trafo_data``.
    """

    _key = 'TemporariesRawStackTransformation'

    def __init__(self, block_dim, horizontal,
                 stack_type_name='STACK', stack_ptr_name='L',
                 stack_end_name='U', stack_size_name='ISTSZ', stack_storage_name='ZSTACK',
                 stack_argument_name='PSTACK', stack_local_var_name='YLSTACK', local_ptr_var_name_pattern='IP_{name}', local_int_var_name_pattern='JD_{name}',
                 directive=None, check_bounds=True, key=None, **kwargs):
        super().__init__(**kwargs)
        self.block_dim = block_dim
        self.horizontal = horizontal
        self.stack_type_name = stack_type_name
        self.stack_ptr_name = stack_ptr_name
        self.stack_end_name = stack_end_name
        self.stack_size_name = stack_size_name
        self.stack_storage_name = stack_storage_name
        self.stack_argument_name = stack_argument_name
        self.stack_local_var_name = stack_local_var_name
        self.local_ptr_var_name_pattern = local_ptr_var_name_pattern
        self.local_int_var_name_pattern = local_int_var_name_pattern
        self.directive = directive
        self.check_bounds = check_bounds

        if key:
            self._key = key

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        item = kwargs.get('item', None)
        targets = kwargs.get('targets', None)

        self.stack_type_kind = 'JPRB'
        if item:
            if (real_kind := item.config.get('real_kind', None)):
                self.stack_type_kind = real_kind

            # Initialize set to store kind imports
            item.trafo_data[self._key] = {'kind_imports': {}}

        successors = kwargs.get('successors', ())

        if role == 'kernel':

            for arg in routine.arguments:
                print(arg, arg.type, arg.__class__)

            stack_size = self.apply_raw_stack_allocator_to_temporaries(routine, item=item)
            if item:
                stack_size = self._determine_stack_size(routine, successors, stack_size, item=item)
                item.trafo_data[self._key]['stack_size'] = stack_size


    def apply_raw_stack_allocator_to_temporaries(self, routine, item=None):
        """
        Apply pool allocator to local temporary arrays

        This appends the relevant argument to the routine's dummy argument list and
        creates the assignment for the local copy of the stack type.
        For all local arrays, a Cray pointer is instantiated and the temporaries
        are mapped via Cray pointers to the pool-allocated memory region.

        The cumulative size of all temporary arrays is determined and returned.
        """

        # Find all temporary arrays
        arguments = routine.arguments
        temporary_arrays = [
            var for var in routine.variables
            if isinstance(var, Array) and var not in arguments
        ]

        # Filter out derived-type objects. Partly because the possibility of derived-type
        # nesting increases the complexity of determing allocation size, and partly because `C_SIZEOF`
        # doesn't account for the size of allocatable/pointer members of derived-types.
        if any(isinstance(var.type.dtype, DerivedType) for var in temporary_arrays):
            warning(f'[Loki::PoolAllocator] Derived-type vars in {routine} not supported in pool allocator')
        temporary_arrays = [
            var for var in temporary_arrays if not isinstance(var.type.dtype, DerivedType)
        ]

        # Filter out unused vars
        with dataflow_analysis_attached(routine):
            temporary_arrays = [
                var for var in temporary_arrays
                if var.name.lower() in routine.body.defines_symbols
            ]

        # Filter out variables whose size is known at compile-time
        temporary_arrays = [
            var for var in temporary_arrays
            if not all(is_dimension_constant(d) for d in var.shape)
        ]

        # Filter out variables whose first dimension is not horizontal
        temporary_arrays = [
            var for var in temporary_arrays if (
            isinstance(var.shape[0], Scalar) and
            var.shape[0].name.lower() == self.horizontal.size.lower())
        ]

        # Create stack argument and local stack var
        stack_var = self._get_local_stack_var(routine)

        stack_type = SymbolAttributes(dtype=BasicType.REAL, intent='inout', 
                                      shape=(RangeIndex((None, None)), RangeIndex((None, None))))

        stack_arg = Array(name=self.stack_argument_name, 
                          type=stack_type, scope=routine)

        # Determine size of temporary arrays
        stack_size = Literal(0)

        # Create Cray pointer declarations and "stack allocations"
        integers = []
        allocations = []
        old_int_var = IntLiteral(0)
        old_dim = IntLiteral(0)

        int_type = SymbolAttributes(dtype=BasicType.INTEGER)
        for arr in temporary_arrays:

            
            int_var = Scalar(name=self.local_int_var_name_pattern.format(name=arr.name), scope=routine, type=int_type)
            integers += [int_var]

            if arr.dimensions[0].name.lower() == self.horizontal.size.lower():
                dim = IntLiteral(1)
            else:
                raise RuntimeError('Found non-horizontal dimension in _create_stack_allocation')

            for d in arr.dimensions[1:]:
                if isinstance(d, RangeIndex):
                    dim = simplify(Product((dim, Sum((d.upper, Product((-1,d.lower)), IntLiteral(1))))))
                else:
                    dim = Product((dim, d))

            stack_size = simplify(Sum((stack_size, dim)))
            allocations += [Assignment(lhs=int_var, rhs=simplify(Sum((old_int_var, old_dim))))]

            old_int_var = int_var
            old_dim = dim

            # Store type information of temporary allocation
            if item and (_kind := arr.type.kind):
                if _kind in routine.imported_symbols:
                    item.trafo_data[self._key]['kind_imports'][_kind] = routine.import_map[_kind.name].module.lower()

        routine.variables += as_tuple(integers)
        routine.body.prepend(allocations)

        stack_arg = stack_arg.clone(dimensions=((Variable(name=self.horizontal.size, scope=routine), stack_size)))

        # Keep optional arguments last; a workaround for the fact that keyword arguments are not supported
        # in device code
        arg_pos = [routine.arguments.index(arg) for arg in routine.arguments if arg.type.optional]
        if arg_pos:
            routine.arguments = routine.arguments[:arg_pos[0]] + (stack_arg,) + routine.arguments[arg_pos[0]:]
        else:
            routine.arguments += (stack_arg,)
        return stack_size


    def _get_local_stack_var(self, routine):
        """
        Utility routine to get the local stack variable

        The variable is created and added to :data:`routine` if it doesn't exist, yet.
        """
        if self.stack_local_var_name in routine.variables:
            return routine.variable_map[self.stack_local_var_name]

        stack_type = SymbolAttributes(dtype=DerivedType(name=self.stack_type_name))
        stack_var = Variable(name=self.stack_local_var_name, type=stack_type, scope=routine)
        routine.variables += (stack_var,)
        return stack_var


    def _get_stack_arg(self, routine):
        """
        Utility routine to get the stack argument

        The argument is created and added to the dummy argument list of :data:`routine`
        if it doesn't exist, yet.
        """
        if self.stack_argument_name in routine.arguments:
            return routine.variable_map[self.stack_argument_name]

        stack_type = SymbolAttributes(dtype=BasicType.REAL, intent='inout')
        stack_arg = Array(name=self.stack_argument_name, type=stack_type, scope=routine)

        # Keep optional arguments last; a workaround for the fact that keyword arguments are not supported
        # in device code
        arg_pos = [routine.arguments.index(arg) for arg in routine.arguments if arg.type.optional]
        if arg_pos:
            routine.arguments = routine.arguments[:arg_pos[0]] + (stack_arg,) + routine.arguments[arg_pos[0]:]
        else:
            routine.arguments += (stack_arg,)

        return stack_arg



    def _get_stack_ptr(self, routine):
        """
        Utility routine to get the stack pointer variable
        """
        return Variable(
            name=f'{self.stack_local_var_name}%{self.stack_ptr_name}',
            parent=self._get_local_stack_var(routine),
            scope=routine
        )


    def _get_stack_end(self, routine):
        """
        Utility routine to get the stack end pointer variable
        """
        return Variable(
            name=f'{self.stack_local_var_name}%{self.stack_end_name}',
            parent=self._get_local_stack_var(routine),
            scope=routine
        )


    def _create_stack_allocation(self, int_var, old_int_var, old_dim, arr, stack_size):
        """
        Utility routine to "allocate" a temporary array on the pool allocator's "stack"

        This creates the pointer assignment, stack pointer increment and adds a stack size check.

        Parameters
        ----------
        stack_ptr : :any:`Variable`
            The stack pointer variable
        arr : :any:`Variable`
            The temporary array to allocate on the pool allocator's "stack"
        stack_size : :any:`Variable`
            The size in bytes of the pool allocator's "stack"

        Returns
        -------
        list
            The IR nodes to add for the stack allocation: an :any:`Assignment` for the pointer
            association, an :any:`Assignment` for the stack pointer increment, and a
            :any:`Conditional` that verifies that the stack is big enough
        """

        print('horizontal size: ', self.horizontal.size)

        # Build expression for array size in bytes.
        # Assert first dimension is horizontal
        if arr.dimensions[0].name.lower() == self.horizontal.size.lower():
            dim = IntLiteral(1)
        else:
            raise RuntimeError('Found non-horizontal dimension in _create_stack_allocation')

        for d in arr.dimensions[1:]:
            if isinstance(d, RangeIndex):
                dim = simplify(Product((dim, Sum((d.upper, Product((-1,d.lower)), IntLiteral(1))))))
            else:
                dim = Product((dim, d))

        print('stack_size: ', stack_size)
        print('dim: ', dim)
        # Increment stack size
        stack_size = simplify(Sum((stack_size, dim)))

        int_increment = Assignment(lhs=int_var, rhs=Sum((old_int_var, dim)))
        print('stack_size: ', stack_size)
        return int_increment, dim, stack_size


    def _determine_stack_size(self, routine, successors, local_stack_size=None, item=None):
        """
        Utility routine to determine the stack size required for the given :data:`routine`,
        including calls to subroutines

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine object for which to determine the stack size
        successors : list of :any:`Item`
            The items corresponding to successor routines called from :data:`routine`
        local_stack_size : :any:`Expression`, optional
            The stack size required for temporaries in :data:`routine`
        item : :any:`Item`
            Scheduler work item corresponding to routine.

        Returns
        -------
        :any:`Expression` :
            The expression representing the required stack size.
        """

        # Collect variable kind imports from successors
        if item:
            item.trafo_data[self._key]['kind_imports'].update({k: v
                                                           for s in successors if isinstance(s, SubroutineItem)
                                                           for k, v in s.trafo_data[self._key]['kind_imports'].items()})

        # Note: we are not using a CaseInsensitiveDict here to be able to search directly with
        # Variable instances in the dict. The StrCompareMixin takes care of case-insensitive
        # comparisons in that case
        successor_map = {successor.routine.name.lower(): successor for successor in successors
                                                         if isinstance(successor, SubroutineItem)}

        # Collect stack sizes for successors
        # Note that we need to translate the names of variables used in the expressions to the
        # local names according to the call signature
        stack_sizes = []
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in successor_map and self._key in successor_map[call.name].trafo_data:
                successor_stack_size = successor_map[call.name].trafo_data[self._key]['stack_size']
                # Replace any occurence of routine arguments in the stack size expression
                arg_map = dict(call.arg_iter())
                expr_map = {
                    expr: DetachScopesMapper()(arg_map[expr]) for expr in FindVariables().visit(successor_stack_size)
                    if expr in arg_map
                }
                if expr_map:
                    expr_map = recursive_expression_map_update(expr_map)
                    successor_stack_size = SubstituteExpressions(expr_map).visit(successor_stack_size)
                stack_sizes += [successor_stack_size]

        # Unwind "max" expressions from successors and inject the local stack size into the expressions
        stack_sizes = [
            d for s in stack_sizes
            for d in (s.parameters if isinstance(s, InlineCall) and s.function == 'MAX' else [s])
        ]
        if local_stack_size:
            local_stack_size = DetachScopesMapper()(simplify(local_stack_size))
            stack_sizes = [simplify(Sum((local_stack_size, s))) for s in stack_sizes]

        if not stack_sizes:
            # Return only the local stack size if there are no callees
            return local_stack_size or Literal(0)

        if len(stack_sizes) == 1:
            # For a single successor, it is sufficient to add the local stack size to the expression
            return stack_sizes[0]

        # Re-build the max expressions, taking into account the local stack size and calls to successors
        stack_size = InlineCall(function=Variable(name='MAX'), parameters=as_tuple(stack_sizes), kw_parameters=())
        return stack_size

