# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pdb
from loki import (
    as_tuple, flatten, warning, simplify, recursive_expression_map_update, get_pragma_parameters,
    Transformation, FindNodes, FindVariables, Transformer, SubstituteExpressions, DetachScopesMapper,
    SymbolAttributes, BasicType, DerivedType, Quotient, IntLiteral, IntrinsicLiteral, LogicLiteral,
    Variable, Array, Sum, Literal, Product, InlineCall, Comparison, RangeIndex, Scalar,
    Intrinsic, Assignment, Conditional, CallStatement, Import, Allocation, Deallocation,
    Loop, Pragma, SubroutineItem, FindInlineCalls, Interface, ProcedureSymbol, LogicalNot
)

__all__ = ['TemporariesPoolAllocatorTransformation']


class TemporariesPoolAllocatorTransformation(Transformation):
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
    stack_module_name : str, optional
        Name of the Fortran module containing the derived type definition
        (default: ``'STACK_MOD'``)
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

    _key = 'TemporariesPoolAllocatorTransformation'

    def __init__(self, block_dim,
                 stack_module_name='STACK_MOD', stack_type_name='STACK', stack_ptr_name='L',
                 stack_end_name='U', stack_size_name='ISTSZ', stack_storage_name='ZSTACK',
                 stack_argument_name='YDSTACK', stack_local_var_name='YLSTACK', local_ptr_var_name_pattern='IP_{name}',
                 directive=None, check_bounds=True, key=None, **kwargs):
        super().__init__(**kwargs)
        self.block_dim = block_dim
        self.stack_module_name = stack_module_name
        self.stack_type_name = stack_type_name
        self.stack_ptr_name = stack_ptr_name
        self.stack_end_name = stack_end_name
        self.stack_size_name = stack_size_name
        self.stack_storage_name = stack_storage_name
        self.stack_argument_name = stack_argument_name
        self.stack_local_var_name = stack_local_var_name
        self.local_ptr_var_name_pattern = local_ptr_var_name_pattern
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
            if item.local_name != routine.name.lower():
                return
            if (real_kind := item.config.get('real_kind', None)):
                self.stack_type_kind = real_kind

        # add iso_c_binding import if necessary
        self.import_iso_c_binding(routine)

        successors = kwargs.get('successors', ())

        self.inject_pool_allocator_import(routine)

        if role == 'kernel':
            stack_size = self.apply_pool_allocator_to_temporaries(routine)
            if item:
                stack_size = self._determine_stack_size(routine, successors, stack_size)
                item.trafo_data[self._key] = stack_size

        elif role == 'driver':
            stack_size = self._determine_stack_size(routine, successors)
            self.create_pool_allocator(routine, stack_size)

        self.inject_pool_allocator_into_calls(routine, targets)

    def import_iso_c_binding(self, routine):
        """
        Add the iso_c_binding import if necesssary.
        """

        imports = FindNodes(Import).visit(routine.spec)
        for imp in imports:
            if imp.module.lower() == 'iso_c_binding':
                if 'c_sizeof' in [s for s in imp.symbols] or not imp.symbols:
                    return

                # Update iso_c_binding import
                imp._update(symbols=as_tuple(imp.symbols + ProcedureSymbol('C_SIZEOF', scope=routine)))

        # add qualified iso_c_binding import
        imp = Import(module='ISO_C_BINDING', symbols=as_tuple(ProcedureSymbol('C_SIZEOF', scope=routine)))
        routine.spec.prepend(imp)

    def inject_pool_allocator_import(self, routine):
        """
        Add the import statement for the pool allocator's "stack" type
        """
        if self.stack_type_name not in routine.imported_symbols:
            routine.spec.prepend(Import(
                module=self.stack_module_name, symbols=(Variable(name=self.stack_type_name, scope=routine),)
            ))

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

        stack_type = SymbolAttributes(dtype=DerivedType(name=self.stack_type_name), intent='inout')
        stack_arg = Variable(name=self.stack_argument_name, type=stack_type, scope=routine)

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

    def _get_stack_storage_and_size_var(self, routine, stack_size):
        """
        Utility routine to obtain storage array and size variable for the pool allocator

        If array or size variable already exist, matching the provided names :attr:`stack_size_name`
        and :attr:`stack_storage_name`, they are used directly. Note that this does not validate
        that :data:`stack_size` matches the allocated array size.

        If array or size variable do not exist, yet, they are created as required and initialized or
        allocated accordingly.
        """
        variable_map = routine.variable_map  # Local copy to look-up variables by name

        # Nodes to prepend/append to the routine's body
        body_prepend = []
        body_append = []

        variables_append = []  # New variables to declare in the routine

        if self.stack_size_name in variable_map:
            # Use an existing stack size declaration
            stack_size_var = routine.variable_map[self.stack_size_name]

        else:
            # Create a variable for the stack size and assign the size
            stack_size_var = Variable(name=self.stack_size_name, type=SymbolAttributes(BasicType.INTEGER))

            # Convert stack_size from bytes to integer
            stack_type_bytes = InlineCall(Variable(name='REAL'), parameters=(
                                         Literal(1), IntrinsicLiteral(f'kind={self.stack_type_kind}')))
            stack_type_bytes = InlineCall(Variable(name='C_SIZEOF'),
                                          parameters=as_tuple(stack_type_bytes))
            stack_size_assign = Assignment(lhs=stack_size_var, rhs=Quotient(stack_size, stack_type_bytes))
            body_prepend += [stack_size_assign]
            if self.check_bounds:
                stack_size_check = Conditional(
                                     condition=LogicalNot(Comparison(InlineCall(Variable(name='MOD'),
                                     parameters=(stack_size_var, stack_type_bytes)),
                                     '==', Literal(0))), inline=True, body=(Intrinsic('STOP'),),
                                     else_body=None
                )
                body_prepend += [stack_size_check]

            variables_append += [stack_size_var]

        if self.stack_storage_name in variable_map:
            # Use an existing stack storage array
            stack_storage = routine.variable_map[self.stack_storage_name]
        else:
            # Create a variable for the stack storage array and create corresponding
            # allocation/deallocation statements
            stack_type = SymbolAttributes(
                dtype=BasicType.REAL,
                kind=Variable(name=self.stack_type_kind, scope=routine),
                shape=(RangeIndex((None, None)), RangeIndex((None, None))),
                allocatable=True,
            )
            stack_storage = Variable(
                name=self.stack_storage_name, type=stack_type,
                dimensions=stack_type.shape, scope=routine
            )
            variables_append += [stack_storage]

            stack_alloc = Allocation(variables=(stack_storage.clone(dimensions=(  # pylint: disable=no-member
                stack_size_var, Variable(name=self.block_dim.size, scope=routine)
            )),))
            stack_dealloc = Deallocation(variables=(stack_storage.clone(dimensions=None),))  # pylint: disable=no-member

            body_prepend += [stack_alloc]
            if self.directive == 'openacc':
                pragma_data_start = Pragma(
                    keyword='acc',
                    content=f'data create({stack_storage.name})' # pylint: disable=no-member
                )
                body_prepend += [pragma_data_start]
                pragma_data_end = Pragma(keyword='acc', content='end data')
                body_append += [pragma_data_end]
            body_append += [stack_dealloc]

        # Inject new variables and body nodes
        if variables_append:
            routine.variables += as_tuple(variables_append)
        if body_prepend:
            routine.body.prepend(body_prepend)
        if body_append:
            routine.body.append(body_append)

        return stack_storage, stack_size_var

    def _determine_stack_size(self, routine, successors, local_stack_size=None):
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

        Returns
        -------
        :any:`Expression` :
            The expression representing the required stack size.
        """
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
                successor_stack_size = successor_map[call.name].trafo_data[self._key]
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

    def _get_c_sizeof_arg(self, arr):
        """
        Return an inline declaration of an intrinsic type, to be used as an argument to
        `C_SIZEOF`.
        """

        if arr.type.dtype == BasicType.REAL:
            param = InlineCall(Variable(name='REAL'), parameters=as_tuple(IntLiteral(1)))
        elif arr.type.dtype == BasicType.INTEGER:
            param = InlineCall(Variable(name='INT'), parameters=as_tuple(IntLiteral(1)))
        elif arr.type.dtype == BasicType.CHARACTER:
            param = InlineCall(Variable(name='CHAR'), parameters=as_tuple(IntLiteral(1)))
        elif arr.type.dtype == BasicType.LOGICAL:
            param = InlineCall(Variable(name='LOGICAL'), parameters=as_tuple(LogicLiteral('.TRUE.')))
        elif arr.type.dtype == BasicType.COMPLEX:
            param = InlineCall(Variable(name='CMPLX'), parameters=as_tuple(IntLiteral(1), IntLiteral(1)))
        elif arr.type.dtype == BasicType.DEFERRED:
            param = InlineCall(Variable(name='REAL'), parameters=as_tuple(IntLiteral(1)))
            warning(f"[Loki::PoolAllocator] {arr} - DeferredType var assumed to be size of 'REAL'")

        if (kind := getattr(arr.type, 'kind', None)):
            param.parameters = param.parameters + as_tuple(IntrinsicLiteral(f'kind={kind}'))

        return param

    def _create_stack_allocation(self, stack_ptr, stack_end, ptr_var, arr, stack_size):
        """
        Utility routine to "allocate" a temporary array on the pool allocator's "stack"

        This creates the pointer assignment, stack pointer increment and adds a stack size check.

        Parameters
        ----------
        stack_ptr : :any:`Variable`
            The stack pointer variable
        stack_end : :any:`Variable`
            The pointer variable that points to the end of the stack, used to verify stack size
        ptr_var : :any:`Variable`
            The pointer variable to use for the temporary array
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

        # Bail if 'arr' is a derived-type object. Partly because the possibility of derived-type
        # nesting increases the complexity of determing allocation size, and partly because `C_SIZEOF`
        # doesn't account for the size of allocatable/pointer members of derived-types.
        if isinstance(arr.type.dtype, DerivedType):
            warning(f'[Loki::PoolAllocator] {arr} - Derived type var not supported in pool allocator')
            return ([], stack_size)

        ptr_assignment = Assignment(lhs=ptr_var, rhs=stack_ptr)

        # Build expression for array size in bytes
        dim = arr.dimensions[0]
        for d in arr.dimensions[1:]:
            dim = Product((dim, d))
        arr_size = Product((dim, InlineCall(Variable(name='C_SIZEOF'), parameters=as_tuple(self._get_c_sizeof_arg(arr)))))

        # Increment stack size
        stack_size = simplify(Sum((stack_size, arr_size)))

        ptr_increment = Assignment(lhs=stack_ptr, rhs=Sum((stack_ptr, arr_size)))
        if self.check_bounds:
            stack_size_check = Conditional(
                condition=Comparison(stack_ptr, '>', stack_end), inline=True,
                body=(Intrinsic('STOP'),), else_body=None
            )
            return ([ptr_assignment, ptr_increment, stack_size_check], stack_size)
        return ([ptr_assignment, ptr_increment], stack_size)

    def apply_pool_allocator_to_temporaries(self, routine):
        """
        Apply pool allocator to local temporary arrays

        This appends the relevant argument to the routine's dummy argument list and
        creates the assignment for the local copy of the stack type.
        For all local arrays, a Cray pointer is instantiated and the temporaries
        are mapped via Cray pointers to the pool-allocated memory region.

        The cumulative size of all temporary arrays is determined and returned.
        """

        def _is_constant(d):
            """Establish if a given dimensions symbol is a compile-time constant"""
            if isinstance(d, IntLiteral):
                return True

            if isinstance(d, RangeIndex):
                if d.lower:
                    return _is_constant(d.lower) and _is_constant(d.upper)
                return _is_constant(d.upper)

            if isinstance(d, Scalar) and isinstance(d.initial , IntLiteral):
                return True

            return False

        # Find all temporary arrays
        arguments = routine.arguments
        temporary_arrays = [
            var for var in routine.variables
            if isinstance(var, Array) and var not in arguments
        ]

        # Filter out variables whose size is known at compile-time
        temporary_arrays = [
            var for var in temporary_arrays
            if not all(_is_constant(d) for d in var.shape)
        ]

        # Create stack argument and local stack var
        stack_var = self._get_local_stack_var(routine)
        stack_arg = self._get_stack_arg(routine)
        allocations = [Assignment(lhs=stack_var, rhs=stack_arg)]

        # Determine size of temporary arrays
        stack_size = Literal(0)

        # Create Cray pointer declarations and "stack allocations"
        declarations = []
        stack_ptr = self._get_stack_ptr(routine)
        stack_end = self._get_stack_end(routine)
        for arr in temporary_arrays:
            ptr_var = Variable(name=self.local_ptr_var_name_pattern.format(name=arr.name), scope=routine)
            declarations += [Intrinsic(f'POINTER({ptr_var.name}, {arr.name})')]  # pylint: disable=no-member
            allocation, stack_size = self._create_stack_allocation(stack_ptr, stack_end, ptr_var, arr, stack_size)
            allocations += allocation

        routine.spec.append(declarations)
        routine.body.prepend(allocations)

        return stack_size

    def create_pool_allocator(self, routine, stack_size):
        """
        Create a pool allocator in the driver
        """
        # Create and allocate the stack
        stack_storage, stack_size_var = self._get_stack_storage_and_size_var(routine, stack_size)
        stack_var = self._get_local_stack_var(routine)
        stack_ptr = self._get_stack_ptr(routine)
        stack_end = self._get_stack_end(routine)

        pragma_map = {}
        if self.directive == 'openacc':
            # Find OpenACC loop statements
            acc_pragmas = [p for p in FindNodes(Pragma).visit(routine.body) if p.keyword.lower() == 'acc']
            for pragma in acc_pragmas:
                if pragma.content.startswith('parallel') and 'gang' in pragma.content.lower():
                    parameters = get_pragma_parameters(pragma, starts_with='parallel', only_loki_pragmas=False)
                    if 'private' in parameters:
                        content = pragma.content.replace('private(', f'private({stack_var.name}, ')
                    else:
                        content = pragma.content + f' private({stack_var.name})'
                    pragma_map[pragma] = pragma.clone(content=content)

        elif self.directive == 'openmp':
            # Find OpenMP parallel statements
            omp_pragmas = [p for p in FindNodes(Pragma).visit(routine.body) if p.keyword.lower() == 'omp']
            for pragma in omp_pragmas:
                if pragma.content.startswith('parallel'):
                    parameters = get_pragma_parameters(pragma, starts_with='parallel', only_loki_pragmas=False)
                    if 'private' in parameters:
                        content = pragma.content.replace('private(', f'private({stack_var.name}, ')
                    else:
                        content = pragma.content + f' private({stack_var.name})'
                    pragma_map[pragma] = pragma.clone(content=content)

        if pragma_map:
            routine.body = Transformer(pragma_map).visit(routine.body)

        # Find first block loop and assign local stack pointers there
        loop_map = {}
        for loop in FindNodes(Loop).visit(routine.body):
            assignments = FindNodes(Assignment).visit(loop.body)
            if loop.variable != self.block_dim.index:
                # Check if block variable is assigned in loop body
                for assignment in assignments:
                    if assignment.lhs == self.block_dim.index:
                        assert assignment in loop.body
                        # Need to insert the pointer assignment after block dimension is set
                        assign_pos = loop.body.index(assignment)
                        break
                else:
                    continue
            else:
                # block variable is the loop variable: pointer assignment can happen
                # at the beginning of the loop body
                assign_pos = -1

            # Check for existing pointer assignment
            if any(a.lhs == stack_ptr.name for a in assignments):  # pylint: disable=no-member
                break

            ptr_assignment = Assignment(
                lhs=stack_ptr, rhs=InlineCall(
                    function=Variable(name='LOC'),
                    parameters=(
                        stack_storage.clone(
                            dimensions=(Literal(1), Variable(name=self.block_dim.index, scope=routine))
                        ),
                    ),
                    kw_parameters=None
                )
            )
            # Stack increment
            stack_incr = Assignment(
                lhs=stack_end, rhs=Sum((stack_ptr, Product((stack_size_var, Literal(8)))))
            )
            loop_map[loop] = loop.clone(
                body=loop.body[:assign_pos + 1] + (ptr_assignment, stack_incr) + loop.body[assign_pos + 1:]
            )

        if loop_map:
            routine.body = Transformer(loop_map).visit(routine.body)
        else:
            warning(
                f'{self.__class__.__name__}: '
                f'Could not find a block dimension loop in {routine.name}; no stack pointer assignment inserted.'
            )

    def inject_pool_allocator_into_calls(self, routine, targets):
        """
        Add the pool allocator argument into subroutine calls
        """
        call_map = {}
        stack_var = self._get_local_stack_var(routine)
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in targets:
               # If call is declared via an explicit interface, the ProcedureSymbol corresponding to the call is the
               # interface block rather than the Subroutine itself. This means we have to update the interface block
               # accordingly
                if call.name in [s for i in FindNodes(Interface).visit(routine.spec) for s in i.symbols]:
                    _ = self._get_stack_arg(call.routine)

                if call.routine != BasicType.DEFERRED and self.stack_argument_name in call.routine.arguments:
                    arg_idx = call.routine.arguments.index(self.stack_argument_name)
                    arguments = call.arguments
                    call_map[call] = call.clone(arguments=arguments[:arg_idx] + (stack_var,) + arguments[arg_idx:])

        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)

        # Now repeat the process for InlineCalls
        call_map = {}
        for call in FindInlineCalls().visit(routine.body):
            if call.name.lower() in [t.lower() for t in targets]:
                parameters = call.parameters
                call_map[call] = call.clone(parameters=parameters + (stack_var,))

        if call_map:
            routine.body = SubstituteExpressions(call_map).visit(routine.body)
