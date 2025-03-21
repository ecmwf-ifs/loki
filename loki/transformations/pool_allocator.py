# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from collections import  defaultdict

from loki.batch import Transformation
from loki.expression import (
    IntLiteral, LogicLiteral, Variable, Array, Sum, Literal,
    Product, InlineCall, Comparison, RangeIndex, Cast,
    ProcedureSymbol, simplify, is_dimension_constant,
    DetachScopesMapper
)
from loki.ir import (
    FindNodes, FindVariables, FindInlineCalls, Transformer, Intrinsic,
    Assignment, Conditional, CallStatement, Import, Allocation,
    Deallocation, Loop, Pragma, Interface, get_pragma_parameters,
    SubstituteExpressions
)
from loki.logging import warning, debug
from loki.tools import as_tuple
from loki.types import SymbolAttributes, BasicType, DerivedType

from loki.transformations.utilities import recursive_expression_map_update


__all__ = ['TemporariesPoolAllocatorTransformation']


class TemporariesPoolAllocatorTransformation(Transformation):
    """
    Transformation to inject a pool allocator that allocates a large scratch space per block
    on the driver and maps temporary arrays in kernels to this scratch space

    The stack is provided via two integer variables, ``<stack name>_L`` and ``<stack name>_U``, which are
    used as a stack pointer and stack end pointer, respectively.
    Naming is flexible and can be changed via options to the transformation.

    The transformation needs to be applied in reverse order, which will do the following for each **kernel**:

    * Add an argument/arguments to the kernel call signature to pass the stack integer(s)
        * either only the stack pointer is passed or the stack end pointer additionally if bound checking is active
    * Create a local copy of the stack derived type inside the kernel
    * Determine the combined size of all local arrays that are to be allocated by the pool allocator,
      taking into account calls to nested kernels. This is reported in :any:`Item`'s ``trafo_data``.
    * Inject Cray pointer assignments and stack pointer increments for all temporaries
    * Pass the local copy/copies of the stack integer(s) as argument to any nested kernel calls

    In a **driver** routine, the transformation will:

    * Determine the required scratch space from ``trafo_data``
    * Allocate the scratch space to that size
    * Insert data transfers (for OpenACC offloading)
    * Insert data sharing clauses into OpenMP or OpenACC pragmas
    * Assign stack base pointer and end pointer for each block (identified via :data:`block_dim`)
    * Pass the stack argument(s) to kernel calls


    With ``cray_ptr_loc_rhs=False`` the following stack/pool allocator will be generated:

    .. code-block:: fortran

        SUBROUTINE DRIVER (...)
          ...
          INTEGER(KIND=8) :: ISTSZ
          REAL(KIND=REAL64), ALLOCATABLE :: ZSTACK(:, :)
          INTEGER(KIND=8) :: YLSTACK_L
          INTEGER(KIND=8) :: YLSTACK_U
          ISTSZ = ISHFT(7 + C_SIZEOF(REAL(1, kind=jprb))**<array dim1>*<array dim2>, -3) + ...
          ALLOCATE (ZSTACK(ISTSZ, nb))
          DO b=1,nb
            YLSTACK_L = LOC(ZSTACK(1, b))
            YLSTACK_U = YLSTACK_L + ISTSZ*C_SIZEOF(REAL(1, kind=REAL64))
            CALL KERNEL(..., YDSTACK_L=YLSTACK_L, YDSTACK_U=YLSTACK_U)
          END DO
          DEALLOCATE (ZSTACK)
        END SUBROUTINE DRIVER

        SUBROUTINE KERNEL(...)
          ...
          INTEGER(KIND=8) :: YLSTACK_L
          INTEGER(KIND=8) :: YLSTACK_U
          INTEGER(KIND=8), INTENT(INOUT) :: YDSTACK_L
          INTEGER(KIND=8), INTENT(INOUT) :: YDSTACK_U
          POINTER(IP_tmp1, tmp1)
          POINTER(IP_tmp2, tmp2)
          ...
          YLSTACK_L = YDSTACK_L
          YLSTACK_U = YDSTACK_U
          IP_tmp1 = YLSTACK_L
          YLSTACK_L = YLSTACK_L + ISHFT(ISHFT(<array dim1>*<array dim2>*C_SIZEOF(REAL(1, kind=JPRB)) + 7, -3), 3)
          IF (YLSTACK_L > YLSTACK_U) STOP
          IP_tmp2 = YLSTACK_L
          YLSTACK_L = YLSTACK_L + ISHFT(ISHFT(...*C_SIZEOF(REAL(1, kind=JPRB)) + 7, -3), 3)
          IF (YLSTACK_L > YLSTACK_U) STOP
        END SUBROUTINE KERNEL

    With ``cray_ptr_loc_rhs=True`` the following stack/pool allocator will be generated:

    .. code-block:: fortran

        SUBROUTINE driver (NLON, NZ, NB, field1, field2)
          ...
          INTEGER(KIND=8) :: ISTSZ
          REAL(KIND=REAL64), ALLOCATABLE :: ZSTACK(:, :)
          INTEGER(KIND=8) :: YLSTACK_L
          INTEGER(KIND=8) :: YLSTACK_U
          ISTSZ = ISTSZ = ISHFT(7 + C_SIZEOF(REAL(1, kind=jprb))**<array dim1>*<array dim2>, -3) + ...
          ALLOCATE (ZSTACK(ISTSZ, nb))
          DO b=1,nb
            YLSTACK_L = 1
            YLSTACK_U = YLSTACK_L + ISTSZ
            CALL KERNEL(..., YDSTACK_L=YLSTACK_L, YDSTACK_U=YLSTACK_U, ZSTACK=ZSTACK(:, b))
          END DO
          DEALLOCATE (ZSTACK)
        END SUBROUTINE driver

        SUBROUTINE KERNEL(...)
          ...
          INTEGER(KIND=8) :: YLSTACK_L
          INTEGER(KIND=8) :: YLSTACK_U
          INTEGER(KIND=8), INTENT(INOUT) :: YDSTACK_L
          INTEGER(KIND=8), INTENT(INOUT) :: YDSTACK_U
          REAL(KIND=REAL64), CONTIGUOUS, INTENT(INOUT) :: ZSTACK(:)
          POINTER(IP_tmp1, tmp1)
          POINTER(IP_tmp2, tmp2)
          ...
          YLSTACK_L = YDSTACK_L
          YLSTACK_U = YDSTACK_U
          IP_tmp1 = LOC(ZSTACK(YLSTACK_L))
          YLSTACK_L = YLSTACK_L + ISHFT(<array dim1>*<array dim2>*C_SIZEOF(REAL(1, kind=JPRB)) + 7, -3)
          IF (YLSTACK_L > YLSTACK_U) STOP
          IP_tmp2 = LOC(ZSTACK(YLSTACK_L))
          YLSTACK_L = YLSTACK_L + ISHFT(...*C_SIZEOF(REAL(1, kind=JPRB)) + 7, -3)
          IF (YLSTACK_L > YLSTACK_U) STOP
        END SUBROUTINE KERNEL


    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension
        to use for hoisted column arrays if hoisting is enabled.
    stack_ptr_name : str, optional
        Name of the stack pointer variable to be appended to the generic
        stack name (default: ``'L'``) resulting in e.g., ``'<stack name>_L'``
    stack_end_name : str, optional
        Name of the stack end pointer variable to be appendend to the generic
        stack name (default: ``'U'``) resulting in e.g., ``'<stack name>_L'``
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
    stack_int_type_kind: :any:`Literal` or :any:`Variable`
        Integer type kind used for the stack pointer variable(s) (default: ``'8'``
        resulting in ``'INTEGER(KIND=8)'``)
    directive : str, optional
        Can be ``'openmp'`` or ``'openacc'``. If given, insert data sharing clauses for
        the stack derived type, and insert data transfer statements (for OpenACC only).
    check_bounds : bool, optional
        Insert bounds-checks in the kernel to make sure the allocated stack size is not
        exceeded (default: `True`)
    cray_ptr_loc_rhs : bool, optional
        Whether to only pass the stack variable as integer to the kernel(s) or
        whether to pass the whole stack array to the driver and the calls to ``LOC()``
        within the kernel(s) itself (default: `False`)
    """

    _key = 'TemporariesPoolAllocatorTransformation'

    # Traverse call tree in reverse when using Scheduler
    reverse_traversal = True

    process_ignored_items = True

    def __init__(
            self, block_dim, horizontal=None, stack_ptr_name='L', stack_end_name='U', stack_size_name='ISTSZ',
            stack_storage_name='ZSTACK', stack_argument_name='YDSTACK', stack_local_var_name='YLSTACK',
            local_ptr_var_name_pattern='IP_{name}', stack_int_type_kind=IntLiteral(8), directive=None,
            check_bounds=True, cray_ptr_loc_rhs=False
    ):
        self.block_dim = block_dim
        self.horizontal = horizontal
        self.stack_ptr_name = stack_ptr_name
        self.stack_end_name = stack_end_name
        self.stack_size_name = stack_size_name
        self.stack_storage_name = stack_storage_name
        self.stack_argument_name = stack_argument_name
        self.stack_local_var_name = stack_local_var_name
        self.local_ptr_var_name_pattern = local_ptr_var_name_pattern
        self.stack_int_type_kind = stack_int_type_kind
        self.directive = directive
        self.check_bounds = check_bounds
        self.cray_ptr_loc_rhs = cray_ptr_loc_rhs

        if self.stack_ptr_name == self.stack_end_name:
            raise ValueError(f'"stack_ptr_name": "{self.stack_ptr_name}" and '
                f'"stack_end_name": "{self.stack_end_name}" must be different!')

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        item = kwargs.get('item', None)
        ignore = item.ignore if item else ()
        targets = as_tuple(kwargs.get('targets', None))

        if item:
            # Initialize set to store kind imports
            item.trafo_data[self._key] = {'kind_imports': {}}

        # add iso_c_binding import if necessary
        self.import_c_sizeof(routine)
        # add iso_fortran_env import if necessary
        self.import_real64(routine)

        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = as_tuple(sub_sgraph.successors(item)) if sub_sgraph is not None else ()

        if role == 'kernel':
            stack_size = self.apply_pool_allocator_to_temporaries(routine, item=item)
            if item:
                stack_size = self._determine_stack_size(routine, successors, stack_size, item=item)
                item.trafo_data[self._key]['stack_size'] = stack_size

        elif role == 'driver':
            stack_size = self._determine_stack_size(routine, successors, item=item)
            if item:
                # import variable type specifiers used in stack allocations
                self.import_allocation_types(routine, item)
            self.create_pool_allocator(routine, stack_size)

        self.inject_pool_allocator_into_calls(routine, targets, ignore, driver=role=='driver')

    @staticmethod
    def import_c_sizeof(routine):
        """
        Import the c_sizeof symbol if necesssary.
        """

        # add qualified iso_c_binding import
        if not 'C_SIZEOF' in routine.imported_symbols:
            imp = Import(
                module='ISO_C_BINDING', symbols=as_tuple(ProcedureSymbol('C_SIZEOF', scope=routine)),
                nature='intrinsic'
            )
            routine.spec.prepend(imp)

    @staticmethod
    def import_real64(routine):
        """
        Import the real64 symbol if necesssary.
        """

        # add qualified iso_fortran_env import
        if not 'REAL64' in routine.imported_symbols:
            imp = Import(
                module='ISO_FORTRAN_ENV', symbols=as_tuple(ProcedureSymbol('REAL64', scope=routine)),
                nature='intrinsic'
            )
            routine.spec.prepend(imp)

    def import_allocation_types(self, routine, item):
        """
        Import all the variable types used in allocations.
        """

        new_imports = defaultdict(set)
        for s, m in item.trafo_data[self._key]['kind_imports'].items():
            new_imports[m] |= set(as_tuple(s))

        import_map = {i.module.lower(): i for i in routine.imports}
        for mod, symbs in new_imports.items():
            if mod in import_map:
                import_map[mod]._update(symbols=as_tuple(set(import_map[mod].symbols + as_tuple(symbs))))
            else:
                _symbs = [s for s in symbs if not (s.name.lower() in routine.variable_map or
                                                   s.name.lower() in routine.imported_symbol_map)]
                if _symbs:
                    imp = Import(module=mod, symbols=as_tuple(_symbs))
                    routine.spec.prepend(imp)

    def _get_local_stack_var(self, routine):
        """
        Utility routine to get the local stack variable

        The variable is created and added to :data:`routine` if it doesn't exist, yet.
        """
        if f'{self.stack_local_var_name}_{self.stack_ptr_name}' in routine.variables:
            return routine.variable_map[f'{self.stack_local_var_name}_{self.stack_ptr_name}']

        stack_type = SymbolAttributes(dtype=BasicType.INTEGER, kind=self.stack_int_type_kind)
        stack_var = Variable(name=f'{self.stack_local_var_name}_{self.stack_ptr_name}', type=stack_type, scope=routine)
        routine.variables += (stack_var,)
        return stack_var

    def _get_local_stack_var_end(self, routine):
        """
        Utility routine to get the local stack variable end

        The variable is created and added to :data:`routine` if it doesn't exist, yet.
        """
        if f'{self.stack_local_var_name}_{self.stack_end_name}' in routine.variables:
            return routine.variable_map[f'{self.stack_local_var_name}_{self.stack_end_name}']

        stack_type = SymbolAttributes(dtype=BasicType.INTEGER, kind=self.stack_int_type_kind)
        var_name = f'{self.stack_local_var_name}_{self.stack_end_name}'
        stack_var_end = Variable(name=var_name, type=stack_type, scope=routine)
        routine.variables += (stack_var_end,)
        return stack_var_end

    def _get_stack_arg(self, routine):
        """
        Utility routine to get the stack argument

        The argument is created and added to the dummy argument list of :data:`routine`
        if it doesn't exist, yet.
        """
        if f'{self.stack_argument_name}_{self.stack_ptr_name}' in routine.arguments:
            return routine.variable_map[f'{self.stack_argument_name}_{self.stack_ptr_name}']

        stack_type = SymbolAttributes(dtype=BasicType.INTEGER, intent='inout', kind=self.stack_int_type_kind)
        var_name = f'{self.stack_argument_name}_{self.stack_ptr_name}'
        stack_arg = Variable(name=var_name, type=stack_type, scope=routine)
        routine.arguments += (stack_arg,)

        return stack_arg

    def _get_stack_arg_end(self, routine):
        """
        Utility routine to get the stack argument end

        The argument is created and added to the dummy argument list of :data:`routine`
        if it doesn't exist, yet.
        """
        if f'{self.stack_argument_name}_{self.stack_end_name}' in routine.arguments:
            return routine.variable_map[f'{self.stack_argument_name}_{self.stack_end_name}']

        stack_type = SymbolAttributes(dtype=BasicType.INTEGER, intent='inout', kind=self.stack_int_type_kind)
        var_name = f'{self.stack_argument_name}_{self.stack_end_name}'
        stack_arg_end = Variable(name=var_name, type=stack_type, scope=routine)
        routine.arguments += (stack_arg_end,)

        return stack_arg_end

    def _get_stack_ptr(self, routine):
        """
        Utility routine to get the stack pointer variable
        """
        return Variable(
                name=f'{self.stack_local_var_name}_{self.stack_ptr_name}',
                scope=routine
                )

    def _get_stack_end(self, routine):
        """
        Utility routine to get the stack end pointer variable
        """
        return Variable(
            name=f'{self.stack_local_var_name}_{self.stack_end_name}',
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
            stack_size_var_type = SymbolAttributes(BasicType.INTEGER, kind=self.stack_int_type_kind)
            stack_size_var = Variable(name=self.stack_size_name, type=stack_size_var_type)

            # Retrieve kind parameter of stack storage
            _kind = routine.symbol_map.get('REAL64', None) or Variable(name='REAL64')

            # Convert stack_size from bytes to integer
            stack_type_bytes = Cast(name='REAL', expression=Literal(1), kind=_kind)
            stack_type_bytes = InlineCall(Variable(name='C_SIZEOF'),
                                          parameters=as_tuple(stack_type_bytes))
            stack_size_assign = Assignment(lhs=stack_size_var, rhs=stack_size)
            body_prepend += [stack_size_assign]
            variables_append += [stack_size_var]

        if self.stack_storage_name in variable_map:
            # Use an existing stack storage array
            stack_storage = routine.variable_map[self.stack_storage_name]
        else:
            # Create a variable for the stack storage array and create corresponding
            # allocation/deallocation statements
            stack_type = SymbolAttributes(
                dtype=BasicType.REAL,
                kind=Variable(name='REAL64', scope=routine),
                shape=(RangeIndex((None, None)), RangeIndex((None, None))),
                allocatable=True,
            )
            stack_storage = Variable(
                name=self.stack_storage_name, type=stack_type,
                dimensions=stack_type.shape, scope=routine
            )
            variables_append += [stack_storage]

            block_size = routine.resolve_typebound_var(self.block_dim.size, routine.symbol_map)
            stack_alloc = Allocation(variables=(stack_storage.clone(dimensions=(  # pylint: disable=no-member
                stack_size_var, block_size)),))
            stack_dealloc = Deallocation(variables=(stack_storage.clone(dimensions=None),))  # pylint: disable=no-member

            body_prepend += [stack_alloc]
            pragma_data_start = Pragma(
                keyword='loki',
                content=f'structured-data create({stack_storage.name})' # pylint: disable=no-member
            )
            body_prepend += [pragma_data_start]
            pragma_data_end = Pragma(keyword='loki', content='end structured-data')
            body_append += [pragma_data_end]
            body_append += [stack_dealloc]

        # Inject new variables and body nodes
        if variables_append:
            routine.variables += as_tuple(variables_append)
        if body_prepend:
            if not self._insert_stack_at_loki_pragma(routine, body_prepend):
                routine.body.prepend(body_prepend)
        if body_append:
            routine.body.append(body_append)

        return stack_storage, stack_size_var

    @staticmethod
    def _insert_stack_at_loki_pragma(routine, insert):
        pragma_map = {}
        for pragma in FindNodes(Pragma).visit(routine.body):
            if pragma.keyword == 'loki' and 'stack-insert' in pragma.content:
                pragma_map[pragma] = insert
        if pragma_map:
            routine.body = Transformer(pragma_map).visit(routine.body)
            return True
        return False

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
            item.trafo_data[self._key]['kind_imports'].update({
                k: v
                for s in successors
                for k, v in s.trafo_data.get(self._key, {}).get('kind_imports', {}).items()
            })

        # Note: we are not using a CaseInsensitiveDict here to be able to search directly with
        # Variable instances in the dict. The StrCompareMixin takes care of case-insensitive
        # comparisons in that case
        successor_map = {
            successor.local_name.lower(): successor
            for successor in successors
        }

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

    def _get_c_sizeof_arg(self, arr):
        """
        Return an inline declaration of an intrinsic type, to be used as an argument to
        `C_SIZEOF`.
        """

        if arr.type.dtype == BasicType.REAL:
            param = Cast(name='REAL', expression=IntLiteral(1))
        elif arr.type.dtype == BasicType.INTEGER:
            param = Cast(name='INT', expression=IntLiteral(1))
        elif arr.type.dtype == BasicType.CHARACTER:
            param = Cast(name='CHAR', expression=IntLiteral(1))
        elif arr.type.dtype == BasicType.LOGICAL:
            param = Cast(name='LOGICAL', expression=LogicLiteral('.TRUE.'))
        elif arr.type.dtype == BasicType.COMPLEX:
            param = Cast(name='CMPLX', expression=(IntLiteral(1), IntLiteral(1)))

        param.kind = getattr(arr.type, 'kind', None) # pylint: disable=possibly-used-before-assignment

        return param

    def _create_stack_allocation(self, stack_ptr, stack_end, ptr_var, arr, stack_size, stack_storage=None):
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

        if self.cray_ptr_loc_rhs:
            ptr_assignment = Assignment(lhs=ptr_var, rhs=InlineCall(
                        function=Variable(name='LOC'),
                        parameters=(
                            stack_storage.clone(
                                dimensions=(stack_ptr.clone(),)
                            ),
                        ),
                        kw_parameters=None
                    )
                )
        else:
            ptr_assignment = Assignment(lhs=ptr_var, rhs=stack_ptr)

        # Build expression for array size in bytes
        dims = ()
        for d in arr.dimensions:
            if isinstance(d, RangeIndex):
                dims += (Sum((d.upper, Product((-1, d.lower)), 1)),)
            else:
                dims += (d,)
        dim = Product(dims)
        arr_type_bytes = InlineCall(Variable(name='C_SIZEOF'),
                                            parameters=as_tuple(self._get_c_sizeof_arg(arr)))
        arr_size = Product((dim, arr_type_bytes))

        # Allocation is expressed in terms of REAL64, i.e., 8 byte values
        # We obtain the allocation size by dividing the required size by 8 and rounding up,
        # i.e., (size + 7) // 8, with the division implemented as bit shifts
        ishift_func = InlineCall(function=Variable(name='ISHFT'))
        arr_size = ishift_func.clone(parameters=(Sum((arr_size, 7)), -3))

        # Increment stack size
        stack_size = simplify(Sum((stack_size, arr_size)))

        if self.cray_ptr_loc_rhs:
            ptr_increment = Assignment(lhs=stack_ptr, rhs=Sum((stack_ptr, arr_size)))
        else:
            ptr_increment = Assignment(lhs=stack_ptr, rhs=Sum((stack_ptr, ishift_func.clone(parameters=(arr_size, 3)))))
        if self.check_bounds:
            stack_size_check = Conditional(
                condition=Comparison(stack_ptr, '>', stack_end), inline=True,
                body=(Intrinsic('STOP'),), else_body=None
            )
            return ([ptr_assignment, ptr_increment, stack_size_check], stack_size)
        return ([ptr_assignment, ptr_increment], stack_size)

    def apply_pool_allocator_to_temporaries(self, routine, item=None):
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
        #  this used to rely on dataflow_analysis only putting temporaries that are defined/written to
        #  however, there exist some cases where temporaries are only read and it is not the
        #  responsibility of this transformation to decide whether that is reasonable.
        #  The following just removes unused temporaries so that they are not put on the stack.
        used_vars = {v.name.lower() for v in FindVariables().visit(routine.body)}
        temporary_arrays = [
                var for var in temporary_arrays
                if var.name.lower() in used_vars
        ]

        # Filter out variables whose size is known at compile-time
        temporary_arrays = [
            var for var in temporary_arrays
            if not all(is_dimension_constant(d) for d in var.shape)
        ]

        # Filter out pointers
        temporary_arrays = [
            var for var in temporary_arrays
            if not var.type.pointer or var.type.allocatable
        ]

        # Create stack argument and local stack var
        stack_var = self._get_local_stack_var(routine)
        stack_var_end = self._get_local_stack_var_end(routine) if self.check_bounds else None
        stack_arg = self._get_stack_arg(routine)
        stack_arg_end = self._get_stack_arg_end(routine) if self.check_bounds else None

        stack_storage = None
        if self.cray_ptr_loc_rhs:
            stack_type = SymbolAttributes(
                    dtype=BasicType.REAL,
                    kind=Variable(name='REAL64', scope=routine),
                    shape=(RangeIndex((None, None)),), intent='inout', contiguous=True,
            )
            stack_storage = Variable(
                    name=self.stack_storage_name, type=stack_type,
                    dimensions=stack_type.shape, scope=routine,
            )
            arg_pos = [routine.arguments.index(arg) for arg in routine.arguments if arg.type.optional]
            if arg_pos:
                routine.arguments = routine.arguments[:arg_pos[0]] + (stack_storage,) + routine.arguments[arg_pos[0]:]
            else:
                routine.arguments += (stack_storage,)

        allocations = [Assignment(lhs=stack_var, rhs=stack_arg)]
        if self.check_bounds:
            allocations.append(Assignment(lhs=stack_var_end, rhs=stack_arg_end))

        # Determine size of temporary arrays
        stack_size = Literal(0)

        # Create Cray pointer declarations and "stack allocations"
        declarations = []
        stack_ptr = self._get_stack_ptr(routine)
        stack_end = self._get_stack_end(routine)
        for arr in temporary_arrays:
            ptr_var = Variable(name=self.local_ptr_var_name_pattern.format(name=arr.name), scope=routine)
            declarations += [Intrinsic(f'POINTER({ptr_var.name}, {arr.name})')]  # pylint: disable=no-member
            allocation, stack_size = self._create_stack_allocation(stack_ptr, stack_end, ptr_var, arr,
                    stack_size, stack_storage)
            allocations += allocation

            # Store type and size information of temporary allocation
            if item:
                if (kind := arr.type.kind):
                    if kind in routine.imported_symbols:
                        item.trafo_data[self._key]['kind_imports'][kind] = routine.import_map[kind.name].module.lower()
                dims = [d for d in arr.shape if d in routine.imported_symbols]
                for d in dims:
                    item.trafo_data[self._key]['kind_imports'][d] = routine.import_map[d.name].module.lower()

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
        stack_var_end = self._get_local_stack_var_end(routine) if self.check_bounds else None
        stack_ptr = self._get_stack_ptr(routine)
        stack_end = self._get_stack_end(routine)

        pragma_map = {}
        pragmas = [p for p in FindNodes(Pragma).visit(routine.body) if p.keyword.lower() == 'loki']
        for pragma in pragmas:
            if pragma.content.lower().startswith('loop gang'):
                parameters = get_pragma_parameters(pragma, starts_with='loop gang', only_loki_pragmas=False)
                if 'private' in [p.lower() for p in parameters]:
                    var_end_str = f' {stack_var_end.name},' if self.check_bounds else ''
                    content = re.sub(r'\bprivate\(', f'private({stack_var.name},{var_end_str} ',
                            pragma.content.lower())
                else:
                    var_end_str = f', {stack_var_end.name}' if self.check_bounds else ''
                    content = pragma.content + f' private({stack_var.name}{var_end_str})'
                pragma_map[pragma] = pragma.clone(content=content)
        # problem being that code, like e.g. ecwam transformed for 'idem-stack', already having
        #  OpenMP pragmas rely on the following. Once we (decide to) implement a
        #  'reverse PragmaModel' trafo that converts e.g., OpenMP pragmas to generic Loki pragmas
        #  we do not longer rely on the following
        omp_pragmas = [p for p in FindNodes(Pragma).visit(routine.body) if p.keyword.lower() == 'omp']
        for pragma in omp_pragmas:
            if pragma.content.lower().startswith('parallel'):
                parameters = get_pragma_parameters(pragma, starts_with='parallel', only_loki_pragmas=False)
                if 'private' in [p.lower() for p in parameters]:
                    var_end_str = f' {stack_var_end.name},' if self.check_bounds else ''
                    content = re.sub(r'\bprivate\(', f'private({stack_var.name},{var_end_str}',
                            pragma.content.lower())
                else:
                    var_end_str = f', {stack_var_end.name}' if self.check_bounds else ''
                    content = pragma.content + f' private({stack_var.name}{var_end_str})'
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
                    warning(
                        f'{self.__class__.__name__}: '
                        f'Could not find a block dimension for loop with variable {loop.variable} and '
                        f'bounds {loop.bounds} in {routine.name}; no stack pointer assignment inserted!'
                    )
                    continue
            else:
                # block variable is the loop variable: pointer assignment can happen
                # at the beginning of the loop body
                assign_pos = -1

            # Check for existing pointer assignment
            if any(a.lhs == f'{self.stack_local_var_name}_{self.stack_ptr_name}' for a in assignments):
                debug(
                    f'{self.__class__.__name__}: '
                    f'Stack (pointer) already exists within/for loop with variable {loop.variable} and '
                    f'bounds {loop.bounds} in {routine.name}; thus no stack pointer assignment inserted!'
                )
                break
            if self.cray_ptr_loc_rhs:
                ptr_assignment = Assignment(lhs=stack_ptr, rhs=IntLiteral(1))
            else:
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

            # Retrieve kind parameter of stack storage
            _kind = routine.imported_symbol_map.get('REAL64')

            # Stack increment
            if self.cray_ptr_loc_rhs:
                stack_incr = Assignment(
                    lhs=stack_end, rhs=Sum((stack_ptr, stack_size_var))
                )
            else:
                _real_size_bytes = Cast(name='REAL', expression=Literal(1), kind=_kind)
                _real_size_bytes = InlineCall(Variable(name='C_SIZEOF'),
                                              parameters=as_tuple(_real_size_bytes))
                stack_incr = Assignment(
                    lhs=stack_end, rhs=Sum((stack_ptr, Product((stack_size_var, _real_size_bytes))))
                )
            new_assignments = (ptr_assignment,)
            if self.check_bounds:
                new_assignments += (stack_incr,)
            loop_map[loop] = loop.clone(
                body=loop.body[:assign_pos + 1] + new_assignments + loop.body[assign_pos + 1:]
            )

        if loop_map:
            routine.body = Transformer(loop_map).visit(routine.body)

    def inject_pool_allocator_into_calls(self, routine, targets, ignore, driver=False):
        """
        Add the pool allocator argument into subroutine calls
        """
        call_map = {}

        # Careful to not use self._get_stack_arg, as it will
        # inject a delaration which the driver cannot do!
        stack_var = self._get_local_stack_var(routine)
        stack_arg_name = f'{self.stack_argument_name}_{self.stack_ptr_name}'
        new_kwarguments = ((stack_arg_name, stack_var),)

        if self.check_bounds:
            stack_var_end = self._get_local_stack_var_end(routine)
            stack_arg_end_name = f'{self.stack_argument_name}_{self.stack_end_name}'
            new_kwarguments += ((stack_arg_end_name, stack_var_end),)

        if self.cray_ptr_loc_rhs:
            stack_storage_var = routine.variable_map[self.stack_storage_name]
            if driver:
                stack_storage_var_dim = list(stack_storage_var.dimensions)
                stack_storage_var_dim[1] = routine.variable_map[self.block_dim.index]
            else:
                stack_storage_var_dim = None
            dimensions = as_tuple(stack_storage_var_dim)
            new_kwarguments += ((stack_storage_var.name, stack_storage_var.clone(dimensions=dimensions)),)

        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in targets or call.routine.name.lower() in ignore:
               # If call is declared via an explicit interface, the ProcedureSymbol corresponding to the call is the
               # interface block rather than the Subroutine itself. This means we have to update the interface block
               # accordingly
                if call.name in [s for i in FindNodes(Interface).visit(routine.spec) for s in i.symbols]:
                    _ = self._get_stack_arg(call.routine)

                if call.routine != BasicType.DEFERRED and stack_arg_name in call.routine.arguments:
                    call_map[call] = call.clone(
                        kwarguments=call.kwarguments + new_kwarguments
                    )

        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)

        # Now repeat the process for InlineCalls
        call_map = {}
        for call in FindInlineCalls().visit(routine.body):
            if call.name.lower() in [t.lower() for t in targets]:
                call_map[call] = call.clone(
                    kw_parameters=as_tuple(call.kw_parameters) + new_kwarguments
                )

        if call_map:
            routine.body = SubstituteExpressions(call_map).visit(routine.body)
