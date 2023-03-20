# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import (
    as_tuple, warning, simplify, recursive_expression_map_update,
    Transformation, FindNodes, FindVariables, Transformer, SubstituteExpressions, DetachScopesMapper,
    SymbolAttributes, BasicType, DerivedType,
    Variable, Array, Sum, Literal, Product, InlineCall, Comparison, RangeIndex,
    Intrinsic, Assignment, Conditional, CallStatement, Import, Allocation, Deallocation,
    Loop
)

__all__ = ['TemporariesPoolAllocatorTransformation']


class TemporariesPoolAllocatorTransformation(Transformation):
    """
    Parameters
    ----------
    block_dim : :any:`Dimension`
        Optional ``Dimension`` object to define the blocking dimension
        to use for hoisted column arrays if hoisting is enabled.
    """

    _key = 'TemporariesPoolAllocatorTransformation'

    def __init__(self, block_dim, stack_module_name='STACK_MOD', stack_type_name='STACK', stack_ptr_name='L',
                 stack_end_name='U', stack_size_name='ISTSZ', stack_storage_name='PSTACK',
                 stack_arg_name='YDSTACK', stack_var_name='YLSTACK', pointer_var_name_pattern='IP_{name}',
                 key=None, **kwargs):
        super().__init__(**kwargs)
        self.block_dim = block_dim
        self.stack_module_name = stack_module_name
        self.stack_type_name = stack_type_name
        self.stack_ptr_name = stack_ptr_name
        self.stack_end_name = stack_end_name
        self.stack_size_name = stack_size_name
        self.stack_storage_name = stack_storage_name
        self.stack_arg_name = stack_arg_name
        self.stack_var_name = stack_var_name
        self.pointer_var_name_pattern = pointer_var_name_pattern
        if key:
            self._key = key

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        item = kwargs.get('item', None)
        targets = kwargs.get('targets', None)

        if item and item.local_name != routine.name.lower():
            return

        successors = kwargs.get('successors', ())

        self.inject_pool_allocator_import(routine)

        if role == 'kernel':
            stack_size = self.apply_temporaries_pool_allocator(routine)
            if item:
                stack_size = self._determine_stack_size(routine, successors, stack_size)
                stack_size = DetachScopesMapper()(stack_size)
                item.trafo_data[self._key] = stack_size

        elif role == 'driver':
            stack_size = self._determine_stack_size(routine, successors)
            self.create_temporaries_pool_allocator(routine, stack_size)

        self.inject_pool_allocator_into_calls(routine, targets)

    def inject_pool_allocator_import(self, routine):
        if self.stack_type_name not in routine.imported_symbols:
            routine.spec.prepend(Import(
                module=self.stack_module_name, symbols=(Variable(name=self.stack_type_name, scope=routine),)
            ))

    def _get_stack_var(self, routine):
        if self.stack_var_name in routine.variables:
            return routine.variable_map[self.stack_var_name]

        stack_type = SymbolAttributes(dtype=DerivedType(name=self.stack_type_name))
        stack_var = Variable(name=self.stack_var_name, type=stack_type, scope=routine)
        routine.variables += (stack_var,)
        return stack_var

    def _get_stack_arg(self, routine):
        if self.stack_arg_name in routine.arguments:
            return routine.variable_map[self.stack_arg_name]

        stack_type = SymbolAttributes(dtype=DerivedType(name=self.stack_type_name), intent='inout')
        stack_arg = Variable(name=self.stack_arg_name, type=stack_type, scope=routine)
        routine.arguments += (stack_arg,)
        return stack_arg

    def _get_stack_ptr(self, routine):
        return Variable(
            name=f'{self.stack_var_name}%{self.stack_ptr_name}',
            parent=self._get_stack_var(routine),
            scope=routine
        )

    def _get_stack_end(self, routine):
        return Variable(
            name=f'{self.stack_var_name}%{self.stack_end_name}',
            parent=self._get_stack_var(routine),
            scope=routine
        )

    def _get_stack_storage_and_size_var(self, routine, stack_size):
        variable_map = routine.variable_map
        body_prepend = []
        body_append = []
        variables_append = []
        if self.stack_size_name in variable_map:
            stack_size_var = routine.variable_map[self.stack_size_name]
        else:
            stack_size_var = Variable(name=self.stack_size_name, type=SymbolAttributes(BasicType.INTEGER))
            stack_size_assign = Assignment(lhs=stack_size_var, rhs=stack_size)
            variables_append += [stack_size_var]
            body_prepend += [stack_size_assign]
        if self.stack_storage_name in variable_map:
            stack_storage = routine.variable_map[self.stack_storage_name]
        else:
            stack_type = SymbolAttributes(
                dtype=BasicType.REAL,
                kind=Variable(name='JPRB', scope=routine),
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
            body_prepend += [stack_alloc]
            stack_dealloc = Deallocation(variables=(stack_storage.clone(dimensions=None),))  # pylint: disable=no-member
            body_append += [stack_dealloc]
        if variables_append:
            routine.variables += as_tuple(variables_append)
        if body_prepend:
            routine.body.prepend(body_prepend)
        if body_append:
            routine.body.append(body_append)
        return stack_storage, stack_size_var

    def _determine_stack_size(self, routine, successors, local_stack_size=None):
        successor_map = {successor.routine.name.lower(): successor for successor in successors}
        # Collect stack sizes for successors
        stack_sizes = []
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in successor_map and self._key in successor_map[call.name].trafo_data:
                successor_stack_size = successor_map[call.name].trafo_data[self._key]
                # Replace any occurence of routine arguments in the stack size expression
                arg_map = {
                    kernel_parameter: call_arg for kernel_parameter, call_arg in call.arg_iter()
                }
                expr_map = {
                    expr: arg_map[expr] for expr in FindVariables().visit(successor_stack_size)
                    if expr in arg_map
                }
                if expr_map:
                    expr_map = recursive_expression_map_update(expr_map)
                    successor_stack_size = SubstituteExpressions(expr_map).visit(successor_stack_size)
                stack_sizes += [successor_stack_size]
        # stack_sizes = [s.trafo_data[self._key] for s in successors if self._key in s.trafo_data]
        # Unwind "max" expressions from successors
        stack_sizes = [
            d for s in stack_sizes
            for d in (s.parameters if isinstance(s, InlineCall) and s.function == 'MAX' else [s])
        ]
        if local_stack_size:
            local_stack_size = simplify(local_stack_size)
            stack_sizes = [simplify(Sum((local_stack_size, s))) for s in stack_sizes]
        if not stack_sizes:
            return local_stack_size or Literal(0)
        if len(stack_sizes) == 1:
            return stack_sizes[0]
        stack_size = InlineCall(function=Variable(name='MAX'), parameters=as_tuple(stack_sizes), kw_parameters=())
        return stack_size

    def _create_stack_allocation(self, stack_ptr, stack_size, ptr_var, arr):
        ptr_assignment = Assignment(lhs=ptr_var, rhs=stack_ptr)
        kind_bytes = Literal(8)  # TODO: Use c_sizeof
        arr_size = InlineCall(Variable(name='SIZE'), parameters=(arr.clone(dimensions=None),))
        ptr_increment = Assignment(lhs=stack_ptr, rhs=Sum((stack_ptr, Product((kind_bytes, arr_size)))))
        stack_size_check = Conditional(
            condition=Comparison(stack_ptr, '>', stack_size), inline=True,
            body=(Intrinsic('STOP'),), else_body=None
        )
        return [ptr_assignment, ptr_increment, stack_size_check]

    def apply_temporaries_pool_allocator(self, routine):
        # Find all temporary arrays
        arguments = routine.arguments
        temporary_arrays = [
            var for var in routine.variables
            if isinstance(var, Array) and var not in arguments
        ]

        # Determine size of temporary arrays
        stack_size = Literal(0)
        for array in temporary_arrays:
            dim = array.dimensions[0]
            for d in array.dimensions[1:]:
                dim = Product((dim, d))
            stack_size = Sum((stack_size, dim))

        # Create stack argument and local stack var
        stack_var = self._get_stack_var(routine)
        stack_arg = self._get_stack_arg(routine)
        allocations = [Assignment(lhs=stack_var, rhs=stack_arg)]

        # Create Cray pointer declarations and "stack allocations"
        declarations = []
        stack_ptr = self._get_stack_ptr(routine)
        stack_end = self._get_stack_end(routine)
        for arr in temporary_arrays:
            ptr_var = Variable(name=self.pointer_var_name_pattern.format(name=arr.name), scope=routine)
            declarations += [Intrinsic(f'POINTER({ptr_var.name}, {arr.name})')]  # pylint: disable=no-member
            allocations += self._create_stack_allocation(stack_ptr, stack_end, ptr_var, arr)

        routine.spec.append(declarations)
        routine.body.prepend(allocations)

        return stack_size

    def create_temporaries_pool_allocator(self, routine, stack_size):
        # Create and allocate the stack
        stack_storage, stack_size = self._get_stack_storage_and_size_var(routine, stack_size)
        self._get_stack_var(routine)
        stack_ptr = self._get_stack_ptr(routine)
        stack_end = self._get_stack_end(routine)

        # Find first block loop and assign local stack pointers there
        loop_map = {}
        for loop in FindNodes(Loop).visit(routine.body):
            assignments = FindNodes(Assignment).visit(loop.body)
            if loop.variable != self.block_dim.index:
                # Check if block variable is assigned in loop body
                if not any(a.lhs == self.block_dim.index for a in assignments):
                    continue

            # Check for existing pointer assignment
            if any(a.lhs == stack_ptr.name for a in FindNodes(Assignment).visit(loop.body)):  # pylint: disable=no-member
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
                lhs=stack_end, rhs=Sum((stack_ptr, Product((stack_size, Literal(8)))))
            )
            loop_map[loop] = loop.clone(body=(ptr_assignment, stack_incr) + loop.body)
            break

        else:
            breakpoint()
            warning(
                f'{self.__class__.__name__}: '
                'Could not find a block dimension loop; no stack pointer assignment inserted.'
            )

        if loop_map:
            routine.body = Transformer(loop_map).visit(routine.body)

    def inject_pool_allocator_into_calls(self, routine, targets):
        call_map = {}
        stack_var = self._get_stack_var(routine)
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in targets:
                if call.routine != BasicType.DEFERRED and self.stack_arg_name in call.routine.arguments:
                    arg_idx = call.routine.arguments.index(self.stack_arg_name)
                    arguments = call.arguments
                    call_map[call] = call.clone(arguments=arguments[:arg_idx] + (stack_var,) + arguments[arg_idx:])
        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)
