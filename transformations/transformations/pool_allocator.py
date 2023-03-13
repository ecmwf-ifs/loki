# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import (
    Transformation, FindNodes, Transformer,
    SymbolAttributes, BasicType, DerivedType,
    Variable, Array, Sum, Literal, Product, InlineCall, Comparison, RangeIndex,
    Intrinsic, Assignment, Conditional, CallStatement, Import
)

__all__ = ['TemporariesPoolAllocatorTransformation']


class TemporariesPoolAllocatorTransformation(Transformation):

    def __init__(self, stack_module_name='STACK_MOD', stack_type_name='STACK', stack_ptr_name='L',
                 stack_size_name='U', stack_storage_name='PSTACK',
                 stack_arg_name='YDSTACK', stack_var_name='YLSTACK', pointer_var_name_pattern='IP_{name}',
                 **kwargs):
        super().__init__(**kwargs)
        self.stack_module_name = stack_module_name
        self.stack_type_name = stack_type_name
        self.stack_ptr_name = stack_ptr_name
        self.stack_size_name = stack_size_name
        self.stack_storage_name = stack_storage_name
        self.stack_arg_name = stack_arg_name
        self.stack_var_name = stack_var_name
        self.pointer_var_name_pattern = pointer_var_name_pattern

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        item = kwargs.get('item', None)
        targets = kwargs.get('targets', None)

        if item  and item.local_name != routine.name.lower():
            return

        self.inject_pool_allocator_import(routine)

        if role == 'kernel':
            self.apply_temporaries_pool_allocator(routine)

        elif role == 'driver':
            self.create_temporaries_pool_allocator(routine)

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

    def _get_stack_size(self, routine):
        return Variable(
            name=f'{self.stack_var_name}%{self.stack_size_name}',
            parent=self._get_stack_var(routine),
            scope=routine
        )

    def _get_stack_storage(self, routine):
        if self.stack_storage_name in routine.variables:
            return routine.variable_map[self.stack_storage_name]

        stack_type = SymbolAttributes(
            dtype=BasicType.REAL,
            kind=Variable(name='JPRB', scope=routine),
            shape=(
                Product((Literal(5000), Variable(name='KLON', scope=routine))),
                Variable(name='NGPBLK', scope=routine)
            ),
            allocatable=True,
        )
        stack_storage = Variable(
            name=self.stack_storage_name, type=stack_type,
            dimensions=(RangeIndex((None, None)), RangeIndex((None, None))),
            scope=routine
        )
        routine.variables += (stack_storage,)
        return stack_storage


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

        # Create stack argument and local stack var
        stack_var = self._get_stack_var(routine)
        stack_arg = self._get_stack_arg(routine)
        allocations = [Assignment(lhs=stack_var, rhs=stack_arg)]

        # Create Cray pointer declarations and "stack allocations"
        declarations = []
        stack_ptr = Variable(name=f'{self.stack_var_name}%{self.stack_ptr_name}', parent=stack_var, scope=routine)
        stack_size = Variable(name=f'{self.stack_var_name}%{self.stack_size_name}', parent=stack_var, scope=routine)
        for arr in temporary_arrays:
            ptr_var = Variable(name=self.pointer_var_name_pattern.format(name=arr.name), scope=routine)
            declarations += [Intrinsic(f'POINTER({ptr_var.name}, {arr.name})')]  # pylint: disable=no-member
            allocations += self._create_stack_allocation(stack_ptr, stack_size, ptr_var, arr)

        routine.spec.append(declarations)
        routine.body.prepend(allocations)

    def create_temporaries_pool_allocator(self, routine):
        # TODO
        # Create local stack var
        # stack_storage = self._get_stack_storage(routine)
        # stack_var = self._get_stack_var(routine)
        # stack_alloc = Allocation(variables=stack_storage.clone(dimensions=stack_storage.shape))
        # ptr_assignment = Assignment(
        #     lhs=self._get_stack_ptr(routine),
        #     rhs=InlineCall(Variable(name='LOC'), parameters=)
        # )
        pass

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
