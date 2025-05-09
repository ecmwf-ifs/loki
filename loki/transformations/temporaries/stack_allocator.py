# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from collections import  defaultdict

from loki.analyse import dataflow_analysis_attached
from loki.batch.item import ProcedureItem
from loki.batch.transformation import Transformation
from loki.expression.symbols import (
    Array, Scalar, Variable, Literal, Product, Sum, InlineCall,
    IntLiteral, RangeIndex, DeferredTypeSymbol
)
from loki.expression.symbolic import is_dimension_constant, simplify
from loki.expression.mappers import DetachScopesMapper
from loki.ir.expr_visitors import FindVariables, SubstituteExpressions
from loki.ir.nodes import (
        Assignment, CallStatement, Pragma, Allocation, Deallocation, VariableDeclaration, Import
)
from loki.ir.find import FindNodes
from loki.ir.transformer import Transformer
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import BasicType, SymbolAttributes
from loki.transformations.utilities import recursive_expression_map_update, single_variable_declaration


__all__ = ['PoolAllocatorFtrPtrTransformation', 'PoolAllocatorRawTransformation']


class PoolAllocatorBaseTransformation(Transformation):

    _key = 'PoolAllocatorBaseTransformation'
    reverse_traversal = True
    process_ignored_items = True # TODO: remove?

    type_name_dict = {
        BasicType.REAL: {'kernel': 'P', 'driver': 'Z'},
        BasicType.LOGICAL: {'kernel': 'LD', 'driver': 'LL'},
        BasicType.INTEGER: {'kernel': 'K', 'driver': 'I'}
    }

    def __init__(self, block_dim, horizontal,
                 stack_name='STACK', local_int_var_name_pattern='JD_{name}',
                 int_kind='JWIM', driver_horizontal=None, **kwargs):

        super().__init__(**kwargs)
        self.block_dim = block_dim
        self.horizontal = horizontal
        self.stack_name = stack_name
        self.local_int_var_name_pattern = local_int_var_name_pattern
        self.int_kind = int_kind
        self.driver_horizontal = driver_horizontal


    def _get_int_type(self, intent=None):
        return SymbolAttributes(
            dtype=BasicType.INTEGER, kind=DeferredTypeSymbol(self.int_kind),
            intent=intent
        )
    int_type = property(_get_int_type)

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        self.role = role
        item = kwargs.get('item', None)

        if item:
            item.trafo_data[self._key] = {'kind_imports': {}}

        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = as_tuple(sub_sgraph.successors(item)) if sub_sgraph is not None else ()

        # TODO: shouldn't happen here ...
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.routine is not BasicType.DEFERRED:
                call.convert_kwargs_to_args()

        if role == 'kernel':
            stack_dict = self.apply_pool_allocator_to_temporaries(routine, item=item)
            if item:
                stack_dict = self._determine_stack_size(routine, successors, stack_dict, item=item)
                item.trafo_data[self._key]['stack_dict'] = stack_dict

            self.create_stacks_kernel(routine, stack_dict, successors)

        if role == 'driver':
            stack_dict = self._determine_stack_size(routine, successors, item=item)
            if item:
                # import variable type specifiers used in stack allocations
                self.import_allocation_types(routine, item)
            self.create_stacks_driver(routine, stack_dict, successors)

    @classmethod
    def import_allocation_types(cls, routine, item):
        new_imports = defaultdict(set)
        for s, m in item.trafo_data[cls._key]['kind_imports'].items():
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

    def _get_stack_int_name(self, prefix, dtype, kind, suffix):
        """
        Construct the name string for stack used and size integers.
        Replace double underscore with single if kind is None
        """
        return (prefix + '_' + self.type_name_dict[dtype][self.role] + '_' +
                self._get_kind_name(kind) + '_' + suffix).replace('__', '_')


    def _get_stack_var(self, routine, dtype, kind):
        """
        Get a stack variable with a name determined by
        the type_name_dict and _get_kind_name().
        intent is determined by whether the routine is a kernel or driver
        """

        stack_name = self.type_name_dict[dtype][self.role] + '_' + self._get_kind_name(kind) + '_' + self.stack_name
        stack_name = stack_name.replace('__', '_')

        stack_intent = 'INOUT' if self.role == 'kernel' else None

        stack_type = SymbolAttributes(dtype = dtype,
                                      kind = kind,
                                      intent = stack_intent,
                                      shape = (RangeIndex((None, None))))

        return Array(name=stack_name, type=stack_type, scope=routine)


    def _get_horizontal_variable(self, routine):
        """
        Get a scalar int variable corresponding to horizontal dimension with routine as scope
        """
        arg_map = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.routine is not BasicType.DEFERRED:
                arg_map.update(dict(call.arg_iter()))

        var = Variable(name=self.horizontal.size, scope=routine, type=self.int_type)
        if var in arg_map:
            return arg_map[var]
        return var

    def _get_horizontal_range(self, routine):
        """
        Get a RangeIndex from one to horizontal dimension
        """
        return RangeIndex((IntLiteral(1), self._get_horizontal_variable(routine)))

    def _get_int_var(self, name, scope, type=None): # pylint: disable=redefined-builtin
        if type is None:
            type = self.int_type
        return Scalar(name=name, scope=scope, type=type)

    def _get_kind_name(self, kind):
        if isinstance(kind, InlineCall):
            kind_name = kind.name
            for p in kind.parameters:
                kind_name += '_' + str(p)
            return kind_name

        return str(kind) if kind is not None else ''

    def _sort_arrays_by_type(self, arrays):
        """
        Go through list of arrays and map each array
        to its type and kind in the the dict type_dict

        Parameters
        ----------
        arrays : List of array objects
        """

        type_dict = {}
        for a in arrays:
            type_dict.setdefault(a.type.dtype, {})
            type_dict[a.type.dtype].setdefault(a.type.kind, []).append(a)
        # for a in arrays:
        #     if a.type.dtype in type_dict:
        #         if a.type.kind in type_dict[a.type.dtype]:
        #             type_dict[a.type.dtype][a.type.kind] += [a]
        #         else:
        #             type_dict[a.type.dtype][a.type.kind] = [a]
        #     else:
        #         type_dict[a.type.dtype] = {a.type.kind: [a]}

        return type_dict

    def insert_stack_in_calls(self, routine, stack_arg_dict, successors):
        """
        Insert stack arguments into calls to successor routines.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The routine in which to transform call statements
        stack_arg_dict : dict
            dict that maps dtype and kind to the sets of stack size variables
            and their corresponding stack array variables
        successors : list of :any:`Item`
            The items corresponding to successor routines called from :data:`routine`
        """
        successor_map = {
            successor.local_name: successor
            for successor in successors if isinstance(successor, ProcedureItem)
        }
        call_map = {}

        #Loop over calls and check if they call a successor routine and if the
        #transformation data is available
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in successor_map and self._key in successor_map[call.name].trafo_data:
                successor_stack_dict = successor_map[call.name].trafo_data[self._key]['stack_dict']

                call_stack_args = []

                #Loop over dtypes and kinds in successor arguments stacks
                #and construct list of stack arguments
                for dtype in successor_stack_dict:
                    for kind in successor_stack_dict[dtype]:
                        call_stack_args += list(stack_arg_dict[dtype][kind])

                #Get position of optional arguments so we can place the stacks in front
                arg_pos = [call.routine.arguments.index(arg) for arg in call.routine.arguments if arg.type.optional]

                arguments = call.arguments
                if arg_pos:
                    #Stack arguments have already been added to the routine call signature
                    #so we have to subtract the number of stack arguments from the optional position
                    arg_pos = min(arg_pos) - len(call_stack_args)
                    arguments = arguments[:arg_pos] + as_tuple(call_stack_args) + arguments[arg_pos:]
                else:
                    arguments += as_tuple(call_stack_args)

                call_map[call] = call.clone(arguments=arguments)

        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)


    def create_stacks_driver(self, routine, stack_dict, successors):
        """
        Create stack variables in the driver routine,
        add pragma directives to create the stacks on the device,
        and add the stack_variables to kernel call arguments.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The driver subroutine to get the stack_variables
        stack_dict : dict
            dict that maps dtype and kind to an expression for the required stack size
        successors : list of :any:`Item`
            The items corresponding to successor routines called from :data:`routine`
        """

        #Block variables
        kgpblock = self._get_int_var(name=self.block_dim.size, scope=routine)
        jgpblock = self._get_int_var(name=self.block_dim.index, scope=routine)

        stack_vars = []
        stack_arg_dict = {}
        assignments = []
        deallocs = []
        pragma_string = ''
        pragma_data_start = None
        for dtype in stack_dict:
            for kind in stack_dict[dtype]:

                #Start integer names in the driver with 'J'
                stack_size_name = self._get_stack_int_name('J', dtype, kind, 'STACK_SIZE')
                stack_size_var = self._get_int_var(name=stack_size_name, scope=routine)

                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                #Create the stack variable and its type with the correct shape
                stack_var = self._get_stack_var(routine, dtype, kind)
                # horizontal_size = self._get_horizontal_variable(routine)
                # if self.driver_horizontal:
                #     # If override is specified, use a separate horizontal in the driver
                #     horizontal_size = Variable(
                #         name=self.driver_horizontal, scope=routine, type=self.int_type
                #     )

                stack_type = stack_var.type.clone(shape=(RangeIndex((None,None)), RangeIndex((None,None))),
                                                  allocatable=True)
                stack_var = stack_var.clone(type=stack_type)

                stack_alloc = Allocation(variables=(stack_var.clone(dimensions=(stack_dict[dtype][kind], kgpblock)),))
                stack_dealloc = Deallocation(variables=(stack_var.clone(dimensions=None),))

                #Add the variables to the stack_arg_dict with dimensions (:,j_block)
                stack_arg_dict.setdefault(dtype, {})
                stack_arg_dict[dtype][kind] = (stack_size_var,
                                               stack_var.clone(dimensions=(RangeIndex((None,None)), jgpblock,)),
                                               stack_used_var)
                stack_var = stack_var.clone(dimensions=stack_type.shape)

                stack_used_var_init = Assignment(lhs=stack_used_var, rhs=IntLiteral(1))
                #Create stack_vars pair and assignment of the size variable
                stack_vars += [stack_size_var, stack_var, stack_used_var]
                assignments += [Assignment(lhs=stack_size_var,
                                           rhs=stack_dict[dtype][kind]), stack_alloc, stack_used_var_init]
                deallocs += [stack_dealloc]
                pragma_string += f'{stack_var.name}, '

        #Add to routine
        routine.variables = routine.variables + as_tuple(stack_vars)
        nodes_to_add = assignments

        if pragma_string:
            pragma_string = pragma_string[:-2].lower()

            pragma_data_start = Pragma(keyword='loki', content=f'unstructured-data create({pragma_string})')
            pragma_data_end = Pragma(keyword='loki', content=f'end unstructured-data delete({pragma_string})')
            nodes_to_add += [pragma_data_start]

            routine.body.append(pragma_data_end)

        if not self._insert_stack_at_loki_pragma(routine, nodes_to_add):
            routine.body.prepend(nodes_to_add)

        if deallocs:
            routine.body.append(deallocs)

        #Insert variables in successor calls
        self.insert_stack_in_calls(routine, stack_arg_dict, successors)


    def create_stacks_kernel(self, routine, stack_dict, successors):
        """
        Create stack variables in kernel routine,
        add pragma directives to create the stacks on the device,
        and add the stack_variables to kernel call arguments.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The kernel subroutine to get the stack_variables
        stack_dict : dict
            dict that maps dtype and kind to an expression for the required stack size
        successors : list of :any:`Item`
            The items corresponding to successor routines called from :data:`routine`
        """

        stack_vars = []
        stack_args = []
        stack_arg_dict = {}
        pragma_string = ''
        assignments = []
        for dtype in stack_dict:
            for kind in stack_dict[dtype]:

                #Start arguments integer names in kernels with 'K'
                stack_size_name = self._get_stack_int_name('K', dtype, kind, 'STACK_SIZE')
                stack_size_var = self._get_int_var(name=stack_size_name, scope=routine,
                                                   type=self._get_int_type(intent='IN'))

                #Local variables start with 'J'
                stack_used_arg_name = self._get_stack_int_name('JD', dtype, kind, 'STACK_USED')
                stack_used_arg = self._get_int_var(name=stack_used_arg_name, scope=routine,
                                                   type=self._get_int_type(intent='INOUT'))
                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)
                assignments += [Assignment(lhs=stack_used_var, rhs=stack_used_arg)]

                #Create the stack variable and its type with the correct shape
                shape = (stack_size_var,)
                stack_var = self._get_stack_var(routine, dtype, kind)
                stack_type = stack_var.type.clone(shape=as_tuple(shape), target=True)
                stack_var = stack_var.clone(type=stack_type)

                #Pass on the stack variable from stack_used + 1 to stack_size
                #Pass stack_size - stack_used to stack size in called kernel
                arg_dims = (RangeIndex((None, None)),)
                stack_arg_dict.setdefault(dtype, {})
                stack_arg_dict[dtype][kind] = (stack_size_var, stack_var.clone(dimensions=arg_dims), stack_used_var)

                #Create stack_vars pair
                stack_args += [stack_size_var,
                               stack_var.clone(dimensions=stack_type.shape, type=stack_var.type.clone(contiguous=True)),
                               stack_used_arg]
                stack_vars += [stack_used_var]
                pragma_string += f'{stack_var.name}, '

        if pragma_string:
            # remove ', '
            pragma_string = pragma_string[:-2].lower()

            present_pragma = None
            acc_pragmas = [p for p in FindNodes(Pragma).visit(routine.body) if p.keyword.lower() == 'loki'] # acc
            for pragma in acc_pragmas:
                if pragma.content.lower().startswith('device-present'):
                    present_pragma = pragma
                    break
            if present_pragma:
                pragma_map = {present_pragma: None}
                routine.body = Transformer(pragma_map).visit(routine.body)
                content = re.sub(r'\bvars\(', f'vars({pragma_string}, ', present_pragma.content.lower())
                present_pragma = present_pragma.clone(content = content)
                pragma_data_end = None
            else:
                present_pragma = Pragma(keyword='loki', content=f'device-present vars({pragma_string})')
                pragma_data_end = Pragma(keyword='loki', content='end device-present')

            routine.body.prepend(present_pragma)
            routine.body.append(pragma_data_end)
        routine.body.prepend(as_tuple(assignments))
        routine.variables += as_tuple(stack_vars)

        # Keep optional arguments last; a workaround for the fact that keyword arguments are not supported
        # in device code
        arg_pos = [routine.arguments.index(arg) for arg in routine.arguments if arg.type.optional]
        if arg_pos:
            routine.arguments = routine.arguments[:arg_pos[0]] + as_tuple(stack_args) + routine.arguments[arg_pos[0]:]
        else:
            routine.arguments += as_tuple(stack_args)

        self.insert_stack_in_calls(routine, stack_arg_dict, successors)


    def apply_pool_allocator_to_temporaries(self, routine, item=None): # pylint: disable=unused-argument
        """
        Apply raw stack allocator to local temporary arrays

        This appends the relevant argument to the routine's dummy argument list and
        creates the assignment for the local copy of the stack type.
        For all local arrays, a Cray pointer is instantiated and the temporaries
        are mapped via Cray pointers to the pool-allocated memory region.

        The cumulative size of all temporary arrays is determined and returned.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine object to apply transformation to

        Returns
        -------
        stack_dict : :any:`dict`
            dict with required stack size mapped to type and kind
        """
        return {}

    def _filter_temporary_arrays(self, routine):
        """
        Find all array variables in routine
        and filter out arguments, unused variables, fixed size arrays,
        and arrays whose lead dimension is not horizontal.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine object to get arrays from
        """

        # Find all temporary arrays
        arguments = routine.arguments
        temporary_arrays = [
            var for var in routine.variables
            if isinstance(var, Array) and var not in arguments
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
        # temporary_arrays = [
        #     var for var in temporary_arrays if (
        #     isinstance(var.shape[0], Scalar) and
        #     var.shape[0].name.lower() == self.horizontal.size.lower())
        # ]

        return temporary_arrays


    def _determine_stack_size(self, routine, successors, local_stack_dict=None, item=None):
        """
        Utility routine to determine the stack size required for the given :data:`routine`,
        including calls to subroutines

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine object for which to determine the stack size
        successors : list of :any:`Item`
            The items corresponding to successor routines called from :data:`routine`
        local_stack_dict : :any:`dict`, optional
            dict mapping type and kind to the corresponding number of elements used
        item : :any:`Item`
            Scheduler work item corresponding to routine.

        Returns
        -------
        stack_dict : :any:`dict`
            dict with required stack size mapped to type and kind
        """

        # Collect variable kind imports from successors
        if item:
            item.trafo_data[self._key]['kind_imports'].update(
                {k: v
                 for s in successors if isinstance(s, ProcedureItem)
                 for k, v in s.trafo_data[self._key]['kind_imports'].items()
                }
            )

        # Note: we are not using a CaseInsensitiveDict here to be able to search directly with
        # Variable instances in the dict. The StrCompareMixin takes care of case-insensitive
        # comparisons in that case
        successor_map = {
            successor.ir.name.lower(): successor
            for successor in successors if isinstance(successor, ProcedureItem)
        }

        # Collect stack sizes for successors
        # Note that we need to translate the names of variables used in the expressions to the
        # local names according to the call signature
        stack_dict = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in successor_map and self._key in successor_map[call.name].trafo_data:
                successor_stack_dict = successor_map[call.name].trafo_data[self._key]['stack_dict']

                # Replace any occurence of routine arguments in the stack size expression
                arg_map = dict(call.arg_iter())
                for dtype in successor_stack_dict:
                    for kind in successor_stack_dict[dtype]:
                        successor_stack_size = SubstituteExpressions(arg_map).visit(successor_stack_dict[dtype][kind])
                        arg_map = dict(call.arg_iter())
                        expr_map = {
                            expr: DetachScopesMapper()(arg_map[expr])\
                                    for expr in FindVariables().visit(successor_stack_size)
                            if expr in arg_map
                        }
                        if expr_map:
                            expr_map = recursive_expression_map_update(expr_map)
                            successor_stack_size = SubstituteExpressions(expr_map).visit(successor_stack_size)
                        stack_dict.setdefault(dtype, {})
                        stack_dict[dtype].setdefault(kind, []).append(successor_stack_size)


        if not stack_dict:
            # Return only the local stack size if there are no callees
            return local_stack_dict or {}

        # Unwind "max" expressions from successors and inject the local stack size into the expressions
        for (dtype, kind_dict) in stack_dict.items():
            for (kind, stack_sizes) in kind_dict.items():
                new_list = []
                for stack_size in stack_sizes:
                    if (isinstance(stack_size, InlineCall) and stack_size.function == 'MAX'):
                        new_list += list(stack_size.parameters)
                    else:
                        new_list += [stack_size]
                stack_sizes = new_list

        #Simplify the local stack sizes and add them to the stack_dict
        if local_stack_dict:
            for dtype in local_stack_dict:
                for kind in local_stack_dict[dtype]:
                    local_stack_dict[dtype][kind] = DetachScopesMapper()(simplify(local_stack_dict[dtype][kind]))

                    if dtype in stack_dict:
                        if kind in stack_dict[dtype]:
                            stack_dict[dtype][kind] = [simplify(Sum((local_stack_dict[dtype][kind], s)))
                                                       for s in stack_dict[dtype][kind]]
                        else:
                            stack_dict[dtype][kind] = [local_stack_dict[dtype][kind]]
                    else:
                        stack_dict[dtype] = {kind: [local_stack_dict[dtype][kind]]}

        #If several expressions, return MAX, else just add the expression
        for (dtype, kind_dict) in stack_dict.items():
            for (kind, stacks) in kind_dict.items():
                if len(stacks) == 1:
                    kind_dict[kind] = stacks[0]
                else:
                    kind_dict[kind] = InlineCall(function = Variable(name = 'MAX'), parameters = as_tuple(stacks))

        return stack_dict

class PoolAllocatorFtrPtrTransformation(PoolAllocatorBaseTransformation):

    def adapt_temp_declarations(self, routine, temporary_arrays):
        # make sure relevant variables are declared in their own statement
        single_variable_declaration(routine, variables=[var.name for var in temporary_arrays])
        # make them 'pointer' and 'contiguous'
        for var in temporary_arrays:
            routine.symbol_attrs[var.name] = var.type.clone(pointer=True, contiguous=True)
        declarations = FindNodes(VariableDeclaration).visit(routine.spec)
        # adapt declarations
        decl_map = {}
        for decl in declarations:
            if decl.symbols[0] in temporary_arrays:
                new_dimensions = as_tuple(RangeIndex((None, None)) for _ in decl.symbols[0].dimensions)
                new_symbol = decl.symbols[0].clone(dimensions=new_dimensions)
                if decl.dimensions is not None:
                    decl_map[decl] = decl.clone(dimensions=as_tuple(RangeIndex((None, None)) for _ in decl.dimensions),
                                                symbols=(new_symbol,))
                else:
                    decl_map[decl] = decl.clone(symbols=(new_symbol,))
        routine.spec = Transformer(decl_map).visit(routine.spec)

    def apply_pool_allocator_to_temporaries(self, routine, item=None):
        """
        Apply raw stack allocator to local temporary arrays

        This appends the relevant argument to the routine's dummy argument list and
        creates the assignment for the local copy of the stack type.
        For all local arrays, a Cray pointer is instantiated and the temporaries
        are mapped via Cray pointers to the pool-allocated memory region.

        The cumulative size of all temporary arrays is determined and returned.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine object to apply transformation to

        Returns
        -------
        stack_dict : :any:`dict`
            dict with required stack size mapped to type and kind
        """

        #Get all temporary dicts and sort them according to dtype and kind
        temporary_arrays = self._filter_temporary_arrays(routine)

        self.adapt_temp_declarations(routine, temporary_arrays)
        temporary_array_dict = self._sort_arrays_by_type(temporary_arrays)

        integers = []
        allocations = []

        stack_dict = {}
        # stack_set = set()

        for (dtype, kind_dict) in temporary_array_dict.items():

            if dtype not in stack_dict:
                stack_dict[dtype] = {}

            for (kind, arrays) in kind_dict.items():

                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                #Initialize stack_used to 0
                stack_used = IntLiteral(1)
                if kind not in stack_dict[dtype]:
                    stack_dict[dtype][kind] = Literal(0)

                # Store type information of temporary allocation
                if item:
                    if kind in routine.imported_symbols:
                        item.trafo_data[self._key]['kind_imports'][kind] = routine.import_map[kind.name].module.lower()
                    for array in arrays:
                        dims = [d for d in array.shape if d in routine.imported_symbols]
                        for d in dims:
                            item.trafo_data[self._key]['kind_imports'][d] = routine.import_map[d.name].module.lower()

                #Get the stack variable
                stack_var = self._get_stack_var(routine, dtype, kind)
                old_int_var = stack_used_var
                old_array_size = ()

                int_var_kind_name = self._get_kind_name(kind)
                int_var_name = 'incr'
                if int_var_kind_name:
                    int_var_name += f'_{int_var_kind_name}'
                int_var = self._get_int_var(name=self.local_int_var_name_pattern.format(name=int_var_name),
                                            scope=routine)
                integers += [int_var]

                #Loop over arrays
                for array in arrays:

                    #Computer array size
                    array_size = IntLiteral(1)
                    for d in array.shape:
                        if isinstance(d, RangeIndex):
                            d_extent = Sum((d.upper, Product((-1,d.lower)), IntLiteral(1)))
                        else:
                            d_extent = d
                        array_size = simplify(Product((array_size, d_extent)))

                    #Add to stack dict and list of allocations
                    stack_dict[dtype][kind] = simplify(Sum((stack_dict[dtype][kind], array_size)))
                    allocations += [Assignment(lhs=int_var, rhs=Sum((old_int_var,) + old_array_size))]

                    #Store the old int variable to calculate offset for next array
                    old_int_var = int_var
                    if isinstance(array_size, Sum):
                        old_array_size = array_size.children
                    else:
                        old_array_size = (array_size,)

                    ptr_assignment = self._get_ptr_assignment(array, int_var, stack_var)
                    allocations += [ptr_assignment]

                #Compute stack used
                stack_used = simplify(Sum((int_var, array_size)))
                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                #List up integers and allocations generated
                allocations += [Assignment(lhs=stack_used_var, rhs=stack_used)]

        #Add  variables to routines and allocations to body
        routine.variables = as_tuple(v for v in routine.variables if v not in temporary_arrays) + as_tuple(integers)
        routine.body.prepend(allocations)

        return stack_dict

    def _get_ptr_assignment(self, array, int_var, stack_var):
        arr_dim = ()
        stack_dim_upper = ()
        for dim in array.shape:
            if isinstance(dim, RangeIndex):
                arr_dim += (dim,)
                stack_dim_upper += (Sum((dim.upper, IntLiteral(1), Product((-1, dim.lower)))),)
            else:
                arr_dim += (RangeIndex((IntLiteral(1), dim)),)
                stack_dim_upper += (dim,)

        if stack_dim_upper:
            stack_dim_upper = Sum((int_var, Product(stack_dim_upper)))
        else:
            stack_dim_upper = Sum((int_var, IntLiteral(1)))
        ptr_assignment = Assignment(lhs=array.clone(dimensions=arr_dim),
                                    rhs=stack_var.clone(dimensions=(RangeIndex((int_var, stack_dim_upper)))),
                                    ptr=True)
        return ptr_assignment

class PoolAllocatorRawTransformation(PoolAllocatorBaseTransformation):

    def apply_pool_allocator_to_temporaries(self, routine, item=None):
        """
        Apply raw stack allocator to local temporary arrays

        This appends the relevant argument to the routine's dummy argument list and
        creates the assignment for the local copy of the stack type.
        For all local arrays, a Cray pointer is instantiated and the temporaries
        are mapped via Cray pointers to the pool-allocated memory region.

        The cumulative size of all temporary arrays is determined and returned.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine object to apply transformation to

        Returns
        -------
        stack_dict : :any:`dict`
            dict with required stack size mapped to type and kind
        """

        #Get all temporary dicts and sort them according to dtype and kind
        temporary_arrays = self._filter_temporary_arrays(routine)
        temporary_array_dict = self._sort_arrays_by_type(temporary_arrays)

        integers = []
        allocations = []
        var_map = {}

        stack_dict = {}

        temp_array_map = CaseInsensitiveDict()

        for (dtype, kind_dict) in temporary_array_dict.items():

            if dtype not in stack_dict:
                stack_dict[dtype] = {}

            for (kind, arrays) in kind_dict.items():

                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                #Initialize stack_used to 0
                stack_used = IntLiteral(1)
                if kind not in stack_dict[dtype]:
                    stack_dict[dtype][kind] = Literal(0)

                # Store type information of temporary allocation
                if item:
                    if kind in routine.imported_symbols:
                        item.trafo_data[self._key]['kind_imports'][kind] = routine.import_map[kind.name].module.lower()
                    for array in arrays:
                        dims = [d for d in array.shape if d in routine.imported_symbols]
                        for d in dims:
                            item.trafo_data[self._key]['kind_imports'][d] = routine.import_map[d.name].module.lower()

                #Get the stack variable
                stack_var = self._get_stack_var(routine, dtype, kind)
                old_int_var = stack_used_var
                old_array_size = ()

                #Loop over arrays
                for array in arrays:

                    int_var_name = self.local_int_var_name_pattern.format(name=array.name)
                    int_var = self._get_int_var(name=int_var_name, scope=routine)
                    integers += [int_var]

                    #Computer array size
                    array_size = IntLiteral(1)
                    for d in array.shape:
                        if isinstance(d, RangeIndex):
                            d_extent = Sum((d.upper, Product((-1,d.lower)), IntLiteral(1)))
                        else:
                            d_extent = d
                        array_size = simplify(Product((array_size, d_extent)))

                    #Add to stack dict and list of allocations
                    stack_dict[dtype][kind] = simplify(Sum((stack_dict[dtype][kind], array_size)))
                    allocations += [Assignment(lhs=int_var, rhs=Sum((old_int_var,) + old_array_size))]

                    #Store the old int variable to calculate offset for next array
                    old_int_var = int_var
                    if isinstance(array_size, Sum):
                        old_array_size = array_size.children
                    else:
                        old_array_size = (array_size,)

                    # save for later usage
                    temp_array_map[array.name] = (array, stack_var, int_var)

                #Compute stack used
                stack_used = simplify(Sum((int_var, array_size)))
                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                #List up integers and allocations generated
                allocations += [Assignment(lhs=stack_used_var, rhs=stack_used)]

        var_map = self._map_temporary_array(temp_array_map, routine)
        if var_map:
            var_map = recursive_expression_map_update(var_map)
            routine.body = SubstituteExpressions(var_map).visit(routine.body)

        #Add  variables to routines and allocations to body
        routine.variables = as_tuple(v for v in routine.variables if v not in temporary_arrays) + as_tuple(integers)
        routine.body.prepend(allocations)

        return stack_dict

    def _map_temporary_array(self, temp_array_map, routine):
        """
        TODO: adapt ...
        Find all instances of temporary array, temp_array, in routine and
        map them to to the corresponding position in stack stack_var.
        Position in stack is stored in int_var.
        Returns a dict mapping all instances of temp_array to corresponding stack position.

        Parameters
        ----------
        temp_array : :any:`Variable`
            Array to be mapped into stack array
        int_var : :any:`Variable`
            Integer variable corresponding to the position in of the array in the stack
        routine : :any:`Subroutine`
            The subroutine object to transform
        stack_var : :any:`Variable`
            The stack array variable

        Returns
        -------
        temp_map : :any:`dict`
            dict mapping variable instances to positions in the stack array
        """

        #List instances of temp_array
        temp_arrays = [v for v in FindVariables().visit(routine.body) if v.name.lower() in temp_array_map.keys()]
        temp_map = {}
        stack_dimensions = [None]

        #Loop over instances of temp_array
        for t in temp_arrays:

            stack_var = temp_array_map[t.name][1]
            int_var = temp_array_map[t.name][2]

            offset = IntLiteral(1)
            stack_size = IntLiteral(1)

            if t.dimensions:
                #If t has dimensions, we must compute the offsets in the stack
                #taking each dimension into account

                #Check if lead dimension is contiguous
                contiguous = (isinstance(t.dimensions[0], RangeIndex) and
                             (t.dimensions[0] == self._get_horizontal_range(routine) or
                             (t.dimensions[0].lower is None and t.dimensions[0].upper is None)))

                s_offset = IntLiteral(1)
                for d, s in zip(t.dimensions, t.shape):

                    #Check if there are range indices in shape to account for
                    if isinstance(s, RangeIndex):
                        s_lower = s.lower
                        s_upper = s.upper
                        s_extent = Sum((s_upper, Product((-1, s_lower)), IntLiteral(1)))
                    else:
                        s_lower = IntLiteral(1)
                        s_upper = s
                        s_extent = s

                    if isinstance(d, RangeIndex):

                        # TODO: introduce warning here
                        #If dimension is a rangeindex, compute the indices
                        #Stop if there is any non contiguous access to the array
                        if not contiguous:
                            # raise RuntimeError(f'Discontiguous access of array {t}')
                            print(f'Discontiguous access of array {t} within {routine}')

                        d_lower = d.lower or s_lower
                        d_upper = d.upper or s_upper

                        #Store if this dimension was contiguous
                        contiguous = (d_upper == s_upper) and (d_lower == s_lower)

                        #Multiply stack_size by current dimension
                        stack_size = Product((stack_size, Sum((d_upper, Product((-1, d_lower)), IntLiteral(1)))))

                    else:

                        #Only need a single index to compute offset
                        d_lower = d

                    #Compute dimension and shape offsets
                    d_offset =  Sum((d_lower, Product((-1, s_lower))))
                    offset = Sum((offset, Product((d_offset, s_offset))))
                    s_offset = Product((s_offset, s_extent))

            else:
                #If t does not have dimensions,
                #we can just access (1:horizontal.size, 1:stack_size)

                for s in t.shape:
                    if isinstance(s, RangeIndex):
                        s_lower = s.lower
                        s_upper = s.upper
                        s_extent = Sum((s_upper, Product((-1, s_lower)), IntLiteral(1)))
                    else:
                        s_lower = IntLiteral(1)
                        s_upper = s
                        s_extent = s

                    stack_size = Product((stack_size, s_extent))

            offset = simplify(offset)
            stack_size = simplify(stack_size)

            #Add offset to int_var
            lower = Sum((int_var,) + offset.children if isinstance(offset, Sum) else (offset,))

            if stack_size == IntLiteral(1):
                #If a single element is accessed, we only need a number
                stack_dimensions[0] = lower

            else:
                #Else we'll  have to construct a range index
                offset = simplify(Sum((offset, stack_size, Product((-1, IntLiteral(1))))))
                upper = Sum((int_var,) + offset.children if isinstance(offset, Sum) else (offset,))
                stack_dimensions[0] = RangeIndex((lower, upper))

            #Finally add to the mapping
            temp_map[t] = stack_var.clone(dimensions=as_tuple(stack_dimensions))

        return temp_map
