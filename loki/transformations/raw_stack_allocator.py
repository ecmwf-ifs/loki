# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

from loki.analyse import dataflow_analysis_attached
from loki.backend.fgen import fgen
from loki.batch.item import ProcedureItem
from loki.batch.transformation import Transformation
from loki.expression.symbols import (
    Array, Scalar, Variable, Literal, Product, Sum, InlineCall,
    IntLiteral, RangeIndex, DeferredTypeSymbol
)
from loki.expression.symbolic import is_dimension_constant, simplify
from loki.expression.mappers import DetachScopesMapper
from loki.expression.expr_visitors import FindVariables, SubstituteExpressions
from loki.ir.nodes import Assignment, CallStatement, Pragma
from loki.ir.find import FindNodes
from loki.ir.transformer import Transformer
from loki.tools import as_tuple
from loki.types import BasicType, SymbolAttributes


__all__ = ['TemporariesRawStackTransformation']

one = IntLiteral(1)


class TemporariesRawStackTransformation(Transformation):
    """
    Transformation to inject stack arrays at the driver level. These, as well
    as corresponding sizes are passed on to the kernels. Any temporary arrays with
    the horizontal dimension as lead dimension are then allocated as offsets
    in the stack array.

    The transformation needs to be applied in reverse order, which will do the following for each **kernel**:

    * Add arguments to the kernel call signature to pass the stack arrays and their (free) size
    * Determine the combined size of all local arrays that are to be allocated on the stack,
      taking into account calls to nested kernels. This is reported in :any:`Item`'s ``trafo_data``.
    * Replace any access to temporary arrays with the corresponding offsets in the stack array
    * Pass the stack arrays as arguments to any nested kernel calls

    In a **driver** routine, the transformation will:

    * Determine the required scratch space from ``trafo_data``
    * Allocate the stack arrays
    * Insert data sharing clauses into OpenMP or OpenACC pragmas
    * Pass the stack arrays and sizes into the kernel calls

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension
    horizontal: :any:`Dimension`
        :any:`Dimension` object to define the horizontal dimension
    stack_name : str, optional
        Name of the scratch space variable that is allocated in the
        driver (default: ``'STACK'``)
    local_int_var_name_pattern : str, optional
        Python format string pattern for the name of the integer variable
        for each temporary (default: ``'JD_{name}'``)
    directive : str, optional
        Can be ``'openmp'`` or ``'openacc'``. If given, insert data sharing clauses for
        the stack derived type, and insert data transfer statements (for OpenACC only).
    driver_horizontal : str, optional
        Override string if a separate variable name should be used for the horizontal
        when allocating the stack in the driver.
    key : str, optional
        Overwrite the key that is used to store analysis results in ``trafo_data``.
    """

    _key = 'TemporariesRawStackTransformation'

    # Traverse call tree in reverse when using Scheduler
    reverse_traversal = True

    type_name_dict = {
        BasicType.REAL: {'kernel': 'P', 'driver': 'Z'},
        BasicType.LOGICAL: {'kernel': 'LD', 'driver': 'LL'},
        BasicType.INTEGER: {'kernel': 'K', 'driver': 'I'}
    }

    def __init__(
            self, block_dim, horizontal, stack_name='STACK',
            local_int_var_name_pattern='JD_{name}', directive=None,
            driver_horizontal=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.block_dim = block_dim
        self.horizontal = horizontal
        self.stack_name = stack_name
        self.local_int_var_name_pattern = local_int_var_name_pattern
        self.directive = directive
        self.driver_horizontal = driver_horizontal

    @property
    def int_type(self):
        return SymbolAttributes(
            dtype=BasicType.INTEGER, kind=DeferredTypeSymbol('JPIM')
        )

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        item = kwargs.get('item', None)

        if item:
            # Initialize set to store kind imports
            item.trafo_data[self._key] = {'kind_imports': {}}

        successors = kwargs.get('successors', ())

        self.role = role

        if role == 'kernel':

            stack_dict = self.apply_raw_stack_allocator_to_temporaries(routine, item=item)
            if item:
                stack_dict = self._determine_stack_size(routine, successors, stack_dict, item=item)
                item.trafo_data[self._key]['stack_dict'] = stack_dict

            self.create_stacks_kernel(routine, stack_dict, successors)

        if role == 'driver':

            stack_dict = self._determine_stack_size(routine, successors, item=item)

            self.create_stacks_driver(routine, stack_dict, successors)


    def _get_stack_int_name(self, prefix, dtype, kind, suffix):
        """
        Construct the name string for stack used and size integers.
        Replace double underscore with single if kind is None
        """
        return (prefix + '_' + self.type_name_dict[dtype][self.role] + '_' +
                self._get_kind_name(kind) + '_' + suffix).replace('__', '_')


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
        add pragma directives to create the stacks on the device (if self.directive),
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
        kgpblock = Scalar(name=self.block_dim.size, scope=routine, type=self.int_type)
        jgpblock = Scalar(name=self.block_dim.index, scope=routine, type=self.int_type)

        #Full dimensions for arguments
        fulldim = (RangeIndex((None,None)), RangeIndex((None,None)))

        stack_vars = []
        stack_arg_dict = {}
        assignments = []
        pragma_string = ''
        pragma_data_start = None
        for dtype in stack_dict:
            for kind in stack_dict[dtype]:

                #Start integer names in the driver with 'J'
                stack_size_name = self._get_stack_int_name('J', dtype, kind, 'STACK_SIZE')
                stack_size_var = Scalar(name=stack_size_name, scope=routine, type=self.int_type)

                #Create the stack variable and its type with the correct shape
                stack_var = self._get_stack_var(routine, dtype, kind)
                horizontal_size = self._get_horizontal_variable(routine)
                if self.driver_horizontal:
                    # If override is specified, use a separate horizontal in the driver
                    horizontal_size = Variable(
                        name=self.driver_horizontal, scope=routine, type=self.int_type
                    )

                stack_type = stack_var.type.clone(
                    shape=(horizontal_size, stack_dict[dtype][kind], kgpblock)
                )
                stack_var = stack_var.clone(type=stack_type)

                #Add the variables to the stack_arg_dict with dimensions (:,:,j_block)
                if dtype in stack_arg_dict:
                    stack_arg_dict[dtype][kind] = (stack_size_var, stack_var.clone(dimensions = fulldim+(jgpblock,)))
                else:
                    stack_arg_dict[dtype] = {kind: (stack_size_var, stack_var.clone(dimensions = fulldim+(jgpblock,)))}
                stack_var = stack_var.clone(dimensions=stack_type.shape)

                #Create stack_vars pair and assignment of the size variable
                stack_vars += [stack_size_var, stack_var]
                assignments += [Assignment(lhs=stack_size_var, rhs=stack_dict[dtype][kind])]
                pragma_string += f'{stack_var.name}, '

        #If self.directive, create or allocate stack on device
        if self.directive:
            if pragma_string:
                pragma_string = pragma_string[:-2].lower()

                if self.directive == 'openacc':
                    pragma_data_start = Pragma(keyword='acc', content=f'data create({pragma_string})')
                    pragma_data_end = Pragma(keyword='acc', content='end data')

                elif self.directive == 'openmp':
                    pragma_data_start = Pragma(keyword='omp', content=f'target allocate({pragma_string})')
                    pragma_data_end = Pragma(keyword='omp', content='end target')

        #Add to routine
        routine.variables = routine.variables + as_tuple(stack_vars)
        routine.body.prepend(assignments)

        #Add directives to beginning and end of routine.body
        if self.directive:
            if pragma_data_start:
                routine.body.prepend(pragma_data_start)
                routine.body.append(pragma_data_end)

        #Insert variables in successor calls
        self.insert_stack_in_calls(routine, stack_arg_dict, successors)


    def create_stacks_kernel(self, routine, stack_dict, successors):
        """
        Create stack variables in kernel routine,
        add pragma directives to create the stacks on the device (if self.directive),
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
        stack_arg_dict = {}
        pragma_string = ''
        for dtype in stack_dict:
            for kind in stack_dict[dtype]:

                #Start arguments integer names in kernels with 'K'
                stack_size_name = self._get_stack_int_name('K', dtype, kind, 'STACK_SIZE')
                stack_size_var = Scalar(name=stack_size_name, scope=routine, type=self.int_type.clone(intent='IN'))

                #Local variables start with 'J'
                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = Scalar(name=stack_used_name, scope=routine, type=self.int_type)

                #Create the stack variable and its type with the correct shape
                stack_var = self._get_stack_var(routine, dtype, kind)
                stack_type = stack_var.type.clone(shape=(self._get_horizontal_variable(routine), stack_size_var))
                stack_var = stack_var.clone(type=stack_type)

                #Pass on the stack variable from stack_used + 1 to stack_size
                #Pass stack_size - stack_used to stack size in called kernel
                arg_dims = (self._get_horizontal_range(routine),
                            RangeIndex((Sum((stack_used_var,IntLiteral(1))), stack_size_var)))
                if dtype in stack_arg_dict:
                    stack_arg_dict[dtype][kind] = (Sum((stack_size_var, Product((-1, stack_used_var)))),
                                                   stack_var.clone(dimensions = arg_dims))
                else:
                    stack_arg_dict[dtype] = {kind: (Sum((stack_size_var, Product((-1, stack_used_var)))),
                                                    stack_var.clone(dimensions = arg_dims))}

                #Create stack_vars pair
                stack_vars += [stack_size_var, stack_var.clone(dimensions=stack_type.shape)]
                pragma_string += f'{stack_var.name}, '

        #If self.directive,s openacc, add present clauses
        if self.directive:
            if pragma_string:
                pragma_string = pragma_string[:-2].lower()

                if self.directive == 'openacc':
                    present_pragma = None
                    acc_pragmas = [p for p in FindNodes(Pragma).visit(routine.body) if p.keyword.lower() == 'acc']
                    for pragma in acc_pragmas:
                        if pragma.content.lower().startswith('data present'):
                            present_pragma = pragma
                            break
                    if present_pragma:
                        pragma_map = {present_pragma: None}
                        routine.body = Transformer(pragma_map).visit(routine.body)
                        content = re.sub(r'\bpresent\(', f'present({pragma_string}, ', present_pragma.content.lower())
                        present_pragma = present_pragma.clone(content = content)
                        pragma_data_end = None
                    else:
                        present_pragma = Pragma(keyword='acc', content=f'data present({pragma_string})')
                        pragma_data_end = Pragma(keyword='acc', content='end data')

                    routine.body.prepend(present_pragma)
                    routine.body.append(pragma_data_end)


        # Keep optional arguments last; a workaround for the fact that keyword arguments are not supported
        # in device code
        arg_pos = [routine.arguments.index(arg) for arg in routine.arguments if arg.type.optional]
        if arg_pos:
            routine.arguments = routine.arguments[:arg_pos[0]] + as_tuple(stack_vars) + routine.arguments[arg_pos[0]:]
        else:
            routine.arguments += as_tuple(stack_vars)

        self.insert_stack_in_calls(routine, stack_arg_dict, successors)


    def apply_raw_stack_allocator_to_temporaries(self, routine, item=None):
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
        stack_set = set()


        for (dtype, kind_dict) in temporary_array_dict.items():

            if dtype not in stack_dict:
                stack_dict[dtype] = {}

            for (kind, arrays) in kind_dict.items():

                #Initialize stack_used to 0
                stack_used = IntLiteral(0)
                if kind not in stack_dict[dtype]:
                    stack_dict[dtype][kind] = Literal(0)

                # Store type information of temporary allocation
                if item:
                    if kind in routine.imported_symbols:
                        item.trafo_data[self._key]['kind_imports'][kind] = routine.import_map[kind.name].module.lower()

                #Get the stack variable
                stack_var = self._get_stack_var(routine, dtype, kind)
                old_int_var = IntLiteral(0)
                old_array_size = ()

                #Loop over arrays
                for array in arrays:

                    int_var = Scalar(name=self.local_int_var_name_pattern.format(name=array.name),
                                     scope=routine, type=self.int_type)
                    integers += [int_var]

                    #Computer array size
                    array_size = one
                    for d in array.shape[1:]:
                        if isinstance(d, RangeIndex):
                            d_extent = Sum((d.upper, Product((-1,d.lower)), one))
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

                    #Map array instances to stack offsets
                    temp_map = self._map_temporary_array(array, int_var, routine, stack_var)
                    var_map = {**var_map, **temp_map}
                    stack_set.add(stack_var)

                #Compute stack used
                stack_used = simplify(Sum((int_var, array_size)))
                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = Scalar(name=stack_used_name, scope=routine, type=self.int_type)

                #List up integers and allocations generated
                integers += [stack_used_var]
                allocations += [Assignment(lhs=stack_used_var, rhs=stack_used)]

        #Substitute temporary arrays if any map
        if var_map:
            routine.body = SubstituteExpressions(var_map).visit(routine.body)

        #Add  variables to routines and allocations to body
        routine.variables = as_tuple(v for v in routine.variables if v not in temporary_arrays) + as_tuple(integers)
        routine.body.prepend(allocations)

        return stack_dict


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
        temporary_arrays = [
            var for var in temporary_arrays if (
            isinstance(var.shape[0], Scalar) and
            var.shape[0].name.lower() == self.horizontal.size.lower())
        ]

        return temporary_arrays


    def _get_kind_name(self, kind):

        if isinstance(kind, InlineCall):
            kind_name = kind.name
            for p in kind.parameters:
                kind_name += '_' + fgen(p)
            return kind_name

        return fgen(kind)


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
            if a.type.dtype in type_dict:
                if a.type.kind in type_dict[a.type.dtype]:
                    type_dict[a.type.dtype][a.type.kind] += [a]
                else:
                    type_dict[a.type.dtype][a.type.kind] = [a]
            else:
                type_dict[a.type.dtype] = {a.type.kind: [a]}

        return type_dict


    def _map_temporary_array(self, temp_array, int_var, routine, stack_var):
        """
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
        temp_arrays = [v for v in FindVariables().visit(routine.body) if v.name == temp_array.name]

        temp_map = {}
        stack_dimensions = [None, None]

        #Loop over instances of temp_array
        for t in temp_arrays:

            offset = one
            stack_size = one

            if t.dimensions:
                #If t has dimensions, we must compute the offsets in the stack
                #taking each dimension into account

                #First dimension is just horizontal
                stack_dimensions[0] = t.dimensions[0]

                #Check if lead dimension is contiguous
                contiguous = (isinstance(t.dimensions[0], RangeIndex) and
                             (t.dimensions[0] == self._get_horizontal_range(routine) or
                             (t.dimensions[0].lower is None and t.dimensions[0].upper is None)))

                s_offset = one
                for d, s in zip(t.dimensions[1:], t.shape[1:]):

                    #Check if there are range indices in shape to account for
                    if isinstance(s, RangeIndex):
                        s_lower = s.lower
                        s_upper = s.upper
                        s_extent = Sum((s_upper, Product((-1, s_lower)), one))
                    else:
                        s_lower = one
                        s_upper = s
                        s_extent = s

                    if isinstance(d, RangeIndex):

                        #If dimension is a rangeindex, compute the indices
                        #Stop if there is any non contiguous access to the array
                        if not contiguous:
                            raise RuntimeError(f'Discontiguous access of array {t}')

                        if d.lower is None:
                            d_lower = s_lower
                        else:
                            d_lower = d.lower

                        if d.upper is None:
                            d_upper = s_upper
                        else:
                            d_upper = d.upper

                        #Store if this dimension was contiguous
                        contiguous = (d_upper == s_upper) and (d_lower == s_lower)

                        #Multiply stack_size by current dimension
                        stack_size = Product((stack_size, Sum((d_upper, Product((-1, d_lower)), one))))

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

                stack_dimensions[0] = self._get_horizontal_range(routine)

                for s in t.shape[1:]:
                    if isinstance(s, RangeIndex):
                        s_lower = s.lower
                        s_upper = s.upper
                        s_extent = Sum((s_upper, Product((-1, s_lower)), one))
                    else:
                        s_lower = one
                        s_upper = s
                        s_extent = s

                    stack_size = Product((stack_size, s_extent))

            offset = simplify(offset)
            stack_size = simplify(stack_size)

            #Add offset to int_var
            if isinstance(offset, Sum):
                lower = Sum((int_var,) + offset.children)
            else:
                lower = Sum((int_var, offset))

            if stack_size == one:
                #If a single element is accessed, we only need a number
                stack_dimensions[1] = lower

            else:
                #Else we'll  have to construct a range index
                offset = simplify(Sum((offset, stack_size, Product((-1,one)))))
                if isinstance(offset, Sum):
                    upper = Sum((int_var,) + offset.children)
                else:
                    upper = Sum((int_var, offset))
                stack_dimensions[1] = RangeIndex((lower, upper))

            #Finally add to the mapping
            temp_map[t] = stack_var.clone(dimensions=as_tuple(stack_dimensions))

        return temp_map


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

                        if dtype in stack_dict:
                            if kind in stack_dict[dtype]:
                                if successor_stack_size not in stack_dict[dtype][kind]:
                                    stack_dict[dtype][kind] += [successor_stack_size]
                            else:
                                stack_dict[dtype][kind] = [successor_stack_size]
                        else:
                            stack_dict[dtype] = {kind: [successor_stack_size]}


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


    def _get_stack_var(self, routine, dtype, kind):
        """
        Get a stack variable with a name determined by
        the type_name_dict and _get_kind_name().
        intent is determined by whether the routine is a kernel or driver
        """

        stack_name = self.type_name_dict[dtype][self.role] + '_' + self._get_kind_name(kind) + '_' + self.stack_name
        stack_name = stack_name.replace('__', '_')

        if self.role == 'kernel':
            stack_intent = 'INOUT'

        if self.role == 'driver':
            stack_intent = None

        stack_type = SymbolAttributes(dtype = dtype,
                                      kind = kind,
                                      intent = stack_intent,
                                      shape = (RangeIndex((None, None))))

        return Array(name=stack_name, type=stack_type, scope=routine)


    def _get_horizontal_variable(self, routine):
        """
        Get a scalar int variable corresponding to horizontal dimension with routine as scope
        """
        return Variable(name=self.horizontal.size, scope=routine, type=self.int_type)

    def _get_horizontal_range(self, routine):
        """
        Get a RangeIndex from one to horizontal dimension
        """
        return RangeIndex((one, self._get_horizontal_variable(routine)))
