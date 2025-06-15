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


__all__ = ['FtrPtrStackTransformation', 'DirectIdxStackTransformation']

class BaseStackTransformation(Transformation):
    """
    Base Transformation to inject a stack that allocates large scratch spaces per block
    and per datatype on the driver and maps temporary arrays in kernels to this scratch space.

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension.
    horizontal : :any:`Dimension`
        :any:`Dimension` object to define the horizontal dimension.
    stack_name : str, optional
        Name of the stack (default: 'STACK')
    local_int_var_name_pattern : str, optional
        Local integer variable names pattern
        (default: 'JD_{name}')
    int_kind : str, optional
        Integer kind (default: 'JWIM')
    """

    _key = 'PoolAllocatorBaseTransformation'

    reverse_traversal = True

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

        # TODO: probably shouldn't happen here ...
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
        new_imports = defaultdict(tuple)
        for s, m in item.trafo_data[cls._key]['kind_imports'].items():
            new_imports[m] += as_tuple(s)
        import_map = {i.module.lower(): i for i in routine.imports}
        for mod, symbs in new_imports.items():
            symbs = tuple(dict.fromkeys(symbs))
            if mod in import_map:
                import_map[mod]._update(symbols=as_tuple(dict.fromkeys(import_map[mod].symbols +symbs)))
            else:
                _symbs = [s for s in symbs if not (s.name.lower() in routine.variable_map or
                                                   s.name.lower() in routine.imported_symbol_map)]
                if _symbs:
                    imp = Import(module=mod, symbols=as_tuple(_symbs))
                    routine.spec.prepend(imp)

    @staticmethod
    def _insert_stack_at_loki_pragma(routine, insert):
        for pragma in FindNodes(Pragma).visit(routine.body):
            if pragma.keyword == 'loki' and 'stack-insert' in pragma.content:
                routine.body = Transformer({pragma: insert}).visit(routine.body)
                return True
        return False


    def _get_stack_int_name(self, prefix, dtype, kind, suffix):
        """
        Construct the name string for stack used and size integers.
        Replace double underscore with single if kind is None
        """
        return (f'{prefix}_{self.type_name_dict[dtype][self.role]}_'
                f'{self._get_kind_name(kind)}_{suffix}'.replace('__', '_'))


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

        # loop over calls and check if they call a successor routine and if the
        # transformation data is available
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in successor_map and self._key in successor_map[call.name].trafo_data:
                successor_stack_dict = successor_map[call.name].trafo_data[self._key]['stack_dict']

                call_stack_args = []

                # loop over dtypes and kinds in successor arguments stacks
                # and construct list of stack arguments
                for dtype in successor_stack_dict:
                    for kind in successor_stack_dict[dtype]:
                        call_stack_args += list(stack_arg_dict[dtype][kind])

                # get position of optional arguments so we can place the stacks in front
                arg_pos = [call.routine.arguments.index(arg) for arg in call.routine.arguments if arg.type.optional]

                arguments = call.arguments
                if arg_pos:
                    # stack arguments have already been added to the routine call signature
                    # so we have to subtract the number of stack arguments from the optional position
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

        # block variables
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
                # start integer names in the driver with 'J'
                stack_size_name = self._get_stack_int_name('J', dtype, kind, 'STACK_SIZE')
                stack_size_var = self._get_int_var(name=stack_size_name, scope=routine)

                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                # create the stack variable and its type with the correct shape
                stack_var = self._get_stack_var(routine, dtype, kind)

                stack_type = stack_var.type.clone(shape=(RangeIndex((None,None)), RangeIndex((None,None))),
                                                  allocatable=True)
                stack_var = stack_var.clone(type=stack_type)

                stack_alloc = Allocation(variables=(stack_var.clone(dimensions=(stack_dict[dtype][kind], kgpblock)),))
                stack_dealloc = Deallocation(variables=(stack_var.clone(dimensions=None),))

                # add the variables to the stack_arg_dict with dimensions (:,j_block)
                stack_arg_dict.setdefault(dtype, {})
                stack_arg_dict[dtype][kind] = (stack_size_var,
                                               stack_var.clone(dimensions=(RangeIndex((None,None)), jgpblock,)),
                                               stack_used_var)
                stack_var = stack_var.clone(dimensions=stack_type.shape)

                stack_used_var_init = Assignment(lhs=stack_used_var, rhs=IntLiteral(1))
                # create stack_vars pair and assignment of the size variable
                stack_vars += [stack_size_var, stack_var, stack_used_var]
                assignments += [Assignment(lhs=stack_size_var,
                                           rhs=stack_dict[dtype][kind]), stack_alloc, stack_used_var_init]
                deallocs += [stack_dealloc]
                pragma_string += f'{stack_var.name}, '

        # add to routine
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

        # insert variables in successor calls
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

                # start arguments integer names in kernels with 'K'
                stack_size_name = self._get_stack_int_name('K', dtype, kind, 'STACK_SIZE')
                stack_size_var = self._get_int_var(name=stack_size_name, scope=routine,
                                                   type=self._get_int_type(intent='IN'))

                # local variables start with 'J'
                stack_used_arg_name = self._get_stack_int_name('JD', dtype, kind, 'STACK_USED')
                stack_used_arg = self._get_int_var(name=stack_used_arg_name, scope=routine,
                                                   type=self._get_int_type(intent='INOUT'))
                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)
                assignments += [Assignment(lhs=stack_used_var, rhs=stack_used_arg)]

                # create the stack variable and its type with the correct shape
                shape = (stack_size_var,)
                stack_var = self._get_stack_var(routine, dtype, kind)
                stack_type = stack_var.type.clone(shape=as_tuple(shape), target=True)
                stack_var = stack_var.clone(type=stack_type)

                # pass on the stack variable from stack_used + 1 to stack_size
                # pass stack_size - stack_used to stack size in called kernel
                arg_dims = (RangeIndex((None, None)),)
                stack_arg_dict.setdefault(dtype, {})
                stack_arg_dict[dtype][kind] = (stack_size_var, stack_var.clone(dimensions=arg_dims), stack_used_var)

                # create stack_vars pair
                stack_args += [stack_size_var,
                               stack_var.clone(dimensions=stack_type.shape, type=stack_var.type.clone(contiguous=True)),
                               stack_used_arg]
                stack_vars += [stack_used_var]
                pragma_string += f'{stack_var.name}, '

        if pragma_string:
            # remove ', '
            pragma_string = pragma_string[:-2].lower()
            present_pragmas = [
                p for p in FindNodes(Pragma).visit(routine.body)
                if p.keyword.lower() == 'loki' and p.content.lower().startswith('device-present')
            ]
            if present_pragmas:
                present_pragma = present_pragmas[0]
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

        # keep optional arguments last; a workaround for the fact that keyword arguments are not supported
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

        # simplify the local stack sizes and add them to the stack_dict
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

        # if several expressions, return MAX, else just add the expression
        for (dtype, kind_dict) in stack_dict.items():
            for (kind, stacks) in kind_dict.items():
                if len(stacks) == 1:
                    kind_dict[kind] = stacks[0]
                else:
                    kind_dict[kind] = InlineCall(function = Variable(name = 'MAX'), parameters = as_tuple(stacks))

        return stack_dict

class FtrPtrStackTransformation(BaseStackTransformation):
    """         
    Transformation to inject a stack that allocates large scratch spaces per block
    and per datatype on the driver and maps temporary arrays in kernels to this scratch space.

    Starting from:

    .. code-block:: fortran
        SUBROUTINE driver (nlon, klev, nb, ydml_phy_mf)

          USE kernel_mod, ONLY: kernel

          IMPLICIT NONE

          INTEGER, INTENT(IN) :: nlon
          INTEGER, INTENT(IN) :: klev
          INTEGER, INTENT(IN) :: nb

          INTEGER :: jstart
          INTEGER :: jend

          INTEGER :: b

          REAL(KIND=jprb), DIMENSION(nlon, klev) :: zzz

          jstart = 1
          jend = nlon

          DO b=1,nb
            CALL kernel(nlon, klev, jstart, jend, zzz)
          END DO

        END SUBROUTINE driver

        SUBROUTINE kernel (nlon, klev, jstart, jend, pzz)

          IMPLICIT NONE

          INTEGER, INTENT(IN) :: nlon
          INTEGER, INTENT(IN) :: klev

          INTEGER, INTENT(IN) :: jstart
          INTEGER, INTENT(IN) :: jend

          REAL, INTENT(IN), DIMENSION(nlon, klev) :: pzz

          REAL, DIMENSION(nlon, klev) :: zzx
          REAL(KIND=SELECTED_REAL_KIND(13, 300)), DIMENSION(nlon, klev) :: zzy
          LOGICAL, DIMENSION(nlon, klev) :: zzl

          INTEGER :: testint
          INTEGER :: jl, jlev

          zzl = .false.
          DO jl=1,nlon
            DO jlev=1,klev
              zzx(jl, jlev) = pzz(jl, jlev)
              zzy(jl, jlev) = pzz(jl, jlev)
            END DO
          END DO

        END SUBROUTINE kernel

    This transformation generates:

    .. code-block:: fortran
        SUBROUTINE driver (nlon, klev, nb)

          USE kernel_mod, ONLY: kernel

          IMPLICIT NONE

          INTEGER, INTENT(IN) :: nlon
          INTEGER, INTENT(IN) :: klev
          INTEGER(KIND=JWIM) :: nb

          INTEGER :: jstart
          INTEGER :: jend

          INTEGER(KIND=JWIM) :: b

          REAL(KIND=jprb), DIMENSION(nlon, klev) :: zzz
          INTEGER(KIND=JWIM) :: J_Z_STACK_SIZE
          REAL, ALLOCATABLE :: Z_STACK(:, :)
          INTEGER(KIND=JWIM) :: J_Z_STACK_USED
          INTEGER(KIND=JWIM) :: J_Z_SELECTED_REAL_KIND_13_300_STACK_SIZE
          REAL(KIND=SELECTED_REAL_KIND(13, 300)), ALLOCATABLE :: Z_SELECTED_REAL_KIND_13_300_STACK(:, :)
          INTEGER(KIND=JWIM) :: J_Z_SELECTED_REAL_KIND_13_300_STACK_USED
          INTEGER(KIND=JWIM) :: J_LL_STACK_SIZE
          LOGICAL, ALLOCATABLE :: LL_STACK(:, :)
          INTEGER(KIND=JWIM) :: J_LL_STACK_USED
          J_Z_STACK_SIZE = klev*nlon
          ALLOCATE (Z_STACK(klev*nlon, nb))
          J_Z_STACK_USED = 1
          J_Z_SELECTED_REAL_KIND_13_300_STACK_SIZE = klev*nlon
          ALLOCATE (Z_SELECTED_REAL_KIND_13_300_STACK(klev*nlon, nb))
          J_Z_SELECTED_REAL_KIND_13_300_STACK_USED = 1
          J_LL_STACK_SIZE = klev*nlon
          ALLOCATE (LL_STACK(klev*nlon, nb))
          J_LL_STACK_USED = 1
        !$loki unstructured-data create( z_stack, z_selected_real_kind_13_300_stack, ll_stack )

          jstart = 1
          jend = nlon

          DO b=1,nb
            CALL kernel(nlon, klev, jstart, jend, zzz, J_Z_STACK_SIZE, Z_STACK(:, b), J_Z_STACK_USED,  &
            & J_Z_SELECTED_REAL_KIND_13_300_STACK_SIZE, Z_SELECTED_REAL_KIND_13_300_STACK(:, b),  &
            & J_Z_SELECTED_REAL_KIND_13_300_STACK_USED, J_LL_STACK_SIZE, LL_STACK(:, b), J_LL_STACK_USED)
          END DO

        !$loki end unstructured-data delete( z_stack, z_selected_real_kind_13_300_stack, ll_stack )
          DEALLOCATE (Z_STACK)
          DEALLOCATE (Z_SELECTED_REAL_KIND_13_300_STACK)
          DEALLOCATE (LL_STACK)
        END SUBROUTINE driver

        SUBROUTINE kernel (nlon, klev, jstart, jend, pzz, K_P_STACK_SIZE, P_STACK, JD_P_STACK_USED,  &
        & K_P_SELECTED_REAL_KIND_13_300_STACK_SIZE, P_SELECTED_REAL_KIND_13_300_STACK, & 
        & JD_P_SELECTED_REAL_KIND_13_300_STACK_USED,  &
        & K_LD_STACK_SIZE, LD_STACK, JD_LD_STACK_USED)

          IMPLICIT NONE

          INTEGER, INTENT(IN) :: nlon
          INTEGER, INTENT(IN) :: klev

          INTEGER, INTENT(IN) :: jstart
          INTEGER, INTENT(IN) :: jend

          REAL, INTENT(IN), DIMENSION(nlon, klev) :: pzz

          REAL, POINTER, CONTIGUOUS, DIMENSION(:, :) :: zzx
          REAL(KIND=SELECTED_REAL_KIND(13, 300)), POINTER, CONTIGUOUS, DIMENSION(:, :) :: zzy
          LOGICAL, POINTER, CONTIGUOUS, DIMENSION(:, :) :: zzl

          INTEGER :: testint
          INTEGER :: jl, jlev
          INTEGER(KIND=JWIM) :: JD_incr
          INTEGER(KIND=JWIM) :: JD_incr_SELECTED_REAL_KIND_13_300
          INTEGER(KIND=JWIM) :: JD_incr
          INTEGER(KIND=JWIM) :: J_P_STACK_USED
          INTEGER(KIND=JWIM) :: J_P_SELECTED_REAL_KIND_13_300_STACK_USED
          INTEGER(KIND=JWIM) :: J_LD_STACK_USED
          INTEGER(KIND=JWIM), INTENT(IN) :: K_P_STACK_SIZE
          REAL, TARGET, CONTIGUOUS, INTENT(INOUT) :: P_STACK(K_P_STACK_SIZE)
          INTEGER(KIND=JWIM), INTENT(INOUT) :: JD_P_STACK_USED
          INTEGER(KIND=JWIM), INTENT(IN) :: K_P_SELECTED_REAL_KIND_13_300_STACK_SIZE
          REAL(KIND=SELECTED_REAL_KIND(13, 300)), TARGET, CONTIGUOUS, INTENT(INOUT) ::  &
          & P_SELECTED_REAL_KIND_13_300_STACK(K_P_SELECTED_REAL_KIND_13_300_STACK_SIZE)
          INTEGER(KIND=JWIM), INTENT(INOUT) :: JD_P_SELECTED_REAL_KIND_13_300_STACK_USED
          INTEGER(KIND=JWIM), INTENT(IN) :: K_LD_STACK_SIZE
          LOGICAL, TARGET, CONTIGUOUS, INTENT(INOUT) :: LD_STACK(K_LD_STACK_SIZE)
          INTEGER(KIND=JWIM), INTENT(INOUT) :: JD_LD_STACK_USED
          J_P_STACK_USED = JD_P_STACK_USED
          J_P_SELECTED_REAL_KIND_13_300_STACK_USED = JD_P_SELECTED_REAL_KIND_13_300_STACK_USED
          J_LD_STACK_USED = JD_LD_STACK_USED
        !$loki device-present vars( p_stack, p_selected_real_kind_13_300_stack, ld_stack )
          JD_incr = J_P_STACK_USED
          zzx(1:nlon, 1:klev) => P_STACK(JD_incr:JD_incr + nlon*klev)
          J_P_STACK_USED = JD_incr + klev*nlon
          JD_incr_SELECTED_REAL_KIND_13_300 = J_P_SELECTED_REAL_KIND_13_300_STACK_USED
          zzy(1:nlon, 1:klev) =>  &
          & P_SELECTED_REAL_KIND_13_300_STACK(JD_incr_SELECTED_REAL_KIND_13_300: &
              & JD_incr_SELECTED_REAL_KIND_13_300 + nlon*klev)
          J_P_SELECTED_REAL_KIND_13_300_STACK_USED = JD_incr_SELECTED_REAL_KIND_13_300 + klev*nlon
          JD_incr = J_LD_STACK_USED
          zzl(1:nlon, 1:klev) => LD_STACK(JD_incr:JD_incr + nlon*klev)
          J_LD_STACK_USED = JD_incr + klev*nlon

          zzl = .false.
          DO jl=1,nlon
            DO jlev=1,klev
              zzx(jl, jlev) = pzz(jl, jlev)
              zzy(jl, jlev) = pzz(jl, jlev)
            END DO
          END DO

        !$loki end device-present
        END SUBROUTINE kernel

    Parameters  
    ----------  
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension.
    horizontal : :any:`Dimension`
        :any:`Dimension` object to define the horizontal dimension.
    stack_name : str, optional
        Name of the stack (default: 'STACK')
    local_int_var_name_pattern : str, optional    
        Local integer variable names pattern
        (default: 'JD_{name}')
    int_kind : str, optional
        Integer kind (default: 'JWIM')
    """

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

        # get all temporary dicts and sort them according to dtype and kind
        temporary_arrays = self._filter_temporary_arrays(routine)

        self.adapt_temp_declarations(routine, temporary_arrays)
        temporary_array_dict = self._sort_arrays_by_type(temporary_arrays)

        integers = []
        allocations = []

        stack_dict = {}

        for (dtype, kind_dict) in temporary_array_dict.items():

            if dtype not in stack_dict:
                stack_dict[dtype] = {}

            for (kind, arrays) in kind_dict.items():

                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                stack_used = IntLiteral(1)
                if kind not in stack_dict[dtype]:
                    stack_dict[dtype][kind] = Literal(0)

                # store type information of temporary allocation
                if item:
                    if kind in routine.imported_symbols:
                        item.trafo_data[self._key]['kind_imports'][kind] = routine.import_map[kind.name].module.lower()
                    for array in arrays:
                        dims = [d for d in array.shape if d in routine.imported_symbols]
                        for d in dims:
                            item.trafo_data[self._key]['kind_imports'][d] = routine.import_map[d.name].module.lower()

                # get the stack variable
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

                # loop over arrays
                for array in arrays:

                    # compute array size
                    array_size = IntLiteral(1)
                    for d in array.shape:
                        if isinstance(d, RangeIndex):
                            d_extent = Sum((d.upper, Product((-1,d.lower)), IntLiteral(1)))
                        else:
                            d_extent = d
                        array_size = simplify(Product((array_size, d_extent)))

                    # add to stack dict and list of allocations
                    stack_dict[dtype][kind] = simplify(Sum((stack_dict[dtype][kind], array_size)))
                    allocations += [Assignment(lhs=int_var, rhs=Sum((old_int_var,) + old_array_size))]

                    # store the old int variable to calculate offset for next array
                    old_int_var = int_var
                    if isinstance(array_size, Sum):
                        old_array_size = array_size.children
                    else:
                        old_array_size = (array_size,)

                    ptr_assignment = self._get_ptr_assignment(array, int_var, stack_var)
                    allocations += [ptr_assignment]

                # compute stack used
                stack_used = simplify(Sum((int_var, array_size)))
                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                # list up integers and allocations generated
                allocations += [Assignment(lhs=stack_used_var, rhs=stack_used)]

        # add variables to routines and allocations to body
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

class DirectIdxStackTransformation(BaseStackTransformation):
    """         
    Transformation to inject a stack that allocates large scratch spaces per block
    and per datatype on the driver and maps temporary arrays in kernels to this scratch space.
                
    Starting from:

    .. code-block:: fortran
        SUBROUTINE driver (nlon, klev, nb, ydml_phy_mf)

          USE kernel_mod, ONLY: kernel

          IMPLICIT NONE

          INTEGER, INTENT(IN) :: nlon
          INTEGER, INTENT(IN) :: klev
          INTEGER, INTENT(IN) :: nb

          INTEGER :: jstart
          INTEGER :: jend

          INTEGER :: b

          REAL(KIND=jprb), DIMENSION(nlon, klev) :: zzz

          jstart = 1
          jend = nlon

          DO b=1,nb
            CALL kernel(nlon, klev, jstart, jend, zzz)
          END DO

        END SUBROUTINE driver

        SUBROUTINE kernel (nlon, klev, jstart, jend, pzz)

          IMPLICIT NONE

          INTEGER, INTENT(IN) :: nlon
          INTEGER, INTENT(IN) :: klev

          INTEGER, INTENT(IN) :: jstart
          INTEGER, INTENT(IN) :: jend

          REAL, INTENT(IN), DIMENSION(nlon, klev) :: pzz

          REAL, DIMENSION(nlon, klev) :: zzx
          REAL(KIND=SELECTED_REAL_KIND(13, 300)), DIMENSION(nlon, klev) :: zzy
          LOGICAL, DIMENSION(nlon, klev) :: zzl

          INTEGER :: testint
          INTEGER :: jl, jlev

          zzl = .false.
          DO jl=1,nlon
            DO jlev=1,klev
              zzx(jl, jlev) = pzz(jl, jlev)
              zzy(jl, jlev) = pzz(jl, jlev)
            END DO
          END DO

        END SUBROUTINE kernel

    This transformation generates:

    .. code-block:: fortran
        SUBROUTINE driver (nlon, klev, nb)

          USE kernel_mod, ONLY: kernel

          IMPLICIT NONE

          INTEGER, INTENT(IN) :: nlon
          INTEGER, INTENT(IN) :: klev
          INTEGER(KIND=JWIM) :: nb

          INTEGER :: jstart
          INTEGER :: jend

          INTEGER(KIND=JWIM) :: b

          REAL(KIND=jprb), DIMENSION(nlon, klev) :: zzz
          INTEGER(KIND=JWIM) :: J_Z_STACK_SIZE
          REAL, ALLOCATABLE :: Z_STACK(:, :)
          INTEGER(KIND=JWIM) :: J_Z_STACK_USED
          INTEGER(KIND=JWIM) :: J_Z_SELECTED_REAL_KIND_13_300_STACK_SIZE
          REAL(KIND=SELECTED_REAL_KIND(13, 300)), ALLOCATABLE :: Z_SELECTED_REAL_KIND_13_300_STACK(:, :)
          INTEGER(KIND=JWIM) :: J_Z_SELECTED_REAL_KIND_13_300_STACK_USED
          INTEGER(KIND=JWIM) :: J_LL_STACK_SIZE
          LOGICAL, ALLOCATABLE :: LL_STACK(:, :)
          INTEGER(KIND=JWIM) :: J_LL_STACK_USED
          J_Z_STACK_SIZE = klev*nlon
          ALLOCATE (Z_STACK(klev*nlon, nb))
          J_Z_STACK_USED = 1
          J_Z_SELECTED_REAL_KIND_13_300_STACK_SIZE = klev*nlon
          ALLOCATE (Z_SELECTED_REAL_KIND_13_300_STACK(klev*nlon, nb))
          J_Z_SELECTED_REAL_KIND_13_300_STACK_USED = 1
          J_LL_STACK_SIZE = klev*nlon
          ALLOCATE (LL_STACK(klev*nlon, nb))
          J_LL_STACK_USED = 1
        !$loki unstructured-data create( z_stack, z_selected_real_kind_13_300_stack, ll_stack )

          jstart = 1
          jend = nlon

          DO b=1,nb
            CALL kernel(nlon, klev, jstart, jend, zzz, J_Z_STACK_SIZE, Z_STACK(:, b), J_Z_STACK_USED,  &
            & J_Z_SELECTED_REAL_KIND_13_300_STACK_SIZE, Z_SELECTED_REAL_KIND_13_300_STACK(:, b),  &
            & J_Z_SELECTED_REAL_KIND_13_300_STACK_USED, J_LL_STACK_SIZE, LL_STACK(:, b), J_LL_STACK_USED)
          END DO

        !$loki end unstructured-data delete( z_stack, z_selected_real_kind_13_300_stack, ll_stack )
          DEALLOCATE (Z_STACK)
          DEALLOCATE (Z_SELECTED_REAL_KIND_13_300_STACK)
          DEALLOCATE (LL_STACK)
        END SUBROUTINE driver

        SUBROUTINE kernel (nlon, klev, jstart, jend, pzz, K_P_STACK_SIZE, P_STACK, JD_P_STACK_USED,  &
        & K_P_SELECTED_REAL_KIND_13_300_STACK_SIZE, P_SELECTED_REAL_KIND_13_300_STACK, & 
        & JD_P_SELECTED_REAL_KIND_13_300_STACK_USED,  &
        & K_LD_STACK_SIZE, LD_STACK, JD_LD_STACK_USED)

          IMPLICIT NONE

          INTEGER, INTENT(IN) :: nlon
          INTEGER, INTENT(IN) :: klev

          INTEGER, INTENT(IN) :: jstart
          INTEGER, INTENT(IN) :: jend

          REAL, INTENT(IN), DIMENSION(nlon, klev) :: pzz


          INTEGER :: testint
          INTEGER :: jl, jlev
          INTEGER(KIND=JWIM) :: JD_zzx
          INTEGER(KIND=JWIM) :: JD_zzy
          INTEGER(KIND=JWIM) :: JD_zzl
          INTEGER(KIND=JWIM) :: J_P_STACK_USED
          INTEGER(KIND=JWIM) :: J_P_SELECTED_REAL_KIND_13_300_STACK_USED
          INTEGER(KIND=JWIM) :: J_LD_STACK_USED
          INTEGER(KIND=JWIM), INTENT(IN) :: K_P_STACK_SIZE
          REAL, TARGET, CONTIGUOUS, INTENT(INOUT) :: P_STACK(K_P_STACK_SIZE)
          INTEGER(KIND=JWIM), INTENT(INOUT) :: JD_P_STACK_USED
          INTEGER(KIND=JWIM), INTENT(IN) :: K_P_SELECTED_REAL_KIND_13_300_STACK_SIZE
          REAL(KIND=SELECTED_REAL_KIND(13, 300)), TARGET, CONTIGUOUS, INTENT(INOUT) ::  &
          & P_SELECTED_REAL_KIND_13_300_STACK(K_P_SELECTED_REAL_KIND_13_300_STACK_SIZE)
          INTEGER(KIND=JWIM), INTENT(INOUT) :: JD_P_SELECTED_REAL_KIND_13_300_STACK_USED
          INTEGER(KIND=JWIM), INTENT(IN) :: K_LD_STACK_SIZE
          LOGICAL, TARGET, CONTIGUOUS, INTENT(INOUT) :: LD_STACK(K_LD_STACK_SIZE)
          INTEGER(KIND=JWIM), INTENT(INOUT) :: JD_LD_STACK_USED
          J_P_STACK_USED = JD_P_STACK_USED
          J_P_SELECTED_REAL_KIND_13_300_STACK_USED = JD_P_SELECTED_REAL_KIND_13_300_STACK_USED
          J_LD_STACK_USED = JD_LD_STACK_USED
        !$loki device-present vars( p_stack, p_selected_real_kind_13_300_stack, ld_stack )
          JD_zzx = J_P_STACK_USED
          J_P_STACK_USED = JD_zzx + klev*nlon
          JD_zzy = J_P_SELECTED_REAL_KIND_13_300_STACK_USED
          J_P_SELECTED_REAL_KIND_13_300_STACK_USED = JD_zzy + klev*nlon
          JD_zzl = J_LD_STACK_USED
          J_LD_STACK_USED = JD_zzl + klev*nlon

          LD_STACK(1:klev*nlon) = .false.
          DO jl=1,nlon
            DO jlev=1,klev
              P_STACK(JD_zzx + jl - nlon + jlev*nlon) = pzz(jl, jlev)
              P_SELECTED_REAL_KIND_13_300_STACK(JD_zzy + jl - nlon + jlev*nlon) = pzz(jl, jlev)
            END DO
          END DO

        !$loki end device-present
        END SUBROUTINE kernel



    Parameters  
    ----------  
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension.
    horizontal : :any:`Dimension`
        :any:`Dimension` object to define the horizontal dimension.
    stack_name : str, optional
        Name of the stack (default: 'STACK')
    local_int_var_name_pattern : str, optional    
        Local integer variable names pattern
        (default: 'JD_{name}')
    int_kind : str, optional
        Integer kind (default: 'JWIM')
    """

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

        # get all temporary dicts and sort them according to dtype and kind
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

                # initialize stack_used to 0
                stack_used = IntLiteral(1)
                if kind not in stack_dict[dtype]:
                    stack_dict[dtype][kind] = Literal(0)

                # store type information of temporary allocation
                if item:
                    if kind in routine.imported_symbols:
                        item.trafo_data[self._key]['kind_imports'][kind] = routine.import_map[kind.name].module.lower()
                    for array in arrays:
                        dims = [d for d in array.shape if d in routine.imported_symbols]
                        for d in dims:
                            item.trafo_data[self._key]['kind_imports'][d] = routine.import_map[d.name].module.lower()

                # get the stack variable
                stack_var = self._get_stack_var(routine, dtype, kind)
                old_int_var = stack_used_var
                old_array_size = ()

                # loop over arrays
                for array in arrays:

                    int_var_name = self.local_int_var_name_pattern.format(name=array.name)
                    int_var = self._get_int_var(name=int_var_name, scope=routine)
                    integers += [int_var]

                    # compute array size
                    array_size = IntLiteral(1)
                    for d in array.shape:
                        if isinstance(d, RangeIndex):
                            d_extent = Sum((d.upper, Product((-1,d.lower)), IntLiteral(1)))
                        else:
                            d_extent = d
                        array_size = simplify(Product((array_size, d_extent)))

                    # add to stack dict and list of allocations
                    stack_dict[dtype][kind] = simplify(Sum((stack_dict[dtype][kind], array_size)))
                    allocations += [Assignment(lhs=int_var, rhs=Sum((old_int_var,) + old_array_size))]

                    # store the old int variable to calculate offset for next array
                    old_int_var = int_var
                    if isinstance(array_size, Sum):
                        old_array_size = array_size.children
                    else:
                        old_array_size = (array_size,)

                    # save for later usage
                    temp_array_map[array.name] = (array, stack_var, int_var)

                # compute stack used
                stack_used = simplify(Sum((int_var, array_size)))
                stack_used_name = self._get_stack_int_name('J', dtype, kind, 'STACK_USED')
                stack_used_var = self._get_int_var(name=stack_used_name, scope=routine)

                # list up integers and allocations generated
                allocations += [Assignment(lhs=stack_used_var, rhs=stack_used)]

        var_map = self._map_temporary_array(temp_array_map, routine)
        if var_map:
            var_map = recursive_expression_map_update(var_map)
            routine.body = SubstituteExpressions(var_map).visit(routine.body)

        # add variables to routines and allocations to body
        routine.variables = as_tuple(v for v in routine.variables if v not in temporary_arrays) + as_tuple(integers)
        routine.body.prepend(allocations)

        return stack_dict

    def _map_temporary_array(self, temp_array_map, routine):
        """
        Find all instances of temporary arrays and
        map them to to the corresponding stack_var and position in stack stack_var.
        Position in stack is stored in the relevant int_var.
        """

        # list instances of temp_array
        temp_arrays = [v for v in FindVariables().visit(routine.body) if v.name.lower() in temp_array_map.keys()]
        temp_map = {}
        stack_dimensions = [None]

        # loop over instances of temp_array
        for t in temp_arrays:

            stack_var = temp_array_map[t.name][1]
            int_var = temp_array_map[t.name][2]

            offset = IntLiteral(1)
            stack_size = IntLiteral(1)

            if t.dimensions:
                # if t has dimensions, we must compute the offsets in the stack
                # taking each dimension into account

                # check if lead dimension is contiguous
                contiguous = (isinstance(t.dimensions[0], RangeIndex) and
                             (t.dimensions[0] == self._get_horizontal_range(routine) or
                             (t.dimensions[0].lower is None and t.dimensions[0].upper is None)))

                s_offset = IntLiteral(1)
                for d, s in zip(t.dimensions, t.shape):

                    # check if there are range indices in shape to account for
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
                        # if dimension is a rangeindex, compute the indices
                        # stop if there is any non contiguous access to the array
                        if not contiguous:
                            # raise RuntimeError(f'Discontiguous access of array {t}')
                            print(f'Discontiguous access of array {t} within {routine}')

                        d_lower = d.lower or s_lower
                        d_upper = d.upper or s_upper

                        # store if this dimension was contiguous
                        contiguous = (d_upper == s_upper) and (d_lower == s_lower)

                        # multiply stack_size by current dimension
                        stack_size = Product((stack_size, Sum((d_upper, Product((-1, d_lower)), IntLiteral(1)))))

                    else:

                        # only need a single index to compute offset
                        d_lower = d

                    # compute dimension and shape offsets
                    d_offset =  Sum((d_lower, Product((-1, s_lower))))
                    offset = Sum((offset, Product((d_offset, s_offset))))
                    s_offset = Product((s_offset, s_extent))

            else:
                # if t does not have dimensions,
                # we can just access (1:horizontal.size, 1:stack_size)

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

            # add offset to int_var
            lower = Sum((int_var,) + offset.children if isinstance(offset, Sum) else (offset,))

            if stack_size == IntLiteral(1):
                # if a single element is accessed, we only need a number
                stack_dimensions[0] = lower

            else:
                # else we'll  have to construct a range index
                offset = simplify(Sum((offset, stack_size, Product((-1, IntLiteral(1))))))
                upper = Sum((int_var,) + offset.children if isinstance(offset, Sum) else (offset,))
                stack_dimensions[0] = RangeIndex((lower, upper))

            # finally add to the mapping
            temp_map[t] = stack_var.clone(dimensions=as_tuple(stack_dimensions))

        return temp_map
