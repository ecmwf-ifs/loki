# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation, ProcedureItem
from loki.ir import nodes as ir, FindNodes, Transformer
from loki.module import Module
from loki.tools import as_tuple # , CaseInsensitiveDict
from loki.types import SymbolAttributes, BasicType
from loki.expression import Variable, Array, RangeIndex, FindVariables, SubstituteExpressions, symbols as sym
from loki.transformations.sanitise import resolve_associates
from loki.transformations.utilities import recursive_expression_map_update
from loki.transformations.single_column.base import SCCBaseTransformation
# from loki import fgen
# from loki.ir.pragma_utils import pragmas_attached

__all__ = ['BlockViewToFieldViewTransformation', 'InjectBlockIndexTransformation',
        'LowerBlockIndexTransformation', 'LowerBlockLoopTransformation']

class BlockViewToFieldViewTransformation(Transformation):
    """
    A very IFS-specific transformation to replace per-block, i.e. per OpenMP-thread, view pointers with per-field
    view pointers. It should be noted that this transformation only replaces the view pointers but does not actually
    insert the block index into the promoted view pointers. Therefore this transformation must always be followed by
    the :any:`InjectBlockIndexTransformation`.

    For example, the following code:

    .. code-block:: fortran

        do jlon=1,nproma
          mystruct%p(jlon,:) = 0.
        enddo

    is transformed to:

    .. code-block:: fortran

        do jlon=1,nproma
          mystruct%p_field(jlon,:) = 0.
        enddo

    As the rank of ``my_struct%p_field`` is one greater than that of ``my_struct%p``, we would need to also apply
    the :any:`InjectBlockIndexTransformation` to obtain semantically correct code: 

    .. code-block:: fortran

        do jlon=1,nproma
          mystruct%p_field(jlon,:,ibl) = 0.
        enddo
    
    Specific arrays in individual routines can also be marked for exclusion from this transformation by assigning
    them to the `exclude_arrays` list in the :any:`SchedulerConfig`.

    This transformation also creates minimal definitions of FIELD API wrappers (i.e. FIELD_RANKSUFF_ARRAY) and
    uses them to enrich the :any:`DataType` of relevant variable declarations and expression nodes. This is
    required because FIELD API can be built independently of library targets Loki would typically operate on.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    global_gfl_ptr: bool
        Toggle whether thread-local gfl_ptr should be replaced with global.
    key : str, optional
        Specify a different identifier under which trafo_data is stored
    """

    _key = 'BlockViewToFieldViewTransformation'
    """Identifier for trafo_data entry"""

    item_filter = (ProcedureItem,)

    def __init__(self, horizontal, global_gfl_ptr=False):
        self.horizontal = horizontal
        self.global_gfl_ptr = global_gfl_ptr

    def transform_subroutine(self, routine, **kwargs):

        if not (item := kwargs.get('item', None)):
            raise RuntimeError('Cannot apply BlockViewToFieldViewTransformation without item to store definitions')
        successors = kwargs.get('successors', ())

        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))

        exclude_arrays = item.config.get('exclude_arrays', [])

        if role == 'kernel':
            self.process_kernel(routine, item, successors, targets, exclude_arrays)
        if role == 'driver':
            self.process_driver(routine, successors)

    @staticmethod
    def _get_parkind_suffix(_type):
        return _type.rsplit('_')[1][1:3]

    def _build_parkind_import(self, field_array_module, wrapper_types):

        deferred_type = SymbolAttributes(BasicType.DEFERRED, imported=True)
        _vars = {Variable(name='JP' + self._get_parkind_suffix(t), type=deferred_type, scope=field_array_module)
                for t in wrapper_types}

        return ir.Import(module='PARKIND1', symbols=as_tuple(_vars))

    def _build_field_array_types(self, field_array_module, wrapper_types):
        """
        Build FIELD_RANKSUFF_ARRAY type-definitions.
        """

        typedefs = ()
        for _type in wrapper_types:
            suff = self._get_parkind_suffix(_type)
            kind = field_array_module.symbol_map['JP' + suff]
            rank = int(_type.rsplit('_')[1][0])

            view_shape = (RangeIndex(children=(None, None)),) * (rank - 1)
            array_shape = (RangeIndex(children=(None, None)),) * rank

            if suff == 'IM':
                basetype = BasicType.INTEGER
            elif suff == 'LM':
                basetype = BasicType.LOGICAL
            else:
                basetype = BasicType.REAL

            pointer_type = SymbolAttributes(basetype, pointer=True, kind=kind, shape=view_shape)
            contig_pointer_type = pointer_type.clone(contiguous=True, shape=array_shape)

            pointer_var = Variable(name='P', type=pointer_type, dimensions=view_shape)
            contig_pointer_var = pointer_var.clone(name='P_FIELD', type=contig_pointer_type, dimensions=array_shape) # pylint: disable=no-member

            decls = (ir.VariableDeclaration(symbols=(pointer_var,)),)
            decls += (ir.VariableDeclaration(symbols=(contig_pointer_var,)),)

            typedefs += (ir.TypeDef(name=_type, body=decls, parent=field_array_module),) # pylint: disable=unexpected-keyword-arg

        return typedefs

    def _create_dummy_field_api_defs(self, field_array_mod_imports):
        """
        Create dummy definitions for FIELD_API wrapper-types to enrich typedefs.
        """

        wrapper_types = {sym.name for imp in field_array_mod_imports for sym in imp.symbols}

        # create dummy module with empty spec
        field_array_module = Module(name='FIELD_ARRAY_MODULE', spec=ir.Section(body=()))

        # build parkind1 import
        parkind_import = self._build_parkind_import(field_array_module, wrapper_types)
        field_array_module.spec.append(parkind_import)

        # build dummy type definitions
        typedefs = self._build_field_array_types(field_array_module, wrapper_types)
        field_array_module.spec.append(typedefs)

        return [field_array_module,]

    @staticmethod
    def propagate_defs_to_children(key, definitions, successors):
        """
        Enrich all successors with the dummy FIELD_API definitions.
        """

        for child in successors:
            child.ir.enrich(definitions)
            child.trafo_data.update({key: {'definitions': definitions}})

    def process_driver(self, routine, successors):

        # create dummy definitions for field_api wrapper types
        field_array_mod_imports = [imp for imp in routine.imports if imp.module.lower() == 'field_array_module']
        definitions = []
        if field_array_mod_imports:
            definitions += self._create_dummy_field_api_defs(field_array_mod_imports)

        # propagate dummy field_api wrapper definitions to children
        self.propagate_defs_to_children(self._key, definitions, successors)

        #TODO: we also need to process any code inside a loki/acdc parallel pragma at the driver layer

    def build_ydvars_global_gfl_ptr(self, var):
        """Replace accesses to thread-local ``YDVARS%GFL_PTR`` with global ``YDVARS%GFL_PTR_G``."""

        if (parent := var.parent):
            parent = self.build_ydvars_global_gfl_ptr(parent)

        _type = var.type
        if 'gfl_ptr' in var.basename.lower():
            _type = parent.variable_map['gfl_ptr_g'].type

        return var.clone(name=var.name.upper().replace('GFL_PTR', 'GFL_PTR_G'),
                         parent=parent, type=_type)

    def process_body(self, body, definitions, successors, targets, exclude_arrays):

        # build list of type-bound array access using the horizontal index
        _vars = [var for var in FindVariables(unique=False).visit(body)
                if isinstance(var, Array) and var.parents and self.horizontal.index in var.dimensions]

        # build list of type-bound view pointers passed as subroutine arguments
        for call in FindNodes(ir.CallStatement).visit(body):
            if call.name in targets:
                _args = {a: d for d, a in call.arg_map.items() if isinstance(d, Array)}
                _vars += [a for a, d in _args.items()
                         if any(v in d.shape for v in self.horizontal.size_expressions) and a.parents]

        # replace per-block view pointers with full field pointers
        vmap = {var: var.clone(name=var.name_parts[-1] + '_FIELD',
                               type=var.parent.variable_map[var.name_parts[-1] + '_FIELD'].type)
                for var in _vars}

        # replace thread-private GFL_PTR with global
        if self.global_gfl_ptr:
            vmap.update({v: self.build_ydvars_global_gfl_ptr(vmap.get(v, v))
                         for v in FindVariables(unique=False).visit(body) if 'ydvars%gfl_ptr' in v.name.lower()})
            vmap = recursive_expression_map_update(vmap)

        # filter out arrays marked for exclusion
        vmap = {k: v for k, v in vmap.items() if not any(e in k for e in exclude_arrays)}

        # propagate dummy field_api wrapper definitions to children
        self.propagate_defs_to_children(self._key, definitions, successors)

        # finally we perform the substitution
        return SubstituteExpressions(vmap).visit(body)


    def process_kernel(self, routine, item, successors, targets, exclude_arrays):

        # Sanitize the subroutine
        resolve_associates(routine)
        v_index = SCCBaseTransformation.get_integer_variable(routine, name=self.horizontal.index)
        SCCBaseTransformation.resolve_masked_stmts(routine, loop_variable=v_index)

        # Bail if routine is marked as sequential or routine has already been processed
        if SCCBaseTransformation.check_routine_pragmas(routine, directive=None):
            print("Bail as already processed!!!")
            return

        bounds = SCCBaseTransformation.get_horizontal_loop_bounds(routine, self.horizontal)
        SCCBaseTransformation.resolve_vector_dimension(routine, loop_variable=v_index, bounds=bounds)

        # for kernels we process the entire body
        routine.body = self.process_body(routine.body, item.trafo_data[self._key]['definitions'],
                                         successors, targets, exclude_arrays)


class InjectBlockIndexTransformation(Transformation):
    """
    A transformation pass to inject the block-index in arrays promoted by a previous transformation pass. As such,
    this transformation also relies on the block-index, or a known alias, being *already* present in routines that
    are to be transformed.

    For array access in a :any:`Subroutine` body, it operates by comparing the local shape of an array with its
    declared shape. If the local shape is of rank one less than the declared shape, then the block-index is appended
    to the array's dimensions.

    For :any:`CallStatement` arguments, if the rank of the argument is one less than that of the corresponding
    dummy-argument, the block-index is appended to the argument's dimensions. It should be noted that this logic relies
    on the :any:`CallStatement` being free of any sequence-association.

    For example, the following code:

    .. code-block:: fortran

        subroutine kernel1(nblks, ...)
           ...
           integer, intent(in) :: nblks
           integer :: ibl
           real :: var(jlon,nlev,nblks)

           do ibl=1,nblks
             do jlon=1,nproma
               var(jlon,:) = 0.
             enddo

             call kernel2(var,...)
           enddo
           ...
        end subroutine kernel1

        subroutine kernel2(var, ...)
           ...
           real :: var(jlon,nlev)
        end subroutine kernel2

    is transformed to:

    .. code-block:: fortran

        subroutine kernel1(nblks, ...)
           ...
           integer, intent(in) :: nblks
           integer :: ibl
           real :: var(jlon,nlev,nblks)

           do ibl=1,nblks
             do jlon=1,nproma
               var(jlon,:,ibl) = 0.
             enddo

             call kernel2(var(:,:,ibl),...)
           enddo
           ...
        end subroutine kernel1

        subroutine kernel2(var, ...)
           ...
           real :: var(jlon,nlev)
        end subroutine kernel2

    Specific arrays in individual routines can also be marked for exclusion from this transformation by assigning
    them to the `exclude_arrays` list in the :any:`SchedulerConfig`.

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the blocking data dimension and iteration space.
    key : str, optional
        Specify a different identifier under which trafo_data is stored
    """

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, block_dim):
        self.block_dim = block_dim

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))

        exclude_arrays = []
        if (item := kwargs.get('item', None)):
            exclude_arrays = item.config.get('exclude_arrays', [])

        if role == 'kernel':
            self.process_kernel(routine, targets, exclude_arrays)

        #TODO: we also need to process any code inside a loki/acdc parallel pragma at the driver layer

    @staticmethod
    def get_call_arg_rank(arg):
        """
        Utility to retrieve the local rank of a :any:`CallStatement` argument.
        """

        rank = len(getattr(arg, 'shape', ()))
        if getattr(arg, 'dimensions', None):
            # We assume here that the callstatement is free of sequence association
            rank = rank - len([d for d in arg.dimensions if not isinstance(d, RangeIndex)])

        return rank

    def get_block_index(self, routine):
        """
        Utility to retrieve the block-index loop induction variable.
        """

        variable_map = routine.variable_map
        if (block_index := variable_map.get(self.block_dim.index, None)):
            return block_index
        if (block_index := [i for i in self.block_dim.index_expressions
                            if i.split('%', maxsplit=1)[0] in variable_map]):
            return routine.resolve_typebound_var(block_index[0], variable_map)
        return None

    def process_body(self, body, block_index, targets, exclude_arrays):
        # The logic for callstatement args differs from other variables in the body,
        # so we build a list to filter
        call_args = [a for call in FindNodes(ir.CallStatement).visit(body) for a in call.arg_map.values()]

        # First get rank mismatched call statement args
        vmap = {}
        for call in FindNodes(ir.CallStatement).visit(body):
            if call.name in targets:
                for dummy, arg in call.arg_map.items():
                    arg_rank = self.get_call_arg_rank(arg)
                    dummy_rank = len(getattr(dummy, 'shape', ()))
                    if arg_rank - 1 == dummy_rank:
                        dimensions = getattr(arg, 'dimensions', None) or ((RangeIndex((None, None)),) * (arg_rank - 1))
                        vmap.update({arg: arg.clone(dimensions=dimensions + as_tuple(block_index))})

        # Now get the rest of the variables
        for var in FindVariables(unique=False).visit(body):
            if getattr(var, 'dimensions', None) and not var in call_args:

                local_rank = len(var.dimensions)
                decl_rank = local_rank
                # we assume here that all derived-type components we wish to transform
                # have been parsed
                if getattr(var, 'shape', None):
                    decl_rank = len(var.shape)

                if local_rank == decl_rank - 1:
                    dimensions = getattr(var, 'dimensions', None) or ((RangeIndex((None, None)),) * (decl_rank - 1))
                    vmap.update({var: var.clone(dimensions=dimensions + as_tuple(block_index))})

        # filter out arrays marked for exclusion
        vmap = {k: v for k, v in vmap.items() if not any(e in k for e in exclude_arrays)}

        # finally we perform the substitution
        return SubstituteExpressions(vmap).visit(body)

    def process_kernel(self, routine, targets, exclude_arrays):

        # we skip routines that do not contain the block index or any known alias
        if not (block_index := self.get_block_index(routine)):
            return

        # for kernels we process the entire subroutine body
        routine.body = self.process_body(routine.body, block_index, targets, exclude_arrays)


class LowerBlockIndexTransformation(Transformation):
    """
    ...

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the blocking data dimension and iteration space.
    recurse_to_kernels : bool, optional
        Recurse/continue with/to (nested) kernels and lower the block index for those
        as well (default: `False`).
    """
    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, block_dim, recurse_to_kernels=True):
        self.block_dim = block_dim
        self.recurse_to_kernels = recurse_to_kernels

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))
        # dispatch driver in any case and recurse to kernels if corresponding flag is set
        if role == 'driver' or self.recurse_to_kernels and role == 'kernel':
            self.process(routine, targets, role)

    def process(self, routine, targets, role):
        processed_routines = ()
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if str(call.name).lower() not in targets:
                continue
            call_arg_map = dict((v,k) for k,v in call.arg_map.items())
            call_block_dim_index = routine.variable_map[self.block_dim.index]
            # add block_dim.index and block_dim.size if not already passed to kernel
            new_args = ()
            if routine.variable_map[self.block_dim.index] not in call_arg_map:
                new_args += (routine.variable_map[self.block_dim.index],)
            else:
                # block_dim.index already defined in relevant kernel, thus remember used name
                call_block_dim_index = call_arg_map[call_block_dim_index]
            call_block_dim_size = routine.variable_map[self.block_dim.size]
            if routine.variable_map[self.block_dim.size] not in call_arg_map:
                new_args += (routine.variable_map[self.block_dim.size],)
            else:
                # block_dim.size already defined in relevant kernel, thus remember used name
                call_block_dim_size = call_arg_map[call_block_dim_size]
            if new_args:
                call._update(kwarguments=call.kwarguments+tuple((new_arg.name, new_arg) for new_arg in new_args))
                if call.routine.name not in processed_routines:
                    call.routine.arguments += tuple((routine.variable_map[new_arg.name].clone(scope=call.routine,
                        type=new_arg.type.clone(intent='in')) for new_arg in new_args))
            # update dimensions and shape
            var_map = {}
            for arg, call_arg in call.arg_iter():
                if isinstance(arg, Array) and len(call_arg.shape) > len(arg.shape):
                    call_routine_var = call.routine.variable_map[arg.name]
                    new_dims = call_routine_var.dimensions + (call.routine.variable_map[call_block_dim_size.name],)
                    new_shape = call_routine_var.shape + (call.routine.variable_map[call_block_dim_size.name],)
                    new_type = call_routine_var.type.clone(shape=new_shape)
                    var_map[call_routine_var] = call_routine_var.clone(dimensions=new_dims, type=new_type)
            call.routine.spec = SubstituteExpressions(var_map).visit(call.routine.spec)
            if role == 'driver':
                _arguments = ()
                for arg in call.arguments:
                    if isinstance(arg, sym.Array):
                        _arguments += (arg.clone(dimensions=tuple(sym.RangeIndex((None, None))
                            if isinstance(dim, sym.Scalar)
                            and dim.name.lower() == self.block_dim.index.lower() else dim for dim in arg.dimensions)),)
                    else:
                        _arguments += (arg,)
                _kwarguments = ()
                for kwarg_name, kwarg in call.kwarguments:
                    if isinstance(kwarg, sym.Array):
                        _kwarguments += ((kwarg_name, kwarg.clone(dimensions=tuple(sym.RangeIndex((None, None))
                            if isinstance(dim, sym.Scalar)
                            and dim.name.lower() == self.block_dim.index.lower()
                            else dim for dim in kwarg.dimensions))),)
                    else:
                        _kwarguments += ((kwarg_name, kwarg),)
                call._update(arguments=_arguments, kwarguments=_kwarguments)
            processed_routines += (call.routine.name,)



class LowerBlockLoopTransformation(Transformation):
    """
    ...

    Parameters
    ----------
    block_dim : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the blocking data dimension and iteration space.
    recurse_to_kernels : bool, optional
        Recurse/continue with/to (nested) kernels and lower the block index for those
        as well (default: `False`).
    """
    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, block_dim): # , recurse_to_kernels=False): # , key=None):
        self.block_dim = block_dim
        # self.remove_loop = True

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))
        if role == 'driver':
            self.process_driver(routine, targets)

    @staticmethod
    def arg_to_local_var(routine, var):
        new_args = tuple(arg for arg in routine.arguments if arg.name.lower() != var.name.lower())
        routine.arguments = new_args
        routine.variables += (routine.variable_map[var.name].clone(scope=routine,
            type=routine.variable_map[var.name].type.clone(intent=None)),)

    def local_var(self, call, var):
        if var.name in call.arg_map:
            self.arg_to_local_var(call.routine, call.arg_map[var.name])
        elif var.name in call.routine.arguments:
            self.arg_to_local_var(call.routine, var)
        else:
            call.routine.variables += (var.clone(scope=call.routine),)
            # (routine.variable_map[var.name].clone(scope=call.routine))

    @staticmethod
    def remove_openmp_pragmas(routine):
        pragmas = [pragma for pragma in FindNodes(ir.Pragma).visit(routine.body) if pragma.keyword.lower() == 'omp']
        pragma_map = {pragma: None for pragma in pragmas}
        routine.body = Transformer(pragma_map).visit(routine.body)

    @staticmethod
    def generate_pragma(loop):
        return ir.Pragma(keyword="loki", content=f"removed_loop var({loop.variable}) \
                    lower({loop.bounds.lower}) upper({loop.bounds.upper}) \
                    step({loop.bounds.step if loop.bounds.step else 1})")

    def process_driver(self, routine, targets):
        # find block loops
        # with pragmas_attached(routine, (ir.Loop, ir.CallStatement)):
        # if True:
        loops = FindNodes(ir.Loop).visit(routine.body)
        loops = [loop for loop in loops if loop.variable == self.block_dim.index
                or loop.variable in self.block_dim._index_aliases]
        # if True:
        # loop_map = {}
        # ignore_routine = []
        driver_loop_map = {}
        to_local_var = {}
        processed_routines = ()
        calls = ()
        for loop in loops:
            for call in  FindNodes(ir.CallStatement).visit(loop.body): #visit(routine.body):
                if str(call.name).lower() not in targets:
                    continue
                calls += (call,)
                # take a copy of the loop that will be lowered
                loop_to_lower = loop.clone()
                call_arg_map = dict((v,k) for k,v in call.arg_map.items())
                # replace/substitute variables according to the caller-callee argument map
                loop_to_lower = SubstituteExpressions(call_arg_map).visit(loop_to_lower)
                # remove calls that are not within targets # TODO: rather a hack to remove
                #  "CALL TIMER%THREAD_LOG(TID, IGPC=ICEND)"
                calls_within_loop = [call for call in  FindNodes(ir.CallStatement).visit(loop_to_lower.body)
                        if str(call.name).lower() not in targets]
                loop_to_lower = Transformer({call: None for call in calls_within_loop}).visit(loop_to_lower)
                # symbols that are defined or rather assigned within the loop
                defined_symbols_loop = [assign.lhs for assign in FindNodes(ir.Assignment).visit(loop_to_lower.body)]
                call_routine_variables = [var.name.lower() for var in FindVariables().visit(call.routine.body)]
                additional_vars = [var for var in FindVariables().visit(loop_to_lower)
                        if var.name.lower() != loop.variable.name.lower()
                        and var not in call_arg_map and var.name not in call.routine.variable_map
                        and var.name.lower() not in call_routine_variables
                        # and var not in call_arg_map
                        and isinstance(var, sym.Scalar) and var not in defined_symbols_loop]
                additional_kwargs = as_tuple([(var.name, var.clone()) for var in additional_vars])
                calls_within_loop = [call for call in  FindNodes(ir.CallStatement).visit(loop_to_lower.body)
                        if str(call.name).lower() in targets]
                if call.routine.name not in processed_routines:
                    call.routine.body = ir.Section(as_tuple(Transformer({calls_within_loop[0]:\
                            call.routine.body}).visit(loop_to_lower)))
                    call.routine.arguments += as_tuple(var.clone(scope=call.routine,
                        type=var.type.clone(intent='in')) for var in additional_vars)
                call_arg_iter = list(call.arg_iter())
                ignore_symbols = [loop.variable.name.lower()] +\
                        [symbol.name.lower() for symbol in defined_symbols_loop]
                _arguments = tuple(arg for i_arg, arg in enumerate(call.arguments)
                        if call_arg_iter[i_arg][0].name.lower() not in ignore_symbols)
                _kwarguments = tuple(kwarg for kwarg in call.kwarguments if kwarg[0].lower() not in ignore_symbols)
                call._update(arguments=_arguments, kwarguments=_kwarguments+additional_kwargs)
                # add a loki pragma to keep loop information in driver
                call_pragmas = (self.generate_pragma(loop),)
                call._update(pragma=(call.pragma if call.pragma else ()) + call_pragmas)
                processed_routines += (call.routine.name,)
                to_local_var[call.routine.name] = defined_symbols_loop + [loop.variable]
            driver_loop_map[loop] = loop.body
        routine.body = Transformer(driver_loop_map).visit(routine.body)
        for call in calls: #  FindNodes(ir.CallStatement).visit(routine.body):
            if str(call.name).lower() not in targets:
                continue
            # self.local_var(routine, call, loop.variable)
            for var in to_local_var[call.routine.name]:
                self.local_var(call, var)
        # TODO: remove
        self.remove_openmp_pragmas(routine)
