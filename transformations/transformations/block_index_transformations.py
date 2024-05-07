# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

## copied from PR: naan-block-index-inject

from loki import (
    Transformation, ProcedureItem, ir, Module, as_tuple, SymbolAttributes, BasicType, Variable,
    RangeIndex, Array, FindVariables, resolve_associates, SubstituteExpressions, FindNodes,
    recursive_expression_map_update, Transformer, symbols as sym
)

from transformations.single_column_coalesced import SCCBaseTransformation

__all__ = ['BlockViewToFieldViewTransformation', 'BlockIndexInjectTransformation',
        'BlockIndexLowerTransformation', 'BlockLoopLowerTransformation']

class BlockViewToFieldViewTransformation(Transformation):
    """
    A very IFS-specific transformation to replace per-block, i.e. per OpenMP-thread, view pointers with per-field
    view pointers. It should be noted that this transformation only replaces the view pointers but does not actually
    insert the block index into the promoted view pointers. Therefore this transformation must always be followed by
    the :any:`BlockIndexInjectTransformation`.

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

    Where the rank of ``my_struct%p_field`` is one greater than that of ``my_struct%p``. Specific arrays in individual
    routines can also be marked for exclusion from this transformation by assigning them to the `exclude_arrays` list
    in the :any:`SchedulerConfig`.

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
    """Default identifier for trafo_data entry"""

    item_filter = (ProcedureItem,)

    def __init__(self, horizontal, global_gfl_ptr=False, key=None):
        self.horizontal = horizontal
        self.global_gfl_ptr = global_gfl_ptr
        if key:
            self._key = key

    @staticmethod
    def get_parent_typedef(var, symbol_map):
        """Utility method to retrieve derived-tyoe definition of parent type."""

        if not var.parent.type.dtype.typedef is BasicType.DEFERRED:
            return var.parent.type.dtype.typedef
        if  (_parent_type := symbol_map.get(var.parent.type.dtype.name, None)):
            if not _parent_type.type.dtype.typedef is BasicType.DEFERRED:
                return _parent_type.type.dtype.typedef
        raise RuntimeError(f'Container data-type {var.parent.type.dtype.name} not enriched')

    def transform_subroutine(self, routine, **kwargs):

        if not (item := kwargs['item']):
            raise RuntimeError('Cannot apply DeprivatiseStructsTransformation without item to store definitions')
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

            typedefs += (ir.TypeDef(name=_type, body=decls, parent=field_array_module),)

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
        if 'gfl_ptr' in var.name.lower().split('%')[-1]:
            _type = parent.type.dtype.typedef.variable_map['gfl_ptr_g'].type

        return var.clone(name=var.name.upper().replace('GFL_PTR', 'GFL_PTR_G'),
                         parent=parent, type=_type)

    def process_body(self, body, symbol_map, definitions, successors, targets, exclude_arrays):

        # build list of type-bound array access using the horizontal index
        _vars = [var for var in FindVariables().visit(body)
                if isinstance(var, Array) and var.parents and self.horizontal.index in getattr(var, 'dimensions', ())]

        # build list of type-bound view pointers passed as subroutine arguments
        for call in [call for call in FindNodes(ir.CallStatement).visit(body) if call.name in targets]:
            _args = {a: d for d, a in call.arg_map.items() if isinstance(d, Array)}
            _vars += [a for a, d in _args.items()
                     if any(v in d.shape for v in self.horizontal.size_expressions) and a.parents]

        # replace per-block view pointers with full field pointers
        vmap = {var:
                var.clone(name=var.name_parts[-1] + '_FIELD',
                type=self.get_parent_typedef(var, symbol_map).variable_map[var.name_parts[-1] + '_FIELD'].type)
                for var in _vars}

        # replace thread-private GFL_PTR with global
        if self.global_gfl_ptr:
            vmap.update({v: self.build_ydvars_global_gfl_ptr(vmap.get(v, v))
                         for v in FindVariables().visit(body) if 'ydvars%gfl_ptr' in v.name.lower()})
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
            return

        bounds = SCCBaseTransformation.get_horizontal_loop_bounds(routine, self.horizontal)
        SCCBaseTransformation.resolve_vector_dimension(routine, loop_variable=v_index, bounds=bounds)

        # for kernels we process the entire body
        routine.body = self.process_body(routine.body, routine.symbol_map, item.trafo_data[self._key]['definitions'],
                                         successors, targets, exclude_arrays)


class BlockIndexInjectTransformation(Transformation):
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

    _key = 'BlockIndexInjectTransformation'
    """Default identifier for trafo_data entry"""

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, block_dim, key=None):
        self.block_dim = block_dim
        if key:
            self._key = key

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
    def _update_expr_map(var, rank, index):
        """
        Return a map with the block-index appended to the variable's dimensions.
        """

        if getattr(var, 'dimensions', None):
            return {var: var.clone(dimensions=var.dimensions + as_tuple(index))}
        return {var:
                var.clone(dimensions=((RangeIndex(children=(None, None)),) * (rank - 1)) + as_tuple(index))}

    @staticmethod
    def get_call_arg_rank(arg):
        """
        Utility to retrieve the local rank of a :any:`CallSatement` argument.
        """

        rank = len(arg.shape) if getattr(arg, 'shape', None) else 0
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
        call_args = [a for call in FindNodes(ir.CallStatement).visit(body) for a in call.arguments]

        # First get rank mismatched call statement args
        vmap = {}
        for call in [call for call in FindNodes(ir.CallStatement).visit(body) if call.name in targets]:
            for dummy, arg in call.arg_map.items():
                arg_rank = self.get_call_arg_rank(arg)
                dummy_rank = len(dummy.shape) if getattr(dummy, 'shape', None) else 0
                if arg_rank - 1 == dummy_rank:
                    vmap.update(self._update_expr_map(arg, arg_rank, block_index))

        # Now get the rest of the variables
        for var in [var for var in FindVariables().visit(body)
                    if getattr(var, 'dimensions', None) and not var in call_args]:

            local_rank = len(var.dimensions)
            decl_rank = local_rank
            # we assume here that all derived-type components we wish to transform
            # have been parsed
            if getattr(var, 'shape', None):
                decl_rank = len(var.shape)

            if local_rank == decl_rank - 1:
                vmap.update(self._update_expr_map(var, decl_rank, block_index))

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

## END: copied from PR: naan-block-index-inject

class BlockIndexLowerTransformation(Transformation):

    _key = 'BlockIndexLowerTransformation'
    """Default identifier for trafo_data entry"""

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, block_dim, recurse_to_kernels=False, key=None):
        self.block_dim = block_dim
        self.recurse_to_kernels = recurse_to_kernels
        if key:
            self._key = key
        print("BlockIndexLowerTransformation ... ")

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))

        # TODO: exclude_arrays?!
        # exclude_arrays = []
        # if (item := kwargs.get('item', None)):
        #     exclude_arrays = item.config.get('exclude_arrays', [])

        # if role == 'kernel':
        #     self.process_kernel(routine, targets, exclude_arrays)
        if role == 'driver' or self.recurse_to_kernels and role == 'kernel':
            self.process_driver(routine, targets)

        #TODO: we also need to process any code inside a loki/acdc parallel pragma at the driver layer


    def process_driver(self, routine, targets):
        # print(f"process driver '{routine.name}'")
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            # print(f"call.name '{call.name}' vs. {targets}")
            if str(call.name).lower() in targets:
                arg_iter = call.arg_iter()
                if self.block_dim.index not in [arg[1] for arg in arg_iter]:
                    new_kwarg = (routine.variable_map[self.block_dim.index].name, routine.variable_map[self.block_dim.index])
                    call._update(kwarguments=call.kwarguments+(new_kwarg,))
                    call.routine.arguments += (routine.variable_map[self.block_dim.index].clone(scope=call.routine).clone(type=routine.variable_map[self.block_dim.index].type.clone(intent='in')),)
                if self.block_dim.size not in [arg[1] for arg in arg_iter]:
                    new_kwarg = (routine.variable_map[self.block_dim.size].name, routine.variable_map[self.block_dim.size])
                    call._update(kwarguments=call.kwarguments+(new_kwarg,))
                    call.routine.arguments += (routine.variable_map[self.block_dim.size].clone(scope=call.routine).clone(type=routine.variable_map[self.block_dim.size].type.clone(intent='in')),)

                var_map = {}
                for arg, call_arg in call.arg_iter(): # arg_iter: # call.arg_iter():
                    print(f"call: {call.name} | arg: {arg} | call_arg: {call_arg}")
                    if isinstance(arg, Array) and len(call_arg.shape) > len(arg.shape):
                        print(f"updating arg {arg}")
                        call_routine_var = call.routine.variable_map[arg.name] # [call_arg.name]
                        new_dims = call_routine_var.dimensions + (call.routine.variable_map[self.block_dim.size],)
                        new_shape = call_routine_var.shape + (call.routine.variable_map[self.block_dim.size],)
                        new_type = call_routine_var.type.clone(shape=new_shape)
                        var_map[call_routine_var] = call_routine_var.clone(dimensions=new_dims, type=new_type)
                        # var_map[call_routine_var] = call_routine_var.clone(type=new_type)
                        print(f"updating '{call_routine_var.name}' for routine 'call.routine.name' with {new_dims}, {new_shape}")
                call.routine.spec = SubstituteExpressions(var_map).visit(call.routine.spec)
                # call.routine.body = SubstituteExpressions(var_map).visit(call.routine.body)

class BlockLoopLowerTransformation(Transformation):

    _key = 'BlockIndexLowerTransformation'
    """Default identifier for trafo_data entry"""

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, block_dim, recurse_to_kernels=False, key=None):
        self.block_dim = block_dim
        self.recurse_to_kernels = recurse_to_kernels
        if key:
            self._key = key
        print("BlockLoopLowerTransformation ... ")

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))
        
        if role == 'driver':
            self.process_driver(routine, targets)

    @staticmethod
    def arg_to_local_var(routine, var):
        new_args = tuple(arg for arg in routine.arguments if arg.name.lower() != var.name.lower())
        routine.arguments = new_args
        routine.variables += (routine.variable_map[var.name].clone(scope=routine, type=routine.variable_map[var.name].type.clone(intent=None)),)

    def process_driver(self, routine, targets):
        # print(f"process driver '{routine.name}'")
        # for call in FindNodes(ir.CallStatement).visit(routine.body):
        #     ...

        loops = FindNodes(ir.Loop).visit(routine.body)
        print(f"loops in routine {routine.name}: {loops}")
        loops = [loop for loop in loops if loop.variable == self.block_dim.index or loop.variable in self.block_dim._index_aliases]
        print(f"loops with block_dim ... in routine {routine.name}: {loops}")
        loop_map = {}
        ignore_routine = []
        for loop in loops:
            start = loop.bounds.children[0]
            end = loop.bounds.children[1]
            step = loop.bounds.children[2] if len(loop.bounds.children) > 2 else sym.IntLiteral(1)
            calls = [call for call in FindNodes(ir.CallStatement).visit(routine.body) if str(call.name).lower() in targets]
            for call in calls:
                ##
                updated_args = ()
                updated_kwargs = ()
                for arg in call.arguments: # updated_args:
                    if isinstance(arg, Array) and any(dim == self.block_dim.index for dim in arg.dimensions):
                        updated_args += (arg.clone(dimensions=tuple(dim if dim != self.block_dim.index else sym.RangeIndex((None, None)) for dim in arg.dimensions)),)
                    else:
                        updated_args += (arg,)
                print(f"call.kwarguments: {call.kwarguments}")
                for name, kwarg in call.kwarguments:
                    if isinstance(kwarg, Array) and any(dim == self.block_dim.index for dim in kwarg.dimensions):
                        updated_kwargs += ((name, kwarg.clone(dimensions=tuple(dim if dim != self.block_dim.index else sym.RangeIndex((None, None)) for dim in kwarg.dimensions))),)
                    else:
                        updated_kwargs += ((name, kwarg),)
                # call._update(arguments=updated_args, kwarguments=updated_kwargs)
                call_pragmas = (ir.Pragma(keyword="loki", content=f"removed_loop var({loop.variable}) \
                        lower({loop.bounds.lower}) upper({loop.bounds.upper}) \
                        step({loop.bounds.step if loop.bounds.step else 1})"),)
                call._update(arguments=updated_args, kwarguments=updated_kwargs, pragma=(call.pragma if call.pragma else ()) + call_pragmas)
                ##
                # call._update(kwarguments=call.kwarguments+((f"{loop.variable.name}_start", start), (f"{loop.variable.name}_end", end), 
                #     (f"{loop.variable.name}_step", step)))
                if call.routine.name.lower() in ignore_routine:
                    continue
                # add loop variables ...
                # arg_type = call.routine.variable_map[self.block_dim.size].type.clone()
                # start_arg = start.clone(name=f"{loop.variable.name}_start", scope=call.routine) if isinstance(start, sym.Variable) else sym.Variable(name=f"{loop.variable.name}_start", type=arg_type)
                # end_arg = end.clone(name=f"{loop.variable.name}_end", scope=call.routine) if isinstance(end, sym.Variable) else sym.Variable(name=f"{loop.variable.name}_end", type=arg_type)
                # step_arg = end.clone(name=f"{loop.variable.name}_step", scope=call.routine) if isinstance(end, sym.Variable) else sym.Variable(name=f"{loop.variable.name}_step", type=arg_type)
                # call.routine.arguments += (start_arg, end_arg, step_arg)
                ignore_routine.append(call.routine.name.lower())
                print(f"call: '{call}' within loop: {loop}")
                arg_iter = list(call.arg_iter())
                # remove loop.variable
                if loop.variable in [arg[1] for arg in arg_iter]: # TODO: for aliases ...
                    # remove from callee arguments
                    self.arg_to_local_var(call.routine, call.arg_map[loop.variable.name])
                    # remove from callstatement
                    # if loop.variable in call.arguments:
                    call._update(arguments=tuple(arg for arg in call.arguments if arg.name.lower() != loop.variable.name.lower()),
                            kwarguments=tuple(kwarg for kwarg in call.kwarguments if kwarg[0].lower() != loop.variable.name.lower()))
                else:
                    if loop.variable not in call.routine.variables:
                        call.routine.variables += (loop.variable.clone(scope=call.routine),)
                additional_assignments = ()
                if self.block_dim.index != loop.variable:
                    # additional arguments and assignment ...
                    assignments = FindNodes(ir.Assignment).visit(loop.body)
                    additional_arguments = ()
                    for assignment in assignments:
                        if assignment.lhs == self.block_dim.index:
                            # TODO: ....
                            # additional_assignments += (assignment.clone(),)
                            for var in FindVariables().visit(assignment.rhs):
                                ignore = [loop.variable, self.block_dim.index, self.block_dim.size]
                                ignore.extend([arg[1] for arg in arg_iter])
                                if var not in ignore: # [loop.variable, self.block_dim.index, self.block_dim.size].extend([arg[1] for arg in arg_iter]): # TODO: sufficient?
                                    additional_arguments += (var,)
                    # remove duplicates in additional_arguments ...
                    additional_arguments = sorted(list(set(additional_arguments)), key=lambda symbol: symbol.name)
                    call.routine.arguments += tuple(arg.clone(scope=call.routine, type=arg.type.clone(intent='in')) for arg in additional_arguments)
                    call._update(kwarguments=call.kwarguments+tuple((arg.name, arg.clone()) for arg in additional_arguments))
                    # end: additional arguments and assignment ...
                    # remove block_dim.index
                    if self.block_dim.index in [arg[1] for arg in arg_iter]: # TODO: for aliases ...
                        print(f"block_dim.index {self.block_dim.index} in arguments ...")
                        # remove from callee arguments
                        print(f"call.arg_map: {call.arg_map}")
                        # TODO: why do I need to comment that?
                        """
                        self.arg_to_local_var(call.routine, call.arg_map[self.block_dim.index.lower()])
                        """
                        # remove from callstatement
                        # if loop.variable in call.arguments:
                        
                        # TODO: why do I need to comment that?
                        """
                        call._update(arguments=tuple(arg for arg in call.arguments if not isinstance(arg, sym.Variable) or arg.name.lower() != self.block_dim.index.lower()),
                                kwarguments=tuple(kwarg for kwarg in call.kwarguments if not isinstance(kwarg[0], sym.Variable) or kwarg[0].lower() != self.block_dim.index.lower()))
                        """

                # call.routine.body = (ir.Comment('! loki inserted loop ...'), loop.clone(bounds=sym.LoopRange((call.routine.variable_map[f"{loop.variable.name}_start"],
                #         call.routine.variable_map[f"{loop.variable.name}_end"],
                #         call.routine.variable_map[f"{loop.variable.name}_step"])), body=additional_assignments + (call.routine.body,)), ir.Comment('! end: loki inserted loop ...'))
                call.routine.body = (ir.Comment('! loki inserted loop ...'), loop.clone(body=additional_assignments + (call.routine.body,)), ir.Comment('! end: loki inserted loop ...'))
            # loop_map[loop] = (ir.Comment('! start removed loop ...'), loop.body, ir.Comment('! end: removed loop ...'))
            loop_map[loop] = (ir.Comment('! start removed loop ...'), loop, ir.Comment('! end: removed loop ...'))
        routine.body = Transformer(loop_map).visit(routine.body)


