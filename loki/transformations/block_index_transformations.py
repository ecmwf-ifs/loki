# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import (
    Transformation, ProcedureItem, ir, Module, as_tuple, SymbolAttributes, BasicType, Variable,
    RangeIndex, Array, FindVariables, resolve_associates, SubstituteExpressions, FindNodes,
    recursive_expression_map_update
)

from transformations.single_column_coalesced import SCCBaseTransformation

__all__ = ['BlockViewToFieldViewTransformation', 'BlockIndexInjectTransformation']

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

        if not var.parent.type.dtype.typedef == BasicType.DEFERRED:
            return var.parent.type.dtype.typedef
        if not symbol_map[var.parent.type.dtype.name].type.dtype.typedef == BasicType.DEFERRED:
            return symbol_map[var.parent.type.dtype.name].type.dtype.typedef
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
            _args = {a: d for a, d in _args.items()
                     if any(v in d.shape for v in self.horizontal.size_expressions) and a.parents}
            _vars += list(_args)

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
        if any(i.rsplit('%')[0] in variable_map for i in self.block_dim._index_aliases):
            index_name = [alias for alias in self.block_dim._index_aliases
                          if alias.rsplit('%')[0] in variable_map][0]

            block_index = routine.resolve_typebound_var(index_name, variable_map)

        return block_index

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
