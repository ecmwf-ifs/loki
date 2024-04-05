# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import (
    Transformation, ProcedureItem, ir, Module, as_tuple, fgen, SymbolAttributes, BasicType, Variable,
    RangeIndex, Array, FindVariables, resolve_associates, SubstituteExpressions, FindNodes, resolve_type_bound_var
)

from transformations.single_column_coalesced import SCCBaseTransformation

__all__ = ['UnprivatiseStructsTransformation', 'BlockIndexInjectTransformation']

def get_parent_typedef(var, routine):

    if not var.parent.type.dtype.typedef == BasicType.DEFERRED:
        return var.parent.type.dtype.typedef
    elif not routine.symbol_map[var.parent.type.dtype.name].type.dtype.typedef == BasicType.DEFERRED:
        return routine.symbol_map[var.parent.type.dtype.name].type.dtype.typedef
    else:
        raise RuntimeError(f'Container data-type {var.parent.type.dtype.name} not enriched')

class UnprivatiseStructsTransformation(Transformation):


    _key = 'UnprivatiseStructsTransformation'

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, horizontal, key=None):
        self.horizontal = horizontal
        if key:
             self._key = key

    def transform_subroutine(self, routine, **kwargs):

        if not (item := kwargs['item']):
            raise RuntimeError('Cannot apply DeprivatiseStructsTransformation without item to store definitions')
        successors = kwargs.get('successors', ())

        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))

        if role == 'kernel':
            self.process_kernel(routine, item, successors, targets)
        if role == 'driver':
           self.process_driver(routine, successors)

    @staticmethod
    def _get_parkind_suffix(type):
        return type.rsplit('_')[1][1:3]

    def _build_parkind_import(self, field_array_module, wrapper_types):

        deferred_type = SymbolAttributes(BasicType.DEFERRED, imported=True)
        vars = {Variable(name='JP' + self._get_parkind_suffix(type), type=deferred_type, scope=field_array_module)
                for type in wrapper_types}

        return ir.Import(module='PARKIND1', symbols=as_tuple(vars))

    def _build_field_array_types(self, field_array_module, wrapper_types):

        typedefs = ()
        for type in wrapper_types:
            suff = self._get_parkind_suffix(type)
            kind = field_array_module.symbol_map['JP' + suff]
            rank = int(type.rsplit('_')[1][0])

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
            contig_pointer_var = pointer_var.clone(name='P_FIELD', type=contig_pointer_type, dimensions=array_shape)

            decls = (ir.VariableDeclaration(symbols=(pointer_var,)),)
            decls += (ir.VariableDeclaration(symbols=(contig_pointer_var,)),)

            typedefs += (ir.TypeDef(name=type, body=decls, parent=field_array_module),)

        return typedefs

    def _create_dummy_field_api_defs(self, field_array_mod_imports):

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

    def process_kernel(self, routine, item, successors, targets):

        # Sanitize the subroutine
        resolve_associates(routine)
        v_index = SCCBaseTransformation.get_integer_variable(routine, name=self.horizontal.index)
        SCCBaseTransformation.resolve_masked_stmts(routine, loop_variable=v_index)

        if self.horizontal.bounds[0] in routine.variables and self.horizontal.bounds[1] in routine.variables:
            _bounds = self.horizontal.bounds
        else:
            _bounds = self.horizontal._bounds_aliases
        SCCBaseTransformation.resolve_vector_dimension(routine, loop_variable=v_index, bounds=_bounds)

        # build list of type-bound array access using the horizontal index
        vars = [var for var in FindVariables().visit(routine.body)
                if isinstance(var, Array) and var.parents and self.horizontal.index in getattr(var, 'dimensions', ())]

        # remove YDCPG_SL1 members, as these are not memory blocked
        vars = [var for var in vars if not 'ydcpg_sl1' in var]

        # build list of type-bound view pointers passed as subroutine arguments
        for call in [call for call in FindNodes(ir.CallStatement).visit(routine.body) if call.name in targets]:
            _args = {a: d for d, a in call.arg_map.items() if isinstance(d, Array)}
            _args = {a: d for a, d in _args.items()
                     if any([v in d.shape for v in self.horizontal.size_expressions]) and a.parents}
            vars += list(_args)

        # check if array pointers are defined
        for var in vars:
            typedef = get_parent_typedef(var, routine)
            name = var.name_parts[-1] + '_FIELD'
            if not name in typedef.variable_map:
                raise RuntimeError(f'Container data-type {typedef.name} does not contain *_FIELD pointer')

        # replace view pointers with array pointers
        vmap = {var: var.clone(name='%'.join([v for v in var.name_parts[:-1]]) + '%' + var.name_parts[-1] + '_FIELD')
                for var in vars}
        routine.body = SubstituteExpressions(vmap).visit(routine.body)

        # propagate dummy field_api wrapper definitions to children
        definitions = item.trafo_data[self._key]['definitions']
        self.propagate_defs_to_children(self._key, definitions, successors)


class BlockIndexInjectTransformation(Transformation):

    _key = 'BlockIndexInjectTransformation'

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, horizontal, block_dim, key=None):
        self.horizontal = horizontal
        self.block_dim = block_dim
        if key:
             self._key = key

    def transform_subroutine(self, routine, **kwargs):

        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))

        if role == 'kernel':
            self.process_kernel(routine, targets)

    @staticmethod
    def get_derived_type_member_rank(a, routine):
        typedef = get_parent_typedef(a, routine)
        return len(typedef.variable_map[a.name_parts[-1]].shape)

    @staticmethod
    def _update_expr_map(var, rank, index):
        if getattr(var, 'dimensions', None):
            return {var: var.clone(dimensions=var.dimensions + as_tuple(index))}
        else:
            return {var:
                    var.clone(dimensions=((RangeIndex(children=(None, None)),) * (rank - 1)) + as_tuple(index))}

    def process_kernel(self, routine, targets):

        # Check that the block index is defined
        if self.block_dim.index in routine.variables:
            block_index = routine.variable_map[self.block_dim.index]
        elif any(i.rsplit('%')[0] in routine.variables for i in self.block_dim._index_aliases):
            index_name = [alias for alias in self.block_dim._index_aliases
                          if alias.rsplit('%')[0] in routine.variables][0]

            child, parent = resolve_type_bound_var(index_name)
            block_index = Variable(name=child, parent=parent, scope=routine)
        else:
            # we skip routines that do not contain the block index
            return

        # The logic for callstatement args differs from other array instances in the body,
        # so we build a list to filter
        call_args = [a for call in FindNodes(ir.CallStatement).visit(routine.body) for a in call.arguments]

        # First get rank mismatched call statement args
        vmap = {}
        for call in [call for call in FindNodes(ir.CallStatement).visit(routine.body) if call.name in targets]:
            _args = {a: d for d, a in call.arg_map.items() if isinstance(d, Array)
                     if any([v in getattr(d, 'shape', None) for v in self.horizontal.size_expressions])}

            for arg, dummy in _args.items():
                if arg.parents:
                    rank = self.get_derived_type_member_rank(arg, routine)
                else:
                    rank = len(arg.shape)

                if rank - 1 == len(dummy.shape):
                    vmap.update(self._update_expr_map(arg, rank, block_index))

        # Now get the rest of the horizontal arrays
        for var in [var for var in FindVariables().visit(routine.body) if isinstance(var, Array)
                    and self.horizontal.index in getattr(var, 'dimensions', ()) and not var in call_args]:

            local_rank = len(var.dimensions)
            if var.parents:
                decl_rank = self.get_derived_type_member_rank(var, routine)
            else:
                decl_rank = len(var.shape)

            if local_rank == decl_rank - 1:
                vmap.update(self._update_expr_map(var, decl_rank, block_index))

        routine.body = SubstituteExpressions(vmap).visit(routine.body)