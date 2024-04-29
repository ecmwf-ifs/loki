# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki.ir import (
    is_loki_pragma, get_pragma_parameters, pragma_regions_attached,
    FindNodes, nodes as ir
)
from loki.transform import Transformation
from loki import (
    FindVariables, DerivedType, SymbolAttributes,
    Array, single_variable_declaration, Transformer,
    BasicType, as_tuple
)
import pickle
import os
from itertools import chain

__all__ = ['ParallelRoutineDispatchTransformation']


class ParallelRoutineDispatchTransformation(Transformation):

    def __init__(self):
        self.is_intent = False #set to True if the intent are read for interface block
        self.horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
        self.map_compute = {
            "OpenMP" : self.create_compute_openmp, 
            "OpenMPSingleColumn" : self.create_compute_openmpscc,
            "OpenACCSingleColumn" : self.create_compute_openaccscc
                       }

        #TODO : do smthg for opening field_index.pkl
        with open(os.getcwd()+"/transformations/transformations/field_index.pkl", 'rb') as fp:
            self.map_index = pickle.load(fp)
        # CALL FIELD_NEW (YL_ZA, UBOUNDS=[KLON, KFLEVG, KGPBLKS], LBOUNDS=[1, 0, 1], PERSISTENT=.TRUE.)
        self.new_calls = []
        # IF (ASSOCIATED (YL_ZA)) CALL FIELD_DELETE (YL_ZA)
        self.delete_calls = []
        # map[name] = [field_ptr, ptr]
        # where : 
        # field_ptr : pointer on field api object
        # ptr : pointer to the data
        self.routine_map_temp = {}  
        self.routine_map_derived = {} 

    def transform_subroutine(self, routine, **kwargs):
        self.get_cpg(routine)
        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if is_loki_pragma(region.pragma):
                    self.process_parallel_region(routine, region)
        single_variable_declaration(routine)
        self.add_temp(routine)
        self.add_field(routine)
        self.add_derived(routine)
        #call add_arrays etc...


    def process_parallel_region(self, routine, region):
        pragma_content = region.pragma.content.split(maxsplit=1)
        pragma_content = [entry.split('=', maxsplit=1) for entry in pragma_content[1].split(',')]
        pragma_attrs = {
            entry[0].lower(): entry[1] if len(entry) == 2 else None
            for entry in pragma_content
        }
        if 'parallel' not in pragma_attrs:
            return
        pragma_attrs['target'] = pragma_attrs['target'].split('/')
        region_name = pragma_attrs['name']
        dr_hook_calls = self.create_dr_hook_calls(
            routine, f"{routine.name}:{region_name}",
            sym.Variable(name='ZHOOK_HANDLE_FIELD_API', scope=routine)
        )

        region.prepend(dr_hook_calls[0])
        region.append(dr_hook_calls[1])

        region_map_temp= self.decl_local_array(routine, region)
        region_map_derived= self.decl_derived_types(routine, region)

        self.get_data = {}
        self.compute = {}
###        self.synchost = {} #synchost same for all the targets
###        self.nullify  = {} #synchost same for all the targets

        self.synchost = self.create_synchost(routine, region_name, region_map_derived, region_map_temp)
        self.nullify = self.create_nullify(routine, region_name, region_map_derived, region_map_temp)

        for target in pragma_attrs['target']:
# Q : I would like get_data, synchost and nullify not be members of the Transformation object, however, I need them to run the test... 
# A : maybe have them as members of the routine while
# Is there an object to handle data that is needed for tests ? 
#        get_data = self.create_pt_sync(routine, region_name, True, region_map_derived, region_map_temp)
#        synchost = self.create_synchost(routine, region_name, True, region_map_derived, region_map_temp)
#        nullify = self.create_nullify(routine, region_name, True, region_map_derived, region_map_temp)
            self.process_target(routine, region, region_name, region_map_temp, region_map_derived, target)
        for var_name in region_map_temp:
            if var_name not in self.routine_map_temp:
                self.routine_map_temp[var_name]=region_map_temp[var_name]

        for var_name in region_map_derived:
            if var_name not in self.routine_map_derived:
                self.routine_map_derived[var_name]=region_map_derived[var_name]

    def process_target(self, routine, region, region_name, region_map_temp, region_map_derived, target):
        self.get_data[target] = self.create_pt_sync(routine, target, region_name, True, region_map_derived, region_map_temp)
        self.compute[target] = self.map_compute[target](routine, region, region_name, region_map_temp, region_map_derived)

    @staticmethod
    def create_dr_hook_calls(scope, cdname, handle):
        dr_hook_calls = []
        for kswitch in (0, 1):
            call_stmt = ir.CallStatement(
                name=sym.Variable(name='DR_HOOK', scope=scope),
                arguments=(sym.StringLiteral(cdname), sym.IntLiteral(kswitch), handle)
            )
            dr_hook_calls += [
                ir.Conditional(
                    condition=sym.Variable(name='LHOOK', scope=scope),
                    inline=True, body=(call_stmt,)
                )
            ]
        return dr_hook_calls


    def create_field_new_delete(self, routine, var, field_ptr_var):
        # Create the FIELD_NEW call
        var_shape = var.shape
        ubounds = [d.upper if isinstance(d, sym.RangeIndex) else d for d in var_shape]
        ubounds += [sym.Variable(name="KGPBLKS", parent=routine.variable_map["YDCPG_OPTS"])]
        # NB: This is presumably what the condition should look like, however, the
        #     logic is flawed in to_parallel and it will only insert lbounds if _the last_
        #     dimension has an lbound. We emulate this with the second line here to
        #     generate identical results, but this line should probably not be there
        has_lbounds = any(isinstance(d, sym.RangeIndex) for d in var_shape)
        has_lbounds = has_lbounds and isinstance(var_shape[-1], sym.RangeIndex)
        if has_lbounds:
            lbounds = [
                d.lower if isinstance(d, sym.RangeIndex) else sym.IntLiteral(0)
                for d in var_shape
            ]
            kwarguments = (
                ('UBOUNDS', sym.LiteralList(ubounds)),
                ('LBOUNDS', sym.LiteralList(lbounds)),
                ('PERSISTENT', sym.LogicLiteral(True))
            )
        else:
            kwarguments = (
                ('UBOUNDS', sym.LiteralList(ubounds)),
                ('PERSISTENT', sym.LogicLiteral(True))
            )
        self.new_calls += [ir.CallStatement(
            name=sym.Variable(name='FIELD_NEW', scope=routine),
            arguments=(field_ptr_var,),
            kwarguments=kwarguments
        )]

        # Create the FIELD_DELETE CALL
        call = ir.CallStatement(sym.Variable(name='FIELD_DELETE', scope=routine), arguments=(field_ptr_var,))
        condition = sym.InlineCall(sym.Variable(name='ASSOCIATED'), parameters=(field_ptr_var,))
        self.delete_calls += [ir.Conditional(condition=condition, inline=True, body=(call,))]

    def decl_local_array(self, routine, region):
        temp_arrays = [var for var in FindVariables(Array).visit(region)  if isinstance(var, Array) and not var.name_parts[0] in routine.arguments and var.shape[0] in self.horizontal]
        region_map_temp={}        
        #check if first dim NPROMA ?
        for var in temp_arrays:
            var_type = var.type
            var_shape = var.shape

            dim = len(var_shape) + 1 # Temporary dimensions + block

            # The FIELD_{d}RB variable
            field_ptr_type = SymbolAttributes(
                dtype=DerivedType(f'FIELD_{dim}RB'),
                pointer=True, polymorphic=True, initial="NULL()"
            )
            field_ptr_var = sym.Variable(name=f'YL_{var.name}', type=field_ptr_type, scope=routine)

            # Create a pointer instead of the array
            shape = (sym.RangeIndex((None, None)),) * dim
          #  var.type = var_type.clone(pointer=True, shape=shape)
            local_ptr_var = var.clone(dimensions=shape)

            region_map_temp[var.name]=[field_ptr_var,local_ptr_var]
            self.create_field_new_delete(routine, var, field_ptr_var)
        return(region_map_temp)

    def add_temp(self, routine):
        # Replace temporary declaration by pointer to array and pointer to field_api object
        map_dcl = {}
        for decl in routine.declarations:
            if len(decl.symbols) == 1:
                var = decl.symbols[0]
                if var.name in self.routine_map_temp:
                    new_vars = self.routine_map_temp[var.name]
                    new_vars[1].type = new_vars[1].type.clone(pointer=True, shape=new_vars[1].dimensions, initial="NULL()")
                    map_dcl.update({decl : (ir.VariableDeclaration(symbols=(new_vars[0],)), ir.VariableDeclaration(symbols=(new_vars[1],)))})
            else:
                raise Exception("Declaration should have only one symbol, please run single_variable_declaration before calling add_temp function.")
        routine.spec = Transformer(map_dcl).visit(routine.spec)
        #decls_to_replace = [var for var in decl.var for decl in routine.declarations if var in self.routine_map_temp]
    
    def add_field(self, routine):
        # Insert the field generation wrapped into a DR_HOOK call
        dr_hook_calls = self.create_dr_hook_calls(
            routine, cdname='CREATE_TEMPORARIES',
            handle=sym.Variable(name='ZHOOK_HANDLE_FIELD_API', scope=routine)
        )
        routine.body.insert(2, (dr_hook_calls[0], ir.Comment(text=''), *self.new_calls, dr_hook_calls[1]))

        # Insert the field deletion wrapped into a DR_HOOK call
        dr_hook_calls = self.create_dr_hook_calls(
            routine, cdname='DELETE_TEMPORARIES',
            handle=sym.Variable(name='ZHOOK_HANDLE_FIELD_API', scope=routine)
        )
        routine.body.insert(-2,(dr_hook_calls[0], ir.Comment(text=''), *self.delete_calls, dr_hook_calls[1]))

    def decl_derived_types(self, routine, region):
        region_map_derived = {}
        derived = [var for var in FindVariables().visit(region) if var.name_parts[0] in routine.arguments]
        for var in derived :
            
            key = f"{routine.variable_map[var.name_parts[0]].type.dtype.name}%{'%'.join(var.name_parts[1:])}"
            #TODO : maybe have a global derive typed table, to avoid to many lookup in the map_index???
            if key in self.map_index:
                value = self.map_index[key]
                # Creating the pointer on the data : YL_A
                data_name = f"Z_{var.name.replace('%', '_')}"
                if "REAL" and "JPRB" in value[0]:
                    data_dim = value[2] + 1
                    data_shape = (sym.RangeIndex((None, None)),) * data_dim
                    data_type = SymbolAttributes(
                        dtype=BasicType.REAL, kind=routine.symbol_map['JPRB'],
                        pointer=True, shape=data_shape
                    )
                    ptr_var = sym.Variable(name=data_name, type=data_type, dimensions=data_shape, scope=routine)

                else:
                    raise NotImplementedError("This type isn't implemented yet")

                # Creating the pointer on the field api object : YL%FA, YL%F_A...
                if routine.variable_map[var.name_parts[0]].type.dtype.name=="MF_PHYS_SURF_TYPE":
                    # YL%PA becomes YL%F_A
                    field_name = f"{'%'.join(var.name_parts[:-1])}%F_{var.name_parts[-1][1:]}"
                elif routine.variable_map[var.name_parts[0]].type.dtype.name=="FIELD_VARIABLES":
                    # YL%A becomes YL%FA
                    field_name = f"{'%'.join(var.name_parts[:-1])}%F{var.name_parts[-1]}"
                    if var.name_parts[-1]=="P": #YL%FP = YL%FT0
                        field_name = f"{field_name[-1]}T0"
                else:
                    # YL%A becomes YL%F_A
                    field_name = f"{'%'.join(var.name_parts[:-1])}%F_{var.name_parts[-1]}"
                field_ptr_var = var.clone(name=field_name)
                region_map_derived[var.name] = [field_ptr_var, ptr_var]
        return(region_map_derived)

    def add_derived(self, routine):
        routine.variables += tuple(v[1] for v in self.routine_map_derived.values())
        
    def create_pt_sync(self, routine, target, region_name, is_get_data, region_map_derived, region_map_temp):
        if is_get_data: #GET_***_DATA
            hook_name = "GET_DATA"
            if target == "OpenMP" or target == "OpenMPSingleColumn" : 
                sync_name = "GET_HOST_DATA"
            elif target == "OpenACCSingleColumn":
                sync_name = "GET_DEVICE_DATA"
            else : 
                raise Exception(f"{target} : this target isn't known!")
        else: #SYNCHOST
            hook_name = "SYNCHOST"
            sync_name = "GET_HOST_DATA"

        dr_hook_calls = self.create_dr_hook_calls(
            routine, cdname=f"{routine.name}:{region_name}:{hook_name}",
            handle=sym.Variable(name='ZHOOK_HANDLE_FIELD_API', scope=routine)
        )
       
        sync_data = [dr_hook_calls[0]]

        for var in chain(region_map_temp.values(), region_map_derived.values()):
            if is_get_data:
                if not self.is_intent : 
                    intent = "RDWR"
                else:
                    raise NotImplementedError("Reading the intent from interface isn't implemented yes")
            else:
                    intent = "RDWR"  # for SYNCHOST, always RDWR

            call = sym.InlineCall(sym.Variable(name=f"{sync_name}_{intent}"), parameters=(var[0],))
            sync_data += [ir.Assignment(lhs=var[1].clone(dimensions=None), rhs=call, ptr=True)]

        sync_data.append(dr_hook_calls[1])
    
        return(sync_data)

    def create_synchost(self, routine, region_name, region_map_derived, region_map_temp):
        synchost = self.create_pt_sync(routine, None, region_name, False, region_map_derived, region_map_temp)
        condition = sym.InlineCall(sym.Variable(name='LSYNCHOST'), parameters=(sym.StringLiteral(value=f"{routine.name}:{region_name}"),))
        return [ir.Conditional(condition=condition, body=tuple(synchost))]

    def create_nullify(self, routine, region_name, region_map_derived, region_map_temp):
        dr_hook_calls = self.create_dr_hook_calls(
            routine, cdname=f"{routine.name}:{region_name}:NULLIFY",
            handle=sym.Variable(name='ZHOOK_HANDLE_FIELD_API', scope=routine)
        )
        nullify= [dr_hook_calls[0]]
        for var in chain(region_map_temp.values(), region_map_derived.values()):
            nullify += [ir.Assignment(lhs=var[1].clone(dimensions=None), rhs=sym.InlineCall(sym.Variable(name='NULL')),ptr=True)]
        nullify.append(dr_hook_calls[1])
        return nullify

    def get_cpg(self,routine):
        #Assuming CPG_OPTS_TYPE and CPG_BNDS_TYPE are the same in all the routine.
        found_opts = False
        found_bnds = False
        for var in FindVariables().visit(routine.spec):
            if var.type.dtype.name=="CPG_OPTS_TYPE":
                self.cpg_opts = var 
                found_opts = True
            if var.type.dtype.name=="CPG_BNDS_TYPE":
                self.cpg_bnds = var
                found_bnds = True
            if (found_opts and found_bnds) :
                if "YD" in self.cpg_bnds.name:
                    lcpg_bnds_name = self.cpg_bnds.name.replace("YD", "YL")
                    self.lcpg_bnds = sym.Variable(name=lcpg_bnds_name, scope=routine)
                    dcl = ir.VariableDeclaration(symbols=as_tuple(self.lcpg_bnds))
                    routine.spec.append(dcl)
                    data_type = SymbolAttributes(
                        dtype=BasicType.INTEGER, kind=routine.symbol_map['JPIM']
                    )
                    self.jblk = sym.Variable(name="JBLK", type=data_type, scope=routine)
                    routine.spec.append(self.jblk)
                    return
                else:
                    raise Exception(f"cpg_bnds unexpected name : {self.cpg_bnds.name}")

    def update_args(self, arg, region_map):
        new_arg = region_map[arg.name][1]
        dim = len(new_arg.dimensions)
        #dim = len(new_arg.shape)
        new_dimensions = (sym.RangeIndex((None, None)),) * (dim-1)
        new_dimensions += (self.jblk,)
        return new_arg.clone(dimensions=new_dimensions)

    def create_compute_openmp(self, routine, region, region_name, region_map_temp, region_map_derived):
    #ylcpg_bnds : new var to add to spec, type(ylcpg)=type(cpg_bnds)=CPG_BNDS_TYPE

    #hook_compute 0
    #call ylcpg_bnds%init(ydcpg_opts)
    #!$omp parallel do private (jblk) firstprivate (ylcpg_bnds)
    #do jblk = 1, ydcpg_opts%kgpblks
    #   call ylcpg_bnds%update(jblk)
    #   call callee(ydgeometry, ydmodel, ylcpg_bnds%kidia, ... (...BLK))
    #enddo
    #hook_compute 1

        init = ir.CallStatement(
            name=routine.resolve_typebound_var(f"{self.lcpg_bnds.name}%INIT"),
            arguments=(self.cpg_opts,))
        #TODO : generate lst_private !!!!
        lst_private = "JBLK"
        pragma = ir.Pragma(keyword="OMP", content=f"PARALLEL DO PRIVATE {lst_private} FIRSTPRIVATE ({self.lcpg_bnds})")
        update = ir.CallStatement(  
            name=routine.resolve_typebound_var(f"{self.lcpg_bnds.name}%UPDATE"), 
            arguments=(self.jblk,)
        )
        #TODO : musn't be call but the body of the region here?? 

        new_calls = []
        for call in FindNodes(ir.CallStatement).visit(region):
            if call.name!="DR_HOOK":
               # for var in chain(region_map_temp.values(), region_map_derived.values()):
                new_arguments = []
                for arg in call.arguments:
                    if arg.name in region_map_temp:
                        new_arguments +=[self.update_args(arg, region_map_temp)]
                    elif arg.name in region_map_derived:
                        new_arguments +=[self.update_args(arg, region_map_derived)]
                    elif arg.name_parts[0]==self.cpg_bnds.name:
                        new_arguments += [routine.resolve_typebound_var(f"{self.lcpg_bnds}%{arg.name_parts[1]}")]
                    else:
                        new_arguments +=[arg]
                new_calls += [call.clone(arguments=as_tuple(new_arguments))]
        
        new_calls = tuple(new_calls)
        
        loop_body = (update,) + new_calls
        loop = ir.Loop(variable=self.jblk, bounds=sym.LoopRange((1,routine.resolve_typebound_var(f"{self.cpg_opts}%KGPBLKS"))), body=loop_body)
        dr_hook_calls = self.create_dr_hook_calls(
            routine, f"{routine.name}:{region_name}:COMPUTE",
            sym.Variable(name='ZHOOK_HANDLE_COMPUTE', scope=routine)
        )
        new_region = (dr_hook_calls[0], init, pragma, loop, dr_hook_calls[1])
        return(new_region)
   # TODO : YLCPG_BNDS%INIT
   # TODO : OMP PARALLEL
  #  sym.DeferredTypeSymbol
 #call : call.clone(name=..., args= tuple of the region var + dimensions!!!)

    def create_compute_openmpscc(self, routine, region, region_name, region_map_temp, region_map_derived):
        pass
    def create_compute_openaccscc(self, routine, region, region_name, region_map_temp, region_map_derived):
        pass