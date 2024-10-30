# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pickle
from itertools import chain

from loki import (
    Array,
    BasicType,
    DerivedType,
    FindVariables,
    SubstituteExpressions,
    SymbolAttributes,
    Transformer,
    as_tuple,
    parse_expr,
    single_variable_declaration,
)
from loki.expression import symbols as sym
from loki.ir import (
    FindNodes,
    is_loki_pragma,
    pragma_regions_attached,
)
from loki.ir import (
    nodes as ir,
)
from loki.transform import Transformation

__all__ = ["ParallelRoutineDispatchTransformation"]


class SubstituteExpressionsIgnoreCallstatements(SubstituteExpressions):
    def visit_CallStatement(self, o, **kwargs):
        return o


class ParallelRoutineDispatchTransformation(Transformation):
    """
    Applying the transformation to create _parallel.F90 routine.
    """
    def __init__(self, is_intent, horizontal, path_map_derived_field):
        self.is_intent = (
            is_intent  # set to True if the intent are read for interface block
        )
        self.horizontal = horizontal

        self.map_call_compute = {
            "OpenMP": self.create_compute_openmp,
            "OpenMPSingleColumn": self.create_compute_openmpscc,
            "OpenACCSingleColumn": self.create_compute_openaccscc,
        }

        self.path_map_derived_field = path_map_derived_field
        with open(path_map_derived_field, "rb") as fp:
            self.map_derived_field = pickle.load(fp)



    def transform_subroutine(self, routine, **kwargs):
        item = kwargs.get("item")
        if item:
            item.trafo_data["create_parallel"] = {}

        # change name first to have right name further
        self.change_routine_name(routine)
        map_routine = self.init_map_routine(routine)

        self.create_imports(routine, map_routine)
        self.create_variables(routine, map_routine)
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        self.process_arrays_routine(routine, map_routine)
        self.process_parallel_regions(routine, map_routine, calls)
        single_variable_declaration(routine)
        self.process_not_region_call(routine, map_routine)
        self.add_arrays(routine, map_routine)
        self.add_field(routine, map_routine)
        self.add_derived(routine, map_routine)
        self.add_routine_imports(routine, map_routine)
        self.update_routine_args(routine, map_routine)
        # sanitise_imports(routine) => bug...
        self.clean_imports(routine, map_routine)

        if item:
            item.trafo_data["create_parallel"]["map_routine"] = map_routine

    def change_routine_name(self, routine):
        routine.name = (
            routine.name + "_PARALLEL"
        )

    def init_map_routine(self, routine):
        """
        Init map_routine dictionnary. This dictionnary contains information useful 
        through the whole routine transformation.
        """
        map_routine = {}
        map_routine["field_new"] = []
        map_routine["field_delete"] = []
        map_routine["map_derived"] = {}
        map_routine["c_imports_scc"] = {}
        map_routine["c_imports"] = {
            imp.module: imp
            for imp in FindNodes(ir.Import).visit(routine.spec)
            if imp.c_import
        }
        map_routine["field_new"] = []
        map_routine["field_delete"] = []
        map_routine["nb_no_name"] = (
            0  # to give unique identifier to parallel regions with no name
        )
        map_routine["not_in_pragma_calls"] = []
        map_routine["c_imports_parallel"] = []
        map_routine["imports_mapper"] = {}
        map_routine["call_mapper"] = {}
    
        return(map_routine)
    

    def process_parallel_regions(self, routine, map_routine, calls):
        map_region = {}
        in_pragma_calls = []
        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(routine.body):
                if is_loki_pragma(region.pragma):
                    in_pragma_calls += FindNodes(ir.CallStatement).visit(region)
                    self.process_parallel_region(
                        routine, region, map_routine, map_region
                    )
        map_routine["not_in_pragma_calls"] = [
            call for call in calls if call not in in_pragma_calls
        ]
        map_routine["map_region"] = map_region #for pytest

    def process_parallel_region(self, routine, region, map_routine, map_region):
        """
        Transform the code in acdc pragmas, creating openmp, openmpscc and openaccscc sections.
        """
        self.init_map_region(map_region)

        pragma_content = region.pragma.content.split(maxsplit=1)
        pragma_content = [entry.replace(" ", "") for entry in pragma_content]
        pragma_content = [
            entry.split("=", maxsplit=1) for entry in pragma_content[1].split(",")
        ]
        pragma_attrs = {
            entry[0].lower(): entry[1] if len(entry) == 2 else None
            for entry in pragma_content
        }
        if "parallel" not in pragma_attrs:
            return
        if "target" not in pragma_attrs: #default value
            pragma_attrs["target"] = "OpenMP"
        if "name" not in pragma_attrs:
            pragma_attrs["name"] = str(map_routine["nb_no_name"])
            map_routine["nb_no_name"] += 1

        pragma_attrs["target"] = pragma_attrs["target"].split("/")
        region_name = pragma_attrs["name"]

        region_map_arrays = self.get_region_arrays(
            map_routine, region
        )  # map_region to store field_new and field_delete
        region_map_derived, region_map_not_field = self.process_derived_types(
            routine, region
        )
        region_map_private = self.get_private(region)
        region_map_var = [
            var
            for var in chain(region_map_arrays.values(), region_map_derived.values())
        ]
        region_map_var_sorted = sorted(region_map_var, key=lambda x: x[1].name)

        map_region["var_sorted"] = region_map_var_sorted
        map_region["map_arrays"] = region_map_arrays
        map_region["map_derived"] = region_map_derived
        map_region["private"] = region_map_private
        region_map_not_field = sorted(
            region_map_not_field
        )  # TODO : uniforme names sorted/unsorted; sort at one point, maybe here?
        map_region["not_field_array"] = region_map_not_field

        self.create_synchost(routine, region_name, map_region)
        self.create_nullify(routine, region_name, map_region)

        targets = pragma_attrs["target"]
        for target in targets:
            self.process_target(
                routine, region, region_name, map_routine, map_region, target
            )

        self.create_new_region(
            routine, region, region_name, map_region, targets
        )

        self.add_derived_to_map_routine(map_routine, map_region)

    def init_map_region(self, map_region):
        map_region["get_data"] = {}
        map_region["compute"] = {}
        map_region["region"] = {}
        map_region["lparallel"] = {}
        map_region["scalar"] = []
        map_region["not_field_array"] = []
        return map_region

    def clean_imports(self,routine, map_routine):
        # sanitise_imports(routine) => bug...
        calls = [
            call.name.name.lower()
            for call in FindNodes(ir.CallStatement).visit(routine.body)
        ]

        map_imports = {}
        for imp in map_routine["c_imports"].values():
            imp_name = imp.module.replace(".intfb.h", "")
            if imp_name not in calls:
                map_imports[imp] = None
        routine.spec = Transformer(map_imports).visit(routine.spec)
    def create_new_region(
        self, routine, region, region_name, map_region, targets
    ):
        """
        Gather the different pieces of the region.
        """
        #todo : nettoyer
        # IF (LPARALLELMETHOD ('OPENMP','APL_ARPEGE_PARALLEL:CPPHINP')) THEN
        if len(targets) == 1:
            cond = ir.Conditional(
                condition=map_region["lparallel"][targets[0]],
                body=map_region["region"][targets[0]],
            )
        elif len(targets) == 2:
            cond1 = ir.Conditional(
                condition=map_region["lparallel"][targets[1]],
                body=map_region["region"][targets[1]],
            )
            cond = ir.Conditional(
                condition=map_region["lparallel"][targets[0]],
                body=map_region["region"][targets[0]],
                else_body=(cond1,),
                has_elseif=True,
            )

        elif len(targets) == 3:
            cond1 = ir.Conditional(
                condition=map_region["lparallel"][targets[2]],
                body=map_region["region"][targets[2]],
            )
            cond2 = ir.Conditional(
                condition=map_region["lparallel"][targets[1]],
                body=map_region["region"][targets[1]],
                else_body=(cond1,),
                has_elseif=True,
            )
            cond = ir.Conditional(
                condition=map_region["lparallel"][targets[0]],
                body=map_region["region"][targets[0]],
                else_body=(cond2,),
                has_elseif=True,
            )
        else:
            raise Exception("They should be 1, 2 or 3 targets.")
        dr_hook_calls = self.create_dr_hook_calls(
            routine,
            f"{routine.name}:{region_name}",
            sym.Variable(name="ZHOOK_HANDLE_PARALLEL", scope=routine),
        )

        new_region = (dr_hook_calls[0], cond, dr_hook_calls[1])
        region._update(pragma=None, pragma_post=None, body=new_region)


    def process_target(
        self, routine, region, region_name, map_routine, map_region, target
    ):
        """
        Build the parallel region for the target (where the target (=lparallelmethod) is openmp, openmpscc or openaccscc).
        IF LPARALLELMETHOD ...
            do computation
        END IF
        """
        #todo : clean
        get_data = True
        map_region["get_data"][target] = self.create_pt_sync(
            routine, target, region_name, get_data, map_region
        )
        map_region["compute"][target] = self.map_call_compute[target](
            routine, region, region_name, map_routine, map_region
        )
        map_region["region"][target] = (
            map_region["get_data"][target]
            + list(map_region["compute"][target])
            + [map_region["synchost"]]
            + map_region["nullify"]
        )
        condition_parameters = (
            sym.StringLiteral(value=f"{target.upper()}"),
            sym.StringLiteral(value=f"{routine.name}:{region_name}"),
        )
        condition = sym.InlineCall(
            sym.Variable(name="LPARALLELMETHOD"), parameters=condition_parameters
        )
        map_region["lparallel"][target] = condition

    @staticmethod
    def create_dr_hook_calls(scope, cdname, handle):
        dr_hook_calls = []
        for kswitch in (0, 1):
            call_stmt = ir.CallStatement(
                name=sym.Variable(name="DR_HOOK", scope=scope),
                arguments=(sym.StringLiteral(cdname), sym.IntLiteral(kswitch), handle),
            )
            dr_hook_calls += [
                ir.Conditional(
                    condition=sym.Variable(name="LHOOK", scope=scope),
                    inline=True,
                    body=(call_stmt,),
                )
            ]
        return dr_hook_calls

    def create_field_new(self, routine, map_routine, var, field_api_ptr):
        """
        Create field_new calls for initializing pointers on field_api objects.

        CALL FIELD_NEW (YL_ZA, UBOUNDS=[KLON, KFLEVG, KGPBLKS], LBOUNDS=[1, 0, 1], PERSISTENT=.TRUE.)
        """

        field_new = map_routine["field_new"]
        var_shape = var.shape
        ubounds = [d.upper if isinstance(d, sym.RangeIndex) else d for d in var_shape]
        ubounds += [
            sym.Variable(name="JBLKMAX", parent=routine.variable_map["YDCPG_OPTS"])
        ]
        # NB: This is presumably what the condition should look like, however, the
        #     logic is flawed in to_parallel and it will only insert lbounds if _the last_
        #     dimension has an lbound. We emulate this with the second line here to
        #     generate identical results, but this line should probably not be there
        has_lbounds = any(isinstance(d, sym.RangeIndex) for d in var_shape)
        has_lbounds = has_lbounds and isinstance(var_shape[-1], sym.RangeIndex)
        has_lbounds = True
        if has_lbounds:
            lbounds = [
                d.lower if isinstance(d, sym.RangeIndex) else sym.IntLiteral(1)
                for d in var_shape
            ]
            lbounds.append(
                sym.Variable(name="JBLKMIN", parent=routine.variable_map["YDCPG_OPTS"])
            )
            kwarguments = (
                ("UBOUNDS", sym.LiteralList(ubounds)),
                ("LBOUNDS", sym.LiteralList(lbounds)),
                ("PERSISTENT", sym.LogicLiteral(True)),
            )
        else:
            kwarguments = (
                ("UBOUNDS", sym.LiteralList(ubounds)),
                ("PERSISTENT", sym.LogicLiteral(True)),
            )
        field_new += [
            ir.CallStatement(
                name=sym.Variable(name="FIELD_NEW", scope=routine),
                arguments=(field_api_ptr,),
                kwarguments=kwarguments,
            )
        ]


    def create_field_delete(self, routine, map_routine, field_api_ptr):
        """
        Create field_delete calls for deleting pointers on field_api objects.

        IF (ASSOCIATED (YL_ZA)) CALL FIELD_DELETE (YL_ZA)
        """
        field_delete = map_routine["field_delete"]
        call = ir.CallStatement(
            sym.Variable(name="FIELD_DELETE", scope=routine), arguments=(field_api_ptr,)
        )
        condition = sym.InlineCall(
            sym.Variable(name="ASSOCIATED"), parameters=(field_api_ptr,)
        )
        field_delete += [ir.Conditional(condition=condition, inline=True, body=(call,))]

    def process_arrays_routine(self, routine, map_routine):
        """Creates the pointers on data by wich the arrays declarations will be replaced, creates pointers on field_api objects.
        Creates field_new/field_delete calls (call to self.create_field_new and self.create_field_delete), to init/delete the pointers on field_api objects.
        """
        #todo : check when var.name_parts[0] in routine.arguments
        arrays = [
            var
            for var in FindVariables(Array).visit(routine.spec)
            if isinstance(var, Array)
            and var.name_parts[0] not in routine.arguments
            and var.shape[0] in self.horizontal
        ]
        routine_map_arrays = {}
        # check if first dim NPROMA ?
        for var in arrays:
            var_shape = var.shape

            dim = len(var_shape) + 1  # Temporary dimensions + block

            if var.name in routine.argnames:
                init = None
                name_prefix = "YD_"
            else:
                init = "NULL()"
                name_prefix = "YL_"

            # The FIELD_{d}RB variable
            if var.type.dtype.name == "LOGICAL":
                field_name = "LM"
            if var.type.dtype.name == "REAL":
                field_name = "RB"
            if var.type.dtype.name == "INTEGER":
                field_name = "IM"
            field_ptr_type = SymbolAttributes(
                dtype=DerivedType(f"FIELD_{dim}{field_name}"),
                pointer=True,
                polymorphic=True,
                initial=init,
            )
            field_api_ptr = sym.Variable(
                name=f"{name_prefix}{var.name}", type=field_ptr_type, scope=routine
            )

            # Create a pointer instead of the array
            shape = (sym.RangeIndex((None, None)),) * dim
            #  var.type = var_type.clone(pointer=True, shape=shape)
            ptr_var = var.clone(dimensions=shape)

            routine_map_arrays[var.name] = [field_api_ptr, ptr_var]
            if var.name not in routine.argnames:
                self.create_field_new(
                    routine, map_routine, var, field_api_ptr
                )  # file in map_routine['field_new']
                self.create_field_delete(
                    routine, map_routine, field_api_ptr
                )  # file in map_routine['field_delete']
        
        map_routine["map_arrays"] = routine_map_arrays

    def get_region_arrays(self, map_routine, region):
        """Finds arrays in map_routine["map_arrays"] for the region.
        region : parallel region or call statement (call outside of parallel region)
        return : region_map_arrays
        """
        arrays = [var for var in FindVariables(Array).visit(region)]
        region_map_arrays = {}
        for var in arrays:
            if var.name in map_routine["map_arrays"]:
                field_api_ptr = map_routine["map_arrays"][var.name][0]
                ptr_var = map_routine["map_arrays"][var.name][1]
                region_map_arrays[var.name] = [field_api_ptr, ptr_var]
        return region_map_arrays

    def add_arrays(self, routine, map_routine):
        routine_map_arrays = map_routine["map_arrays"]
        # Replace temporary declaration by pointer to array and pointer to field_api object
        map_dcl = {}
        for decl in routine.declarations:
            if len(decl.symbols) == 1:
                var = decl.symbols[0]
                if var.name in routine_map_arrays:
                    if var.name in routine.argnames:
                        init = None
                    else:
                        init = "NULL()"
                    new_vars = routine_map_arrays[var.name]
                    new_vars[1].type = new_vars[1].type.clone(
                        pointer=True,
                        shape=new_vars[1].dimensions,
                        initial=init,
                        intent=None,
                    )
                    map_dcl.update(
                        {
                            decl: (
                                ir.VariableDeclaration(symbols=(new_vars[0],)),
                                ir.VariableDeclaration(symbols=(new_vars[1],)),
                            )
                        }
                    )
            else:
                raise Exception(
                    "Declaration should have only one symbol, please run single_variable_declaration before calling add_arrays function."
                )
        routine.spec = Transformer(map_dcl).visit(routine.spec)
        # decls_to_replace = [var for var in decl.var for decl in routine.declarations if var in self.routine_map_arrays]

    def add_field(self, routine, map_routine):
        field_new = map_routine["field_new"]
        field_delete = map_routine["field_delete"]

        field_new_sorted = sorted(field_new, key=lambda X: X.arguments[0].name)
        field_delete_sorted = sorted(
            field_delete, key=lambda X: X.body[0].arguments[0].name
        )

        # Insert the field generation wrapped into a DR_HOOK call
        dr_hook_calls = self.create_dr_hook_calls(
            routine,
            cdname="CREATE_TEMPORARIES",
            handle=sym.Variable(name="ZHOOK_HANDLE_FIELD_API", scope=routine),
        )
        routine.body.insert(
            2,
            (
                dr_hook_calls[0],
                ir.Comment(text=""),
                *field_new_sorted,
                dr_hook_calls[1],
            ),
        )

        # Insert the field deletion wrapped into a DR_HOOK call
        dr_hook_calls = self.create_dr_hook_calls(
            routine,
            cdname="DELETE_TEMPORARIES",
            handle=sym.Variable(name="ZHOOK_HANDLE_FIELD_API", scope=routine),
        )
        routine.body.insert(
            -2,
            (
                dr_hook_calls[0],
                ir.Comment(text=""),
                *field_delete_sorted,
                dr_hook_calls[1],
            ),
        )

    def process_derived_types(self, routine, region):
        """
        I - Finds the derived types of the region that are in map_derived_field and creates:

        1) a pointer on the field api object corresponding to that derived type 
        2) a pointer on the data

        Example : 
        
        derived type :
        YDMF_PHYS_BASE_STATE%YCPG_PHY%PREHYDF
        
        pointer on the field api object : 
        YDMF_PHYS_BASE_STATE%YCPG_PHY%F_PREHYDF
        
        pointer on the data : 
        REAL(KIND=JPRB), POINTER :: Z_YDMF_PHYS_BASE_STATE_YCPG_PHY_PREHYDF(:, :, :)
        
        the pointer on the field api object and the pointer on the data "interact"
        this way : 
        Z_YDMF_PHYS_BASE_STATE_YCPG_PHY_PREHYDF => 
        GET_HOST_DATA_RDWR(YDMF_PHYS_BASE_STATE%YCPG_PHY%F_PREHYDF)

        II - Find derived types that don't contain fields 
        """
        region_map_derived = {}
        not_field_array = []
        basename_derived = []
        derived = [
            var
            for var in FindVariables().visit(region)
            if var.name_parts[0] in routine.arguments
        ]
        for var in derived:
            key = f"{routine.variable_map[var.name_parts[0]].type.dtype.name}%{'%'.join(var.name_parts[1:])}"
            # TODO : maybe have a global derive typed table, to avoid to many lookup in the map_derived_field???
            if key in self.map_derived_field:
                if var.name_parts[0] not in basename_derived:
                    basename_derived.append(var.name_parts[0])
                value = self.map_derived_field[key]
                # Creating the pointer on the data : YL_A
                data_name = f"Z_{var.name.replace('%', '_')}"
                if "REAL" and "JPRB" in value[0]:
                    data_dim = value[2] + 1
                    data_shape = (sym.RangeIndex((None, None)),) * data_dim
                    data_type = SymbolAttributes(
                        dtype=BasicType.REAL,
                        kind=routine.symbol_map["JPRB"],
                        pointer=True,
                        shape=data_shape,
                    )
                    ptr_var = sym.Variable(
                        name=data_name,
                        type=data_type,
                        dimensions=data_shape,
                        scope=routine,
                    )

                else:
                    raise NotImplementedError("This type isn't implemented yet")

                # Creating the pointer on the field api object : YL%FA, YL%F_A...
                if (
                    routine.variable_map[var.name_parts[0]].type.dtype.name
                    == "MF_PHYS_SURF_TYPE"
                ):
                    # YL%PA becomes YL%F_A
                    field_name = (
                        f"{'%'.join(var.name_parts[:-1])}%F_{var.name_parts[-1][1:]}"
                    )
                elif (
                    routine.variable_map[var.name_parts[0]].type.dtype.name
                    == "FIELD_VARIABLES"
                ):
                    # YL%A becomes YL%FA
                    field_name = (
                        f"{'%'.join(var.name_parts[:-1])}%F{var.name_parts[-1]}"
                    )
                    if var.name_parts[-1] == "P":  # YL%FP = YL%FT0
                        field_name = "FT0"
                else:
                    # YL%A becomes YL%F_A
                    field_name = (
                        f"{'%'.join(var.name_parts[:-1])}%F_{var.name_parts[-1]}"
                    )
                field_api_ptr = var.clone(name=field_name, dimensions=None)
                region_map_derived[var.name] = [field_api_ptr, ptr_var]
            elif var.name_parts[0] not in not_field_array:
                if (
                    routine.variable_map[var.name_parts[0]].type.dtype.name
                    != "CPG_BNDS_TYPE"
                ):
                    if (
                        routine.variable_map[var.name_parts[0]].type.dtype.name
                        != "TYP_DDH"
                    ):
                        if isinstance(
                            routine.variable_map[var.name_parts[0]].type.dtype,
                            sym.DerivedType,
                        ):
                            not_field_array.append(var.name_parts[0])
        not_field_array_ = [
            var for var in not_field_array if var not in basename_derived
        ]
        return (region_map_derived, not_field_array_)

    def get_private(self, region):
        lhs = [a.lhs for a in FindNodes(ir.Assignment).visit(region)]
        scalars = [var for var in lhs if isinstance(var, sym.Scalar)]
        scalars_ = [
            var.name for var in scalars if not isinstance(var.type.dtype, DerivedType)
        ]

        loop_variables = [
            loop.variable.name for loop in FindNodes(ir.Loop).visit(region)
        ]
        scalars_ += loop_variables

        scalars_sorted = sorted(scalars_, key=lambda X: X)

        return scalars_sorted

    def add_derived(self, routine, map_routine):
        routine_map_derived = map_routine["map_derived"]
        routine_map_derived_sorted = sorted(
            routine_map_derived.values(), key=lambda X: X[1].name
        )
        routine.variables += tuple(v[1] for v in routine_map_derived_sorted)

    def add_routine_imports(self, routine, map_routine):
        routine.spec = Transformer(map_routine["imports_mapper"]).visit(routine.spec)

    def update_routine_args(self, routine, map_routine):
        routine_map_arrays = map_routine["map_arrays"]
        lst_routine_args = list(routine.arguments)
        idx = 0
        for arg in lst_routine_args:
            if arg.name in routine_map_arrays:
                lst_routine_args[idx] = routine_map_arrays[arg.name][0]
            idx += 1
        tuple_routine_args = tuple(lst_routine_args)
        routine.arguments = tuple_routine_args

    def create_pt_sync(self, routine, target, region_name, is_get_data, map_region):
        region_map_var_sorted = map_region["var_sorted"]
        if is_get_data:  # GET_***_DATA
            hook_name = "GET_DATA"
            if target == "OpenMP" or target == "OpenMPSingleColumn":
                sync_name = "GET_HOST_DATA"
            elif target == "OpenACCSingleColumn":
                sync_name = "GET_DEVICE_DATA"
            else:
                raise Exception(f"{target} : this target isn't known!")
        else:  # SYNCHOST
            hook_name = "SYNCHOST"
            sync_name = "GET_HOST_DATA"

        dr_hook_calls = self.create_dr_hook_calls(
            routine,
            cdname=f"{routine.name}:{region_name}:{hook_name}",
            handle=sym.Variable(name="ZHOOK_HANDLE_FIELD_API", scope=routine),
        )

        sync_data = [dr_hook_calls[0]]

        for var in region_map_var_sorted:
            if is_get_data:
                if not self.is_intent:
                    intent = "RDWR"
                else:
                    raise NotImplementedError(
                        "Reading the intent from interface isn't implemented yes"
                    )
            else:
                intent = "RDWR"  # for SYNCHOST, always RDWR

            call = sym.InlineCall(
                sym.Variable(name=f"{sync_name}_{intent}"), parameters=(var[0],)
            )
            sync_data += [
                ir.Assignment(lhs=var[1].clone(dimensions=None), rhs=call, ptr=True)
            ]

        sync_data.append(dr_hook_calls[1])

        return sync_data

    def create_synchost(self, routine, region_name, map_region):
        get_data = False
        synchost = self.create_pt_sync(routine, None, region_name, get_data, map_region)
        condition = sym.InlineCall(
            sym.Variable(name="LSYNCHOST"),
            parameters=(sym.StringLiteral(value=f"{routine.name}:{region_name}"),),
        )
        map_region["synchost"] = ir.Conditional(
            condition=condition, body=tuple(synchost)
        )

    def create_nullify(self, routine, region_name, map_region):
        """
        Create the piece of code that null the pointers on the data at the end of each parallel region.

        Example:

        IF (LHOOK) CALL DR_HOOK ('APL_ARPEGE_PARALLEL:CPPHINP:NULLIFY',0,ZHOOK_HANDLE_FIELD_API)
        ZRDG_CVGQ => NULL ()
        ...
        IF (LHOOK) CALL DR_HOOK ('APL_ARPEGE_PARALLEL:CPPHINP:NULLIFY',1,ZHOOK_HANDLE_FIELD_API)
        """
        region_map_var_sorted = map_region["var_sorted"]
        dr_hook_calls = self.create_dr_hook_calls(
            routine,
            cdname=f"{routine.name}:{region_name}:NULLIFY",
            handle=sym.Variable(name="ZHOOK_HANDLE_FIELD_API", scope=routine),
        )
        nullify = [dr_hook_calls[0]]
        for var in region_map_var_sorted:
            nullify += [
                ir.Assignment(
                    lhs=var[1].clone(dimensions=None),
                    rhs=sym.InlineCall(sym.Variable(name="NULL")),
                    ptr=True,
                )
            ]
        nullify.append(dr_hook_calls[1])
        map_region["nullify"] = nullify

    def get_cpg(self, routine, map_routine):
        """
        I  - Find objects with CPG_OPTS_TYPE and CPG_BNDS_TYPE types. 
        II - Create lcpg_bnds and return lcpg_bnds_declaration
        """
        # Assuming CPG_OPTS_TYPE and CPG_BNDS_TYPE are the same in all the routine.
        found_opts = False
        found_bnds = False
        for var in FindVariables().visit(routine.spec):
            if var.type.dtype.name == "CPG_OPTS_TYPE":
                cpg_opts = var
                map_routine["cpg_opts"] = cpg_opts
                found_opts = True
            if var.type.dtype.name == "CPG_BNDS_TYPE":
                cpg_bnds = var
                map_routine["cpg_bnds"] = cpg_bnds
                found_bnds = True
            if found_opts and found_bnds:
                if "YD" in cpg_bnds.name:
                    lcpg_bnds_name = cpg_bnds.name.replace("YD", "YL")
                    lcpg_bnds_type = cpg_bnds.type.clone(intent=None)
                    lcpg_bnds = sym.Variable(
                        name=lcpg_bnds_name, type=lcpg_bnds_type, scope=routine
                    )
                    map_routine["lcpg_bnds"] = lcpg_bnds
                    lcpg_bnds_declaration = ir.VariableDeclaration(symbols=(lcpg_bnds,))
                    return [lcpg_bnds_declaration]
                raise Exception(f"cpg_bnds unexpected name : {self.cpg_bnds.name}")

    def create_variables(self, routine, map_routine):
        """
        Add some variables to the end of the routine spec.
        
          TYPE(CPG_BNDS_TYPE) :: YLCPG_BNDS
          TYPE(STACK) :: YLSTACK
          INTEGER(KIND=JPIM) :: JBLK
          REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_FIELD_API
          REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_PARALLEL
          REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_COMPUTE
        
         if not present:
          INTEGER(KIND=JPIM) :: JLON
        """

        variable_declarations = []
        variable_declarations += self.get_cpg(routine, map_routine)
        stack_type = DerivedType(name="STACK")
        var_type = SymbolAttributes(name="STACK", dtype=stack_type)
        ylstack = sym.Variable(name="YLSTACK", type=var_type, scope=routine)
        variable_declarations += [ir.VariableDeclaration(symbols=(ylstack,))]
        integer_type = SymbolAttributes(
            dtype=BasicType.INTEGER, kind=routine.symbol_map["JPIM"]
        )
        self.jblk = sym.Variable(name="JBLK", type=integer_type, scope=routine)
        variable_declarations += [ir.VariableDeclaration(symbols=(self.jblk,))]
        hook_type = SymbolAttributes(
            dtype=BasicType.REAL, kind=routine.symbol_map["JPHOOK"]
        )

        hook_var = sym.Variable(
            name="ZHOOK_HANDLE_FIELD_API", type=hook_type, scope=routine
        )
        variable_declarations += [ir.VariableDeclaration(symbols=(hook_var,))]
        hook_var = sym.Variable(
            name="ZHOOK_HANDLE_PARALLEL", type=hook_type, scope=routine
        )
        variable_declarations += [ir.VariableDeclaration(symbols=(hook_var,))]
        hook_var = sym.Variable(
            name="ZHOOK_HANDLE_COMPUTE", type=hook_type, scope=routine
        )
        variable_declarations += [ir.VariableDeclaration(symbols=(hook_var,))]

        if "JLON" not in routine.variable_map:
            jlon = sym.Variable(name="JLON", type=integer_type, scope=routine)
            variable_declarations += [ir.VariableDeclaration(symbols=(jlon,))]

        routine.spec.append(variable_declarations)
        map_routine["variable_declarations"] = variable_declarations

    def create_imports(self, routine, map_routine):
        """
        Add imports to the routine spec.
        """
        imports = []
        imports += [ir.Import(module="ACPY_MOD", scope=routine)]
        imports += [ir.Import(module="STACK_MOD", scope=routine)]
        imports += [ir.Import(module="YOMPARALLELMETHOD", scope=routine)]
        imports += [ir.Import(module="FIELD_ACCESS_MODULE", scope=routine)]
        imports += [ir.Import(module="FIELD_FACTORY_MODULE", scope=routine)]
        imports += [ir.Import(module="FIELD_MODULE", scope=routine)]
        imports += [ir.Import(module="stack.h", c_import=True)]
        routine.spec.prepend(imports)
        map_routine["imports"] = imports

    def update_arg_dims(self, arg, region_map):
        new_arg = region_map[arg.name][1]
        dim = len(new_arg.dimensions)
        # dim = len(new_arg.shape)
        if isinstance(arg, Array):
            if len(arg.dimensions) != 0:
                new_dimensions = arg.dimensions
            else:
                new_dimensions = (sym.RangeIndex((None, None)),) * (dim - 1)
        else:
            new_dimensions = (sym.RangeIndex((None, None)),) * (dim - 1)
        new_dimensions += (self.jblk,)
        return new_arg.clone(dimensions=new_dimensions)

    def update_vars(self, routine, var, region_map, scc):
        new_var = region_map[var.name][1]
        if isinstance(var.dimensions[0], sym.RangeIndex):
            if scc:
                jlon = (routine.variable_map["JLON"],)
                new_dimensions = jlon + var.dimensions[1:]
            else:
                new_dimensions = var.dimensions
        else:
            new_dimensions = var.dimensions
        new_dimensions += (self.jblk,)
        return new_var.clone(dimensions=new_dimensions)

    def process_not_call(self, routine, region, map_routine, map_region, scc):
        verbose = False
        region_map_arrays = map_region["map_arrays"]
        region_map_derived = map_region["map_derived"]
        map_not_call = {}
        #                if not (
        #                    isinstance(var, sym.LogicalOr) or isinstance(var, sym.LogicalAnd)):
        calls = [call for call in FindNodes(ir.CallStatement).visit(region)]
        var_calls = []
        for call in calls:
            for var in FindVariables(Array).visit(call):
                var_calls.append(var)
        var_not_calls = [
            var for var in FindVariables(Array).visit(region) if var not in var_calls
        ]
        var_map = {}
        for var in var_not_calls:
            if var.name in region_map_arrays:
                if verbose:
                    print(f"var_arrays={var}")
                new_var = self.update_vars(routine, var, region_map_arrays, scc)
                if verbose:
                    print(f"new_var_arrays={new_var}")
                var_map[var] = new_var
            #                var._update(name=new_var.name, dimensions=new_var.dimensions)
            elif var.name in region_map_derived:
                if verbose:
                    print(f"var_derived={var}")
                new_var = self.update_vars(routine, var, region_map_derived, scc)
                if verbose:
                    print(f"new_var_derived={new_var}")
                var_map[var] = new_var

        # new_region = SubstituteExpressions(var_map).visit(region.body)
        # new_region = SubstituteExpressions(var_map).visit(region)
        return var_map

    def process_loops(self, routine, region, map_routine, map_region, scc):
        loops = [
            loop
            for loop in FindNodes(ir.Loop).visit(region)
            if loop.variable.name == "JLON"
        ]
        #        for loop in loops:
        #            loop_map[loop] = loop #call scc transformation here

        lst_horizontal_idx = ["JLON", "JROF"]
        loop_map = {}

        if scc:
            for loop in FindNodes(ir.Loop).visit(region.body):
                if loop.variable.name in lst_horizontal_idx:
                    loop_map[loop] = loop.body
        else:
            lcpg_bnds = map_routine["lcpg_bnds"]
            for loop in FindNodes(ir.Loop).visit(region.body):
                if loop.variable.name == "JLON":
                    lower_bound = routine.resolve_typebound_var(f"{lcpg_bnds}%KIDIA")
                    upper_bound = routine.resolve_typebound_var(f"{lcpg_bnds}%KFDIA")
                    new_bounds = sym.LoopRange((lower_bound, upper_bound))
                    new_loop = loop.clone(bounds=new_bounds)
                    loop_map[loop] = new_loop
                # else:
                #    loop_map[loop] = loop
        #         new_region_body=Transformer(loop_map).visit(new_region_body)

        return loop_map

    def process_argument(
        self,
        routine,
        arg,
        region_map_arrays,
        region_map_derived,
        cpg_bnds,
        lcpg_bnds,
        vars_call,
    ):
        if arg not in vars_call:
            vars_call.append(arg)
        if not (
            isinstance(arg, sym.LogicalOr)
            or isinstance(arg, sym.LogicalAnd)
            or isinstance(arg, sym.IntLiteral)
            or isinstance(arg, sym.LogicLiteral)
            or isinstance(arg, sym.Product)
            or isinstance(arg, sym.Sum)
        ):
            #             or isinstance(arg, sym.StringLiteral)):

            if arg.name in region_map_arrays:
                return self.update_arg_dims(arg, region_map_arrays)
            if arg.name in region_map_derived:
                return self.update_arg_dims(arg, region_map_derived)
            if arg.name_parts[0] == cpg_bnds.name:
                if len(arg.name_parts) == 2:
                    return routine.resolve_typebound_var(
                        f"{lcpg_bnds}%{arg.name_parts[1]}"
                    )
                return lcpg_bnds
            return arg
        return arg

    def process_call(self, routine, region, map_routine, map_region, scc):
        region_map_arrays = map_region["map_arrays"]
        region_map_derived = map_region["map_derived"]
        vars_call = []
        c_imports = map_routine["c_imports"]
        cpg_bnds = map_routine["cpg_bnds"]
        lcpg_bnds = map_routine["lcpg_bnds"]
        map_new_calls = {}
        calls = [call for call in FindNodes(ir.CallStatement).visit(region)]
        for call in calls:
            if call.name != "DR_HOOK" and call.name != "ABOR1":
                new_arguments = []
                new_kwarguments = []
                for arg in call.arguments:
                    new_arg = self.process_argument(
                        routine,
                        arg,
                        region_map_arrays,
                        region_map_derived,
                        cpg_bnds,
                        lcpg_bnds,
                        vars_call,
                    )
                    new_arguments.append(new_arg)
                for arg in call.kwarguments:
                    new_kwarg = self.process_argument(
                        routine,
                        arg[1],
                        region_map_arrays,
                        region_map_derived,
                        cpg_bnds,
                        lcpg_bnds,
                        vars_call,
                    )
                    new_kwarguments.append(
                        (
                            arg[0],
                            new_kwarg,
                        )
                    )

                if scc:
                    new_kwarguments += (("YDSTACK", routine.variable_map["YLSTACK"]),)
                    new_call = call.clone(
                        name=sym.ProcedureSymbol(name=f"{call.name.name}_OPENACC"),
                        arguments=as_tuple(new_arguments),
                        kwarguments=as_tuple(new_kwarguments),
                    )

                    map_new_calls[call] = new_call
                    c_import_name = f"{call.name.name.lower()}.intfb.h"
                    if c_import_name not in c_imports:
                        print(f'{map_routine["c_imports"]=}')
                        raise Exception(
                            f"{call.name.name} should have an interface block"
                        )
                    c_import = c_imports[c_import_name]
                    if c_import not in map_routine["imports_mapper"]:
                        new_c_import_name = (
                            f"{call.name.name.lower()}_openacc.intfb.h"
                        )
                        new_c_import = ir.Import(
                            module=new_c_import_name, c_import=True
                        )
                        map_routine["imports_mapper"][c_import] = (
                            c_import,
                            new_c_import,
                        )

                else:
                    new_call = call.clone(
                        arguments=as_tuple(new_arguments),
                        kwarguments=as_tuple(new_kwarguments),
                    )
                    map_new_calls[call] = new_call

        # map_region["vars_call"] = vars_call
        return map_new_calls

    def create_compute_openmp(
        self, routine, region, region_name, map_routine, map_region
    ):
        lcpg_bnds = map_routine["lcpg_bnds"]
        cpg_bnds = map_routine["cpg_bnds"]
        cpg_opts = map_routine["cpg_opts"]
        init = ir.CallStatement(
            name=routine.resolve_typebound_var(f"{lcpg_bnds.name}%INIT"),
            arguments=(cpg_opts,),
        )
        # ==============================================================
        # ==============================================================
        # TODO : generate lst_private !!!! see LLHMT in CALL ACSOL
        # ==============================================================
        # ==============================================================
        lst_private = ["JBLK"]
        for private in map_region["private"]:
            if private not in lst_private:
                lst_private += [private]
        lst_private = sorted(lst_private)
        str_private = ", ".join(lst_private)
        pragma = ir.Pragma(
            keyword="OMP",
            content=f"PARALLEL DO PRIVATE ({str_private}) FIRSTPRIVATE ({lcpg_bnds.name})",
        )
        update = ir.CallStatement(
            name=routine.resolve_typebound_var(f"{lcpg_bnds.name}%UPDATE"),
            arguments=(self.jblk,),
        )
        # TODO : musn't be call but the body of the region here??

        map_new_calls = self.process_call(
            routine, region, map_routine, map_region, scc=False
        )  # call : new_call
        map_not_calls = self.process_not_call(
            routine, region, map_routine, map_region, scc=False
        )  # var : new_var
        map_new_loops = self.process_loops(
            routine, region, map_routine, map_region, scc=False
        )  # loop : new_loop
        map_new_region = map_new_calls | map_new_loops
        new_region_body = Transformer(map_new_region).visit(region.body)
        new_region_body = SubstituteExpressionsIgnoreCallstatements(
            map_not_calls
        ).visit(new_region_body)

        loop_body = (update,) + new_region_body
        lower_bound = routine.resolve_typebound_var(f"{cpg_opts}%JBLKMIN")
        upper_bound = routine.resolve_typebound_var(f"{cpg_opts}%JBLKMAX")
        loop = ir.Loop(
            variable=self.jblk,
            bounds=sym.LoopRange((lower_bound, upper_bound)),
            body=loop_body,
        )
        dr_hook_calls = self.create_dr_hook_calls(
            routine,
            f"{routine.name}:{region_name}:COMPUTE",
            sym.Variable(name="ZHOOK_HANDLE_COMPUTE", scope=routine),
        )
        new_region = (dr_hook_calls[0], init, pragma, loop, dr_hook_calls[1])
        return new_region

    def create_scc(
        self, routine, region, region_name, map_routine, map_region, pragma1, pragma2
    ):
        # ==============================================================
        # ==============================================================
        # TODO : generate lst_private !!!! see LLHMT in CALL ACSOL
        # ==============================================================
        # ==============================================================
        cpg_opts = map_routine["cpg_opts"]
        lower_bound = routine.resolve_typebound_var(f"{cpg_opts}%JBLKMIN")
        upper_bound = routine.resolve_typebound_var(f"{cpg_opts}%JBLKMAX")

        cpg_opts_kgpblks = routine.resolve_typebound_var("YDCPG_OPTS%KGPBLKS")
        kidia = ir.Assignment(
            lhs=routine.resolve_typebound_var("YLCPG_BNDS%KIDIA"),
            rhs=routine.variable_map["JLON"],
        )
        kfdia = ir.Assignment(
            lhs=routine.resolve_typebound_var("YLCPG_BNDS%KFDIA"),
            rhs=routine.variable_map["JLON"],
        )
        jblk_param = parse_expr("JBLK-YDCPG_OPTS%JBLKMIN+1")

        stack_param = (
            sym.Variable(name="YSTACK", scope=routine),
            jblk_param,
            cpg_opts_kgpblks,
        )
        ylstack_l8 = ir.Assignment(
            lhs=routine.resolve_typebound_var("YLSTACK%L8"),
            rhs=sym.InlineCall(sym.Variable(name="stack_l8"), parameters=stack_param),
        )
        ylstack_u8 = ir.Assignment(
            lhs=routine.resolve_typebound_var("YLSTACK%U8"),
            rhs=sym.InlineCall(sym.Variable(name="stack_u8"), parameters=stack_param),
        )
        ylstack_l4 = ir.Assignment(
            lhs=routine.resolve_typebound_var("YLSTACK%L4"),
            rhs=sym.InlineCall(sym.Variable(name="stack_l4"), parameters=stack_param),
        )
        ylstack_u4 = ir.Assignment(
            lhs=routine.resolve_typebound_var("YLSTACK%U4"),
            rhs=sym.InlineCall(sym.Variable(name="stack_u4"), parameters=stack_param),
        )

        # new_calls = self.process_call(routine, region, map_routine, map_region, scc=True)

        # TODO save the new_region_body in order to apply Transformer once instead of twice

        map_new_calls = self.process_call(
            routine, region, map_routine, map_region, scc=True
        )  # call : new_call
        map_not_calls = self.process_not_call(
            routine, region, map_routine, map_region, scc=True
        )  # var : new_var
        map_new_loops = self.process_loops(
            routine, region, map_routine, map_region, scc=True
        )  # loop : new_loop
        map_new_region = map_new_calls | map_new_loops
        new_region_body = Transformer(map_new_region).visit(region.body)
        new_region_body = SubstituteExpressionsIgnoreCallstatements(
            map_not_calls
        ).visit(new_region_body)

        loop_jlon_body = [
            kidia,
            kfdia,
            ylstack_l8,
            ylstack_u8,
            ylstack_l4,
            ylstack_u4,
        ] + list(new_region_body)
        min_rhs = parse_expr("YDCPG_OPTS%KGPCOMP - (JBLK - 1) * YDCPG_OPTS%KLON")
        loop_jlon_bounds = (
            1,
            sym.InlineCall(
                sym.DeferredTypeSymbol(name="MIN"),
                parameters=(routine.resolve_typebound_var(f"{cpg_opts}%KLON"), min_rhs),
            ),
        )
        loop_jlon = ir.Loop(
            variable=routine.variable_map["JLON"],
            bounds=sym.LoopRange(loop_jlon_bounds),
            body=loop_jlon_body,
        )
        loop_jblk_body = [pragma2, loop_jlon]
        loop_jblk = ir.Loop(
            variable=routine.variable_map["JBLK"],
            bounds=sym.LoopRange((lower_bound, upper_bound)),
            body=loop_jblk_body,
        )
        dr_hook_calls = self.create_dr_hook_calls(
            routine,
            f"{routine.name}:{region_name}:COMPUTE",
            sym.Variable(name="ZHOOK_HANDLE_COMPUTE", scope=routine),
        )
        computescc = (dr_hook_calls[0], pragma1, loop_jblk, dr_hook_calls[1])
        return computescc

    def create_compute_openmpscc(
        self, routine, region, region_name, map_routine, map_region
    ):
        lst_private = ["JBLK", "JLON", "YLCPG_BNDS", "YLSTACK"]
        for private in map_region["private"]:
            if private not in lst_private:
                lst_private += [private]
        lst_private = sorted(lst_private)
        str_private = ", ".join(lst_private)
        pragma1 = ir.Pragma(
            keyword="OMP", content=f"PARALLEL DO PRIVATE ({str_private})"
        )
        pragma2 = None
        compute_openmpscc = self.create_scc(
            routine, region, region_name, map_routine, map_region, pragma1, pragma2
        )
        return compute_openmpscc

    def create_compute_openaccscc(
        self, routine, region, region_name, map_routine, map_region
    ):
        region_map_var_sorted = map_region["var_sorted"]
        not_field_array = map_region["not_field_array"]
        cpg_opts = map_routine["cpg_opts"]
        lst_private = ["JLON", "YLCPG_BNDS", "YLSTACK"]
        lst_private = sorted(lst_private)
        str_private = ", ".join(lst_private)
        for private in map_region["private"]:
            if private not in lst_private:
                lst_private += [private]
        lst_present = ["YDCPG_OPTS", "YSTACK"]
        for var_name in not_field_array:
            if var_name not in lst_present:
                lst_present += [var_name]

        for var in region_map_var_sorted:
            if var[1].name not in lst_present:
                lst_present += [var[1].name]
        lst_present = sorted(lst_present)
        str_present = ", ".join(lst_present)
        acc_vector_length = f"{cpg_opts}%KLON"
        pragma1 = ir.Pragma(
            keyword="ACC",
            content=f"PARALLEL LOOP GANG PRESENT ({str_present}) PRIVATE (JBLK) VECTOR_LENGTH({acc_vector_length})",
        )
        pragma2 = ir.Pragma(
            keyword="ACC", content=f"LOOP VECTOR PRIVATE ({str_private})"
        )
        compute_openaccscc = self.create_scc(
            routine, region, region_name, map_routine, map_region, pragma1, pragma2
        )
        return compute_openaccscc

    def process_not_region_call(self, routine, map_routine):
        """
        Process calls that aren't in ACDC pragma regions.
        """
        c_imports = map_routine["c_imports"]
        map_region = {}
        map_region["map_arrays"] = {}
        map_region["map_derived"] = {}

        calls = map_routine["not_in_pragma_calls"]

        call_mapper = {}  # mapper for transformation
        for call in calls:
            if call.name != "DR_HOOK" and call.name != "ABOR1":
                region_map_arrays = self.get_region_arrays(
                    map_routine, call
                )  # map_region to store field_new and field_delete
                region_map_derived, region_map_not_field = self.process_derived_types(
                    routine, call
                )
                map_region["map_arrays"] = map_region["map_arrays"] | region_map_arrays
                map_region["map_derived"] = (
                    map_region["map_derived"] | region_map_derived
                )
                # TODO 2 cases : derived type used just outside acdc region : no field, or use in both : field...
                #                map_region['map_derived'][1] = None #no field to add to routine dcl
                # TODO fix the field_new = map_region[...] .... this is because map_region['field_new'] is init in get_region_arrays.
                # an other part of the code needs this initialization ...

                new_arguments = []
                for arg in call.arguments:
                    if not (
                        isinstance(arg, sym.LogicalOr)
                        or isinstance(arg, sym.LogicalAnd)
                        or isinstance(arg, sym.StringLiteral)
                    ):
                        if arg.name in region_map_arrays:
                            new_arguments += [region_map_arrays[arg.name][0]]
                        elif arg.name in region_map_derived:
                            new_arguments += [region_map_derived[arg.name][0]]
                        else:
                            new_arguments += [arg]
                    else:
                        new_arguments += [arg]

                import_name = f"{call.name.name.lower()}.intfb.h"
                if import_name in c_imports:
                    c_import = c_imports[import_name]
                    import_name_parallel = f"{call.name.name.lower()}_parallel.intfb.h"
                    #                    if import_name not in c_imports:
                    new_c_import = c_import.clone(module=import_name_parallel)
                    map_routine["imports_mapper"][c_import] = new_c_import
                    # map_routine['c_imports_parallel'].append(new_c_import)
                else:
                    raise Exception(f"{call.name.name} should have an interface block")

                new_call = call.clone(
                    name=sym.ProcedureSymbol(name=f"{call.name.name}_PARALLEL"),
                    arguments=as_tuple(new_arguments),
                )
                call._update(
                    name=sym.ProcedureSymbol(name=f"{call.name.name}_PARALLEL"),
                    arguments=as_tuple(new_arguments),
                )
                call_mapper[call] = new_call
        # pointers on field associated to derived type musn't be added to routine dcl
        map_region["map_derived"] = [] 


        self.add_derived_to_map_routine(map_routine, map_region)
        map_routine["call_mapper"] = call_mapper

    def add_derived_to_map_routine(self, map_routine, map_region):
        """
        Add derived types that were found in the region to all the derived types of the routine.
        """
        if bool(map_region["map_derived"]):
            map_routine["map_derived"] = map_region["map_derived"]|map_routine["map_derived"]



# TODO : fix create ylstack!!!!!!!!!!
