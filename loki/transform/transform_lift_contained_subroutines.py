from loki import (
    Subroutine, FindNodes, FindVariables,
    CallStatement, VariableDeclaration, Import,
    FindInlineCalls, Transformer, CaseInsensitiveDict
)
from functools import reduce
from loki.transform import resolve_associates

def lift_contained_subroutines(outer):
    def find_var_defining_import(varname: str, impos):
        for impo in impos: 
            varnames_in_import = [var.name.lower() for var in FindVariables().visit(impo)]
            if varname.lower() in varnames_in_import:
                return impo.clone()
        raise Exception(f"variable '{varname}' was not found from the specified import objects.")

    outer = outer.clone()
    # Get all inner routines and resolve all associate statements in them.
    # This is strictly not needed, but is done to simplify the logic in later steps.
    inner_routines = []
    for r in outer.contains.body:
        if isinstance(r, Subroutine):
            newinner = r.clone()
            resolve_associates(newinner)
            inner_routines.append(newinner)
    assert len(inner_routines) > 0

    # Get all calls.
    all_calls = FindNodes(CallStatement).visit(outer.body)
    call_transmap = {}

    # Construct a list of routines returned from this function.
    routines = []

    outer_decls = FindNodes(VariableDeclaration).visit(outer.spec)

    for inner in inner_routines:
        # Find all local variables (names). 
        # This includes:
        # 1. Declared non-array variables in spec.
        # 2. Array-variables, EXCLUDING any variables that only exist in array dimensions.
        # NOTE: Such dimension variables will be in group 1, if they are actually defined.
        # 3. Imported variables. NOTE: Here collecting also imports that correspond to functions,
        # but they are removed later. 
        ldecls = FindNodes(VariableDeclaration).visit(inner.spec)
        limports_vars = FindVariables().visit(FindNodes(Import).visit(inner.spec))
        lvars = list(reduce(lambda x, y: x + y, (decl.symbols for decl in ldecls))) 
        lvars = set(var.clone() for var in lvars + list(limports_vars))
        lvar_names = [var.name.lower() for var in lvars]

        # Find all variables (names), including also:
        # 1. Declared non-array and array variables in spec.
        # 2. Variables in body.
        # 3. Variable kinds.
        # 4. Array dimension variables.
        all_vars_body = FindVariables().visit(inner.body)
        all_vars_spec = FindVariables().visit(inner.spec)
        all_vars = set(var.clone() for var in all_vars_body.union(all_vars_spec))
        kinds = set()
        for var in all_vars:
            if var.type.kind:
                kinds.add(var.type.kind.clone())
        all_vars = all_vars.union(kinds)

        # Get the names of all global variables from the perspective of `inner`. 
        gvar_names = set()
        for var in all_vars:
            # The next while loop is in case `var` is a member of a derived type. 
            # For example for variable named `a%b%c`, 
            # the global variable we want is `a`, not `a%b%c` itself.
            varname = var.name.lower()
            vartmp = var.clone()
            while vartmp.parent is not None:
                varname = vartmp.parent.name.lower()
                vartmp = vartmp.parent
            if not varname in lvar_names: # Only add variables that are not defined locally.
                gvar_names.add(varname)

        # `gvar_names` might still contain inline calls to functions (say, log or exp) 
        # that show up as variables. Here they get removed.
        inline_call_names = set([call.name.lower() for call in FindInlineCalls().visit(inner.body)])
        gvar_names = gvar_names.difference(inline_call_names)

        defs_to_add = [] # List of new arguments that should be added to the inner routine. 
        imports_to_add = [] # List of imports that should be added to the inner routine. 
        outer_spec_vars = FindVariables().visit(outer.spec) 
        outer_imports = FindNodes(Import).visit(outer.spec)
        outer_import_vars = FindVariables().visit(outer_imports)
        outer_import_names = [var.name.lower() for var in outer_import_vars]
        
        global_vars = CaseInsensitiveDict()
        inner_varnames = [var.name.lower() for var in inner.variables]
        for varname in outer.variable_map:
            if not (varname.lower() in inner_varnames):
                global_vars[varname] = outer.variable_map[varname].clone() 
        breakpoint()
        gvar_names = list(gvar_names)
        # This is the main loop that processes `gvar_names`, i.e decides how each global variable in 
        # `inner` should be resolved.
        # NOTE: It is very important to process `gvar_names` as a list (with an order) so that new elements may 
        # be added to the end (if necessary) in the following loop.
        for gn in gvar_names:  
            same_named_globals = [var.clone() for var in outer_spec_vars if var.name.lower() == gn]
            assert len(same_named_globals) == 1 # The above should be length one, or there is a bug somewhere.
            var = same_named_globals[0].clone() 
            #var.type = same_named_globals[0].type.clone() 
            if not var.name.lower() in outer_import_names:
                # Global is not an import, so need to add it as an argument.

                # If there is no intent, set intent as 'inout', so that variable can be (possibly) modified inside
                # the lifted `inner` routine. Without further analysis, it is not possibly to
                # say whether this is the true intent.
                if not var.type.intent:
                    var.type = var.type.clone(intent = 'inout') 
                defs_to_add.append(var)

                # Subtle cornercase: the definition of `var` itself might contain further variables that
                # also need to be defined in `inner`. Therefore, search `var` for further variables
                # and add them to `gvar_names`, if they are not there.
                further_vars = FindVariables().visit(var)
                for fv in further_vars: # NOTE: var itself is also in `further_vars`
                    if not fv.name.lower() in gvar_names:
                        gvar_names.append(fv.name.lower())

                # Related to the above case: 
                # this is an ugly hack to check if `var` has a derived type whose definition is also needed.
                dtname = var.type.dtype.name.lower()
                do_not_consider = ('integer', 'logical', 'real', 'complex', 'character')
                if not (dtname in do_not_consider or dtname in gvar_names):
                    gvar_names.append(dtname)   
                        
            else:
                # Global is an import, so need to add the import. 
                # Change the import to only include the symbols that are needed. 
                matching_import = find_var_defining_import(var.name, outer_imports)
                imports_to_add.append(matching_import.clone(symbols = (var.clone(),)))

        # Change `inner` to take the globals as argument or add the corresponding import
        # for the global. After these lines, `inner` should have no global variables or there is a bug. 
        inner.arguments += tuple(defs_to_add)
        inner.variables += tuple(defs_to_add)
        inner.spec.prepend(imports_to_add)

        # Construct transformation map to modify all calls to `inner` to include the globals. 
        # Here the dimensions from `defs_to_add` are dropped, since they should not appear in the call. 
        inner_calls = [call for call in all_calls if call.name.name == inner.name]
        for call in inner_calls:
            newargs = call.arguments + tuple(map(lambda x: x.clone(dimensions = None), defs_to_add)) 
            call_transmap[call] = call.clone(arguments = newargs) 

        routines.append(inner) 

    # Transform calls in `outer`.
    outer.body = Transformer(call_transmap).visit(outer.body)

    # Remove contained subroutines from `outer`.
    outer.contains = None

    routines.append(outer)
    return routines

