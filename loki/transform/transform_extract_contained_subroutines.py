# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.subroutine import Subroutine
from loki.expression import (
    FindVariables,
    FindInlineCalls,
)
from loki.ir import (
    CallStatement, VariableDeclaration, Import,
    CaseInsensitiveDict, DerivedType
)
from loki.visitors import (
    Transformer, FindNodes, 
)
from functools import reduce
from loki.transform import resolve_associates

def extract_contained_subroutines(routine):
    new_routines = []
    for r in routine.subroutines:
        new_routines += [extract_contained_subroutine(routine, r.name)]
    routine.contains = Transformer({r: None for r in routine.subroutines}).visit(routine.contains)
    return new_routines

def extract_contained_subroutine(routine, name):
    """
        TODO: Update docs of this function.
    """
    inner = routine.subroutine_map[name] # Fetch the subroutine to extract (or crash with 'KeyError').
    resolve_associates(inner) # Resolving associate statements is done to simplify logic in future steps.

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
    outer_spec_vars = FindVariables().visit(routine.spec) 
    outer_imports = FindNodes(Import).visit(routine.spec)
    outer_import_vars = FindVariables().visit(outer_imports)
    outer_import_names = [var.name.lower() for var in outer_import_vars]
    
    gvar_names = list(gvar_names)
    # This is the main loop that processes `gvar_names`, i.e decides how each global variable in 
    # `inner` should be resolved.
    # NOTE: It is very important to process `gvar_names` as a list (with an order) because the
    # following loop may add new items that to `gvar_names` that need to be resolved. 
    for gn in gvar_names:  
        same_named_globals = [var.clone() for var in outer_spec_vars if var.name.lower() == gn]
        if len(same_named_globals) != 1:
            # If the code crashes to the next assert, there is a bug in the implementation of this function.
            assert len(same_named_globals) == 0 
            msg = f"Could not resolve symbol '{gn}' "\
            f"in the contained subroutine '{inner.name}'. "\
            f"The symbol '{gn}' is undefined in the parent subroutine '{routine.name}'."
            raise RuntimeError(msg)

        var = same_named_globals[0].rescope(inner) 
        if not var.name.lower() in outer_import_names:
            # Global is not an import, so it needs to be added as an argument. 

            # If there is no intent, set intent as 'inout', so that variable can be (possibly) modified inside
            # the lifted `inner` routine. Without further analysis, it is not possibly to
            # say whether this is the "true" intent.
            if not var.type.intent:
                var.type = var.type.clone(intent = 'inout') 
            defs_to_add.append(var)

            # Subtle cornercase: the definition of `var` itself might contain further variables that
            # also need to be defined in `inner`. Therefore, here we search `var` for further variables
            # and add them to `gvar_names`, if they are not there.
            further_vars = FindVariables().visit(var)
            for fv in further_vars: # NOTE: var itself is also in `further_vars`
                if not fv.name.lower() in gvar_names:
                    gvar_names.append(fv.name.lower())

            # Related to the above case: 
            # This checks if `var` has a derived type whose definition is also needed.
            is_derived_type = isinstance(var.type.dtype, DerivedType)
            dtname = var.type.dtype.name.lower()
            if is_derived_type and not dtname in gvar_names:
                gvar_names.append(dtname)   
                    
        else:
            # Global is an import, so need to add the import. 
            # Change the import to only include the symbols that are needed. 
            matching_import = routine.import_map[var.name] 
            imports_to_add.append(matching_import.clone(symbols = (var.clone(),)))

    # Change `inner` to take the globals as argument or add the corresponding import
    # for the global. After these lines, `inner` should have no global variables or there is a bug. 
    inner.arguments += tuple(defs_to_add)
    inner.variables += tuple(defs_to_add)
    inner.spec.prepend(imports_to_add)

    # Construct transformation map to modify all calls to `inner` in the parent to include the globals. 
    # Here the dimensions from `defs_to_add` are dropped, since they should not appear in the call. 
    call_transmap = {}
    all_calls = FindNodes(CallStatement).visit(routine.body)
    inner_calls = [call for call in all_calls if call.name.name == inner.name]
    for call in inner_calls:
        newargs = call.arguments + tuple(map(lambda x: x.clone(dimensions = None), defs_to_add)) 
        call_transmap[call] = call.clone(arguments = newargs) 

    # Transform calls in parent.
    routine.body = Transformer(call_transmap).visit(routine.body)

    return inner


