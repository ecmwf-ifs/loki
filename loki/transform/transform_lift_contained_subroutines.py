from loki import (
    Subroutine, FindNodes, FindVariables,
    CallStatement, VariableDeclaration, Import,
    FindInlineCalls, Transformer, CaseInsensitiveDict,
    DerivedType
)
from functools import reduce
from loki.transform import resolve_associates

def lift_contained_subroutines(routine):
    """
    This transform creates "standalone" `Subroutine`s from the contained subroutines of `routine`.
    A list of `Subroutine`s is returned, where the first `Subroutine` is the modified parent (i.e `routine` itself),
    and the modified contained subroutines come next.
    In summary, this function does the following transforms:
    1. all global bindings from the point of view of the contained subroutine(s) are introduced 
    as imports or dummy arguments to the modified contained subroutine(s) to make them standalone.
    2. all calls to the contained subroutines in parent are modified accordingly.

    As a basic example of this transformation, the Fortran subroutine:
    ```
    subroutine outer()
        integer :: y
        integer :: o
        o = 0
        y = 1
        call inner(o)
        contains
        subroutine inner(o)
           integer, intent(inout) :: o
           integer :: x
           x = 4
           o = x + y ! Note, 'y' is "global" here!
        end subroutine inner
    end subroutine outer
    ```
    is transformed to (modified) parent:
    ```
    subroutine outer()
        integer :: y
        integer :: o
        o = 0
        y = 1
        call inner(o, y) ! 'y' now passed as argument.
        contains
    end subroutine outer
    ```
    and the (modified) child:
    ```
    subroutine inner(o, y)
           integer, intent(inout) :: o
           integer, intent(inout) :: y
           integer :: x
           x = 4
           o = x + y ! Note, 'y' is no longer "global"
    end subroutine inner
    ```
    """

    def find_var_defining_import(varname: str, impos):
        for impo in impos: 
            varnames_in_import = [var.name.lower() for var in FindVariables().visit(impo)]
            if varname.lower() in varnames_in_import:
                return impo.clone()
        raise Exception(f"variable '{varname}' was not found from the specified import objects.")

    routine = routine.clone()

    # Get all inner routines and resolve all associate statements in them.
    # This is strictly not needed, but is done to simplify the logic in later steps.
    inner_routines = []
    for r in routine.contains.body:
        if isinstance(r, Subroutine):
            newinner = r.clone()
            resolve_associates(newinner)
            inner_routines.append(newinner)
    assert len(inner_routines) > 0

    # Get all calls.
    all_calls = FindNodes(CallStatement).visit(routine.body)
    call_transmap = {}

    # Construct a list of routines returned from this function.
    routines = []

    outer_decls = FindNodes(VariableDeclaration).visit(routine.spec)

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
                msg = f"Could not resolve the symbol '{gn}' "\
                f"in the contained subroutine '{inner.name}'. "\
                f"The symbol '{gn}' is undefined in the parent subroutine '{routine.name}'."
                raise Exception(msg)

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

    # Transform calls in `routine`.
    routine.body = Transformer(call_transmap).visit(routine.body)

    # Remove contained subroutines from `routine`.
    contains_body = tuple(n for n in routine.contains.body if not isinstance(n, Subroutine))
    routine.contains._update(body = contains_body)
    routines.insert(0, routine) # Insert the parent to the beginning of returned routines.
    return routines

