# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.subroutine import Subroutine
from loki.expression import (
    FindVariables, FindInlineCalls, SubstituteExpressions
)
from loki.ir import (
    CallStatement, Import, DerivedType
)
from loki.visitors import (
    Transformer, FindNodes,
)
from loki.transform import resolve_associates

def extract_contained_procedures(procedure):
    """
    This transform creates "standalone" :any:`Subroutine`s
    from the contained procedures (subroutines or functions) of ``procedure``.

    A list of :any:`Subroutine`s corresponding to each contained subroutine of
    ``procedure`` is returned and ``procedure`` itself is
    modified (see below).
    This function does the following transforms:
    1. all global bindings from the point of view of the contained procedures(s) are introduced
    as imports or dummy arguments to the modified contained procedures(s) to make them standalone.
    2. all calls or invocations of the contained procedures in parent are modified accordingly.
    3. All procedures are removed from the CONTAINS block of ``procedure``.

    As a basic example of this transformation, the Fortran subroutine:
    .. code-block::
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
    is modified to:
    .. code-block::
        subroutine outer()
            integer :: y
            integer :: o
            o = 0
            y = 1
            call inner(o, y) ! 'y' now passed as argument.
            contains
        end subroutine outer
    and the (modified) child:
    .. code-block::
        subroutine inner(o, y)
               integer, intent(inout) :: o
               integer, intent(inout) :: y
               integer :: x
               x = 4
               o = x + y ! Note, 'y' is no longer "global"
        end subroutine inner
    is returned.
    """
    new_procedures = []
    for r in procedure.subroutines:
        new_procedures += [extract_contained_procedure(procedure, r.name)]

    # Remove all subroutines (or functions) from the CONTAINS section.
    newbody = tuple(r for r in procedure.contains.body if not isinstance(r, Subroutine))
    procedure.contains = procedure.contains.clone(body = newbody)
    return new_procedures

def extract_contained_procedure(procedure, name):
    """
    Extract a single contained procedure with name ``name`` from the parent procedure ``procedure``.

    This function does the following transforms:
    1. all global bindings from the point of view of the contained procedure are introduced
    as imports or dummy arguments to the modified contained procedures returned from this function.
    2. all calls or invocations of the contained procedure in the parent are modified accordingly.

    See also the "driver" function ``extract_contained_procedures``, which applies this function to each
    contained procedure of a parent procedure and additionally empties the CONTAINS section of subroutines.
    """
    inner = procedure.subroutine_map[name] # Fetch the subprocedure to extract (or crash with 'KeyError').
    resolve_associates(inner) # Resolving associate statements is done to simplify logic in future steps.

    # Find all local variables names.
    lvar_names = [var.name.lower() for var in inner.symbols]

    # Find all variables (names), including also:
    # 1. Declared non-array and array variables in spec.
    # 2. Variables in body.
    # 3. Variable kinds.
    # 4. Array dimension variables.
    all_vars_body = FindVariables().visit(inner.body)
    all_vars_spec = FindVariables().visit(inner.spec)
    all_vars = set(var for var in all_vars_body.union(all_vars_spec))
    kinds = set()
    for var in all_vars:
        if var.type.kind:
            kinds.add(var.type.kind)
    all_vars = all_vars.union(kinds)

    # Get the names of all global variables from the perspective of `inner`.
    gvar_names = set()
    for var in all_vars:
        # The next while loop is in case `var` is a member of a derived type.
        # For example for variable named `a%b%c`,
        # the global variable we want is `a`, not `a%b%c` itself.
        varname = var.name.lower()
        vartmp = var
        while vartmp.parent is not None:
            varname = vartmp.parent.name.lower()
            vartmp = vartmp.parent
        if not varname in lvar_names: # Only add variables that are not defined locally.
            gvar_names.add(varname)
    # `gvar_names` might still contain inline calls to functions (say, log or exp)
    # that show up as variables. Here they get removed.
    inline_call_names = set(call.name.lower() for call in FindInlineCalls().visit(inner.body))
    gvar_names = gvar_names.difference(inline_call_names)

    defs_to_add = [] # List of new arguments that should be added to the inner procedure.
    imports_to_add = [] # List of imports that should be added to the inner procedure.
    outer_spec_vars = FindVariables().visit(procedure.spec)
    outer_imports = FindNodes(Import).visit(procedure.spec)
    outer_import_vars = FindVariables().visit(outer_imports)
    outer_import_names = [var.name.lower() for var in outer_import_vars]

    # This is the main loop that processes `gvar_names`, i.e decides how each global variable in
    # `inner` should be resolved.
    # NOTE: It is very important to process `gvar_names` as a list (with an order) because the
    # following loop may add new items to `gvar_names` that need to be resolved.
    # (in case a definition in the parent depends on something)
    gvar_names = list(gvar_names)
    for gn in gvar_names:
        same_named_globals = [var for var in outer_spec_vars if var.name.lower() == gn]
        if len(same_named_globals) != 1:
            # If the code crashes to the next assert, there is a bug in the implementation of this function.
            assert len(same_named_globals) == 0
            msg = f"Could not resolve symbol '{gn}' "\
            f"in the contained subroutine '{inner.name}'. "\
            f"The symbol '{gn}' is undefined in the parent subroutine '{procedure.name}'."
            raise RuntimeError(msg)

        var = same_named_globals[0].rescope(inner)
        if not var.name.lower() in outer_import_names:
            # Global is not an import, so it needs to be added as an argument.

            # If there is no intent, set intent as 'inout', so that variable can be (possibly) modified inside
            # the extracted `inner` procedure. Without further analysis, it is not possibly to
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
            matching_import = procedure.import_map[var.name]
            imports_to_add.append(matching_import.clone(symbols = (var,)))

    # Change `inner` to take the globals as argument or add the corresponding import
    # for the global. After these lines, `inner` should have no global variables or there is a bug.
    inner.arguments += tuple(defs_to_add)
    inner.variables += tuple(defs_to_add)
    inner.spec.prepend(imports_to_add)

    # Construct transformation map to modify all calls / function invocations to `inner` in the parent to
    # include the globals.
    # Here the dimensions from `defs_to_add` are dropped, since they should not appear in the call.
    # Functions need to be treated differently than Subroutines, hence the "mapping object" `m`.
    m = {
        'call_finder': FindInlineCalls() if inner.is_function else FindNodes(CallStatement),
        'argname': 'parameters' if inner.is_function else 'arguments',
        'transformer': SubstituteExpressions if inner.is_function else Transformer
    }
    call_transmap = {}
    all_calls = m['call_finder'].visit(procedure.body)
    inner_calls = (call for call in all_calls if call.routine == inner)
    for call in inner_calls:
        newargs = getattr(call, m['argname']) + tuple(map(lambda x: x.clone(dimensions = None), defs_to_add))
        call_transmap[call] = call.clone(**{m['argname']: newargs})

    # Transform calls in parent.
    procedure.body = m['transformer'](call_transmap).visit(procedure.body)

    return inner
