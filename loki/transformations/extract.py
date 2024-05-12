# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.subroutine import Subroutine
from loki.expression import (
    FindVariables, FindInlineCalls, SubstituteExpressions,
    DeferredTypeSymbol, Array
)
from loki.ir import CallStatement, Transformer, FindNodes
from loki.types import DerivedType


__all__ = ['extract_contained_procedures', 'extract_contained_procedure']


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
    procedure.contains = procedure.contains.clone(body=newbody)
    return new_procedures

def extract_contained_procedure(procedure, name):
    """
    Extract a single contained procedure with name ``name`` from the parent procedure ``procedure``.

    This function does the following transforms:
    1. all global bindings from the point of view of the contained procedure are introduced
    as imports or dummy arguments to the modified contained procedure returned from this function.
    2. all calls or invocations of the contained procedure in the parent are modified accordingly.

    See also the "driver" function ``extract_contained_procedures``, which applies this function to each
    contained procedure of a parent procedure and additionally empties the CONTAINS section of subroutines.
    """
    inner = procedure.subroutine_map[name] # Fetch the subprocedure to extract (or crash with 'KeyError').

    # Check if there are variables that don't have a scope. This means that they are not defined anywhere
    # and execution cannot continue.
    undefined = tuple(v for v in FindVariables().visit(inner.body) if not v.scope)
    if undefined:
        msg = f"The following variables appearing in the contained procedure '{inner.name}' are undefined "
        msg += f"in both '{inner.name}' and the parent procedure '{procedure.name}': "
        for u in undefined:
            msg += f"{u.name}, "
        raise RuntimeError(msg)

    ## PRODUCING VARIABLES TO INTRODUCE AS DUMMY ARGUMENTS TO `inner`.
    # Produce a list of variables defined in the scope of `procedure` that need to be resolved in `inner`'s scope
    # by introducing them as dummy arguments.
    # The second line drops any derived type fields, don't want them, since want to resolve the derived type itself.
    vars_to_resolve = [v for v in FindVariables().visit(inner.body) if v.scope is procedure]
    vars_to_resolve = [v for v in vars_to_resolve if not v.parent]

    # Save any `DeferredTypeSymbol`s for later, they are in fact defined through imports in `procedure`,
    # and therefore not to be added as arguments to `inner`. (the next step removes them from `vars_to_resolve`)
    var_imports_to_add = tuple(v for v in vars_to_resolve if isinstance(v, DeferredTypeSymbol))

    # Lookup the definition of the variables in `vars_to_resolve` from the scope of `procedure`.
    # This provides maximal information on them.
    vars_to_resolve = [proc_var for v in vars_to_resolve if \
        (proc_var := procedure.variable_map.get(v.name))]

    # For each array in `vars_to_resolve`, append any non-literal shape variables to `vars_to_resolve`,
    # if not already there.
    arr_shapes = []
    for var in vars_to_resolve:
        if isinstance(var, Array):
            # Dropping variables with parents here to handle the case that the array dimension(s)
            # are defined through the field of a derived type.
            arr_shapes += list(v for v in FindVariables().visit(var.shape) if not v.parent)
    for v in arr_shapes:
        if v.name not in vars_to_resolve:
            vars_to_resolve.append(v)
    vars_to_resolve = tuple(vars_to_resolve)

    ## PRODUCING IMPORTS TO INTRODUCE TO `inner`.
    # Get all variables from `inner.spec`. Need to search them for resolving kinds and derived types for
    # variables that do not need resolution.
    inner_spec_vars = tuple(FindVariables().visit(inner.spec))

    # Produce derived types appearing in `vars_to_resolve` or in `inner.spec` that need to be resolved
    # from imports of `procedure`.
    dtype_imports_to_add = tuple(v.type.dtype for v in vars_to_resolve + inner_spec_vars \
        if isinstance(v.type.dtype, DerivedType))

    # Produce kinds appearing in `vars_to_resolve` or in `inner.spec` that need to be resolved
    # from imports of `procedure`.
    kind_imports_to_add = tuple(v.type.kind for v in vars_to_resolve + inner_spec_vars \
        if v.type.kind and v.type.kind.scope is procedure)

    # Produce all imports to add.
    # Here the imports are also tidied to only import what is strictly necessary, and with single
    # USE statements for each module.
    imports_to_add = []
    to_lookup_from_imports = dtype_imports_to_add + kind_imports_to_add + var_imports_to_add
    for val in to_lookup_from_imports:
        imp = procedure.import_map[val.name]
        matching_import = tuple(i for i in imports_to_add if i.module == imp.module)
        if matching_import:
            # Have already encountered module name, modify existing.
            matching_import = matching_import[0]
            imports_to_add.remove(matching_import)
            newimport = matching_import.clone(symbols=tuple(set(matching_import.symbols + imp.symbols)))
        else:
            # Have not encountered the module yet, add new one.
            newsyms = tuple(s for s in imp.symbols if s.name == val.name)
            newimport = imp.clone(symbols=newsyms)
        imports_to_add.append(newimport)

    ## MAKING THE CHANGES TO `inner`
    # Change `inner` to take `vars_to_resolve` as dummy arguments and add all necessary imports.
    # Here also rescoping all variables to the scope of `inner` and specifying intent as "inout",
    # if not set in `procedure` scope.
    # Note: After these lines, `inner` should be self-contained or there is a bug.
    inner.arguments += tuple(
        v.clone(type=v.type.clone(intent=v.type.intent or 'inout'), scope=inner)
        for v in vars_to_resolve
    )
    inner.spec.prepend(imports_to_add)

    ## TRANSFORMING CALLS TO `inner` in `procedure`.
    # The resolved variables are all added as keyword arguments to each call.
    # (to avoid further modification of the call if it already happens to contain kwargs).
    # Here any dimensions in the variables are dropped, since they should not appear in the call.
    # Note that functions need different visitors and mappers than subroutines.
    call_map = {}
    if inner.is_function:
        for call in FindInlineCalls().visit(procedure.body):
            if call.routine is inner:
                newkwargs = tuple((v.name, v.clone(dimensions=None, scope=procedure)) for v in vars_to_resolve)
                call_map[call] = call.clone(kw_parameters=call.kwarguments + newkwargs)
        procedure.body = SubstituteExpressions(call_map).visit(procedure.body)
    else:
        for call in FindNodes(CallStatement).visit(procedure.body):
            if call.routine is inner:
                newkwargs = tuple((v.name, v.clone(dimensions=None, scope=procedure)) for v in vars_to_resolve)
                call_map[call] = call.clone(kwarguments=tuple(call.kwarguments) + newkwargs)
        procedure.body = Transformer(call_map).visit(procedure.body)

    return inner
