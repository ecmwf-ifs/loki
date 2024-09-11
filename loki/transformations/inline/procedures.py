# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict, ChainMap

from loki.ir import (
    Import, Comment, VariableDeclaration, CallStatement, Transformer,
    FindNodes, FindVariables, FindInlineCalls, SubstituteExpressions,
    pragmas_attached, is_loki_pragma, Interface, Pragma
)
from loki.expression import symbols as sym
from loki.types import BasicType
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.logging import error
from loki.subroutine import Subroutine

from loki.transformations.sanitise import transform_sequence_association_append_map
from loki.transformations.utilities import (
    single_variable_declaration, recursive_expression_map_update
)


__all__ = [
    'inline_internal_procedures', 'inline_member_procedures',
    'inline_marked_subroutines',
    'resolve_sequence_association_for_inlined_calls'
]


def resolve_sequence_association_for_inlined_calls(routine, inline_internals, inline_marked):
    """
    Resolve sequence association in calls to all member procedures (if ``inline_internals = True``)
    or in calls to procedures that have been marked with an inline pragma (if ``inline_marked = True``).
    If both ``inline_internals`` and ``inline_marked`` are ``False``, no processing is done.
    """
    call_map = {}
    with pragmas_attached(routine, node_type=CallStatement):
        for call in FindNodes(CallStatement).visit(routine.body):
            condition = (
                (inline_marked and is_loki_pragma(call.pragma, starts_with='inline')) or
                (inline_internals and call.routine in routine.routines)
            )
            if condition:
                if call.routine == BasicType.DEFERRED:
                    # NOTE: Throwing error here instead of continuing, because the user has explicitly
                    # asked sequence assoc to happen with inlining, so source for routine should be
                    # found in calls to be inlined.
                    raise ValueError(
                        f"Cannot resolve sequence association for call to ``{call.name}`` " +
                        f"to be inlined in routine ``{routine.name}``, because " +
                        f"the ``CallStatement`` referring to ``{call.name}`` does not contain " +
                        "the source code of the procedure. " +
                        "If running in batch processing mode, please recheck Scheduler configuration."
                    )
                transform_sequence_association_append_map(call_map, call)
        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)


def map_call_to_procedure_body(call, caller, callee=None):
    """
    Resolve arguments of a call and map to the called procedure body.

    Parameters
    ----------
    call : :any:`CallStatment` or :any:`InlineCall`
         Call object that defines the argument mapping
    caller : :any:`Subroutine`
         Procedure (scope) into which the callee's body gets mapped
    callee : :any:`Subroutine`, optional
         Procedure (scope) called. Provide if it differs from
         call.routine.
    """

    def _map_unbound_dims(var, val):
        """
        Maps all unbound dimension ranges in the passed array value
        ``val`` with the indices from the local variable ``var``. It
        returns the re-mapped symbol.

        For example, mapping the passed array ``m(:,j)`` to the local
        expression ``a(i)`` yields ``m(i,j)``.
        """
        new_dimensions = list(val.dimensions)

        indices = [index for index, dim in enumerate(val.dimensions) if isinstance(dim, sym.Range)]

        for index, dim in enumerate(var.dimensions):
            new_dimensions[indices[index]] = dim

        return val.clone(dimensions=tuple(new_dimensions))

    # Get callee from the procedure type
    callee = callee or call.routine
    if callee is BasicType.DEFERRED:
        error(
            '[Loki::TransformInline] Need procedure definition to resolve '
            f'call to {call.name} from {caller}'
        )
        raise RuntimeError('Procedure definition not found! ')

    argmap = {}
    callee_vars = FindVariables().visit(callee.body)

    # Match dimension indexes between the argument and the given value
    # for all occurences of the argument in the body
    for arg, val in call.arg_map.items():
        if isinstance(arg, sym.Array):
            # Resolve implicit dimension ranges of the passed value,
            # eg. when passing a two-dimensional array `a` as `call(arg=a)`
            # Check if val is a DeferredTypeSymbol, as it does not have a `dimensions` attribute
            if not isinstance(val, sym.DeferredTypeSymbol) and val.dimensions:
                qualified_value = val
            else:
                qualified_value = val.clone(
                    dimensions=tuple(sym.Range((None, None)) for _ in arg.shape)
                )

            # If sequence association (scalar-to-array argument passing) is used,
            # we cannot determine the right re-mapped iteration space, so we bail here!
            if not any(isinstance(d, sym.Range) for d in qualified_value.dimensions):
                error(
                    '[Loki::TransformInline] Cannot find free dimension resolving '
                    f' array argument for value "{qualified_value}"'
                )
                raise RuntimeError(
                    f'[Loki::TransformInline] Cannot resolve procedure call to {call.name}'
                )
            arg_vars = tuple(v for v in callee_vars if v.name == arg.name)
            argmap.update((v, _map_unbound_dims(v, qualified_value)) for v in arg_vars)
        else:
            argmap[arg] = val

    # Deal with PRESENT check for optional arguments
    present_checks = tuple(
        check for check in FindInlineCalls().visit(callee.body) if check.function == 'PRESENT'
    )
    present_map = {
        check: sym.Literal('.true.') if check.arguments[0] in [arg.name for arg in call.arg_map]
                                     else sym.Literal('.false.')
        for check in present_checks
    }
    argmap.update(present_map)

    # Recursive update of the map in case of nested variables to map
    argmap = recursive_expression_map_update(argmap, max_iterations=10)

    # Substitute argument calls into a copy of the body
    callee_body = SubstituteExpressions(argmap, rebuild_scopes=True).visit(
        callee.body.body, scope=caller
    )

    # Remove 'loki routine' pragmas
    callee_body = Transformer(
        {pragma: None for pragma in FindNodes(Pragma).visit(callee_body)
         if is_loki_pragma(pragma, starts_with='routine')}
    ).visit(callee_body)

    # Inline substituted body within a pair of marker comments
    comment = Comment(f'! [Loki] inlined child subroutine: {callee.name}')
    c_line = Comment('! =========================================')
    return (comment, c_line) + as_tuple(callee_body) + (c_line, )


def inline_subroutine_calls(routine, calls, callee, allowed_aliases=None):
    """
    Inline a set of call to an individual :any:`Subroutine` at source level.

    This will replace all :any:`Call` objects to the specified
    subroutine with an adjusted equivalent of the member routines'
    body. For this, argument matching, including partial dimension
    matching for array references is performed, and all
    member-specific declarations are hoisted to the containing
    :any:`Subroutine`.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to inline all calls to the member routine
    calls : tuple or list of :any:`CallStatement`
    callee : :any:`Subroutine`
        The called target subroutine to be inlined in the parent
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope, even
        if they alias with a local declaration.
    """
    allowed_aliases = as_tuple(allowed_aliases)

    # Ensure we process sets of calls to the same callee
    assert all(call.routine == callee for call in calls)
    assert isinstance(callee, Subroutine)

    # Prevent shadowing of callee's variables by renaming them a priori
    parent_variables = routine.variable_map
    duplicates = tuple(
        v for v in callee.variables
        if v.name in parent_variables and v.name.lower() not in callee._dummies
    )
    # Filter out allowed aliases to prevent suffixing
    duplicates = tuple(v for v in duplicates if v.symbol not in allowed_aliases)
    shadow_mapper = SubstituteExpressions(
        {v: v.clone(name=f'{callee.name}_{v.name}') for v in duplicates}
    )
    callee.spec = shadow_mapper.visit(callee.spec)

    var_map = {}
    duplicate_names = {dl.name.lower() for dl in duplicates}
    for v in FindVariables(unique=False).visit(callee.body):
        if v.name.lower() in duplicate_names:
            var_map[v] = v.clone(name=f'{callee.name}_{v.name}')
    var_map = recursive_expression_map_update(var_map)
    callee.body = SubstituteExpressions(var_map).visit(callee.body)

    # Separate allowed aliases from other variables to ensure clean hoisting
    if allowed_aliases:
        single_variable_declaration(callee, variables=allowed_aliases)

    # Get local variable declarations and hoist them
    decls = FindNodes(VariableDeclaration).visit(callee.spec)
    decls = tuple(d for d in decls if all(s.name.lower() not in callee._dummies for s in d.symbols))
    decls = tuple(d for d in decls if all(s not in routine.variables for s in d.symbols))
    # Rescope the declaration symbols
    decls = tuple(d.clone(symbols=tuple(s.clone(scope=routine) for s in d.symbols)) for d in decls)

    # Find and apply symbol remappings for array size expressions
    symbol_map = dict(ChainMap(*[call.arg_map for call in calls]))
    decls = SubstituteExpressions(symbol_map).visit(decls)

    routine.spec.append(decls)

    # Resolve the call by mapping arguments into the called procedure's body
    call_map = {
        call: map_call_to_procedure_body(call, caller=routine) for call in calls
    }

    # Replace calls to child procedure with the child's body
    routine.body = Transformer(call_map).visit(routine.body)

    # We need this to ensure that symbols, as well as nested scopes
    # are correctly attached to each other (eg. nested associates).
    routine.rescope_symbols()


def inline_internal_procedures(routine, allowed_aliases=None):
    """
    Inline internal subroutines contained in an individual :any:`Subroutine`.

    Please note that internal functions are not yet supported!

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to inline all member routines
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope, even
        if they alias with a local declaration.
    """

    from loki.transformations.inline import inline_functions  # pylint: disable=cyclic-import,import-outside-toplevel

    # Run through all members and invoke individual inlining transforms
    for child in routine.members:
        if child.is_function:
            inline_functions(routine, functions=(child,))
        else:
            calls = tuple(
                call for call in FindNodes(CallStatement).visit(routine.body)
                if call.routine == child
            )
            inline_subroutine_calls(routine, calls, child, allowed_aliases=allowed_aliases)

        # Can't use transformer to replace subroutine/function, so strip it manually
        contains_body = tuple(n for n in routine.contains.body if not n == child)
        routine.contains._update(body=contains_body)


inline_member_procedures = inline_internal_procedures


def inline_marked_subroutines(routine, allowed_aliases=None, adjust_imports=True):
    """
    Inline :any:`Subroutine` objects guided by pragma annotations.

    When encountering :any:`CallStatement` objects that are marked with a
    ``!$loki inline`` pragma, this utility will attempt to replace the call
    with the body of the called procedure and remap all passed arguments
    into the calling procedures scope.

    Please note that this utility requires :any:`CallStatement` objects
    to be "enriched" with external type information.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to look for pragma-marked procedures to inline
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope, even
        if they alias with a local declaration.
    adjust_imports : bool
        Adjust imports by removing the symbol of the inlined routine or adding
        imports needed by the imported routine (optional, default: True)
    """

    with pragmas_attached(routine, node_type=CallStatement):

        # Group the marked calls by callee routine
        call_sets = defaultdict(list)
        no_call_sets = defaultdict(list)
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.routine == BasicType.DEFERRED:
                continue

            if is_loki_pragma(call.pragma, starts_with='inline'):
                call_sets[call.routine].append(call)
            else:
                no_call_sets[call.routine].append(call)

        # Trigger per-call inlining on collected sets
        for callee, calls in call_sets.items():
            if callee:  # Skip the unattached calls (collected under None)
                inline_subroutine_calls(
                    routine, calls, callee, allowed_aliases=allowed_aliases
                )

    # Remove imported symbols that have become obsolete
    if adjust_imports:
        callees = tuple(callee.procedure_symbol for callee in call_sets.keys())
        not_inlined = tuple(callee.procedure_symbol for callee in no_call_sets.keys())

        import_map = {}
        for impt in FindNodes(Import).visit(routine.spec):
            # Remove interface header imports
            if any(f'{c.name.lower()}.intfb.h' == impt.module for c in callees):
                import_map[impt] = None

            if any(s.name in callees for s in impt.symbols):
                new_symbols = tuple(
                    s for s in impt.symbols if s.name not in callees or s.name in not_inlined
                )
                # Remove import if no further symbols used, otherwise clone with new symbols
                import_map[impt] = impt.clone(symbols=new_symbols) if new_symbols else None

        # Remove explicit interfaces of inlined routines
        for intf in routine.interfaces:
            if not intf.spec:
                _body = tuple(
	                    s.type.dtype.procedure for s in intf.symbols
	                    if s.name not in callees or s.name in not_inlined
                )
                if _body:
                    import_map[intf] = intf.clone(body=_body)
                else:
                    import_map[intf] = None

        # Now move any callee imports we might need over to the caller
        new_imports = set()
        imported_module_map = CaseInsensitiveDict((im.module, im) for im in routine.imports)
        for callee in call_sets.keys():
            for impt in callee.imports:

                # Add any callee module we do not yet know
                if impt.module not in imported_module_map:
                    new_imports.add(impt)

                # If we're importing the same module, check for missing symbols
                if m := imported_module_map.get(impt.module):
                    _m = import_map.get(m, m)
                    if not all(s in _m.symbols for s in impt.symbols):
                        new_symbols = tuple(s.rescope(routine) for s in impt.symbols)
                        import_map[m] = m.clone(symbols=tuple(set(_m.symbols + new_symbols)))

        # Finally, apply the import remapping
        routine.spec = Transformer(import_map).visit(routine.spec)

        # Add missing explicit interfaces from inlined subroutines
        new_intfs = []
        intf_symbols = routine.interface_symbols
        for callee in call_sets.keys():
            for intf in callee.interfaces:
                for s in intf.symbols:
                    if not s in intf_symbols:
                        new_intfs += [s.type.dtype.procedure,]

        if new_intfs:
            routine.spec.append(Interface(body=as_tuple(new_intfs)))

        # Add Fortran imports to the top, and C-style interface headers at the bottom
        c_imports = tuple(im for im in new_imports if im.c_import)
        f_imports = tuple(im for im in new_imports if not im.c_import)
        routine.spec.prepend(f_imports)
        routine.spec.append(c_imports)
