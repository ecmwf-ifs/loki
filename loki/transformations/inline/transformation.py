# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict

from loki.batch import Transformation
from loki.transformations.remove_code import do_remove_dead_code
from loki.ir import pragmas_attached, CallStatement, FindNodes, is_loki_pragma
from loki.types import BasicType

from loki.tools.util import is_iterable, as_tuple, CaseInsensitiveDict
from loki.transformations.inline.constants import inline_constant_parameters
from loki.transformations.inline.functions import (
    inline_elemental_functions, inline_statement_functions
)
from loki.transformations.inline.procedures import (
    inline_internal_procedures, inline_marked_subroutines,
    resolve_sequence_association_for_inlined_calls
)


__all__ = ['InlineTransformation']


class InlineTransformation(Transformation):
    """
    :any:`Transformation` class to apply several types of source inlining
    when batch-processing large source trees via the :any:`Scheduler`.

    Parameters
    ----------
    inline_constants : bool
        Replace instances of variables with known constant values by
        :any:`Literal` (see :any:`inline_constant_parameters`); default: False.
    inline_elementals : bool
        Replaces :any:`InlineCall` expression to elemental functions
        with the called function's body (see :any:`inline_elemental_functions`);
        default: True.
    inline_stmt_funcs: bool
        Replaces  :any:`InlineCall` expression to statement functions
        with the corresponding rhs of the statement function if
        the statement function declaration is available; default: False.
    inline_internals : bool
        Inline internal procedure (see :any:`inline_internal_procedures`);
        default: False.
    inline_marked : bool
        Inline :any:`Subroutine` objects marked by pragma annotations
        (see :any:`inline_marked_subroutines`); default: True.
    remove_dead_code : bool
        Perform dead code elimination, where unreachable branches are
        trimmed from the code (see :any:`dead_code_elimination`); default: True
    allowed_aliases : tuple or list of str or :any:`Expression`, optional
        List of variables that will not be renamed in the parent scope during
        internal and pragma-driven inlining.
    adjust_imports : bool
        Adjust imports by removing the symbol of the inlined routine or adding
        imports needed by the imported routine (optional, default: True)
    external_only : bool, optional
        Do not replace variables declared in the local scope when
        inlining constants (default: True)
    resolve_sequence_association: bool
        Resolve sequence association for routines that contain calls to inline (default: False)
    """

    # Ensure correct recursive inlining by traversing from the leaves
    reverse_traversal = True

    # This transformation will potentially change the edges in the callgraph
    creates_items = False

    def __init__(
            self, inline_constants=False, inline_elementals=True,
            inline_stmt_funcs=False, inline_internals=False,
            inline_marked=True, remove_dead_code=True,
            allowed_aliases=None, adjust_imports=True,
            external_only=True, resolve_sequence_association=False
    ):
        self.inline_constants = inline_constants
        self.inline_elementals = inline_elementals
        self.inline_stmt_funcs = inline_stmt_funcs
        self.inline_internals = inline_internals
        self.inline_marked = inline_marked
        self.remove_dead_code = remove_dead_code
        self.allowed_aliases = allowed_aliases
        self.adjust_imports = adjust_imports
        self.external_only = external_only
        self.resolve_sequence_association = resolve_sequence_association
        if self.inline_marked:
            self.creates_items = True

    def transform_subroutine(self, routine, **kwargs):

        # Resolve sequence association in calls that are about to be inlined.
        # This step runs only if all of the following hold:
        # 1) it is requested by the user
        # 2) inlining of "internals" or "marked" routines is activated
        # 3) there is an "internal" or "marked" procedure to inline.
        if self.resolve_sequence_association:
            resolve_sequence_association_for_inlined_calls(
                routine, self.inline_internals, self.inline_marked
            )

        # Replace constant parameter variables with explicit values
        if self.inline_constants:
            inline_constant_parameters(routine, external_only=self.external_only)

        # Inline elemental functions
        if self.inline_elementals:
            inline_elemental_functions(routine)

        # Inline Statement Functions
        if self.inline_stmt_funcs:
            inline_statement_functions(routine)

        # Inline internal (contained) procedures
        if self.inline_internals:
            inline_internal_procedures(routine, allowed_aliases=self.allowed_aliases)

        # Inline explicitly pragma-marked subroutines
        if self.inline_marked:
            inline_marked_subroutines(
                routine, allowed_aliases=self.allowed_aliases,
                adjust_imports=self.adjust_imports
            )

        # After inlining, attempt to trim unreachable code paths
        if self.remove_dead_code:
            do_remove_dead_code(routine)

    def plan_subroutine(self, routine, **kwargs):
        item = kwargs.get('item')
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = sub_sgraph.successors(item) if sub_sgraph is not None else ()
        successor_map = CaseInsensitiveDict(
            (successor.local_name, successor) for successor in successors
        )
        # from loki import pprint
        # print('------------------------------')
        # print(f"routine: {routine}")
        # print(f"{pprint(routine.ir)}")
        # print('------------------------------')
        
        # calls = FindNodes(CallStatement).visit(routine.ir)
        # print(f"INLINE plan_subroutine routine {routine} | calls: {calls}")
        with pragmas_attached(routine, node_type=CallStatement):
            # Group the marked calls by callee routine
            # call_sets = defaultdict(list)
            # no_call_sets = defaultdict(list)
            inline_calls = set()
            calls = FindNodes(CallStatement).visit(routine.ir)
            for call in calls: # FindNodes(CallStatement).visit(routine.body):
                # if call.routine == BasicType.DEFERRED:
                #     continue

                if is_loki_pragma(call.pragma, starts_with='inline'):
                    # call_sets[call.routine].append(call)
                    inline_calls.add(call)
                # else:
                #     no_call_sets[call.routine].append(call)
        # print(f"INLINE plan_subroutine routine {routine} | calls: {calls} | call_sets: {call_sets} | no_call_sets: {no_call_sets}")
        # print(f"INLINE plan_subroutine routine {routine} | call_sets: {call_sets} | no_call_sets: {no_call_sets}")
        # print(f"INLINE plan_subroutine routine {routine} | inline_calls: {inline_calls}")
        # print(f"  successors: {successor_map}")
        inline_items = [successor_map[call.name] for call in inline_calls]
        # print(f"  inline_items: {inline_items}")
        # new_item.plan_data.setdefault('additional_dependencies', ())

        if inline_items:
            item.plan_data.setdefault('removed_dependencies', ())
            item.plan_data.setdefault('additional_dependencies', ())
            item.plan_data['removed_dependencies'] += as_tuple(inline_items)
            additional_dep = ()
            for inline_item in inline_items:
                print(f"[{item}] - inline_item: {inline_item} | removed dep {inline_item.plan_data.get('removed_dependencies', None)}")
                inlined_successors = sub_sgraph.successors(inline_item) + inline_item.plan_data.get('additional_dependencies', ())
                print(f"  successors: {inlined_successors}")
                for inlined_successor in inlined_successors:
                    # print(f"[{item}] inlined_successor: {inlined_successor}") #  | removed inlined successor: {inlined_successor.plan_data['removed_dependencies']}")
                    # if 'removed_dependencies' in inlined_successor.plan_data:
                    #     print(f"  removed inlined successor: {inlined_successor.plan_data['removed_dependencies']}")
                    # if 'removed_dependencies' in inlined_successor.plan_data and inlined_successor not in inlined_successor.plan_data['removed_dependencies']:
                    # print(f"ADDING ADDITIONAL DEP {inlined_successors}")
                    if inlined_successor not in inline_item.plan_data.get('removed_dependencies', ()):
                        additional_dep += (inlined_successor,)
            print(f"  additional_dependencies: {as_tuple(set(additional_dep))}")
            item.plan_data['additional_dependencies'] += as_tuple(set(additional_dep))
            print(f"  item.plan_data['removed_dependencies']: {item.plan_data['removed_dependencies']}")

