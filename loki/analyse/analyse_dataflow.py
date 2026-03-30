# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of dataflow analysis schema routines.
"""

from collections import defaultdict
from contextlib import contextmanager
from loki.expression import Array, ProcedureSymbol
import loki.expression.symbols as sym
from loki.expression.symbolic import simplify, is_constant, is_minus_prefix, strip_minus_prefix
from loki.ir.expr_visitors import FindLiterals
from loki.tools import as_tuple, flatten, OrderedSet
from loki.types import BasicType
from loki.ir import (
    Visitor, Transformer, FindVariables, FindInlineCalls, FindTypedSymbols,
    FindNodes, Assignment, Loop
)
from loki.subroutine import Subroutine
from loki.tools.util import CaseInsensitiveDict

__all__ = [
    'dataflow_analysis_attached', 'read_after_write_vars',
    'loop_carried_dependencies', 'classify_array_access_offsets',
    'array_loop_carried_dependencies', 'detect_loop_carry_variables',
    'classify_nonzero_offset_arrays'
]


def strip_nested_dimensions(expr):
    """
    Strip dimensions from array expressions of arbitrary derived-type
    nesting depth.
    """

    if not hasattr(expr, 'parent'):
        return expr
    parent = expr.parent
    if parent:
        parent = strip_nested_dimensions(parent)
    return expr.clone(dimensions=None, parent=parent)


class DataflowAnalysisAttacher(Transformer):
    """
    Analyse and attach in-place the definition, use and live status of
    symbols.

    Parameters
    ----------
    include_literal_kinds : bool (default : True)
       Include kind specifiers for literals in dataflow analysis.
    """

    # group of functions that only query memory properties and don't read/write variable value
    _mem_property_queries = ('size', 'lbound', 'ubound', 'present')

    def __init__(self, include_literal_kinds=True, **kwargs):
        super().__init__(inplace=True, invalidate_source=False, **kwargs)
        self.include_literal_kinds = include_literal_kinds

    # Utility routines

    def _visit_body(self, body, live=None, defines=None, uses=None, **kwargs):
        """
        Iterate through the tuple that is a body and update defines and
        uses along the way.
        """
        if live is None:
            live = OrderedSet()
        if defines is None:
            defines = OrderedSet()
        if uses is None:
            uses = OrderedSet()
        visited = []
        for i in flatten(body):
            visited += [self.visit(i, live_symbols=live|defines, **kwargs)]
            uses |= visited[-1].uses_symbols.copy() - defines
            defines |= visited[-1].defines_symbols.copy()
        return as_tuple(visited), defines, uses

    @staticmethod
    def _symbols_from_expr(expr, condition=None):
        """
        Return set of symbols found in an expression.
        """
        variables = OrderedSet(strip_nested_dimensions(v) for v in FindVariables().visit(expr))
        parents = OrderedSet(p for var in variables for p in var.parents)
        variables -= parents
        if condition is not None:
            return OrderedSet(v for v in variables if condition(v))
        return variables

    @classmethod
    def _symbols_from_lhs_expr(cls, expr):
        """
        Determine symbol use and symbol definition from a left-hand side expression.

        Parameters
        ----------
        expr : :any:`Scalar` or :any:`Array`
            The left-hand side expression of an assignment.

        Returns
        -------
        (defines, uses) : (set, set)
            The sets of defined and used symbols (in that order).
        """
        defines = {strip_nested_dimensions(expr)}
        uses = cls._symbols_from_expr(getattr(expr, 'dimensions', ()))
        return defines, uses

    # Abstract node (also called from every node type for integration)

    def visit_Node(self, o, **kwargs):
        # Live symbols are determined on InternalNode handler levels and
        # get passed down to all child nodes
        o._update(_live_symbols=kwargs.get('live_symbols', OrderedSet()))

        # Symbols defined or used by this node are determined by their individual
        # handler routines and passed on to visitNode from there
        o._update(_defines_symbols=kwargs.get('defines_symbols', OrderedSet()))
        o._update(_uses_symbols=kwargs.get('uses_symbols', OrderedSet()))
        return o

    # Internal nodes

    def visit_Interface(self, o, **kwargs):
        # Subroutines/functions calls defined in an explicit interface
        defines = OrderedSet()
        for b in o.body:
            if isinstance(b, Subroutine):
                defines = defines | OrderedSet(as_tuple(b.procedure_symbol))
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    def visit_InternalNode(self, o, **kwargs):
        # An internal node defines all symbols defined by its body and uses all
        # symbols used by its body before they are defined in the body
        live = kwargs.pop('live_symbols', OrderedSet())
        body, defines, uses = self._visit_body(o.body, live=live, **kwargs)
        o._update(body=body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Associate(self, o, **kwargs):
        # An associate block defines all symbols defined by its body and uses all
        # symbols used by its body before they are defined in the body
        live = kwargs.pop('live_symbols', OrderedSet())
        body, defines, uses = self._visit_body(o.body, live=live, **kwargs)
        o._update(body=body)

        # reverse the mapping of names before assinging lives, defines, uses sets for Associate node itself
        invert_assoc = CaseInsensitiveDict({v.name: k for k, v in o.associations})
        _live = OrderedSet(invert_assoc[v.name] if v.name in invert_assoc else v for v in live)
        _defines = OrderedSet(invert_assoc[v.name] if v.name in invert_assoc else v for v in defines)
        _uses = OrderedSet(invert_assoc[v.name] if v.name in invert_assoc else v for v in uses)

        return self.visit_Node(o, live_symbols=_live, defines_symbols=_defines, uses_symbols=_uses, **kwargs)

    def visit_Loop(self, o, **kwargs):
        # A loop defines the induction variable for its body before entering it
        live = kwargs.pop('live_symbols', OrderedSet())
        mem_calls = as_tuple(i for i in FindInlineCalls().visit(o.bounds) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_calls))
        uses = self._symbols_from_expr(o.bounds)
        uses = OrderedSet(v for v in uses if not v in query_args)
        body, defines, uses = self._visit_body(o.body, live=live|{o.variable.clone()}, uses=uses, **kwargs)
        o._update(body=body)
        # Make sure the induction variable is not considered outside the loop
        uses.discard(o.variable)
        defines.discard(o.variable)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_WhileLoop(self, o, **kwargs):
        # A while loop uses variables in its condition
        live = kwargs.pop('live_symbols', OrderedSet())
        uses = self._symbols_from_expr(o.condition)
        body, defines, uses = self._visit_body(o.body, live=live, uses=uses, **kwargs)
        o._update(body=body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Conditional(self, o, **kwargs):
        live = kwargs.pop('live_symbols', OrderedSet())

        # exclude arguments to functions that just check the memory attributes of a variable
        mem_call = as_tuple(i for i in FindInlineCalls().visit(o.condition) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_call))
        cset = OrderedSet(v for v in FindVariables().visit(o.condition) if not v in query_args)

        if not self.include_literal_kinds:
            # Filter out any symbols used to qualify literals e.g. 0._JPRB
            literals = FindLiterals().visit(o.condition)
            literal_vars = FindVariables().visit(literals)
            cset -= OrderedSet(literal_vars)

        condition = self._symbols_from_expr(as_tuple(cset))
        body, defines, uses = self._visit_body(o.body, live=live, uses=condition, **kwargs)
        else_body, else_defines, uses = self._visit_body(o.else_body, live=live, uses=uses, **kwargs)
        o._update(body=body, else_body=else_body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines|else_defines, uses_symbols=uses, **kwargs)

    def visit_MultiConditional(self, o, **kwargs):
        live = kwargs.pop('live_symbols', OrderedSet())

        # exclude arguments to functions that just check the memory attributes of a variable
        mem_calls = as_tuple(i for i in FindInlineCalls().visit(o.expr) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_calls))
        eset = OrderedSet(v for v in FindVariables().visit(o.expr) if not v in query_args)

        mem_calls = as_tuple(i for i in FindInlineCalls().visit(o.values) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_calls))
        vset = OrderedSet(v for v in FindVariables().visit(o.values) if not v in query_args)

        uses = self._symbols_from_expr(as_tuple(eset)) | self._symbols_from_expr(as_tuple(vset))
        body = ()
        defines = OrderedSet()
        for b in o.bodies:
            _b, _d, uses = self._visit_body(b, live=live, uses=uses, **kwargs)
            body += (as_tuple(_b),)
            defines |= _d
        else_body, else_defines, uses = self._visit_body(o.else_body, live=live, uses=uses, **kwargs)
        o._update(bodies=body, else_body=else_body)
        defines = defines | else_defines
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    visit_TypeConditional = visit_MultiConditional

    def visit_MaskedStatement(self, o, **kwargs):
        live = kwargs.pop('live_symbols', OrderedSet())

        conditions = self._symbols_from_expr(o.conditions)
        if not self.include_literal_kinds:
            # Filter out any symbols used to qualify literals e.g. 0._JPRB
            literals = as_tuple(FindLiterals().visit(o.conditions))
            literal_vars = FindVariables().visit(literals)
            conditions -= OrderedSet(literal_vars)

        body = ()
        defines = OrderedSet()
        uses = OrderedSet(conditions)
        for b in o.bodies:
            _b, defines, uses = self._visit_body(b, live=live, uses=uses, defines=defines, **kwargs)
            body += (_b,)

        default, default_defs, uses = self._visit_body(o.default, live=live, uses=uses, **kwargs)
        o._update(bodies=body, default=default)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines|default_defs, uses_symbols=uses, **kwargs)

    # Leaf nodes

    def visit_Assignment(self, o, **kwargs):
        # exclude arguments to functions that just check the memory attributes of a variable
        mem_calls = as_tuple(i for i in FindInlineCalls().visit(o.rhs) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_calls))
        rset = OrderedSet(v for v in FindVariables().visit(o.rhs) if not v in query_args)

        if not self.include_literal_kinds:
            # Filter out any symbols used to qualify literals e.g. 0._JPRB
            literals = FindLiterals().visit(o.rhs)
            literal_vars = FindVariables().visit(literals)
            rset -= OrderedSet(literal_vars)

        # The left-hand side variable is defined by this statement
        defines, uses = self._symbols_from_lhs_expr(o.lhs)

        # Anything on the right-hand side is used before assigning to it
        uses |= self._symbols_from_expr(as_tuple(rset))
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_ConditionalAssignment(self, o, **kwargs):
        # The left-hand side variable is defined by this statement
        defines, uses = self._symbols_from_lhs_expr(o.lhs)
        # Anything on the right-hand side is used before assigning to it
        uses |= self._symbols_from_expr((o.condition, o.rhs, o.else_rhs))
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_CallStatement(self, o, **kwargs):
        if o.routine is not BasicType.DEFERRED:
            # With a call context provided we can determine which arguments
            # are potentially defined and which are definitely only used by
            # this call
            defines, uses = OrderedSet(), OrderedSet()
            outvals = [val for arg, val in o.arg_iter() if str(arg.type.intent).lower() in ('inout', 'out')]
            invals = [val for arg, val in o.arg_iter() if str(arg.type.intent).lower() in ('inout', 'in')]

            arrays = [v for v in FindVariables().visit(outvals) if isinstance(v, Array)]
            dims = OrderedSet(v for a in arrays for v in self._symbols_from_expr(a.dimensions))
            for val in outvals:
                exprs = self._symbols_from_expr(val)
                defines |= OrderedSet(e for e in exprs if not e in dims)
                uses |= dims

            uses |= OrderedSet(s for val in invals for s in self._symbols_from_expr(val))
        else:
            # We don't know the intent of any of these arguments and thus have
            # to assume all of them are potentially used or defined by this
            # statement
            arrays = [v for v in FindVariables().visit(o.arguments) if isinstance(v, Array)]
            arrays += [v for arg, val in o.kwarguments for v in FindVariables().visit(val) if isinstance(v, Array)]

            dims = OrderedSet(v for a in arrays for v in FindVariables().visit(a.dimensions))
            defines = self._symbols_from_expr(o.arguments, condition=lambda x: x not in dims)
            for arg, val in o.kwarguments:
                defines |= self._symbols_from_expr(val, condition=lambda x: x not in dims)
            uses = defines.copy() | dims

        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Allocation(self, o, **kwargs):
        arrays = [v for v in FindVariables().visit(o.variables) if isinstance(v, Array)]
        dims = OrderedSet(v for a in arrays for v in FindVariables().visit(a.dimensions))
        defines = self._symbols_from_expr(o.variables, condition=lambda x: x not in dims)
        uses = self._symbols_from_expr(o.data_source or ()) | dims
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Deallocation(self, o, **kwargs):
        defines = self._symbols_from_expr(o.variables)
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    visit_Nullify = visit_Deallocation

    def visit_Import(self, o, **kwargs):
        defines = OrderedSet(s.clone(dimensions=None) for s in FindTypedSymbols().visit(o.symbols or ())
                      if isinstance(s, ProcedureSymbol))
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    def visit_VariableDeclaration(self, o, **kwargs):
        defines = self._symbols_from_expr(o.symbols, condition=lambda v: v.type.initial is not None)
        uses = OrderedSet(v for a in o.symbols if isinstance(a, Array) for v in self._symbols_from_expr(a.dimensions))
        if o.symbols[0].type.kind:
            uses |= {o.symbols[0].type.kind}
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    # The definition of the function has no effect on data flow
    visit_StatementFunction = visit_Node


class DataflowAnalysisDetacher(Transformer):
    """
    Remove in-place any dataflow analysis properties.
    """

    def __init__(self, **kwargs):
        super().__init__(inplace=True, invalidate_source=False, **kwargs)

    def visit_Node(self, o, **kwargs):
        o._update(_live_symbols=None, _defines_symbols=None, _uses_symbols=None)
        return super().visit_Node(o, **kwargs)


def attach_dataflow_analysis(module_or_routine):
    """
    Determine and attach to each IR node dataflow analysis metadata.

    This makes for each IR node the following properties available:

    * :attr:`Node.live_symbols`: symbols defined before the node;
    * :attr:`Node.defines_symbols`: symbols (potentially) defined by the
      node, i.e., live in subsequent nodes;
    * :attr:`Node.uses_symbols`: symbols used by the node (that had to be
      defined before).

    The IR nodes are updated in-place and thus existing references to IR
    nodes remain valid.
    """
    live_symbols = OrderedSet()
    if hasattr(module_or_routine, 'arguments'):
        live_symbols = DataflowAnalysisAttacher._symbols_from_expr(
            module_or_routine.arguments,
            condition=lambda a: a.type.intent and a.type.intent.lower() in ('in', 'inout')
        )

    if hasattr(module_or_routine, 'spec'):
        DataflowAnalysisAttacher().visit(module_or_routine.spec, live_symbols=live_symbols)
        live_symbols |= module_or_routine.spec.defines_symbols

    if hasattr(module_or_routine, 'body'):
        DataflowAnalysisAttacher().visit(module_or_routine.body, live_symbols=live_symbols)


def detach_dataflow_analysis(module_or_routine):
    """
    Remove from each IR node the stored dataflow analysis metadata.

    Accessing the relevant attributes afterwards raises :py:class:`RuntimeError`.
    """
    if hasattr(module_or_routine, 'spec'):
        DataflowAnalysisDetacher().visit(module_or_routine.spec)
    if hasattr(module_or_routine, 'body'):
        DataflowAnalysisDetacher().visit(module_or_routine.body)


@contextmanager
def dataflow_analysis_attached(module_or_routine):
    r"""
    Create a context in which information about defined, live and used symbols
    is attached to each IR node

    This makes for each IR node the following properties available:

    * :attr:`Node.live_symbols`: symbols defined before the node;
    * :attr:`Node.defines_symbols`: symbols (potentially) defined by the
      node;
    * :attr:`Node.uses_symbols`: symbols used by the node that had to be
      defined before.

    This is an in-place update of nodes and thus existing references to IR
    nodes remain valid. When leaving the context the information is removed
    from IR nodes, while existing references remain valid.

    The analysis is based on a rather crude regions-based analysis, with the
    hierarchy implied by (nested) :any:`InternalNode` IR nodes used as regions
    in the reducible flow graph (cf. Chapter 9, in particular 9.7 of Aho, Lam,
    Sethi, and Ulliman (2007)). Our implementation shares some similarities
    with a full reaching definitions dataflow analysis but is not quite as
    powerful.

    In reaching definitions dataflow analysis (cf. Chapter 9.2.4 Aho et. al.),
    the transfer function of a definition :math:`d` can be expressed as:

    .. math:: f_d(x) = \operatorname{gen}_d \cup (x - \operatorname{kill}_d)

    with the set of definitions generated :math:`\operatorname{gen}_d` and the
    set of definitions killed/invalidated :math:`\operatorname{kill}_d`.

    We, however, do not record definitions explicitly and instead operate on
    consolidated sets of defined symbols, i.e., effectively evaluate the
    chained transfer functions up to the node. This yields a set of active
    definitions at this node. The symbols defined by these definitions are
    in :any:`Node.live_symbols`, and the symbols defined by the node (i.e.,
    symbols defined by definitions in :math:`\operatorname{gen}_d`) are in
    :any:`Node.defines_symbols`.

    The advantage of this approach is that it avoids the need to introduce
    a layer for definitions and dependencies. A downside is that this focus
    on symbols instead of definitions precludes, in particular, the ability
    to take data space into account, which makes it less useful for arrays.

    .. note::
        The context manager operates only on the module or routine itself
        (i.e., its spec and, if applicable, body), not on any contained
        subroutines or functions.

    Parameters
    ----------
    module_or_routine : :any:`Module` or :any:`Subroutine`
        The object for which the IR is to be annotated.
    """
    attach_dataflow_analysis(module_or_routine)
    try:
        yield module_or_routine
    finally:
        detach_dataflow_analysis(module_or_routine)


class FindReads(Visitor):
    """
    Look for reads in a specified part of a control flow tree.

    Parameters
    ----------
    start : (iterable of) :any:`Node`, optional
        Visitor is only active after encountering one of the nodes in
        :data:`start` and until encountering a node in :data:`stop`.
    stop : (iterable of) :any:`Node`, optional
        Visitor is no longer active after encountering one of the nodes in
        :data:`stop` until it encounters again a node in :data:`start`.
    active : bool, optional
        Set the visitor active right from the beginning.
    candidate_set : set of :any:`Node`, optional
        If given, only reads for symbols in this set are considered.
    clear_candidates_on_write : bool, optional
        If enabled, writes of a symbol remove it from the :data:`candidate_set`.
    """

    def __init__(self, start=None, stop=None, active=False,
                 candidate_set=None, clear_candidates_on_write=False, **kwargs):
        super().__init__(**kwargs)
        self.start = OrderedSet(as_tuple(start))
        self.stop = OrderedSet(as_tuple(stop))
        self.active = active
        self.candidate_set = candidate_set
        self.clear_candidates_on_write = clear_candidates_on_write
        self.reads = OrderedSet()

    @staticmethod
    def _symbols_from_expr(expr):
        """
        Return set of symbols found in an expression.
        """
        return {v.clone(dimensions=None) for v in FindVariables().visit(expr)}

    def _register_reads(self, read_symbols):
        if self.active:
            if self.candidate_set is None:
                self.reads |= read_symbols
            else:
                self.reads |= read_symbols & self.candidate_set

    def _register_writes(self, write_symbols):
        if self.active and self.clear_candidates_on_write and self.candidate_set is not None:
            self.candidate_set -= write_symbols

    def visit(self, o, *args, **kwargs):
        self.active = (self.active and o not in self.stop) or o in self.start
        return super().visit(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):  # pylint: disable=unused-argument
        pass

    def visit_LeafNode(self, o, **kwargs):  # pylint: disable=unused-argument
        self._register_reads(o.uses_symbols)
        self._register_writes(o.defines_symbols)

    def visit_Conditional(self, o, **kwargs):
        self._register_reads(self._symbols_from_expr(o.condition))
        # Visit each branch with the original candidate set and then take the
        # union of both afterwards to include all potential read-after-writes
        candidate_set = self.candidate_set.copy() if self.candidate_set is not None else None
        self.visit(o.body, **kwargs)
        self.candidate_set, candidate_set = candidate_set, self.candidate_set
        self.visit(o.else_body, **kwargs)
        if self.candidate_set is not None:
            self.candidate_set |= candidate_set

    def visit_Loop(self, o, **kwargs):
        self._register_reads(self._symbols_from_expr(o.bounds))
        active = self.active
        if self.active and self.candidate_set is not None:
            # remove the loop variable as a variable of interest
            self.candidate_set.discard(o.variable)
        self.visit(o.children, **kwargs)
        if active:
            self.reads.discard(o.variable)

    def visit_WhileLoop(self, o, **kwargs):
        self._register_reads(self._symbols_from_expr(o.condition))
        self.visit(o.children, **kwargs)


class FindWrites(Visitor):
    """
    Look for writes in a specified part of a control flow tree.

    Parameters
    ----------
    start : (iterable of) :any:`Node`, optional
        Visitor is only active after encountering one of the nodes in
        :data:`start` and until encountering a node in :data:`stop`.
    stop : (iterable of) :any:`Node`, optional
        Visitor is no longer active after encountering one of the nodes in
        :data:`stop` until it encounters again a node in :data:`start`.
    active : bool, optional
        Set the visitor active right from the beginning.
    candidate_set : set of :any:`Node`, optional
        If given, only writes for symbols in this set are considered.
    """

    def __init__(self, start=None, stop=None, active=False,
                 candidate_set=None, **kwargs):
        super().__init__(**kwargs)
        self.start = OrderedSet(as_tuple(start))
        self.stop = OrderedSet(as_tuple(stop))
        self.active = active
        self.candidate_set = candidate_set
        self.writes = OrderedSet()

    @staticmethod
    def _symbols_from_expr(expr):
        """
        Return set of symbols found in an expression.
        """
        return {v.clone(dimensions=None) for v in FindVariables().visit(expr)}

    def _register_writes(self, write_symbols):
        if self.candidate_set is None:
            self.writes |= write_symbols
        else:
            self.writes |= write_symbols & self.candidate_set

    def visit(self, o, *args, **kwargs):
        self.active = (self.active and o not in self.stop) or o in self.start
        return super().visit(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):  # pylint: disable=unused-argument
        pass

    def visit_LeafNode(self, o, **kwargs):  # pylint: disable=unused-argument
        if self.active:
            self._register_writes(o.defines_symbols)

    def visit_Loop(self, o, **kwargs):
        if self.active:
            # remove the loop variable as a variable of interest
            if self.candidate_set is not None:
                self.candidate_set.discard(o.variable)
            self.writes.discard(o.variable)
        super().visit_Node(o, **kwargs)


def read_after_write_vars(ir, inspection_node):
    """
    Find variables that are read after being written in the given IR.

    This requires prior application of :meth:`dataflow_analysis_attached` to
    the corresponding :any:`Module` or :any:`Subroutine`.

    The result is the set of variables with a data dependency across the
    :data:`inspection_node`.

    See the remarks about implementation and limitations in the description of
    :meth:`dataflow_analysis_attached`. In particular, this does not take into
    account data space and iteration space for arrays.

    Parameters
    ----------
    ir : :any:`Node`
        The root of the control flow (sub-)tree to inspect.
    inspection_node : :any:`Node`
        Only variables with a write before and a read at or after this node
        are considered.

    Returns
    -------
    :any:`set` of :any:`Scalar` or :any:`Array`
        The list of read-after-write variables.
    """
    write_visitor = FindWrites(stop=inspection_node, active=True)
    write_visitor.visit(ir)
    read_visitor = FindReads(start=inspection_node, candidate_set=write_visitor.writes,
                             clear_candidates_on_write=True)
    read_visitor.visit(ir)
    return read_visitor.reads


def loop_carried_dependencies(loop):
    """
    Find variables that are potentially loop-carried dependencies.

    This requires prior application of :meth:`dataflow_analysis_attached` to
    the corresponding :any:`Module` or :any:`Subroutine`.

    See the remarks about implementation and limitations in the description of
    :meth:`dataflow_analysis_attached`. In particular, this does not take into
    account data space and iteration space for arrays. For cases with a
    linear mapping from iteration to data space and no overlap, this will
    falsely report loop-carried dependencies when there are in fact none.
    However, the risk of false negatives should be low.

    Parameters
    ----------
    loop : :any:`Loop`
        The loop node to inspect.

    Returns
    -------
    :any:`set` of :any:`Scalar` or :any:`Array`
        The list of variables that potentially have a loop-carried dependency.
    """
    return loop.uses_symbols & loop.defines_symbols


def _extract_offset(dim, loop_var):
    """
    Extract the integer offset of an array subscript expression relative to
    a loop variable.

    For example, if ``dim`` is ``JK - 1`` and ``loop_var`` is ``JK``, this
    returns ``-1``. If ``dim`` is exactly ``JK``, this returns ``0``.
    If ``dim`` does not involve ``loop_var`` or is non-affine, returns
    :data:`None`.

    Parameters
    ----------
    dim : expression
        The subscript expression for one dimension of an array access.
    loop_var : :any:`Scalar` or str
        The loop induction variable.

    Returns
    -------
    int or None
        The integer offset, or ``None`` if the expression is not an affine
        function of :data:`loop_var`.
    """
    loop_var_name = loop_var.name.lower() if hasattr(loop_var, 'name') else str(loop_var).lower()

    # Check whether loop_var appears in this dimension at all
    found_vars = FindVariables().visit(dim)
    if not any(v.name.lower() == loop_var_name for v in found_vars):
        return None

    # If dim is exactly the loop variable, offset is 0
    if isinstance(dim, (sym.Scalar, sym.DeferredTypeSymbol)):
        if dim.name.lower() == loop_var_name:
            return 0
        return None

    # Compute dim - loop_var and simplify; if the result is a compile-time
    # constant, the offset is that constant value.
    try:
        diff = simplify(dim - loop_var)
    except (TypeError, AttributeError):
        return None

    if isinstance(diff, sym.IntLiteral):
        return diff.value
    if isinstance(diff, int):
        return diff
    if is_minus_prefix(diff):
        inner = strip_minus_prefix(diff)
        if isinstance(inner, sym.IntLiteral):
            return -inner.value
        if isinstance(inner, int):
            return -inner
    if is_constant(diff):
        # Try to evaluate numerically as a last resort
        try:
            from loki.expression.evaluation import LokiEvaluationMapper
            val = LokiEvaluationMapper()(diff)
            if isinstance(val, (int, float)) and val == int(val):
                return int(val)
        except Exception:  # pylint: disable=broad-except
            pass

    return None


def _collect_array_accesses(node, loop_var):
    """
    Walk an IR subtree and collect all array accesses, classifying each
    subscript dimension that involves the loop variable by its integer offset.

    Parameters
    ----------
    node : :any:`Node`
        The IR subtree to walk (e.g., a loop body or an assignment).
    loop_var : :any:`Scalar` or str
        The loop induction variable.

    Returns
    -------
    list of tuple
        Each entry is ``(array_name, dim_index, offset, access_type)`` where:
        - ``array_name`` : str (lowercased)
        - ``dim_index`` : int (which dimension of the array, 0-based)
        - ``offset`` : int (the offset relative to ``loop_var``)
        - ``access_type`` : str, either ``'read'`` or ``'write'``
    """
    accesses = []
    loop_var_name = loop_var.name.lower() if hasattr(loop_var, 'name') else str(loop_var).lower()

    for assign in FindNodes(Assignment).visit(node):
        # --- LHS: write access ---
        lhs = assign.lhs
        if isinstance(lhs, sym.Array) and lhs.dimensions:
            for dim_idx, dim in enumerate(lhs.dimensions):
                offset = _extract_offset(dim, loop_var)
                if offset is not None:
                    accesses.append((lhs.name.lower(), dim_idx, offset, 'write'))

        # --- RHS: read accesses ---
        rhs_vars = FindVariables().visit(assign.rhs)
        for var in rhs_vars:
            if isinstance(var, sym.Array) and var.dimensions:
                for dim_idx, dim in enumerate(var.dimensions):
                    offset = _extract_offset(dim, loop_var)
                    if offset is not None:
                        accesses.append((var.name.lower(), dim_idx, offset, 'read'))

        # Also check for read accesses on the LHS (e.g., arr(JK-1) on LHS
        # means the array is read at the subscript level for other dimensions,
        # and more importantly, if LHS is arr(JK) but rhs references arr too,
        # the LHS subscript dimensions are used for the write address)
        # Additionally, the LHS variable may appear on the RHS (self-update)
        if isinstance(lhs, sym.Array) and lhs.dimensions:
            rhs_var_names = {v.name.lower() for v in rhs_vars if isinstance(v, sym.Array)}
            if lhs.name.lower() in rhs_var_names:
                # The array is both read and written; reads from RHS are
                # already captured above, so nothing extra needed
                pass

    return accesses


def classify_array_access_offsets(loop, loop_var=None):
    """
    For a given loop, classify all array accesses in the loop body by their
    subscript offset relative to the loop induction variable.

    This is a subscript-aware analysis that goes beyond the symbol-level
    :func:`loop_carried_dependencies`. It examines which dimension of each
    array uses the loop variable and at what offset (e.g., ``JK``, ``JK-1``,
    ``JK+1``).

    Parameters
    ----------
    loop : :any:`Loop`
        The loop node to analyse.
    loop_var : expression, optional
        The loop induction variable. If not given, uses ``loop.variable``.

    Returns
    -------
    dict
        A dict mapping ``(array_name, dim_index)`` to a dict of
        ``{offset: set_of_access_types}`` where ``access_types`` are
        ``'read'`` and/or ``'write'``. For example::

            {
                ('zpfplsx', 2): {0: {'read'}, 1: {'write'}},
                ('za', 1):      {0: {'read', 'write'}, -1: {'read'}},
            }
    """
    if loop_var is None:
        loop_var = loop.variable

    accesses = _collect_array_accesses(loop.body, loop_var)

    result = {}
    for arr_name, dim_idx, offset, access_type in accesses:
        key = (arr_name, dim_idx)
        if key not in result:
            result[key] = {}
        if offset not in result[key]:
            result[key][offset] = set()
        result[key][offset].add(access_type)

    return result


def array_loop_carried_dependencies(loop, loop_var=None):
    """
    Find arrays that have true loop-carried dependencies based on subscript
    analysis.

    Unlike :func:`loop_carried_dependencies`, this function examines the
    actual array subscript offsets relative to the loop induction variable to
    determine whether different iterations of the loop access overlapping
    data elements.

    A loop-carried dependency exists when:

    - **Flow (RAW)**: An array is written at offset ``w`` and read at
      offset ``r`` where ``w != r`` (the read at iteration ``k`` accesses
      data written at iteration ``k + (r - w)``).
    - **Anti (WAR)**: An array is read at offset ``r`` and written at
      offset ``w`` where ``w != r``.
    - **Output (WAW)**: An array is written at two different offsets.

    The dependency distance is ``r - w`` for flow dependencies.

    Parameters
    ----------
    loop : :any:`Loop`
        The loop node to analyse.
    loop_var : expression, optional
        The loop induction variable. If not given, uses ``loop.variable``.

    Returns
    -------
    dict
        A dict mapping ``array_name`` to a list of dependency descriptors.
        Each descriptor is a dict with keys:

        - ``'type'``: one of ``'flow'`` (RAW), ``'anti'`` (WAR), ``'output'`` (WAW)
        - ``'dim_index'``: which dimension (0-based) carries the dependency
        - ``'write_offset'``: integer offset of the write access
        - ``'read_offset'``: integer offset of the read access (for flow/anti)
          or second write offset (for output)
        - ``'distance'``: the dependency distance (positive means the read
          depends on data from an earlier iteration in an ascending loop)

        Example::

            {
                'zpfplsx': [
                    {'type': 'flow', 'dim_index': 2, 'write_offset': 1,
                     'read_offset': 0, 'distance': -1}
                ],
                'za': [
                    {'type': 'flow', 'dim_index': 1, 'write_offset': 0,
                     'read_offset': -1, 'distance': -1}
                ]
            }
    """
    access_map = classify_array_access_offsets(loop, loop_var)

    deps = defaultdict(list)

    for (arr_name, dim_idx), offset_map in access_map.items():
        write_offsets = [off for off, types in offset_map.items() if 'write' in types]
        read_offsets = [off for off, types in offset_map.items() if 'read' in types]

        # Flow dependencies (RAW): written at w, read at r, with w != r
        for w in write_offsets:
            for r in read_offsets:
                if w != r:
                    deps[arr_name].append({
                        'type': 'flow',
                        'dim_index': dim_idx,
                        'write_offset': w,
                        'read_offset': r,
                        'distance': r - w
                    })

        # Anti dependencies (WAR): read at r, written at w, with w != r
        # (these are distinct from flow deps when considering ordering)
        # Note: for the same pair (w, r) with w != r, we already have a flow dep.
        # An anti dep is the reverse direction -- read first, then write.
        # In a single loop body executed top-to-bottom, both can exist.
        # We report anti deps only for pairs where there is a read but NOT
        # a write at the same offset (otherwise it is a self-update, not anti).
        # Actually, for loop-carried dependency analysis, both flow and anti
        # matter. The flow deps are already captured above. Anti deps with
        # different offsets are the same pairs but with reversed roles.
        # We avoid double-reporting: flow covers (write_off, read_off),
        # anti would be the same pair interpreted differently.
        # For clarity, we only report flow and output dependencies here.
        # The flow dependency direction already captures both RAW and WAR
        # depending on the sign of the distance.

        # Output dependencies (WAW): written at two different offsets
        for i, w1 in enumerate(write_offsets):
            for w2 in write_offsets[i+1:]:
                deps[arr_name].append({
                    'type': 'output',
                    'dim_index': dim_idx,
                    'write_offset': w1,
                    'read_offset': w2,
                    'distance': w2 - w1
                })

    return dict(deps)


def detect_loop_carry_variables(loop, loop_var=None):
    """
    Detect variables that serve as inter-iteration state carriers within
    a loop.

    This identifies two patterns commonly found in iterative
    computations:

    1. **Scalar carries**: Variables with no loop-variable dimension (e.g.,
       1-D arrays or scalars) that are both read and written within the
       loop body. These propagate state from one iteration to the next.

    2. **Shift registers**: Arrays with a loop-variable dimension that are
       written at one offset and read at a different offset of the loop
       variable (e.g., written at ``JK+1`` and read at ``JK``).

    Parameters
    ----------
    loop : :any:`Loop`
        The loop node to analyse.
    loop_var : expression, optional
        The loop induction variable. If not given, uses ``loop.variable``.

    Returns
    -------
    dict
        A dict with two keys:

        - ``'scalar_carries'``: list of dicts, each with:
            - ``'name'``: variable name (lowercased)
        - ``'shift_registers'``: list of dicts, each with:
            - ``'name'``: array name (lowercased)
            - ``'dim_index'``: which dimension carries the shift
            - ``'write_offset'``: integer offset of the write
            - ``'read_offset'``: integer offset of the read
            - ``'direction'``: ``'downward'`` if write_offset > read_offset
              (data flows from lower to higher index in an ascending loop),
              ``'upward'`` otherwise
    """
    if loop_var is None:
        loop_var = loop.variable
    loop_var_name = loop_var.name.lower() if hasattr(loop_var, 'name') else str(loop_var).lower()

    # --- 1. Scalar carries ---
    # Find variables (scalars or arrays without the loop variable in
    # any subscript) that are both read and written.
    write_names = set()
    read_names = set()

    for assign in FindNodes(Assignment).visit(loop.body):
        lhs = assign.lhs
        # Check if LHS is a variable that does NOT use the loop var
        # in any of its subscript dimensions (i.e., it's a "horizontal-only"
        # or scalar variable relative to this loop).
        if isinstance(lhs, sym.Array) and lhs.dimensions:
            uses_loop_var = False
            for dim in lhs.dimensions:
                found = FindVariables().visit(dim)
                if any(v.name.lower() == loop_var_name for v in found):
                    uses_loop_var = True
                    break
            if not uses_loop_var:
                write_names.add(lhs.name.lower())
        elif isinstance(lhs, (sym.Scalar, sym.DeferredTypeSymbol)):
            write_names.add(lhs.name.lower())

        # Check RHS for reads of variables without loop-var subscripts
        for var in FindVariables().visit(assign.rhs):
            if isinstance(var, sym.Array) and var.dimensions:
                uses_loop_var = False
                for dim in var.dimensions:
                    found = FindVariables().visit(dim)
                    if any(v.name.lower() == loop_var_name for v in found):
                        uses_loop_var = True
                        break
                if not uses_loop_var:
                    read_names.add(var.name.lower())
            elif isinstance(var, (sym.Scalar, sym.DeferredTypeSymbol)):
                if var.name.lower() != loop_var_name:
                    read_names.add(var.name.lower())

    scalar_carries = [
        {'name': name}
        for name in sorted(write_names & read_names)
    ]

    # --- 2. Shift registers ---
    access_map = classify_array_access_offsets(loop, loop_var)
    shift_registers = []

    for (arr_name, dim_idx), offset_map in access_map.items():
        write_offsets = [off for off, types in offset_map.items() if 'write' in types]
        read_offsets = [off for off, types in offset_map.items() if 'read' in types]

        for w in write_offsets:
            for r in read_offsets:
                if w != r:
                    direction = 'downward' if w > r else 'upward'
                    shift_registers.append({
                        'name': arr_name,
                        'dim_index': dim_idx,
                        'write_offset': w,
                        'read_offset': r,
                        'direction': direction
                    })

    return {
        'scalar_carries': scalar_carries,
        'shift_registers': shift_registers
    }


def classify_nonzero_offset_arrays(routine_or_node, loop_var):
    """
    Scan all loops in a routine (or IR subtree) whose induction variable
    matches *loop_var* and return the set of array names that are accessed
    at any non-zero offset of that variable.

    This is a routine-wide version of the per-loop
    :func:`classify_array_access_offsets`.  It collects information from
    **every** loop whose induction variable matches *loop_var* (by name,
    case-insensitive).  An array is classified as "non-zero offset" if,
    in *any* of those loops, it is accessed at an offset other than ``0``
    (e.g., ``JK-1``, ``JK+1``).

    The result can be used to decide which arrays in a mixed init loop need
    to remain in a separate (non-fused) loop and which can safely participate
    in fusion and subsequent demotion.

    Parameters
    ----------
    routine_or_node : :any:`Subroutine` or :any:`Node`
        The routine or IR subtree to scan.  If a :any:`Subroutine`, the
        routine's body is scanned.
    loop_var : :any:`Scalar`, str, or :any:`DeferredTypeSymbol`
        The loop induction variable whose offsets are of interest.

    Returns
    -------
    set of str
        Lowercased array names that have at least one access at a non-zero
        offset of *loop_var* anywhere in the scanned IR.
    """
    loop_var_name = loop_var.name.lower() if hasattr(loop_var, 'name') else str(loop_var).lower()

    # Determine the IR node to walk
    body = routine_or_node.body if hasattr(routine_or_node, 'body') else routine_or_node

    multilevel = set()

    for loop in FindNodes(Loop).visit(body):
        if loop.variable.name.lower() != loop_var_name:
            continue
        access_map = classify_array_access_offsets(loop, loop.variable)
        for (arr_name, _dim_idx), offset_map in access_map.items():
            if any(off != 0 for off in offset_map):
                multilevel.add(arr_name)

    return multilevel
