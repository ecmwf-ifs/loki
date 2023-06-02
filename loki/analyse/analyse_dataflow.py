# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of dataflow analysis schema routines.
"""

from contextlib import contextmanager
from loki.expression import FindVariables, Array, FindInlineCalls
from loki.tools import as_tuple, flatten
from loki.types import BasicType
from loki.visitors import Visitor, Transformer
from loki.subroutine import Subroutine
from loki.tools.util import CaseInsensitiveDict

__all__ = [
    'dataflow_analysis_attached', 'read_after_write_vars',
    'loop_carried_dependencies'
]


class DataflowAnalysisAttacher(Transformer):
    """
    Analyse and attach in-place the definition, use and live status of
    symbols.
    """

    # group of functions that only query memory properties and don't read/write variable value
    _mem_property_queries = ('size', 'lbound', 'ubound', 'present')

    def __init__(self, **kwargs):
        super().__init__(inplace=True, **kwargs)

    # Utility routines

    def _visit_body(self, body, live=None, defines=None, uses=None, **kwargs):
        """
        Iterate through the tuple that is a body and update defines and
        uses along the way.
        """
        if live is None:
            live = set()
        if defines is None:
            defines = set()
        if uses is None:
            uses = set()
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
        if condition is not None:
            return {v.clone(dimensions=None) for v in FindVariables().visit(expr) if condition(v)}
        return {v.clone(dimensions=None) for v in FindVariables().visit(expr)}

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
        defines = {expr.clone(dimensions=None)}
        uses = cls._symbols_from_expr(getattr(expr, 'dimensions', ()))
        return defines, uses

    # Abstract node (also called from every node type for integration)

    def visit_Node(self, o, **kwargs):
        # Live symbols are determined on InternalNode handler levels and
        # get passed down to all child nodes
        o._update(_live_symbols=kwargs.get('live_symbols', set()))

        # Symbols defined or used by this node are determined by their individual
        # handler routines and passed on to visitNode from there
        o._update(_defines_symbols=kwargs.get('defines_symbols', set()))
        o._update(_uses_symbols=kwargs.get('uses_symbols', set()))
        return o

    # Internal nodes

    def visit_Interface(self, o, **kwargs):
        # Subroutines/functions calls defined in an explicit interface
        defines = set()
        for b in o.body:
            if isinstance(b, Subroutine):
                defines = defines | set(as_tuple(b.procedure_symbol))
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    def visit_InternalNode(self, o, **kwargs):
        # An internal node defines all symbols defined by its body and uses all
        # symbols used by its body before they are defined in the body
        live = kwargs.pop('live_symbols', set())
        body, defines, uses = self._visit_body(o.body, live=live, **kwargs)
        o._update(body=body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Associate(self, o, **kwargs):
        # An associate block defines all symbols defined by its body and uses all
        # symbols used by its body before they are defined in the body
        live = kwargs.pop('live_symbols', set())
        body, defines, uses = self._visit_body(o.body, live=live, **kwargs)
        o._update(body=body)

        # reverse the mapping of names before assinging lives, defines, uses sets for Associate node itself
        invert_assoc = CaseInsensitiveDict({v.name: k for k, v in o.associations})
        _live = set(invert_assoc[v.name] if v.name in invert_assoc else v for v in live)
        _defines = set(invert_assoc[v.name] if v.name in invert_assoc else v for v in defines)
        _uses = set(invert_assoc[v.name] if v.name in invert_assoc else v for v in uses)

        return self.visit_Node(o, live_symbols=_live, defines_symbols=_defines, uses_symbols=_uses, **kwargs)

    def visit_Loop(self, o, **kwargs):
        # A loop defines the induction variable for its body before entering it
        live = kwargs.pop('live_symbols', set())
        uses = self._symbols_from_expr(o.bounds)
        body, defines, uses = self._visit_body(o.body, live=live|{o.variable.clone()}, uses=uses, **kwargs)
        o._update(body=body)
        # Make sure the induction variable is not considered outside the loop
        uses.discard(o.variable)
        defines.discard(o.variable)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_WhileLoop(self, o, **kwargs):
        # A while loop uses variables in its condition
        live = kwargs.pop('live_symbols', set())
        uses = self._symbols_from_expr(o.condition)
        body, defines, uses = self._visit_body(o.body, live=live, uses=uses, **kwargs)
        o._update(body=body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Conditional(self, o, **kwargs):
        live = kwargs.pop('live_symbols', set())

        # exclude arguments to functions that just check the memory attributes of a variable
        mem_call = as_tuple(i for i in FindInlineCalls().visit(o.condition) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_call))
        cset = set(v for v in FindVariables().visit(o.condition) if not v in query_args)

        condition = self._symbols_from_expr(as_tuple(cset))
        body, defines, uses = self._visit_body(o.body, live=live, uses=condition, **kwargs)
        else_body, else_defines, uses = self._visit_body(o.else_body, live=live, uses=uses, **kwargs)
        o._update(body=body, else_body=else_body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines|else_defines, uses_symbols=uses, **kwargs)

    def visit_MultiConditional(self, o, **kwargs):
        live = kwargs.pop('live_symbols', set())

        # exclude arguments to functions that just check the memory attributes of a variable
        mem_calls = as_tuple(i for i in FindInlineCalls().visit(o.expr) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_calls))
        eset = set(v for v in FindVariables().visit(o.expr) if not v in query_args)

        mem_calls = as_tuple(i for i in FindInlineCalls().visit(o.values) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_calls))
        vset = set(v for v in FindVariables().visit(o.values) if not v in query_args)

        uses = self._symbols_from_expr(as_tuple(eset)) | self._symbols_from_expr(as_tuple(vset))
        body = ()
        defines = set()
        for b in o.bodies:
            _b, _d, uses = self._visit_body(b, live=live, uses=uses, **kwargs)
            body += (as_tuple(_b),)
            defines |= _d
        else_body, else_defines, uses = self._visit_body(o.else_body, live=live, uses=uses, **kwargs)
        o._update(bodies=body, else_body=else_body)
        defines = defines | else_defines
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_MaskedStatement(self, o, **kwargs):
        live = kwargs.pop('live_symbols', set())
        conditions = self._symbols_from_expr(o.conditions)
        body, defines, uses = self._visit_body(o.bodies, live=live, uses=conditions, **kwargs)
        body = tuple(as_tuple(b,) for b in body)
        default, default_defs, uses = self._visit_body(o.default, live=live, uses=uses, **kwargs)
        o._update(bodies=body, default=default)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines|default_defs, uses_symbols=uses, **kwargs)

    # Leaf nodes

    def visit_Assignment(self, o, **kwargs):
        # exclude arguments to functions that just check the memory attributes of a variable
        mem_calls = as_tuple(i for i in FindInlineCalls().visit(o.rhs) if i.function in self._mem_property_queries)
        query_args = as_tuple(flatten(FindVariables().visit(i.parameters) for i in mem_calls))
        rset = set(v for v in FindVariables().visit(o.rhs) if not v in query_args)

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
            defines, uses = set(), set()
            outvals = [val for arg, val in o.arg_iter() if str(arg.type.intent).lower() in ('inout', 'out')]
            invals = [val for arg, val in o.arg_iter() if str(arg.type.intent).lower() in ('inout', 'in')]

            for val in outvals:
                arrays = [v for v in FindVariables().visit(outvals) if isinstance(v, Array)]
                dims = set(v for a in arrays for v in FindVariables().visit(a.dimensions))
                exprs = self._symbols_from_expr(val)
                defines |= {e for e in exprs if not e in dims}
                uses |= dims

            uses |= {s for val in invals for s in self._symbols_from_expr(val)}
        else:
            # We don't know the intent of any of these arguments and thus have
            # to assume all of them are potentially used or defined by this
            # statement
            arrays = [v for v in FindVariables().visit(o.arguments) if isinstance(v, Array)]
            arrays += [v for arg, val in o.kwarguments for v in FindVariables().visit(val) if isinstance(v, Array)]

            dims = set(v for a in arrays for v in FindVariables().visit(a.dimensions))
            defines = self._symbols_from_expr(o.arguments, condition=lambda x: x not in dims)
            for arg, val in o.kwarguments:
                defines |= self._symbols_from_expr(val, condition=lambda x: x not in dims)
            uses = defines.copy() | dims

        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Allocation(self, o, **kwargs):
        arrays = [v for v in FindVariables().visit(o.variables) if isinstance(v, Array)]
        dims = set(v for a in arrays for v in FindVariables().visit(a.dimensions))
        defines = self._symbols_from_expr(o.variables, condition=lambda x: x not in dims)
        uses = self._symbols_from_expr(o.data_source or ()) | dims
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Deallocation(self, o, **kwargs):
        defines = self._symbols_from_expr(o.variables)
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    visit_Nullify = visit_Deallocation

    def visit_Import(self, o, **kwargs):
        defines = self._symbols_from_expr(o.symbols or ())
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    def visit_VariableDeclaration(self, o, **kwargs):
        defines = self._symbols_from_expr(o.symbols, condition=lambda v: v.type.initial is not None)
        return self.visit_Node(o, defines_symbols=defines, **kwargs)


class DataflowAnalysisDetacher(Transformer):
    """
    Remove in-place any dataflow analysis properties.
    """

    def __init__(self, **kwargs):
        super().__init__(inplace=True, **kwargs)

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
    live_symbols = set()
    if hasattr(module_or_routine, 'arguments'):
        live_symbols = DataflowAnalysisAttacher._symbols_from_expr(
            module_or_routine.arguments,
            condition=lambda a: a.type.intent and a.type.intent.lower() in ('in', 'inout')
        )

    if hasattr(module_or_routine, 'spec'):
        module_or_routine.spec = DataflowAnalysisAttacher().visit(module_or_routine.spec, live_symbols=live_symbols)
        live_symbols |= module_or_routine.spec.defines_symbols

    if hasattr(module_or_routine, 'body'):
        module_or_routine.body = DataflowAnalysisAttacher().visit(module_or_routine.body, live_symbols=live_symbols)


def detach_dataflow_analysis(module_or_routine):
    """
    Remove from each IR node the stored dataflow analysis metadata.

    Accessing the relevant attributes afterwards raises :py:class:`RuntimeError`.
    """
    if hasattr(module_or_routine, 'spec'):
        module_or_routine.spec = DataflowAnalysisDetacher().visit(module_or_routine.spec)
    if hasattr(module_or_routine, 'body'):
        module_or_routine.body = DataflowAnalysisDetacher().visit(module_or_routine.body)


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
        self.start = set(as_tuple(start))
        self.stop = set(as_tuple(stop))
        self.active = active
        self.candidate_set = candidate_set
        self.clear_candidates_on_write = clear_candidates_on_write
        self.reads = set()

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
        self.start = set(as_tuple(start))
        self.stop = set(as_tuple(stop))
        self.active = active
        self.candidate_set = candidate_set
        self.writes = set()

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
