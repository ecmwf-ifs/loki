"""
Collection of dataflow analysis schema routines.
"""

from contextlib import contextmanager
from loki.expression import FindVariables
from loki.visitors import Visitor, Transformer
from loki.tools import as_tuple, flatten


__all__ = [
    'dataflow_analysis_attached', 'read_after_write_vars'
]


class DataflowAnalysisAttacher(Transformer):
    """
    Analyse and attach in-place the definition, use and live status of
    symbols.
    """

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
        setattr(o, '_live_symbols', kwargs.get('live_symbols', set()))

        # Symbols defined or used by this node are determined by their individual
        # handler routines and passed on to visitNode from there
        setattr(o, '_defines_symbols', kwargs.get('defines_symbols', set()))
        setattr(o, '_uses_symbols', kwargs.get('uses_symbols', set()))
        return o

    # Internal nodes

    def visit_InternalNode(self, o, **kwargs):
        # An internal node defines all symbols defined by its body and uses all
        # symbols used by its body before they are defined in the body
        live = kwargs.pop('live_symbols', set())
        body, defines, uses = self._visit_body(o.body, live=live, **kwargs)
        o._update(body=body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

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
        body, defines, uses = self._visit_body(o.body, live=live|{o.variable.clone()}, uses=uses, **kwargs)
        o._update(body=body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Conditional(self, o, **kwargs):
        live = kwargs.pop('live_symbols', set())
        body, defines, uses = self._visit_body(o.body, live=live, uses=self._symbols_from_expr(o.condition), **kwargs)
        else_body, else_defines, uses = self._visit_body(o.else_body, live=live, uses=uses, **kwargs)
        o._update(body=body, else_body=else_body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines|else_defines, uses_symbols=uses, **kwargs)

    def visit_MaskedStatement(self, o, **kwargs):
        live = kwargs.pop('live_symbols', set())
        body, defines, uses = self._visit_body(o.body, live=live, uses=self._symbols_from_expr(o.condition), **kwargs)
        default, default_defs, uses = self._visit_body(o.default, live=live, uses=uses, **kwargs)
        o._update(body=body, default=default)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines|default_defs, uses_symbols=uses, **kwargs)

    # Leaf nodes

    def visit_Assignment(self, o, **kwargs):
        # The left-hand side variable is defined by this statement
        defines, uses = self._symbols_from_lhs_expr(o.lhs)
        # Anything on the right-hand side is used before assigning to it
        uses |= self._symbols_from_expr(o.rhs)
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_ConditionalAssignment(self, o, **kwargs):
        # The left-hand side variable is defined by this statement
        defines, uses = self._symbols_from_lhs_expr(o.lhs)
        # Anything on the right-hand side is used before assigning to it
        uses |= self._symbols_from_expr((o.condition, o.rhs, o.else_rhs))
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_CallStatement(self, o, **kwargs):
        if o.context:
            # With a call context provided we can determine which arguments
            # are potentially defined and which are definitely only used by
            # this call
            defines, uses = set(), set()
            for arg, val in o.context.arg_iter(o):
                if o.context.routine.arguments[arg].intent.lower() in ('inout', 'out'):
                    defines |= self._symbols_from_expr(val)
                if o.context.routine.arguments[arg].intent.lower() in ('in', 'inout'):
                    uses |= self._symbols_from_expr(val)
        else:
            # We don't know the intent of any of these arguments and thus have
            # to assume all of them are potentially used or defined by this
            # statement
            defines = self._symbols_from_expr(o.children)
            uses = defines.copy()
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Allocation(self, o, **kwargs):
        defines = self._symbols_from_expr(o.variables)
        uses = self._symbols_from_expr(o.data_source or ())
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Deallocation(self, o, **kwargs):
        defines = self._symbols_from_expr(o.variables)
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    visit_Nullify = visit_Deallocation

    def visit_Import(self, o, **kwargs):
        defines = self._symbols_from_expr(o.symbols or ())
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    def visit_Declaration(self, o, **kwargs):
        defines = self._symbols_from_expr(o.variables, condition=lambda v: v.type.initial is not None)
        return self.visit_Node(o, defines_symbols=defines, **kwargs)


class DataflowAnalysisDetacher(Transformer):
    """
    Remove in-place any dataflow analysis properties.
    """

    def __init__(self, **kwargs):
        super().__init__(inplace=True, **kwargs)

    def visit_Node(self, o, **kwargs):
        for attr in ('_live_symbols', '_defines_symbols', '_uses_symbols'):
            if hasattr(o, attr):
                delattr(o, attr)
        return super().visit_Node(o, **kwargs)


def attach_dataflow_analysis(module_or_routine):
    """
    Determine and attach to each IR node dataflow analysis metadata.

    This makes for each IR node the following properties available:

    * :py:attr:``Node.live_symbols`: symbols defined before the node;
    * :py:attr:``Node.defines_symbols`: symbols (potentially) defined by the
      node;
    * :py:attr:``Node.uses_symbols`: symbols used by the node that had to be
      defined before.

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
    """
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

    Parameters
    ----------
    module_or_routine : :any:`Module` or :any:`Subroutine`
        The object for which the IR is to be annotated.

    .. note::
        The context manager operates only on the module or routine itself
        (i.e., its spec and, if applicable, body), not on any contained
        subroutines or functions.
    """
    attach_dataflow_analysis(module_or_routine)
    try:
        yield module_or_routine
    finally:
        detach_dataflow_analysis(module_or_routine)


class FindReadAfterWrite(Visitor):

    def __init__(self, inspection_node, **kwargs):
        super().__init__(**kwargs)
        self.inspection_node = inspection_node
        self.writes = set()
        self.reads = set()
        self.find_reads = False

    @staticmethod
    def _symbols_from_expr(expr):
        """
        Return set of symbols found in an expression.
        """
        return {v.clone(dimensions=None) for v in FindVariables().visit(expr)}

    def visit(self, o, *args, **kwargs):
        self.find_reads = self.find_reads or o is self.inspection_node
        super().visit(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):  # pylint: disable=unused-argument
        pass

    def visit_LeafNode(self, o, **kwargs):  # pylint: disable=unused-argument
        if self.find_reads:
            self.reads |= o.uses_symbols & self.writes
        else:
            self.writes |= o.defines_symbols

    def visit_Conditional(self, o, **kwargs):
        if self.find_reads:
            self.reads |= self._symbols_from_expr(o.condition) & self.writes
        self.visit(o.children, **kwargs)

    def visit_Loop(self, o, **kwargs):
        if self.find_reads:
            self.reads |= self._symbols_from_expr(o.bounds) & self.writes
        self.visit(o.children, **kwargs)

    def visit_WhileLoop(self, o, **kwargs):
        if self.find_reads:
            self.reads |= self._symbols_from_expr(o.condition) & self.writes
        self.visit(o.children, **kwargs)


def read_after_write_vars(ir, inspection_node):
    """
    Find variables that are read after being written in the given IR.

    This requires prior application of :meth:`dataflow_analysis_attached` to
    the corresponding :any:`Module` or :any:`Subroutine`.

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
    visitor = FindReadAfterWrite(inspection_node)
    visitor.visit(ir)
    return visitor.reads
