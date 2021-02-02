"""
Collection of dataflow analysis schema routines.
"""

from contextlib import contextmanager
from loki.expression import FindVariables
from loki.visitors import Transformer
from loki.tools import as_tuple


__all__ = [
    'dataflow_analysis_attached'
]


class DataflowAnalysisAttacher(Transformer):
    """
    Analyse and attach in-place the definition, use and live status of
    symbols.
    """

    def __init__(self, **kwargs):
        super().__init__(inplace=True, **kwargs)

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

    def _visit_body(self, body, live=None, defines=None, uses=None, **kwargs):
        if live is None:
            live = set()
        if defines is None:
            defines = set()
        if uses is None:
            uses = set()
        visited = []
        for i in body:
            visited += [self.visit(i, live_symbols=live|defines, **kwargs)]
            uses |= visited[-1].uses_symbols.copy() - defines
            defines |= visited[-1].defines_symbols.copy()
        return as_tuple(visited), defines, uses

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
        uses = {v.clone() for v in FindVariables().visit(o.bounds)}
        body, defines, uses = self._visit_body(o.body, live=live|{o.variable.clone()}, uses=uses, **kwargs)
        o._update(body=body)
        # Make sure the induction variable is not considered outside the loop
        uses.discard(o.variable)
        defines.discard(o.variable)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_WhileLoop(self, o, **kwargs):
        # A while loop uses variables in its condition
        live = kwargs.pop('live_symbols', set())
        uses = {v.clone() for v in FindVariables().visit(o.condition)}
        body, defines, uses = self._visit_body(o.body, live=live|{o.variable.clone()}, uses=uses, **kwargs)
        o._update(body=body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Conditional(self, o, **kwargs):
        live = kwargs.pop('live_symbols', set())
        uses = {v.clone() for v in FindVariables().visit(o.condition)}
        body, defines, uses = self._visit_body(o.body, live=live, uses=uses, **kwargs)
        else_body, else_defines, uses = self._visit_body(o.else_body, live=live, uses=uses, **kwargs)
        defines |= else_defines
        o._update(body=body, else_body=else_body)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_MaskedStatement(self, o, **kwargs):
        live = kwargs.pop('live_symbols', set())
        uses = {v.clone() for v in FindVariables().visit(o.condition)}
        body, defines, uses = self._visit_body(o.body, live=live, uses=uses, **kwargs)
        default, default_defines, uses = self._visit_body(o.default, live=live, uses=uses, **kwargs)
        defines |= default_defines
        o._update(body=body, default=default)
        return self.visit_Node(o, live_symbols=live, defines_symbols=defines, uses_symbols=uses, **kwargs)

    # Leaf nodes

    def visit_Assignment(self, o, **kwargs):
        # The left-hand side variable is defined by this statement
        defines = {o.lhs.clone()}
        # Anything on the right-hand side is used before assigning to it
        uses = {v.clone() for v in FindVariables().visit(o.rhs)}
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_ConditionalAssignment(self, o, **kwargs):
        # The left-hand side variable is defined by this statement
        defines = {o.lhs.clone()}
        # Anything on the right-hand side is used before assigning to it
        uses = {v.clone() for v in FindVariables().visit((o.condition, o.rhs, o.else_rhs))}
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_CallStatement(self, o, **kwargs):
        if o.context:
            # With a call context provided we can determine which arguments
            # are potentially defined and which are definitely only used by
            # this call
            defines, uses = set(), set()
            for arg, val in o.context.arg_iter(o):
                if o.context.routine.arguments[arg].intent.lower() in ('inout', 'out'):
                    defines |= {v.clone() for v in FindVariables().visit(val)}
                if o.context.routine.arguments[arg].intent.lower() in ('in', 'inout'):
                    uses |= {v.clone() for v in FindVariables().visit(val)}
        else:
            # We don't know the intent of any of these arguments and thus have
            # to assume all of them are potentially used or defined by this
            # statement
            defines = {v.clone() for v in FindVariables().visit(o.children)}
            uses = defines.copy()
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Allocation(self, o, **kwargs):
        defines = {v.clone() for v in o.variables}
        uses = {v.clone() for v in o.data_source or ()}
        return self.visit_Node(o, defines_symbols=defines, uses_symbols=uses, **kwargs)

    def visit_Deallocation(self, o, **kwargs):
        defines = {v.clone() for v in o.variables}
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    visit_Nullify = visit_Deallocation

    def visit_Import(self, o, **kwargs):
        defines = set()
        if o.symbols:
            defines = {v.clone() for v in o.symbols}
        return self.visit_Node(o, defines_symbols=defines, **kwargs)

    def visit_Declaration(self, o, **kwargs):
        defines = {v.clone() for v in o.variables if v.type.initial is not None}
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
        live_symbols = {a.clone() for a in module_or_routine.arguments
                        if a.type.intent and a.type.intent.lower() in ('in', 'inout')}

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

    * :py:attr:``Node.live_symbols`: symbols defined before the node;
    * :py:attr:``Node.defines_symbols`: symbols (potentially) defined by the
      node;
    * :py:attr:``Node.uses_symbols`: symbols used by the node that had to be
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
