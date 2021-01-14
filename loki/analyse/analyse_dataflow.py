"""
Collection of dataflow analysis schema routines.
"""

from contextlib import contextmanager
from loki.visitors import Visitor


__all__ = [
    'attach_defined_symbols', 'detach_defined_symbols', 'defined_symbols_attached'
]


class DefinedSymbolsAttacher(Visitor):
    """
    Purpose-built visitor that collects symbol definition information while
    traversing the tree and attaches this information as ``_defined_symbols``
    property (that is then available via ``defined_symbols``) to all IR nodes.
    """

    def visit_object(self, o, **kwargs):
        return set()

    def visit_tuple(self, o, **kwargs):
        defined_symbols = kwargs.pop('defined_symbols', set())
        new_symbols = set()
        for i in o:
            new_symbols |= self.visit(i, defined_symbols=defined_symbols | new_symbols, **kwargs)
        return new_symbols

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        setattr(o, '_defined_symbols', kwargs.get('defined_symbols', set()))
        return self.visit(o.children, **kwargs)

    def visit_Assignment(self, o, **kwargs):  # pylint: disable=no-self-use
        setattr(o, '_defined_symbols', kwargs.get('defined_symbols', set()))
        return {o.lhs.name.lower()}

    visit_ConditionalAssignment = visit_Assignment

    def visit_Loop(self, o, **kwargs):
        defined_symbols = kwargs.pop('defined_symbols', set())
        setattr(o, '_defined_symbols', defined_symbols)
        return self.visit(o.children, defined_symbols=defined_symbols | {o.variable.name.lower()}, **kwargs)

    def visit_Associate(self, o, **kwargs):
        defined_symbols = kwargs.pop('defined_symbols', set())
        setattr(o, '_defined_symbols', defined_symbols)
        associations = {a.name.lower() for a, _ in o.associations}
        return self.visit(o.children, defined_symbols=defined_symbols | associations, **kwargs)

    def visit_Import(self, o, **kwargs):  # pylint: disable=no-self-use
        setattr(o, '_defined_symbols', kwargs.get('defined_symbols', set()))
        imports = {a.name.lower() for a in o.symbols}
        return imports

    def visit_Declaration(self, o, **kwargs):  # pylint: disable=no-self-use
        setattr(o, '_defined_symbols', kwargs.get('defined_symbols', set()))
        variables = {v.name.lower() for v in o.variables if v.type.initial is not None}
        return variables


class DefinedSymbolsDetacher(Visitor):
    """
    Purpose-built visitor to remove the ``_defined_symbols`` property (that is
    used by ``defined_symbols``) from all IR nodes.
    """

    def visit_object(self, o, **kwargs):
        pass

    def visit_tuple(self, o, **kwargs):
        for i in o:
            self.visit(i, **kwargs)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        if hasattr(o, '_defined_symbols'):
            delattr(o, '_defined_symbols')
        self.visit(o.children, **kwargs)


def attach_defined_symbols(module_or_routine):
    """
    Determine and attach to each IR node the set of defined variables.

    The information is then accessible via the ``defined_symbols`` property
    on each IR node.

    This is in in-place update of nodes and thus existing references to IR
    nodes remain valid.
    """
    defined_symbols = set()
    if hasattr(module_or_routine, 'arguments'):
        defined_symbols = {a.name.lower() for a in module_or_routine.arguments
                           if a.type.intent and a.type.intent.lower() in ('in', 'inout')}

    if hasattr(module_or_routine, 'spec'):
        defined_symbols |= DefinedSymbolsAttacher().visit(module_or_routine.spec, defined_symbols=defined_symbols)

    if hasattr(module_or_routine, 'body'):
        DefinedSymbolsAttacher().visit(module_or_routine.body, defined_symbols=defined_symbols)


def detach_defined_symbols(module_or_routine):
    """
    Remove from each IR node the stored set of defined variables.

    Accessing the ``defined_symbols`` property of a node afterwards raises
    ``RuntimeError``.
    """
    if hasattr(module_or_routine, 'spec'):
        DefinedSymbolsDetacher().visit(module_or_routine.spec)
    if hasattr(module_or_routine, 'body'):
        DefinedSymbolsDetacher().visit(module_or_routine.body)


@contextmanager
def defined_symbols_attached(module_or_routine):
    """
    Create a context in which information about defined symbols for each IR
    node is attached to the node. Afterwards this information is accessible
    via the ``defined_symbols`` property.

    This requires a single traversal of the tree and updates IR nodes in-place,
    meaning any existing references to node objects remain valid.

    When leaving the context this information is removed from IR nodes, while
    existing references remain valid.

    NB: Defined symbol information is only done for the object itself (i.e. its
    spec and body), not for any contained subroutines.
    """
    attach_defined_symbols(module_or_routine)
    try:
        yield module_or_routine
    finally:
        detach_defined_symbols(module_or_routine)
