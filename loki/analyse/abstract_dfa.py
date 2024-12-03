# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from contextlib import contextmanager

from loki import Transformer, FindVariables

__all__ = ['AbstractDataflowAnalysis', 'dataflow_analysis_attached']


class AbstractDataflowAnalysis(ABC):

    class _Attacher(Transformer):
        pass

    class Detacher(Transformer):
        pass

    @staticmethod
    def _symbols_from_expr(expr, condition=None):
        """
        Return set of symbols found in an expression.
        """
        if condition is not None:
            return {v.clone(dimensions=None) for v in FindVariables().visit(expr) if condition(v)}
        return {v.clone(dimensions=None) for v in FindVariables().visit(expr)}

    @abstractmethod
    def attach_dataflow_analysis(self, module_or_routine):
        pass

    def detach_dataflow_analysis(self, module_or_routine):
        """
        Remove from each IR node the stored dataflow analysis metadata.

        Accessing the relevant attributes afterwards raises :py:class:`RuntimeError`.
        """

        if hasattr(module_or_routine, 'spec'):
            self.Detacher().visit(module_or_routine.spec)
        if hasattr(module_or_routine, 'body'):
            self.Detacher().visit(module_or_routine.body)


@contextmanager
def dataflow_analysis_attached(module_or_routine, dfa=None):
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
    if dfa is None:
        from loki.analyse.dataflow_analysis import DataflowAnalysis  # pylint: disable=no-toplevel-import
        dfa = DataflowAnalysis()

    dfa.attach_dataflow_analysis(module_or_routine)
    try:
        yield module_or_routine
    finally:
        dfa.detach_dataflow_analysis(module_or_routine)
