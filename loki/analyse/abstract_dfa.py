# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from loki.ir import Transformer

__all__ = ['AbstractDataflowAnalysis', 'dfa_attached']


class AbstractDataflowAnalysis(ABC):
    """
    Base interface for dataflow analyses attaching metadata to IR nodes.

    This abstract class defines the common entry points for providing
    "Attacher" and "Detacher" :any:`Transformation` objects that
    populate the implicit sets of symbols attached to each IR node
    when the analysis has been performed.

    For more info, see the :any:`dfa_attached` context.
    """

    class _Attacher(Transformer):
        """Default base class for analysis attachers."""

    class _Detacher(Transformer):
        """Default base class for analysis detachers."""

    def get_attacher(self) -> Any:
        """ Returns an instance of the associated "Attacher" :any:`Transformation`. """
        return self._Attacher()

    def get_detacher(self) -> Any:
        """ Returns an instance of the associated "Detacher" :any:`Transformation`. """
        return self._Detacher()

    @abstractmethod
    def attach_dataflow_analysis(self, module_or_routine):
        pass

    @abstractmethod
    def detach_dataflow_analysis(self, module_or_routine):
        pass


@contextmanager
def dfa_attached(module_or_routine, dfa):
    """
    Create a context in which information about defined, live and used symbols
    is attached to each IR node.

    This makes for each IR node the following properties available:

    * :attr:`Node.live_symbols`: symbols defined before the node;
    * :attr:`Node.defines_symbols`: symbols (potentially) defined by the
      node;
    * :attr:`Node.uses_symbols`: symbols used by the node that had to be
      defined before.

    This is an in-place update of nodes and thus existing references to IR
    nodes remain valid. When leaving the context the information is removed
    from IR nodes, while existing references remain valid.

    The default analysis, which uses a simplified Reaching Definitions
    Analysis (see :any:`DataflowAnalysis`) is used, unless a more
    specialised implementation is provided via the ``dfa`` argument.

    Parameters
    ----------
    module_or_routine : :any:`Module` or :any:`Subroutine`
        The object for which the IR is to be annotated.
    dfa : :any:`AbstractDataflowAnalysis`, optional
        Instance of a dataflow analysis object providing "attacher"
        and "detacher" mechanisms.
    """
    dfa.attach_dataflow_analysis(module_or_routine)
    try:
        yield module_or_routine
    finally:
        dfa.detach_dataflow_analysis(module_or_routine)
