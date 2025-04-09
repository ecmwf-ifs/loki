# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from contextlib import contextmanager

from loki import Transformer

__all__ = ['AbstractDataflowAnalysis', 'dataflow_analysis_attached']

class AbstractDataflowAnalysis(ABC):
    class _Attacher(Transformer):
        pass

    class _Detacher(Transformer):
        pass

    def get_attacher(self):
        return self._Attacher()

    def get_detacher(self):
        return self._Detacher()

    @abstractmethod
    def attach_dataflow_analysis(self, module_or_routine):
        pass

    def detach_dataflow_analysis(self, module_or_routine):
        """
        Remove from each IR node the stored dataflow analysis metadata.

        Accessing the relevant attributes afterwards raises :py:class:`RuntimeError`.
        """

        if hasattr(module_or_routine, 'spec'):
            self.get_detacher().visit(module_or_routine.spec)
        if hasattr(module_or_routine, 'body'):
            self.get_detacher().visit(module_or_routine.body)

@contextmanager
def dataflow_analysis_attached(module_or_routine, dfa=None):
    if dfa is None:
        from loki.analyse.dataflow_analysis import DataflowAnalysis  # pylint: disable=no-toplevel-import
        dfa = DataflowAnalysis()
    dfa.attach_dataflow_analysis(module_or_routine)
    yield
    dfa.detach_dataflow_analysis(module_or_routine)