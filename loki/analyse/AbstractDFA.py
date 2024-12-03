# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from contextlib import contextmanager

from loki import Transformer, FindVariables

__all__ = ['AbstractDataflowAnalysis']

class AbstractDataflowAnalysis(Transformer, ABC):
    class Attacher(Transformer):
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
    def dataflow_analysis_attached(self, module_or_routine):
        self.attach_dataflow_analysis(module_or_routine)
        try:
            yield module_or_routine
        finally:
            self.detach_dataflow_analysis(module_or_routine)