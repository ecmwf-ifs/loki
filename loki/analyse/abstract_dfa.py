# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from contextlib import contextmanager

from loki import Transformer

__all__ = ['AbstractDataflowAnalysis', 'dfa_attached']


class AbstractDataflowAnalysis(ABC):
    """Base interface for dataflow analyses attaching metadata to IR nodes."""

    class _Attacher(Transformer):
        """Default base class for analysis attachers."""

    class _Detacher(Transformer):
        """Default base class for analysis detachers."""

    def get_attacher(self):
        return self._Attacher()

    def get_detacher(self):
        return self._Detacher()

    @abstractmethod
    def attach_dataflow_analysis(self, module_or_routine):
        pass

    @abstractmethod
    def detach_dataflow_analysis(self, module_or_routine):
        pass


@contextmanager
def dfa_attached(module_or_routine, dfa):
    dfa.attach_dataflow_analysis(module_or_routine)
    try:
        yield module_or_routine
    finally:
        dfa.detach_dataflow_analysis(module_or_routine)
