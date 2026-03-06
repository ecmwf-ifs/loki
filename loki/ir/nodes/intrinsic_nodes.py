# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Intrinsic node type definitions. """

from dataclasses import dataclass

from loki.ir.nodes.abstract_nodes import LeafNode
from loki.tools import dataclass_strict, truncate_string


__all__ = ['Intrinsic']


@dataclass(frozen=True)
class _IntrinsicBase():
    """ Type definitions for :any:`Intrinsic` node type. """

    text: str


@dataclass_strict(frozen=True)
class Intrinsic(LeafNode, _IntrinsicBase):
    """
    Catch-all generic node for corner-cases.

    This is provided as a fallback for any statements that do not have
    an appropriate representation in the IR. These can either be language
    features for which support was not yet added, or statements that are not
    relevant in Loki's scope of applications. This node retains the text of
    the statement in the original source as-is.

    Parameters
    ----------
    text : str
        The statement as a string.
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.text, str)

    def __repr__(self):
        return f'Intrinsic:: {truncate_string(self.text)}'
