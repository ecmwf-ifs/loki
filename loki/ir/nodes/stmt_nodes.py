# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Generic and specific statement node type definitions. """

from typing import Optional, Union, Tuple

from pymbolic.primitives import Expression
from pydantic import field_validator, ValidationError

from loki.ir.nodes.abstract_nodes import LeafNode
from loki.tools import dataclass_strict, truncate_string, sanitize_tuple


__all__ = [
    'GenericStmt', 'ImplicitStmt', 'SaveStmt', 'PublicStmt',
    'PrivateStmt', 'ContainsStmt', 'ReturnStmt', 'CycleStmt',
    'GotoStmt'
]


@dataclass_strict(frozen=True)
class _GenericStmtBase():
    """ Type definitions for :any:`GenericStmt` node type. """

    text: str


@dataclass_strict(frozen=True)
class GenericStmt(LeafNode, _GenericStmtBase):
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

    keyword = None

    def __repr__(self):
        return f'GenericStmt:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class ImplicitStmt(GenericStmt):
    """
    :any:`GenericStmt` node that represents the ``IMPLICIT`` statement.

    Parameters
    ----------
    text : str or :any:`Expression`, optional
        Either a tuple of variable specifiers or a string; default: ``NONE``
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    keyword = 'IMPLICIT'

    text: Optional[Union[str, Tuple[Expression, ...]]] = 'NONE'

    @field_validator('text', mode='before')
    @classmethod
    def ensure_str_or_tuple(cls, value):
        if isinstance(value, str):
            return value
        return sanitize_tuple(value)

    def __repr__(self):
        return f'Implicit:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class SaveStmt(GenericStmt):
    """
    :any:`GenericStmt` node that represents the ``SAVE`` statement.

    Parameters
    ----------
    text : str or :any:`Expression`, optional
        Either a tuple of variable specifiers or a string; default: ``NONE``
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    keyword = 'SAVE'

    text: Optional[Tuple[Expression, ...]] = ()

    @field_validator('text', mode='before')
    @classmethod
    def ensure_tuple(cls, value):
        return sanitize_tuple(value)

    def __repr__(self):
        return f'Save:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class PublicStmt(GenericStmt):
    """
    :any:`GenericStmt` node that represents the ``PUBLIC`` specifier.

    Parameters
    ----------
    text : str or tuple of :any:`Expression`, optional
        Either a tuple of variable specifiers or a string; default: ``NONE``
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    keyword = 'PUBLIC'

    text: Optional[Tuple[Expression, ...]] = ()

    @field_validator('text', mode='before')
    @classmethod
    def ensure_tuple(cls, value):
        return sanitize_tuple(value)

    def __repr__(self):
        return f'Public:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class PrivateStmt(GenericStmt):
    """
    :any:`GenericStmt` node that represents the ``PRIVATE`` specifier.

    Parameters
    ----------
    text : str or tuple of :any:`Expression`, optional
        Either a tuple of variable specifiers or a string; default: ``NONE``
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    keyword = 'PRIVATE'

    text: Optional[Tuple[Expression, ...]] = ()

    @field_validator('text', mode='before')
    @classmethod
    def ensure_tuple(cls, value):
        return sanitize_tuple(value)

    def __repr__(self):
        return f'Private:: {truncate_string(self.text)}'


@dataclass_strict(frozen=True)
class ContainsStmt(GenericStmt):
    """
    :any:`GenericStmt` node that represents the ``CONTAINS`` specifier.

    Parameters
    ----------
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    keyword = 'CONTAINS'

    text: Optional[None] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.text is None:
            raise ValidationError('[Loki] ContainsStmt takes no constructor arguments')

    def __repr__(self):
        return 'Contains::'


@dataclass_strict(frozen=True)
class ReturnStmt(GenericStmt):
    """
    :any:`GenericStmt` node that represents the ``RETURN`` specifier.

    Parameters
    ----------
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    keyword = 'RETURN'

    text: Optional[None] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.text is None:
            raise ValidationError('[Loki] ReturnStmt takes no constructor arguments')

    def __repr__(self):
        return 'Return::'


@dataclass_strict(frozen=True)
class CycleStmt(GenericStmt):
    """
    :any:`GenericStmt` node that represents the ``CYCLE`` specifier.

    Parameters
    ----------
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    keyword = 'CYCLE'

    text: Optional[None] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.text is None:
            raise ValidationError('[Loki] CycleStmt takes no constructor arguments')

    def __repr__(self):
        return 'Cycle::'


@dataclass_strict(frozen=True)
class GotoStmt(GenericStmt):
    """
    :any:`GenericStmt` node that represents the ``GO TO`` specifier.

    Parameters
    ----------
    **kwargs : optional
        Other parameters that are passed on to the parent class constructor.
    """

    keyword = 'GO TO'

    text: str

    def __repr__(self):
        return f'Cycle:: {truncate_string(self.text)}'
