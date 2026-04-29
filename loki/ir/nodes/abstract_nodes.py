# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Abstract base classes for node definitions in the Loki IR. """

from abc import abstractmethod
from collections import OrderedDict
from typing import Tuple, Union, Optional

from pydantic import field_validator

from loki.expression import Variable, parse_expr
from loki.frontend.source import Source
from loki.tools import (
    dataclass_strict, sanitize_tuple, CaseInsensitiveDict
)
from loki.types import Scope


__all__ = ['Node', 'InternalNode', 'LeafNode', 'ScopedNode']


@dataclass_strict(frozen=True)
class Node:
    """
    Base class for all node types in Loki's internal representation.

    Provides the common functionality shared by all node types; specifically,
    this comprises functionality to update or rebuild a node, and source
    metadata.

    Attributes
    ----------
    traversable : list of str
        The traversable fields of the Node; that is, fields walked over by
        a :any:`Visitor`. All arguments in :py:meth:`__init__` whose
        name appear in this list are treated as traversable fields.

    Parameters
    ----------
    source : :any:`Source`, optional
        the information about the original source for the Node.
    label : str, optional
        the label assigned to the statement in the original source
        corresponding to the Node.

    """

    source: Optional[Union[Source, str]] = None
    label: Optional[str] = None

    _traversable = []

    @field_validator('label', mode='before')
    @classmethod
    def ensure_string(cls, value):
        return str(value) if value else value

    def __post_init__(self):
        # Create private placeholders for dataflow analysis fields that
        # do not show up in the dataclass field definitions, as these
        # are entirely transient.
        self._update(_live_symbols=None, _defines_symbols=None, _uses_symbols=None)

    @property
    def children(self):
        """
        The traversable children of the node.
        """
        return tuple(getattr(self, i) for i in self._traversable)

    def _rebuild(self, *args, **kwargs):
        """
        Rebuild the node.

        Constructs an identical copy of the node from when it was first
        created. Optionally, some or all of the arguments for it can
        be overwritten.

        Parameters
        ----------
        *args : optional
            The traversable arguments used to create the node. By default,
            ``args`` are used.
        **kwargs : optional
            The non-traversable arguments used to create the node, By
            default, ``args_frozen`` are used.
        """
        handle = self.args
        argnames = [i for i in self._traversable if i not in kwargs]
        handle.update(OrderedDict(zip(argnames, args)))
        handle.update(kwargs)
        return type(self)(**handle)

    clone = _rebuild

    def _update(self, *args, **kwargs):
        """
        In-place update that modifies (re-initializes) the node
        without rebuilding it. Use with care!

        Parameters
        ----------
        *args : optional
            The traversable arguments used to create the node. By default,
            ``args`` are used.
        **kwargs : optional
            The non-traversable arguments used to create the node, By
            default, ``args_frozen`` are used.

        """
        argnames = [i for i in self._traversable if i not in kwargs]
        kwargs.update(zip(argnames, args))
        self.__dict__.update(kwargs)

    @property
    def args(self):
        """
        Arguments used to construct the Node.
        """
        return {k: v for k, v in self.__dict__.items() if k in self.__dataclass_fields__.keys()}  # pylint: disable=no-member

    @property
    def args_frozen(self):
        """
        Arguments used to construct the Node that cannot be traversed.
        """
        return {k: v for k, v in self.args.items() if k not in self._traversable}

    def __repr__(self):
        raise NotImplementedError

    def view(self):
        """
        Pretty-print the node hierachy under this node.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from loki.backend.pprint import pprint
        pprint(self)

    def ir_graph(self, show_comments=False, show_expressions=False, linewidth=40, symgen=str):
        """
        Get the IR graph to visualize the node hierachy under this node.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from loki.ir.ir_graph import ir_graph

        return ir_graph(self, show_comments, show_expressions,linewidth, symgen)

    @property
    def live_symbols(self):
        """
        Yield the list of live symbols at this node, i.e., variables that
        have been defined (potentially) prior to this point in the control flow
        graph.

        This property is attached to the Node by
        :py:func:`loki.analyse.dataflow_analysis.attach_dataflow_analysis` or
        when using the
        :py:func:`loki.analyse.dataflow_analysis.dataflow_analysis_attached`
        context manager.
        """
        if self.__dict__['_live_symbols'] is None:
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self.__dict__['_live_symbols']

    @property
    def defines_symbols(self):
        """
        Yield the list of symbols (potentially) defined by this node.

        This property is attached to the Node by
        :py:func:`loki.analyse.dataflow_analysis.attach_dataflow_analysis` or
        when using the
        :py:func:`loki.analyse.dataflow_analysis.dataflow_analysis_attached`
        context manager.
        """
        if self.__dict__['_defines_symbols'] is None:
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self.__dict__['_defines_symbols']

    @property
    def uses_symbols(self):
        """

        Yield the list of symbols used by this node before defining it.

        This property is attached to the Node by
        :py:func:`loki.analyse.dataflow_analysis.attach_dataflow_analysis` or
        when using the
        :py:func:`loki.analyse.dataflow_analysis.dataflow_analysis_attached`
        context manager.
        """
        if self.__dict__['_uses_symbols'] is None:
            raise RuntimeError('Need to run dataflow analysis on the IR first.')
        return self.__dict__['_uses_symbols']


@dataclass_strict(frozen=True)
class _InternalNode():
    """ Type definitions for :any:`InternalNode` node type. """

    body: Tuple[Union[Node, Scope], ...] = ()


@dataclass_strict(frozen=True)
class InternalNode(Node, _InternalNode):
    """
    Internal representation of a control flow node that has a traversable
    `body` property.

    Parameters
    ----------
    body : tuple
        The nodes that make up the body.
    """

    _traversable = ['body']

    @field_validator('body', mode='before')
    @classmethod
    def ensure_tuple(cls, value):
        return sanitize_tuple(value)

    def __repr__(self):
        raise NotImplementedError


@dataclass_strict(frozen=True)
class LeafNode(Node):
    """
    Internal representation of a control flow node without a `body`.
    """

    def __repr__(self):
        raise NotImplementedError


# Mix-ins

class ScopedNode(Scope):
    """
    Mix-in to attache a scope to an IR :any:`Node`

    Additionally, this specializes the node's :meth:`_update` and
    :meth:`_rebuild` methods to make sure that an existing symbol table
    is carried over correctly.
    """

    @property
    def args(self):
        """
        Arguments used to construct the :any:`ScopedNode`, excluding
        the symbol table.
        """
        keys = tuple(k for k in self.__dataclass_fields__.keys() if k not in ('symbol_attrs', ))  # pylint: disable=no-member
        return {k: v for k, v in self.__dict__.items() if k in keys}

    def _update(self, *args, **kwargs):
        if 'symbol_attrs' not in kwargs:
            # Retain the symbol table (unless given explicitly)
            kwargs['symbol_attrs'] = self.symbol_attrs
        super()._update(*args, **kwargs)  # pylint: disable=no-member

    def _rebuild(self, *args, **kwargs):
        # Retain the symbol table (unless given explicitly)
        symbol_attrs = kwargs.pop('symbol_attrs', self.symbol_attrs)
        rescope_symbols = kwargs.pop('rescope_symbols', False)

        # Ensure 'parent' is always explicitly set
        kwargs['parent'] = kwargs.get('parent', None)

        new_obj = super()._rebuild(*args, **kwargs)  # pylint: disable=no-member
        new_obj.symbol_attrs.update(symbol_attrs)

        if rescope_symbols:
            new_obj.rescope_symbols()
        return new_obj

    def __getstate__(self):
        s = self.args
        s['symbol_attrs'] = self.symbol_attrs
        return s

    def __setstate__(self, s):
        symbol_attrs = s.pop('symbol_attrs', None)
        self._update(**s, symbol_attrs=symbol_attrs, rescope_symbols=True)

    @property
    @abstractmethod
    def variables(self):
        """
        Return the variables defined in this :any:`ScopedNode`.
        """

    @property
    def variable_map(self):
        """
        Map of variable names to :any:`Variable` objects
        """
        return CaseInsensitiveDict((v.name, v) for v in self.variables)

    def get_symbol(self, name):
        """
        Returns the symbol for a given name as defined in its declaration.

        The returned symbol might include dimension symbols if it was
        declared as an array.

        Parameters
        ----------
        name : str
            Base name of the symbol to be retrieved
        """
        return self.get_symbol_scope(name).variable_map.get(name)

    def Variable(self, **kwargs):
        """
        Factory method for :any:`TypedSymbol` or :any:`MetaSymbol` classes.

        This invokes the :any:`Variable` with this node as the scope.

        Parameters
        ----------
        name : str
            The name of the variable.
        type : optional
            The type of that symbol. Defaults to :any:`BasicType.DEFERRED`.
        parent : :any:`Scalar` or :any:`Array`, optional
            The derived type variable this variable belongs to.
        dimensions : :any:`ArraySubscript`, optional
            The array subscript expression.
        """
        kwargs['scope'] = self
        return Variable(**kwargs)

    def parse_expr(self, expr_str, strict=False, evaluate=False, context=None):
        """
        Uses :meth:`parse_expr` to convert expression(s) represented
        in a string to Loki expression(s)/IR.

        Parameters
        ----------
        expr_str : str
            The expression as a string
        strict : bool, optional
            Whether to raise exception for unknown variables/symbols when
            evaluating an expression (default: `False`)
        evaluate : bool, optional
            Whether to evaluate the expression or not (default: `False`)
        context : dict, optional
            Symbol context, defining variables/symbols/procedures to help/support
            evaluating an expression

        Returns
        -------
        :any:`Expression`
            The expression tree corresponding to the expression
        """
        return parse_expr(expr_str, scope=self, strict=strict, evaluate=evaluate, context=context)
