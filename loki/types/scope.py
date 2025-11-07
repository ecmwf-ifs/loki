# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Representation of symbol tables and scopes in
:doc:`internal_representation`
"""

from dataclasses import dataclass, field, InitVar
import weakref

from loki.tools import WeakrefProperty
from loki.types.symbol_table import SymbolTable


__all__ = ['Scope']


@dataclass(frozen=True)
class Scope:
    """
    Scoping object that manages type caching and derivation for typed symbols.

    The :any:`Scope` provides a symbol table that uniquely maps a symbol's name
    to its :any:`SymbolAttributes` or, for a derived type definition, directly
    to its :any:`DerivedType`.

    See :any:`SymbolTable` for more details on how to look-up symbols.

    Parameters
    ----------
    parent : :any:`Scope`, optional
        The enclosing scope, thus allowing recursive look-ups
    symbol_attrs : :any:`SymbolTable`, optional
        Use the given symbol table instead of instantiating a new
    """

    symbol_attrs: SymbolTable = field(default_factory=SymbolTable, init=False)
    parent: InitVar[object] = WeakrefProperty(default=None, frozen=True)

    def __post_init__(self, parent=None):
        self._reset_parent(parent)

        assert isinstance(self.symbol_attrs, SymbolTable)
        self.symbol_attrs.parent = None if self.parent is None else self.parent.symbol_attrs

    def __repr__(self):
        """
        String representation.
        """
        return f'Scope<{id(self)}>'

    @property
    def parents(self):
        """
        All parent scopes enclosing the current scope, with the top-level
        scope at the end of the list

        Returns
        -------
        tuple
            The list of parent scopes
        """
        parent = self.parent
        if parent:
            return parent.parents + (parent,)
        return ()

    def rescope_symbols(self):
        """
        Make sure all symbols declared and used inside this node belong
        to a scope in the scope hierarchy
        """
        from loki.ir import AttachScopes  # pylint: disable=import-outside-toplevel,cyclic-import
        AttachScopes().visit(self, scope=self)

    def make_complete(self, **frontend_args):
        """
        Trigger a re-parse of the object if incomplete to produce a full Loki IR

        See :any:`ProgramUnit.make_complete` for more details.

        This method relays the call only to the :attr:`parent`.
        """
        if hasattr(super(), 'make_complete'):
            super().make_complete(**frontend_args)
        self.parent.make_complete(**frontend_args)

    def clone(self, **kwargs):
        """
        Create a copy of the scope object with the option to override individual
        parameters

        Note that this will also create a copy of the symbol table via
        :any:`SymbolTable.clone` and force rescoping of variables,
        unless :attr:`symbol_attrs` and :attr:`rescope_symbols` are explicitly
        specified.

        Parameters
        ----------
        **kwargs : Any parameter from the constructor

        Returns
        -------
        `type(self)`
            The cloned scope object
        """
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent
        if 'symbol_attrs' not in kwargs:
            kwargs['symbol_attrs'] = self.symbol_attrs.clone(parent=kwargs.get('parent'))
            kwargs['rescope_symbols'] = True

        if hasattr(self, '_rebuild'):
            # When cloning IR nodes with a Scope mix-in we need to use the
            # rebuild mechanism
            return self._rebuild(**kwargs)  # pylint: disable=no-member
        return type(self)(**kwargs)

    def get_symbol_scope(self, name):
        """
        Find the scope in which :attr:`name` is declared

        This performs a recursive lookup in the :any:`SymbolTable` to find
        the scope in which :attr:`name` is declared. Note, that this may be
        the scope with a :any:`Import` of this name and not the original
        declaration.

        Parameters
        ----------
        name : `str`
            The name of the symbol to look for

        Returns
        -------
        :any:`Scope` or `None`
            The scope object in which the symbol is declared, or `None` if
            not found
        """
        scope = self
        while scope is not None:
            if name in scope.symbol_attrs:
                return scope
            scope = scope.parent
        return None

    def _reset_parent(self, parent):
        """
        Private method to reset the parent of a :any:`Scope` and
        update the symbol table accordingly.

        Parameters
        ----------
        parent : :any:`Scope`, optional
            The enclosing scope, thus allowing recursive look-ups
        """
        self.__dict__['_parent'] = weakref.ref(parent) if parent is not None else None

        if self.parent is not None:
            self.symbol_attrs.parent = self.parent.symbol_attrs
