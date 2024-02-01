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

from loki.types import SymbolAttributes, BasicType
from loki.tools import WeakrefProperty


__all__ = ['SymbolTable', 'Scope']


class SymbolTable(dict):
    """
    Lookup table for symbol types that maps symbol names to :any:`SymbolAttributes`

    It is used to store types for declared variables, defined types or imported
    symbols within their respective scope. If its associated scope is nested
    into an enclosing scope, it allows to perform recursive look-ups in parent
    scopes.

    The interface of this table behaves like a :any:`dict`.

    Parameters
    ----------
    parent : :any:`SymbolTable`, optional
        The symbol table of the parent scope for recursive look-ups.
    case_sensitive : bool, optional
        Respect the case of symbol names in lookups (default: `False`).
    """

    def __new__(cls, *args, case_sensitive=False, **kwargs):
        """
        Set the lookup function on object creation, so that they are safe to pickle
        """
        obj = super(SymbolTable, cls).__new__(cls, *args, **kwargs)
        obj._case_sensitive = case_sensitive
        if obj.case_sensitive:
            obj.format_lookup_name = SymbolTable._case_sensitive_format_lookup_name
        else:
            obj.format_lookup_name = SymbolTable._not_case_sensitive_format_lookup_name
        return obj

    def __init__(self, parent=None, **kwargs):
        super().__init__(**kwargs)
        self._parent = weakref.ref(parent) if parent is not None else None

    @property
    def parent(self):
        """
        The symbol table of the parent scope

        Returns
        -------
        :any:`SymbolTable` or `None`
        """
        return self._parent() if self._parent is not None else None

    @parent.setter
    def parent(self, parent):
        self._parent = weakref.ref(parent) if parent is not None else None

    @property
    def case_sensitive(self):
        """
        Indicate if the :any:`SymbolTable` is case-sensitive when looking up
        names

        Returns
        -------
        `bool`
        """
        return self._case_sensitive  # pylint: disable=no-member

    def format_lookup_name(self, name):  # pylint: disable=method-hidden
        """
        Format a variable name for look-up (e.g., convert to lower case if
        case-insensitive)

        Parameters
        ----------
        name : `str`
            the name to look up

        Returns
        -------
        str :
            the name used for look-ups
        """

    @staticmethod
    def _case_sensitive_format_lookup_name(name):
        name = name.partition('(')[0]  # Remove any dimension parameters
        return name

    @staticmethod
    def _not_case_sensitive_format_lookup_name(name):
        name = name.lower()
        name = name.partition('(')[0]  # Remove any dimension parameters
        return name

    def _lookup_formatted_name(self, name, recursive):
        """
        Helper routine to recursively look for a symbol in the table.

        Look-ups should always be done via :meth:`lookup` as this makes sure
        the look-up name is formatted according to the expected format.

        Parameters
        ----------
        name : `str`
            the name to look for, formatted according to :meth:`format_lookup_name`
        recursive : `bool`
            recursive look-up in parent tables
        """
        value = super().get(name, None)
        if value is None and recursive and self.parent is not None:
            return self.parent._lookup_formatted_name(name, recursive)
        return value.clone() if value is not None else None

    def lookup(self, name, recursive=True):
        """
        Look-up a symbol in the symbol table and return the type or `None` if not found.

        Parameters
        ----------
        name : `str`
            Name of the type or symbol
        recursive : `bool`, optional
            If no entry by that name is found, try to find it in the table of the parent scope

        Returns
        -------
        :any:`SymbolAttributes` or `None`
        """
        formatted_name = self.format_lookup_name(name)  # pylint: disable=assignment-from-no-return
        value = self._lookup_formatted_name(formatted_name, recursive)
        return value

    def __contains__(self, key):
        return super().__contains__(self.format_lookup_name(key))

    def __getitem__(self, key):
        value = self.lookup(key, recursive=False)
        if value is None:
            raise KeyError(key)
        return value.clone()

    def get(self, key, default=None):
        """
        Get a symbol's entry without recursive lookup

        Parameters
        ----------
        key : `str`
            Name of the type or symbol
        default : optional
            Return this value if :attr:`key` is not found in the table
        """
        value = self.lookup(key, recursive=False)
        return value.clone() if value is not None else default

    def __setitem__(self, key, value):
        assert isinstance(value, SymbolAttributes)
        name_parts = self.format_lookup_name(key)  # pylint: disable=assignment-from-no-return
        super().__setitem__(name_parts, value.clone())

    def __hash__(self):
        return hash(tuple(self.keys()))

    def __repr__(self):
        return f'<loki.types.SymbolTable object at {hex(id(self))}>'

    def __getstate__(self):
        _ignored = ('_parent', )
        return {k: v for k, v in self.__dict__.items() if k not in _ignored}

    def __setstate__(self, s):
        self.__dict__.update(s)

        self._parent = None

    def setdefault(self, key, default=None):
        """
        Insert a default value for a key into the table if it does not exist

        Parameters
        ----------
        key : `str`
            Name of the type or symbol
        default : optional
            The default value to store for the key. Defaults to
            ``SymbolAttributes(BasicType.DEFERRED)``.
        """
        if default is None:
            default = SymbolAttributes(BasicType.DEFERRED)
        assert isinstance(default, SymbolAttributes)
        super().setdefault(self.format_lookup_name(key), default.clone())

    def update(self, other):
        """
        Update this symbol table with entries from :attr:`other`
        """
        if isinstance(other, dict):
            other = {self.format_lookup_name(k): v.clone() for k, v in other.items()}
        else:
            other = {self.format_lookup_name(k): v.clone() for k, v in other}
        super().update(other)

    def clone(self, **kwargs):
        """
        Create a copy of the symbol table with the option to override individual
        parameters

        Parameters
        ----------
        **kwargs :
            Any parameters from the constructor of :any:`SymbolTable`

        Returns
        -------
        :any:`SymbolTable`
            The clone symbol table with copies of all :any:`SymbolAttributes`
        """
        if self.case_sensitive and 'case_sensitive' not in kwargs:
            kwargs['case_sensitive'] = self.case_sensitive
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent
        obj = type(self)(**kwargs)
        obj.update(self)
        return obj


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
        from loki.expression import AttachScopes  # pylint: disable=import-outside-toplevel,cyclic-import
        AttachScopes().visit(self)

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
