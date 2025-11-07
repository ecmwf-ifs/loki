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

import weakref

from loki.types.datatypes import SymbolAttributes, BasicType


__all__ = ['SymbolTable']


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
