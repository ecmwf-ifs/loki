# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki.frontend import (
    parse_omni_ast, parse_fparser_ast, get_fparser_node,
    parse_regex_source
)
from loki.ir import (
    nodes as ir, FindNodes, Transformer, ExpressionTransformer,
    pragmas_attached
)
from loki.logging import debug
from loki.program_unit import ProgramUnit
from loki.subroutine import Subroutine
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import BasicType, ProcedureType, SymbolAttributes


__all__ = ['Function']


class Function(Subroutine):
    """
    Class to handle and manipulate a single function.

    Parameters
    ----------
    name : str
        Name of the subroutine.
    args : iterable of str, optional
        The names of the dummy args.
    docstring : tuple of :any:`Node`, optional
        The subroutine docstring in the original source.
    spec : :any:`Section`, optional
        The spec of the subroutine.
    body : :any:`Section`, optional
        The body of the subroutine.
    contains : :any:`Section`, optional
        The internal-subprogram part following a ``CONTAINS`` statement
        declaring member procedures
    prefix : iterable, optional
        Prefix specifications for the procedure
    bind : optional
        Bind information (e.g., for Fortran ``BIND(C)`` annotation).
    result_name : str, optional
        The name of the result variable for functions.
    ast : optional
        Frontend node for this subroutine (from parse tree of the frontend).
    source : :any:`Source`
        Source object representing the raw source string information from the
        read file.
    parent : :any:`Scope`, optional
        The enclosing parent scope of the subroutine, typically a :any:`Module`
        or :any:`Subroutine` object. Declarations from the parent scope remain
        valid within the subroutine's scope (unless shadowed by local
        declarations).
    rescope_symbols : bool, optional
        Ensure that the type information for all :any:`TypedSymbol` in the
        subroutine's IR exist in the subroutine's scope or the scope's parents.
        Defaults to `False`.
    symbol_attrs : :any:`SymbolTable`, optional
        Use the provided :any:`SymbolTable` object instead of creating a new
    incomplete : bool, optional
        Mark the object as incomplete, i.e. only partially parsed. This is
        typically the case when it was instantiated using the :any:`Frontend.REGEX`
        frontend and a full parse using one of the other frontends is pending.
    parser_classes : :any:`RegexParserClass`, optional
        Provide the list of parser classes used during incomplete regex parsing
    """

    is_function = True

    def __repr__(self):
        """ String representation """
        return f'Function:: {self.name}'

    @property
    def return_type(self):
        """ Return the return_type of this subroutine """
        return self.symbol_attrs.get(self.result_name)
