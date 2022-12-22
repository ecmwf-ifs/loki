# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Contains the declaration of :any:`Module` to represent Fortran modules.
"""
from loki.frontend import (
    get_fparser_node, parse_omni_ast, parse_ofp_ast, parse_fparser_ast,
    parse_regex_source
)
from loki.ir import VariableDeclaration
from loki.pragma_utils import pragmas_attached, process_dimension_pragmas
from loki.program_unit import ProgramUnit
from loki.scope import Scope
from loki.subroutine import Subroutine
from loki.tools import as_tuple
from loki.types import ModuleType, SymbolAttributes


__all__ = ['Module']


class Module(ProgramUnit):
    """
    Class to handle and manipulate source modules.

    Parameters
    ----------
    name : str
        Name of the module.
    docstring : :any:`CommentBlock` or list of :any:`Comment`
        The module docstring
    spec : :any:`Section`, optional
        The spec section of the module.
    contains : tuple of :any:`Subroutine`, optional
        The module-subprogram part following a ``CONTAINS`` statement declaring
        member procedures.
    default_access_spec : str, optional
        The default access attribute for variables as defined by an access-spec
        statement without access-id-list, i.e., ``public`` or ``private``.
        Default value is `None` corresponding to the absence of an access-spec
        statement for default accessibility (which is equivalent to ``public``).
    public_access_spec : tuple of str, optional
        List of identifiers that are declared ``public`` in an access-spec statement.
        Default value is `None` which is stored as an empty tuple.
    private_access_spec : tuple of str, optional
        List of identifiers that are declared ``private`` in an access-spec statement.
        Default value is `None` which is stored as an empty tuple.
    ast : optional
        The node for this module from the parse tree produced by the frontend.
    source : :any:`Source`, optional
        Object representing the raw source string information from the read file.
    parent : :any:`Scope`, optional
        The enclosing parent scope of the module. Declarations from the parent
        scope remain valid within the module's scope (unless shadowed by local
        declarations).
    rescope_symbols : bool, optional
        Ensure that the type information for all :any:`TypedSymbol` in the
        module's IR exist in the module's scope. Defaults to `False`.
    symbol_attrs : :any:`SymbolTable`, optional
        Use the provided :any:`SymbolTable` object instead of creating a new
    incomplete : bool, optional
        Mark the object as incomplete, i.e. only partially parsed. This is
        typically the case when it was instantiated using the :any:`Frontend.REGEX`
        frontend and a full parse using one of the other frontends is pending.
    """

    def __init__(self, name=None, docstring=None, spec=None, contains=None,
                 default_access_spec=None, public_access_spec=None, private_access_spec=None,
                 ast=None, source=None, parent=None, rescope_symbols=False, symbol_attrs=None,
                 incomplete=False):

        super().__init__(parent=parent, symbol_attrs=symbol_attrs)

        self.__initialize__(
            name=name, docstring=docstring, spec=spec, contains=contains,
            default_access_spec=default_access_spec, public_access_spec=public_access_spec,
            private_access_spec=private_access_spec, ast=ast, source=source,
            rescope_symbols=rescope_symbols, incomplete=incomplete
        )

    def __initialize__(self, name=None, docstring=None, spec=None, contains=None,
                       default_access_spec=None, public_access_spec=None, private_access_spec=None,
                       ast=None, source=None, rescope_symbols=False, incomplete=False):

        # Apply dimension pragma annotations to declarations
        if spec:
            with pragmas_attached(self, VariableDeclaration):
                spec = process_dimension_pragmas(spec)

        # Store the access spec properties
        self.default_access_spec = None if not default_access_spec else default_access_spec.lower()
        if not public_access_spec:
            self.public_access_spec = ()
        else:
            self.public_access_spec = tuple(v.lower() for v in as_tuple(public_access_spec))
        if not private_access_spec:
            self.private_access_spec = ()
        else:
            self.private_access_spec = tuple(v.lower() for v in as_tuple(private_access_spec))

        super().__initialize__(
            name=name, docstring=docstring, spec=spec, contains=contains, ast=ast,
            source=source, rescope_symbols=rescope_symbols, incomplete=incomplete
        )

    @classmethod
    def from_omni(cls, ast, raw_source, definitions=None, parent=None, type_map=None):
        """
        Create :any:`Module` from :any:`OMNI` parse tree

        Parameters
        ----------
        ast :
            The OMNI parse tree
        raw_source : str
            Fortran source string
        definitions : list, optional
            List of external :any:`Module` to provide derived-type and procedure declarations
        parent : :any:`Scope`, optional
            The enclosing parent scope of the module
        type_map : dict, optional
            A mapping from type hash identifiers to type definitions, as provided in
            OMNI's ``typeTable`` parse tree node
        """
        type_map = type_map or {}
        if ast.tag != 'FmoduleDefinition':
            ast = ast.find('globalDeclarations/FmoduleDefinition')
        return parse_omni_ast(
            ast=ast, definitions=definitions, raw_source=raw_source,
            type_map=type_map, scope=parent
        )

    @classmethod
    def from_ofp(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        """
        Create :any:`Module` from :any:`OFP` parse tree

        Parameters
        ----------
        ast :
            The OFP parse tree
        raw_source : str
            Fortran source string
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        pp_info :
            Preprocessing info as obtained by :any:`sanitize_input`
        parent : :any:`Scope`, optional
            The enclosing parent scope of the module.
        """
        if ast.tag != 'module':
            ast = ast.find('file/module')
        return parse_ofp_ast(
            ast=ast, pp_info=pp_info, raw_source=raw_source,
            definitions=definitions, scope=parent
        )

    @classmethod
    def from_fparser(cls, ast, raw_source, definitions=None, pp_info=None, parent=None):
        """
        Create :any:`Module` from :any:`FP` parse tree

        Parameters
        ----------
        ast :
            The FParser parse tree
        raw_source : str
            Fortran source string
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        pp_info :
            Preprocessing info as obtained by :any:`sanitize_input`
        parent : :any:`Scope`, optional
            The enclosing parent scope of the module.
        """
        if ast.__class__.__name__ != 'Module':
            ast = get_fparser_node(ast, 'Module')
        # Note that our Fparser interface returns a tuple with the
        # Module object always last but potentially containing
        # comments before the Module object
        return parse_fparser_ast(
            ast, pp_info=pp_info, definitions=definitions,
            raw_source=raw_source, scope=parent
        )[-1]

    @classmethod
    def from_regex(cls, raw_source, parser_classes=None, parent=None):
        """
        Create :any:`Module` from source regex'ing

        Parameters
        ----------
        raw_source : str
            Fortran source string
        parent : :any:`Scope`, optional
            The enclosing parent scope of the subroutine, typically a :any:`Module`.
        """
        ir_ = parse_regex_source(raw_source, parser_classes=parser_classes, scope=parent)
        return [node for node in ir_.body if isinstance(node, cls)][0]

    def register_in_parent_scope(self):
        """
        Insert the type information for this object in the parent's symbol table

        If :attr:`parent` is `None`, this does nothing.
        """
        if self.parent:
            self.parent.symbol_attrs[self.name] = SymbolAttributes(self.module_type)

    def clone(self, **kwargs):
        """
        Create a copy of the module with the option to override individual
        parameters.

        Parameters
        ----------
        **kwargs :
            Any parameters from the constructor of :any:`Module`.

        Returns
        -------
        :any:`Module`
            The cloned module object.
        """
        # Collect all properties bespoke to Subroutine
        if self.default_access_spec and 'default_access_spec' not in kwargs:
            kwargs['default_access_spec'] = self.default_access_spec
        if self.public_access_spec and 'public_access_spec' not in kwargs:
            kwargs['public_access_spec'] = self.public_access_spec
        if self.private_access_spec and 'private_access_spec' not in kwargs:
            kwargs['private_access_spec'] = self.private_access_spec

        # Escalate to parent class
        return super().clone(**kwargs)

    @property
    def module_type(self):
        """
        Return the :any:`ModuleType` of this module
        """
        return ModuleType(module=self)

    @property
    def _canonical(self):
        """
        Base definition for comparing :any:`Module` objects.
        """
        return (
            self.name, self.docstring, self.spec, self.contains, self.symbol_attrs,
            self.default_access_spec, self.public_access_spec, self.private_access_spec,
        )

    def __eq__(self, other):
        if isinstance(other, Module):
            return self._canonical == other._canonical
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._canonical)

    def __getstate__(self):
        s = self.__dict__.copy()
        # TODO: We need to remove the AST, as certain AST types
        # (eg. FParser) are not pickle-safe.
        del s['_ast']
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)

        # Re-register all contained procedures in symbol table and update parentage
        if self.contains:
            for node in self.contains.body:
                if isinstance(node, Subroutine):
                    node.parent = self
                    node.register_in_parent_scope()

                if isinstance(node, Scope):
                    node.parent = self

        # Ensure that we are attaching all symbols to the newly create ``self``.
        self.rescope_symbols()
