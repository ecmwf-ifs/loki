# pylint: disable=too-many-lines
import re

try:
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import get_child, walk, BlockBase
    from fparser.two import Fortran2003
    from fparser.common.readfortran import FortranStringReader

    HAVE_FP = True
    """Indicate whether fparser frontend is available."""
except ImportError:
    HAVE_FP = False

from loki.visitors import GenericVisitor, Transformer, FindNodes
from loki.frontend.source import Source
from loki.frontend.preprocessing import sanitize_registry
from loki.frontend.util import read_file, FP, inject_statement_functions, sanitize_ir
from loki import ir
import loki.expression.symbols as sym
from loki.expression.operations import (
    StringConcat, ParenthesisedAdd, ParenthesisedMul, ParenthesisedDiv, ParenthesisedPow
)
from loki.expression import (
    ExpressionDimensionsMapper, FindTypedSymbols, SubstituteExpressions, AttachScopesMapper
)
from loki.logging import DEBUG, warning, error, info
from loki.tools import timeit, as_tuple, flatten, CaseInsensitiveDict
from loki.pragma_utils import (
    attach_pragmas, process_dimension_pragmas, detach_pragmas, pragmas_attached
)
from loki.types import BasicType, DerivedType, ProcedureType, SymbolAttributes
from loki.config import config


__all__ = ['HAVE_FP', 'FParser2IR', 'parse_fparser_file', 'parse_fparser_source',
           'parse_fparser_ast', 'parse_fparser_expression', 'get_fparser_node']


@timeit(log_level=DEBUG)
def parse_fparser_file(filename):
    """
    Generate a parse tree from file via fparser
    """
    info(f'[Loki::FP] Parsing {filename}')
    fcode = read_file(filename)
    return parse_fparser_source(source=fcode)


@timeit(log_level=DEBUG)
def parse_fparser_source(source):
    """
    Generate a parse tree from string
    """
    if not HAVE_FP:
        error('Fparser is not available. Try "pip install fparser".')
        raise RuntimeError

    # Clear FParser's symbol tables if the FParser version is new enough to have them
    try:
        from fparser.two.symbol_table import SYMBOL_TABLES  # pylint: disable=import-outside-toplevel
        SYMBOL_TABLES.clear()
    except ImportError:
        pass

    reader = FortranStringReader(source, ignore_comments=False)
    f2008_parser = ParserFactory().create(std='f2008')

    return f2008_parser(reader)


@timeit(log_level=DEBUG)
def parse_fparser_ast(ast, raw_source, pp_info=None, definitions=None, scope=None):
    """
    Generate an internal IR from fparser parse tree

    Parameters
    ----------
    ast :
        The fparser parse tree as created by :any:`parse_fparser_source` or :any:`parse_fparser_file`
    raw_source : str
        The raw source string from which :attr:`ast` was generated
    pp_info : optional
        Information from internal preprocessing step that was applied to work around
        parser limitations and that should be re-inserted
    definitions : list of :any:`Module`, optional
        List of external module definitions to attach upon use
    scope : :any:`Scope`
        Scope object for which to parse the AST.

    Returns
    -------
    :any:`Node`
        The control flow tree
    """
    # Parse the raw FParser language AST into our internal IR
    _ir = FParser2IR(raw_source=raw_source, definitions=definitions, pp_info=pp_info, scope=scope).visit(ast)
    _ir = sanitize_ir(_ir, FP, pp_registry=sanitize_registry[FP], pp_info=pp_info)
    return _ir


def parse_fparser_expression(source, scope):
    """
    Parse an expression string into an expression tree.

    This exploits Fparser's internal parser structure that relies on recursively
    matching strings against a list of node types. Usually, this would start
    by matching against module, subroutine or program. Here, we shortcut this
    hierarchy by directly matching against a primary expression, thus this
    should be able to parse any syntactically correct Fortran expression.

    Parameters
    ----------
    source : str
        The expression as a string
    scope : :any:`Scope`
        The scope to which symbol names inside the expression belong

    Returns
    -------
    :any:`Expression`
        The expression tree corresponding to the expression
    """
    if not HAVE_FP:
        error('Fparser is not installed')
        raise RuntimeError

    _ = ParserFactory().create(std='f2008')
    # Wrap source in brackets to make sure it appears like a valid expression
    # for fparser, and strip that Parenthesis node from the ast immediately after
    ast = Fortran2003.Primary('(' + source + ')').children[1]
    _ir = parse_fparser_ast(ast, source, scope=scope)
    # TODO: use rescope visitor for this
    rescope_map = {v: v.clone(scope=scope) for v in FindTypedSymbols().visit(_ir)}
    _ir = SubstituteExpressions(rescope_map).visit(_ir)
    return _ir


def get_fparser_node(ast, node_type_name, first_only=True, recurse=False):
    """
    Extract child nodes with type given by :attr:`node_type_name` from an fparser
    parse tree

    Parameters
    ----------
    ast :
        The fparser parse tree as created by :any:`parse_fparser_source` or
        :any:`parse_fparser_file`
    node_type_name : str or list of str
        The name of the node type to extract, e.g. `Module`,
        `Specification_Part` etc.
    first_only : bool, optional
        Return only first instance matching :attr:`node_type_name`.
        Defaults to `True`.
    recurse : bool, optional
        Walk the entire parse tree instead of looking only in the children
        of :attr:`ast`. Defaults to `False`.

    Returns
    -------
    :class:`fparser.two.util.Base`
        The node of requested type (or a list of these nodes if :attr:`all` is `True`)
    """
    node_types = tuple(getattr(Fortran2003, name) for name in as_tuple(node_type_name))

    if recurse:
        nodes = walk(ast, node_types)
    else:
        nodes = [c for c in ast.children if isinstance(c, node_types)]

    if first_only:
        return nodes[0] if nodes else None
    return nodes


def node_sublist(nodelist, starttype, endtype):
    """
    Extract a subset of nodes from a list that sits between marked
    start and end nodes.
    """
    sublist = []
    active = False
    for node in nodelist:
        if isinstance(node, endtype):
            active = False

        if active:
            sublist += [node]

        if isinstance(node, starttype):
            active = True
    return sublist


def rget_child(node, node_type):
    """
    Searches for the last, immediate child of the supplied node that is of
    the specified type.

    :param node: the node whose children will be searched.
    :type node: :py:class:`fparser.two.utils.Base`
    :param node_type: the class(es) of child node to search for.
    :type node_type: type or tuple of type

    :returns: the last child node of type node_type that is encountered or None.
    :rtype: py:class:`fparser.two.utils.Base`

    """
    for child in reversed(node.children):
        if isinstance(child, node_type):
            return child
    return None


def extract_fparser_source(node, raw_source):
    """
    Extract the :any:`Source` object for any py:class:`fparser.two.utils.BlockBase`
    from the raw source string.
    """
    assert isinstance(node, BlockBase)
    if node.item is not None:
        lines = node.item.span
    else:
        start_type = getattr(Fortran2003, node.use_names[0], None)
        if start_type is None:
            # If we don't have any starting point we have to bail out
            return None
        start_node = get_child(node, start_type)
        end_node = node.children[-1]
        if any(i is None or i.item is None for i in [start_node, end_node]):
            # If we don't have source information for start/end we have to bail out
            return None
        lines = (start_node.item.span[0], end_node.item.span[1])
    string = None
    if raw_source is not None:
        string = ''.join(raw_source.splitlines(keepends=True)[lines[0]-1:lines[1]])
    return Source(lines, string=string)


class FParser2IR(GenericVisitor):
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    def __init__(self, raw_source, definitions=None, pp_info=None, scope=None):
        super().__init__()
        self.raw_source = raw_source.splitlines(keepends=True)
        self.definitions = CaseInsensitiveDict((d.name, d) for d in as_tuple(definitions))
        self.pp_info = pp_info
        self.default_scope = scope

    @staticmethod
    def warn_or_fail(msg):
        if config['frontend-strict-mode']:
            error(msg)
            raise NotImplementedError
        warning(msg)

    def get_source(self, o, source):
        """
        Helper method that builds the source object for the node.
        """
        if not isinstance(o, str) and o.item is not None:
            lines = (o.item.span[0], o.item.span[1])
            string = ''.join(self.raw_source[lines[0] - 1:lines[1]]).strip('\n')
            source = Source(lines=lines, string=string)
        return source

    def get_block_source(self, start_node, end_node):
        """
        Helper method that builds the source object for a block node.
        """
        # Extract source by looking at everything between start_type and end_type nodes
        lines = (start_node.item.span[0], end_node.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)
        return source

    def get_label(self, o):
        """
        Helper method that returns the label of the node.
        """
        if not isinstance(o, str) and o.item is not None:
            return getattr(o.item, 'label', None)
        return None

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        kwargs['source'] = self.get_source(o, kwargs.get('source'))
        kwargs['label'] = self.get_label(o)
        kwargs.setdefault('scope', self.default_scope)
        return super().visit(o, **kwargs)

    def visit_List(self, o, **kwargs):
        """
        Universal routine for auto-generated ``*_List`` types in fparser

        ``*_List`` types have their items children
        """
        return tuple(self.visit(i, **kwargs) for i in o.children)

    def visit_Intrinsic_Stmt(self, o, **kwargs):
        """
        Universal routine to capture nodes as plain string in the IR
        """
        return ir.Intrinsic(text=o.tostr(), label=kwargs.get('label'), source=kwargs.get('source'))

    #
    # Base blocks
    #

    def visit_Specification_Part(self, o, **kwargs):
        """
        The specification part of a program-unit

        :class:`fparser.two.Fortran2003.Specification_Part` has variable number
        of children making up the body of the spec.
        """
        children = as_tuple(flatten(self.visit(c, **kwargs) for c in o.children))
        return ir.Section(body=children, source=kwargs.get('source'))

    visit_Implicit_Part = visit_List

    visit_Program = visit_Specification_Part
    visit_Execution_Part = visit_Specification_Part
    visit_Internal_Subprogram_Part = visit_Specification_Part
    visit_Module_Subprogram_Part = visit_Specification_Part

    #
    # Variable, procedure and type names
    #

    def visit_Name(self, o, **kwargs):
        """
        A symbol name

        :class:`fparser.two.Fortran2003.Name` has no children.
        """
        return sym.Variable(name=o.tostr(), source=kwargs.get('source'))

    def visit_Type_Name(self, o, **kwargs):
        """
        A derived type name

        :class:`fparser.two.Fortran2003.Type_Name` has no children.
        """
        return DerivedType(o.tostr())

    def visit_Part_Ref(self, o, **kwargs):
        """
        A part of a data ref (e.g., flat variable or array name, or name of a
        derived type variable or member) and, optionally, a subscript list

        :class:`fparser.two.Fortran2003.Part_Ref` has two children:

        * :class:`fparser.two.Fortran2003.Name`: the part name
        * :class:`fparser.two.Fortran2003.Section_Subscript_List`: the
          subscript (or `None`)
        """
        name = self.visit(o.children[0], **kwargs)
        dimensions = self.visit(o.children[1], **kwargs)
        if dimensions:
            name = name.clone(dimensions=dimensions)

        # Fparser wrongfully interprets function calls as Part_Ref sometimes
        # This should go away once fparser has a basic symbol table, see
        # https://github.com/stfc/fparser/issues/201 for some details
        _type = kwargs['scope'].symbol_attrs.lookup(name.name)
        if _type and isinstance(_type.dtype, ProcedureType):
            name = name.clone(dimensions=None)
            call = sym.InlineCall(name, parameters=dimensions, kw_parameters=(), source=kwargs.get('source'))
            return call
        return name

    def visit_Data_Ref(self, o, **kwargs):
        """
        A fully qualified name for accessing a derived type or class member,
        composed from individual :class:`fparser.two.Fortran2003.Part_Ref` as
        ``part-ref [% part-ref [% part-ref ...] ]``

        :class:`fparser.two.Fortran2003.Data_Ref` has variable number of children,
        depending on the number of part-ref.
        """
        var = self.visit(o.children[0], **kwargs)
        for c in o.children[1:]:
            parent = var
            var = self.visit(c, **kwargs)
            if isinstance(var, sym.InlineCall):
                # This is a function call with a type-bound procedure, so we need to
                # update the name slightly different
                function = var.function.clone(name=f'{parent.name}%{var.function.name}', parent=parent)
                var = var.clone(function=function)
            else:
                var = var.clone(name=f'{parent.name}%{var.name}', parent=parent)
        return var

    #
    # Imports of external names
    #

    def visit_Use_Stmt(self, o, **kwargs):
        """
        An import of symbol names via ``USE``

        :class:`fparser.two.Fortran2003.Use_Stmt` has five children:

        * module-nature (`str`: 'INTRINSIC' or 'NON_INTRINSIC' or `None` if absent)
        * '::' (`str`) if a double colon is used, otherwise `None`
        * module-name :class:`fparser.two.Fortran2003.Module_Name`

        followed by

        * ', ONLY:' (`str`) and :class:`fparser.two.Fortran2003.Only_List`, or
        * ',' (`str`) and :class:`fparser.two.Fortran2003.Rename_List`, or
        * '' (`str`) and no only-list or rename-list
        """
        if o.children[0] is not None:
            # Module nature
            nature = str(o.children[0])
        else:
            nature = None
        name = o.children[2].tostr()
        if nature and nature.lower() == 'intrinsic':
            # Do not use module ref if we refer to an intrinsic module
            module = None
        else:
            module = self.definitions.get(name)
        scope = kwargs['scope']
        if o.children[3] == '' or o.children[3] == ',':
            # No ONLY list (import all)
            symbols = None
            # Rename list
            if o.children[4]:
                rename_list = dict(self.visit(o.children[4], **kwargs))
            else:
                rename_list = {}
            if module is not None:
                # Import symbol attributes from module, if available
                for k, v in module.symbol_attrs.items():
                    if k in rename_list:
                        local_name = rename_list[k].name
                        scope.symbol_attrs[local_name] = v.clone(imported=True, module=module, use_name=k)
                    else:
                        # Need to explicitly reset use_name in case we are importing a symbol
                        # that stems from an import with a rename-list
                        scope.symbol_attrs[k] = v.clone(imported=True, module=module, use_name=None)
            elif rename_list:
                # Module not available but some information via rename-list
                scope.symbol_attrs.update({
                    v.name: v.type.clone(imported=True, use_name=k) for k, v in rename_list.items()
                })
            rename_list = tuple(rename_list.items()) if rename_list else None
        elif o.children[3] == ', ONLY:':
            # ONLY list given (import only selected symbols)
            symbols = () if o.children[4] is None else self.visit(o.children[4], **kwargs)
            # No rename-list
            rename_list = None
            if module is None:
                # Initialize symbol attributes as DEFERRED
                for s in symbols:
                    _type = SymbolAttributes(BasicType.DEFERRED, imported=True)
                    if isinstance(s, tuple):  # Renamed symbol
                        scope.symbol_attrs[s[1].name] = _type.clone(use_name=s[0])
                    else:
                        scope.symbol_attrs[s.name] = _type
            else:
                # Import symbol attributes from module
                for s in symbols:
                    if isinstance(s, tuple):  # Renamed symbol
                        scope.symbol_attrs[s[1].name] = module.symbol_attrs[s[0]].clone(
                            imported=True, module=module, use_name=s[0]
                        )
                    else:
                        # Need to explicitly reset use_name in case we are importing a symbol
                        # that stems from an import with a rename-list
                        scope.symbol_attrs[s.name] = module.symbol_attrs[s.name].clone(
                            imported=True, module=module, use_name=None
                        )
            symbols = tuple(
                s[1].rescope(scope=scope) if isinstance(s, tuple) else s.rescope(scope=scope) for s in symbols
            )
        else:
            raise ValueError(f'Unexpected only/rename-list value in USE statement: {o.children[3]}')

        return ir.Import(module=name, symbols=symbols, nature=nature, rename_list=rename_list,
                         source=kwargs.get('source'), label=kwargs.get('label'))

    visit_Only_List = visit_List
    visit_Rename_List = visit_List

    def visit_Rename(self, o, **kwargs):
        """
        A rename of an imported symbol

        :class:`fparser.two.Fortran2003.Rename` has three children:

        * 'OPERATOR' (`str`) or `None`
        * :class:`fparser.two.Fortran2003.Local_Name` or
          :class:`fparser.two.Fortran2003.Local_Defined_Operator`
        * :class:`fparser.two.Fortran2003.Use_Name` or
          :class:`fparser.two.Fortran2003.Use_Defined_Operator`
        """
        if o.children[0] == 'OPERATOR':
            self.warn_or_fail('OPERATOR in rename-list not yet implemented')
            return ()
        assert o.children[0] is None
        return (str(o.children[2]), self.visit(o.children[1], **kwargs))

    #
    # Variable declarations
    #

    def visit_Type_Declaration_Stmt(self, o, **kwargs):
        """
        Variable declaration statement

        :class:`fparser.two.Fortran2003.Type_Declaration_Stmt` has 3 children:

        * :class:`fparser.two.Fortran2003.Declaration_Type_Spec`
          (:class:`fparser.two.Fortran2003.Intrinsic_Type_Spec` or
          :class:`fparser.two.Fortran2003.Derived_Type_Spec`)
        * :class:`fparser.two.Fortran2003.Attr_Spec_List`
        * :class:`fparser.two.Fortran2003.Entity_Decl_List`
        """
        # First, obtain data type and attributes
        _type = self.visit(o.children[0], **kwargs)
        attrs = self.visit(o.children[1], **kwargs) if o.children[1] else ()
        attrs = dict(attrs)

        # Then, build the common symbol type for all variables
        _type = _type.clone(**attrs)

        # Last, instantiate declared variables
        variables = as_tuple(self.visit(o.children[2], **kwargs))

        # DIMENSION is called shape for us
        if _type.dimension:
            _type = _type.clone(shape=_type.dimension, dimension=None)
            # Attach dimension attribute to variable declaration for uniform
            # representation of variables in declarations
            variables = as_tuple(v.clone(dimensions=_type.shape) for v in variables)

        # Make sure KIND and INITIAL (which can be a name) are in the right scope
        scope = kwargs['scope']
        if _type.kind is not None:
            kind = AttachScopesMapper()(_type.kind, scope=scope)
            _type = _type.clone(kind=kind)
        if _type.initial is not None:
            initial = AttachScopesMapper()(_type.initial, scope=scope)
            _type = _type.clone(initial=initial)

        # EXTERNAL attribute means this is actually a function or subroutine
        # Since every symbol refers to a different function we have to update the
        # type definition for every symbol individually
        if _type.external:
            for var in variables:
                type_kwargs = _type.__dict__.copy()
                return_type = SymbolAttributes(_type.dtype) if _type.dtype is not None else None
                external_type = scope.symbol_attrs.lookup(var.name)
                if external_type is None:
                    type_kwargs['dtype'] = ProcedureType(
                        var.name, is_function=return_type is not None, return_type=return_type
                    )
                else:
                    type_kwargs['dtype'] = external_type.dtype
                scope.symbol_attrs[var.name] = var.type.clone(**type_kwargs)

            variables = tuple(var.rescope(scope=scope) for var in variables)
            return ir.ProcedureDeclaration(
                symbols=variables, external=True, source=kwargs.get('source'), label=kwargs.get('label')
            )

        # Update symbol table entries and rescope
        scope.symbol_attrs.update({var.name: var.type.clone(**_type.__dict__) for var in variables})
        variables = tuple(var.rescope(scope=scope) for var in variables)

        return ir.VariableDeclaration(
            symbols=variables, dimensions=_type.shape,
            source=kwargs.get('source'), label=kwargs.get('label')
        )

    def visit_Intrinsic_Type_Spec(self, o, **kwargs):
        """
        An intrinsic type

        :class:`fparser.two.Fortran2003.Intrinsic_Type_Spec` has 2 children:

        * type name (str)
        * kind (:class:`fparser.two.Fortran2003.Kind_Selector`) or length
          (:class:`fparser.two.Fortran2003.Length_Selector`)
        """
        dtype = BasicType.from_str(o.children[0])
        if o.children[1]:
            if dtype not in (
                BasicType.INTEGER, BasicType.REAL, BasicType.COMPLEX, BasicType.LOGICAL, BasicType.CHARACTER
            ):
                raise ValueError(f'Unknown kind for intrinsic type: {o.children[0]}')

            attr = self.visit(o.children[1], **kwargs)
            if attr:
                attr = dict(attr)
                return SymbolAttributes(dtype, **attr)
        return SymbolAttributes(dtype)

    def visit_Kind_Selector(self, o, **kwargs):
        """
        A kind selector of an intrinsic type

        :class:`fparser.two.Fortran2003.Kind_Selector` has 2 or 3 children:

        * ``'*'`` (str) and :class:`fparser.two.Fortran2003.Char_Length`, or
        * ``'('`` (str), :class:`fparser.two.Fortran2003.Scalar_Int_Initialization_Expr`,
          and ``')'`` (str)
        """
        if len(o.children) in (2, 3) and (o.children[0] == '*' or o.children[0] + str(o.children[-1]) == '()'):
            return (('kind', self.visit(o.children[1], **kwargs)),)
        self.warn_or_fail('Unknown kind selector')
        return None

    def visit_Length_Selector(self, o, **kwargs):
        """
        A length selector for intrinsic character type

        :class:`fparser.two.Fortran2003.Length_Selector` has 3 children:

        * '(' (str)
        * :class:`fparser.two.Fortran2003.Char_Length` or
          :class:`fparser.two.Fortran2003.Type_Param_Value`
        * ')' (str)
        """
        assert o.children[0] == '*' or (o.children[0] == '(' and o.children[2] == ')')
        return (('length', self.visit(o.children[1], **kwargs)),)

    def visit_Char_Length(self, o, **kwargs):
        """
        Length specifier in the Length_Selector

        :class:`fparser.two.Fortran2003.Length_Selector` has one child:

        * length value (str)
        """
        assert o.children[0] == '(' and o.children[2] == ')'
        return self.visit(o.children[1], **kwargs)

    def visit_Char_Selector(self, o, **kwargs):
        """
        Length- and kind-selector for intrinsic character type

        :class:`fparser.two.Fortran2003.Char_Selector` has 2 children:

        * :class:`fparser.two.Fortran2003.Length_Selector`
        * some scalar expression for the kind
        """
        length = None
        if o.children[0] is not None:
            length = self.visit(o.children[0], **kwargs)
        if o.children[1] is not None:
            kind = self.visit(o.children[1], **kwargs)
        return (('length', length), ('kind', kind))

    def visit_Type_Param_Value(self, o, **kwargs):
        """
        The value of a type parameter in a type spefication (such as
        length of a CHARACTER)

        :class:`fparser.two.Fortran2003.Type_Param_Value` has only 1 attribute:

          * :attr:`string` : the value of the parameter (str)
        """
        if o.string in '*:':
            return o.string
        return self.visit(o.string, **kwargs)

    def visit_Declaration_Type_Spec(self, o, **kwargs):
        """
        A derived type specifier in a declaration

        :class:`fparser.two.Fortran2003.Declaration_Type_Spec` has 2 children:

        * keyword 'TYPE' or 'CLASS' (str)
        * :class:`fparser.two.Fortran2003.Derived_Type_Spec`
        """
        if o.children[0].upper() in ('TYPE', 'CLASS'):
            dtype = self.visit(o.children[1], **kwargs)

            # Look for a previous definition of this type
            _type = kwargs['scope'].symbol_attrs.lookup(dtype.name)
            if _type is None or _type.dtype is BasicType.DEFERRED:
                _type = SymbolAttributes(dtype)

            if o.children[0].upper() == 'CLASS':
                _type.polymorphic = True

            # Strip import annotations
            return _type.clone(imported=None, module=None)

        return self.visit_Base(o, **kwargs)

    def visit_Dimension_Attr_Spec(self, o, **kwargs):
        """
        The dimension specification as attribute in a declaration

        :class:`fparser.two.Fortran2003.Dimensions_Attr_Spec` has 2 children:

        * attribute name (str)
        * :class:`fparser.two.Fortran2003.Array_Spec`
        """
        return (o.children[0].lower(), self.visit(o.children[1], **kwargs))

    def visit_Intent_Attr_Spec(self, o, **kwargs):
        """
        The intent specification in a declaration

        :class:`fparser.two.Fortran2003.Intent_Attr_Spec` has 2 children:

        * 'INTENT' keyword
        * :class:`fparser.two.Fortran2003.Intent_Spec`
        """
        return (o.children[0].lower(), o.children[1].tostr().lower())

    visit_Attr_Spec_List = visit_List

    def visit_Attr_Spec(self, o, **kwargs):
        """
        A declaration attribute

        :class:`fparser.two.Fortran2003.Attr_Spec` has no children.
        """
        return (str(o).lower(), True)

    def visit_Access_Spec(self, o, **kwargs):
        """
        A declaration attribute for access specification (PRIVATE, PUBLIC)

        :class:`fparser.two.Fortran2003.Access_Spec` has no children.
        """
        return (o.string.lower(), True)

    visit_Entity_Decl_List = visit_List

    def visit_Entity_Decl(self, o, **kwargs):
        """
        A variable entity in a declaration

        :class:`fparser.two.Fortran2003.Entity_Decl` has 4 children:

        * object name (:class:`fparser.two.Fortran2003.Name`)
        * array spec (:class:`fparser.two.Fortran2003.Array_Spec`)
        * char length (:class:`fparser.two.Fortran2003.Char_Length`)
        * init (:class:`fparser.two.Fortran2003.Initialization`)
        """
        var = self.visit(o.children[0], **kwargs)

        if o.children[1]:
            dimensions = as_tuple(self.visit(o.children[1], **kwargs))
            var = var.clone(dimensions=dimensions, type=var.type.clone(shape=dimensions))

        if o.children[2]:
            char_length = self.visit(o.children[2], **kwargs)
            var = var.clone(type=var.type.clone(length=char_length))

        if o.children[3]:
            init = self.visit(o.children[3], **kwargs)
            var = var.clone(type=var.type.clone(initial=init))

        return var

    def visit_Explicit_Shape_Spec(self, o, **kwargs):
        """
        Explicit shape specification for arrays

        :class:`fparser.two.Fortran2003.Explicit_Shape_Spec` has 2 children:

        * lower bound (if explicitly given)
        * upper bound
        """
        lower_bound, upper_bound = None, None
        if o.children[1] is not None:
            upper_bound = self.visit(o.children[1], **kwargs)
        if o.children[0] is not None:
            lower_bound = self.visit(o.children[0], **kwargs)
        if upper_bound is not None and lower_bound is None:
            return upper_bound
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.RangeIndex((lower_bound, upper_bound), source=source)

    visit_Explicit_Shape_Spec_List = visit_List
    visit_Assumed_Shape_Spec = visit_Explicit_Shape_Spec
    visit_Assumed_Shape_Spec_List = visit_List
    visit_Assumed_Size_Spec = visit_Explicit_Shape_Spec
    visit_Deferred_Shape_Spec = visit_Explicit_Shape_Spec
    visit_Deferred_Shape_Spec_List = visit_List

    def visit_Initialization(self, o, **kwargs):
        """
        Variable initialization in declaration

        :class:`fparser.two.Fortran2003.Initialization` has 2 children:

        * '=' or '=>' (str)
        * init expr
        """
        if o.children[0] == '=':
            return self.visit(o.items[1], **kwargs)
        if o.children[0] == '=>':
            return self.visit(o.items[1], **kwargs)
        raise ValueError(f'Invalid assignment operator {o.children[0]}')

    visit_Component_Initialization = visit_Initialization

    def visit_External_Stmt(self, o, **kwargs):
        """
        An ``EXTERNAL`` statement to specify the external attribute for a list of names

        :class:`fparser.two.Fortran2003.External_Stmt` has 2 children:

        * keyword 'EXTERNAL (`str`)
        * the list of names :class:`fparser.two.Fortran2003.External_Name_List`
        """
        assert o.children[0].upper() == 'EXTERNAL'

        # Compile the list of names...
        symbols = self.visit(o.children[1], **kwargs)

        # ...and update their symbol table entry...
        scope = kwargs['scope']
        for var in symbols:
            _type = scope.symbol_attrs.lookup(var.name)
            if _type is None:
                dtype = ProcedureType(var.name, is_function=False)
            else:
                dtype = _type.dtype
            scope.symbol_attrs[var.name] = SymbolAttributes(dtype, external=True)

        symbols = tuple(v.rescope(scope=scope) for v in symbols)
        declaration = ir.ProcedureDeclaration(symbols=symbols, external=True,
                                              source=kwargs.get('source'), label=kwargs.get('label'))
        return declaration

    visit_External_Name_List = visit_List

    def visit_Access_Stmt(self, o, **kwargs):
        """
        An access-spec statement that specifies accessibility of symbols in a module

        :class:`faprser.two.Fortran2003.Access_Stmt` has 2 children:

        * keyword ``PRIVATE`` or ``PUBLIC`` (`str`)
        * optional list of names (:class:`fparser.two.Fortran2003.Access_Id_List`) or `None`
        """
        from loki.module import Module  # pylint: disable=import-outside-toplevel,cyclic-import
        assert isinstance(kwargs['scope'], Module)
        assert o.children[0] in ('PUBLIC', 'PRIVATE')

        if o.children[1] is None:
            assert kwargs['scope'].default_access_spec is None
            kwargs['scope'].default_access_spec = o.children[0].lower()
        else:
            access_id_list = [str(name).lower() for name in o.children[1].children]
            if o.children[0] == 'PUBLIC':
                kwargs['scope'].public_access_spec += as_tuple(access_id_list)
            else:
                kwargs['scope'].private_access_spec += as_tuple(access_id_list)

    #
    # Procedure declarations
    #

    def visit_Procedure_Declaration_Stmt(self, o, **kwargs):
        """
        Procedure declaration statement

        :class:`fparser.two.Fortran2003.Procedure_Declaration_Stmt` has 3 children:

        * :class:`fparser.two.Fortran2003.Name`: the name of the procedure interface
        * :class:`fparser.two.Fortran2003.Proc_Attr_Spec_List` or `None`:
          the declared attributes (if any)
        * :class:`fparser.two.Fortran2003.Proc_Decl_List`: the local procedure names
        """
        scope = kwargs['scope']

        # Instantiate declared symbols
        symbols = as_tuple(self.visit(o.children[2], **kwargs))

        # Any additional declared attributes
        attrs = self.visit(o.children[1], **kwargs) if o.children[1] else ()
        attrs = dict(attrs)

        # Find out which procedure we are declaring (i.e., PROCEDURE(<func_name>))
        assert o.children[0] is not None
        try:
            # This could be an implicit interface or dummy routine...
            return_type = SymbolAttributes(BasicType.from_str(o.children[0].tostr()))
        except ValueError:
            return_type = None

        if return_type is None:
            interface = self.visit(o.children[0], **kwargs)
            interface = AttachScopesMapper()(interface, scope=scope)
            _type = interface.type.clone(**attrs)
        else:
            interface = return_type.dtype
            _type = SymbolAttributes(BasicType.DEFERRED, **attrs)

        # Make sure any "initial" symbol (i.e. the procedure we're binding to) is in the right scope
        if _type.initial is not None:
            initial = AttachScopesMapper()(_type.initial, scope=scope)
            _type = _type.clone(initial=initial)

        # Update symbol table entries
        if return_type is None:
            scope.symbol_attrs.update({var.name: var.type.clone(**_type.__dict__) for var in symbols})
        else:
            for var in symbols:
                dtype = ProcedureType(var.name, is_function=True, return_type=return_type)
                scope.symbol_attrs[var.name] = _type.clone(dtype=dtype)

        symbols = tuple(var.rescope(scope=scope) for var in symbols)
        return ir.ProcedureDeclaration(
            symbols=symbols, interface=interface, source=kwargs.get('source'), label=kwargs.get('label')
        )

    visit_Proc_Attr_Spec_List = visit_List

    def visit_Proc_Attr_Spec(self, o, **kwargs):
        """
        Procedure declaration attribute

        :class:`fparser.two.Fortran2003.Proc_Attr_Spec` has 2 children:

        * attribute name (`str`)
        * attribute value (such as ``IN``, ``OUT``, ``INOUT``) or `None`
        """
        return (o.children[0].lower(), o.children[1].lower() if o.children[1] is not None else True)

    visit_Proc_Decl_List = visit_List

    def visit_Proc_Decl(self, o, **kwargs):
        """
        A symbol entity in a procedure declaration with initialization

        :class:`fparser.two.Fortran2003.Proc_Decl` has 3 children:

        * object name (:class:`fparser.two.Fortran2003.Name`)
        * operator ``=>`` (`str`)
        * initializer (:class:`fparser.two.Fortran2003.Function_Reference`)
        """
        var = self.visit(o.children[0], **kwargs)
        assert o.children[1] == '=>'
        init = self.visit(o.children[2], **kwargs)
        return var.clone(type=var.type.clone(initial=init))

    #
    # Array constructor
    #

    def visit_Array_Constructor(self, o, **kwargs):
        """
        An array constructor expression

        :class:`fparser.two.Fortran2003.Array_Constructor` has three children:

        * left bracket (`str`): ``(/`` or ``[``
        * the spec: :class:`fparser.two.Fortran2003.Ac_Spec`
        * right bracket (`str`): ``/)`` or ``]``
        """
        return self.visit(o.children[1], **kwargs)

    def visit_Ac_Spec(self, o, **kwargs):
        """
        The spec in an array constructor

        :class:`fparser.two.Fortran2003.Ac_Spec` has two children:

        * :class:`fparser.two.Fortran2003.Type_Spec`
        * :class:`fparser.two.Fortran2003.Ac_Value_List`
        """
        if o.children[0] is not None:
            # TODO: implement Type_Spec support
            return self.visit_Base(o, **kwargs)
        return self.visit(o.children[1], **kwargs)

    def visit_Ac_Value_List(self, o, **kwargs):
        """
        The list of values in an array constructor
        """
        values = as_tuple(self.visit(c, **kwargs) for c in o.children)
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.LiteralList(values=values, source=source)

    def visit_Ac_Implied_Do(self, o, **kwargs):
        """
        An implied-do for array constructors
        """
        # TODO: implement implied-do
        return self.visit_Base(o, **kwargs)

    visit_Ac_Implied_Do_Control = visit_Ac_Implied_Do

    #
    # DATA statements
    #

    def visit_Data_Stmt(self, o, **kwargs):
        """
        A ``DATA`` statement

        :class:`fparser.two.Fortran2003.Data_Stmt` has variable number of
        children :class:`fparser.two.Fortran2003.Data_Stmt_Set`.
        """
        data_statements = tuple(self.visit(data_set, **kwargs) for data_set in o.children)
        return data_statements

    def visit_Data_Stmt_Set(self, o, **kwargs):
        """
        A data-stmt-set in a data-stmt

        :class:`fparser.two.Fortran2003.Data_Stmt_Set` has two children:

        * the object to initialize :class:`fparser.two.Fortran2003.Data_Stmt_Object`
        * the value list :class:`fparser.two.Fortran2003.Data_Stmt_Value_List`
        """
        variable = self.visit(o.children[0], **kwargs)
        values = self.visit(o.children[1], **kwargs)
        return ir.DataDeclaration(variable=variable, values=values,
                                  label=kwargs.get('label'), source=kwargs.get('source'))

    def visit_Data_Implied_Do(self, o, **kwargs):
        """
        An implied-do for data-stmt
        """
        # TODO: Implement implied-do
        return self.visit_Base(o, **kwargs)

    visit_Data_Stmt_Object_List = visit_List
    visit_Data_Stmt_Value_List = visit_List

    def visit_Data_Stmt_Value(self, o, **kwargs):
        """
        A value in a data-stmt-set

        :class:`fparser.two.Fortran2003.Data_Stmt_Value` has two children:

        * the repeat value :class:`fparser.two.Fortran2003.Data_Stmt_Repeat`
        * the constant :class:`fparser.two.Fortran2003.Data_Stmt_Constant`
        """
        constant = self.visit(o.children[1], **kwargs)
        if o.children[0] is None:
            return constant

        repeat = self.visit(o.children[0], **kwargs)
        return self.create_operation('*', (repeat, constant), kwargs.get('source'))

    #
    # Subscripts
    #

    visit_Section_Subscript_List = visit_List

    def visit_Subscript_Triplet(self, o, **kwargs):
        """
        A subscript expression with ``[start] : [stop] [: stride]``

        :class:`fparser.two.Fortran2003.Subscript_Triplet` has three children:

        * start :class:`fparser.two.Fortran2003.Subscript` or `None`
        * stop :class:`fparser.two.Fortran2003.Subscript` or `None`
        * stride :class:`fparser.two.Fortran2003.Stride` or `None`
        """
        start = self.visit(o.children[0], **kwargs) if o.children[0] is not None else None
        stop = self.visit(o.children[1], **kwargs) if o.children[1] is not None else None
        stride = self.visit(o.children[2], **kwargs) if o.children[2] is not None else None
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.RangeIndex((start, stop, stride), source=source)

    def visit_Array_Section(self, o, **kwargs):
        # TODO: Implement Array_Section
        return self.visit_Base(o, **kwargs)

    def visit_Substring_Range(self, o, **kwargs):
        # TODO: Implement Substring_Range
        return self.visit_Base(o, **kwargs)

    def visit_Stride(self, o, **kwargs):
        # TODO: Implement Stride
        return self.visit_Base(o, **kwargs)

    #
    # Derived Type definition
    #

    def visit_Derived_Type_Def(self, o, **kwargs):
        """
        A derived type definition

        :class:`fparser.two.Fortran2003.Derived_Type_Def` has variable number of children:

        * header stmt (:class:`fparser.two.Fortran2003.Derived_Type_Stmt`)
        * all of body (list of :class:`fparser.two.Fortran2003.Type_Param_Def_Stmt`,
          :class:`fparser.two.Fortran2003.Private_Or_Sequence`,
          :class:`fparser.two.Fortran2003.Component_Part`,
          :class:`fparser.two.Fortran2003.Type_Bound_Procedure_Part`)
        * end stmt (:class:`fparser.two.Fortran2003.End_Type_Stmt`)
        """
        # Find start and end of construct
        derived_type_stmt = get_child(o, Fortran2003.Derived_Type_Stmt)
        derived_type_stmt_index = o.children.index(derived_type_stmt)
        end_type_stmt = get_child(o, Fortran2003.End_Type_Stmt)
        end_type_stmt_index = o.children.index(end_type_stmt)

        # Everything before the construct
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:derived_type_stmt_index])

        # Instantiate the TypeDef without its body
        # Note: This creates the symbol table for the declarations and
        # the typedef object registers itself in the parent scope
        typedef = self.visit(derived_type_stmt, **kwargs)

        # Pass down the typedef scope when building the body
        kwargs['scope'] = typedef
        body = [self.visit(c, **kwargs) for c in o.children[derived_type_stmt_index+1:end_type_stmt_index]]
        body = as_tuple(flatten(body))

        # Infer any additional shape information from `!$loki dimension` pragmas
        body = attach_pragmas(body, ir.VariableDeclaration)
        body = process_dimension_pragmas(body)
        body = detach_pragmas(body, ir.VariableDeclaration)

        # Finally: update the typedef with its body and make sure all symbols
        # are in the right scope
        typedef._update(body=body, source=kwargs['source'])
        typedef.rescope_symbols()
        return (*pre, typedef)

    def visit_Derived_Type_Stmt(self, o, **kwargs):
        """
        The block header for the derived type definition

        :class:`fparser.two.Fortran2003.Derived_Type_Stmt` has 3 children:

        * attribute spec list (:class:`fparser.two.Fortran2003.Type_Attr_Spec_List`)
        * type name (:class:`fparser.two.Fortran2003.Type_Name`)
        * parameter name list (:class:`fparser.two.Fortran2003.Type_Param_Name_List`)
        """
        if o.children[0] is not None:
            attrs = dict(self.visit(o.children[0], **kwargs))
            abstract = attrs.get('abstract', False)
            extends = attrs.get('extends')
            bind_c = attrs.get('bind') == 'c'
            private = attrs.get('private', False)
            public = attrs.get('public', False)
        else:
            abstract = False
            extends = None
            bind_c = False
            private = False
            public = False
        name = o.children[1].tostr()
        if o.children[2] is not None:
            self.warn_or_fail('parameter-name-list not implemented for derived types')
        return ir.TypeDef(
            name=name, body=(), abstract=abstract, extends=extends, bind_c=bind_c,
            private=private, public=public, label=kwargs['label'], parent=kwargs['scope']
        )

    visit_Type_Attr_Spec_List = visit_List

    def visit_Type_Attr_Spec(self, o, **kwargs):
        """
        A component declaration attribute

        :class:`fparser.two.Fortran2003.Type_Attr_Spec` has 2 children:

        * keyword (`str`)
        * value (`str`) or `None`
        """
        if o.children[1] is not None:
            return (str(o.children[0]).lower(), str(o.children[1]).lower())
        return (str(o.children[0]).lower(), True)

    def visit_Type_Param_Def_Stmt(self,o , **kwargs):
        self.warn_or_fail('Parameterized types not implemented')

    visit_Binding_Attr_List = visit_List

    def visit_Binding_Attr(self, o, **kwargs):
        """
        A binding attribute

        :class:`fparser.two.Fortran2003.Binding_Attr_Spec` has no children
        """
        keyword = str(o).lower()
        if keyword == 'pass':
            return ('pass_attr', True)
        if keyword == 'nopass':
            return ('pass_attr', False)

        if keyword in ('non_overridable', 'deferred'):
            return (keyword, True)

        self.warn_or_fail(f'Unsupported binding attribute: {str(o)}')
        return None

    def visit_Binding_PASS_Arg_Name(self, o, **kwargs):
        """
        Named PASS attribute

        :class:`fparser.two.Fortran2003.Binding_PASS_Arg_Name` has two children:

        * `str`: 'PASS'
        * `Name`: the argument name
        """
        return ('pass_attr', str(o.children[1]))

    def visit_Component_Part(self, o, **kwargs):
        """
        Derived type definition components

        :class:`fparser.two.Fortran2003.Component_Part` has a list of
        :class:`fparser.two.Fortran2003.Data_Component_Def_Stmt` or
        :class:`fparser.two.Fortran2003.Proc_Component_Def_Stmt` as children
        """
        return tuple(self.visit(c, **kwargs) for c in o.children)

    # The definition stmts (= components of a derived type) look identical to regular
    # variable and procedure declarations in the parse tree and are represented by
    # the same IR nodes in Loki
    visit_Data_Component_Def_Stmt = visit_Type_Declaration_Stmt
    visit_Component_Attr_Spec_List = visit_List
    visit_Component_Attr_Spec = visit_Attr_Spec
    visit_Dimension_Component_Attr_Spec = visit_Dimension_Attr_Spec
    visit_Component_Decl_List = visit_List
    visit_Component_Decl = visit_Entity_Decl
    visit_Proc_Component_Def_Stmt = visit_Procedure_Declaration_Stmt
    visit_Proc_Component_Attr_Spec_List = visit_List
    visit_Proc_Component_Attr_Spec = visit_Attr_Spec

    def visit_Type_Bound_Procedure_Part(self, o, **kwargs):
        """
        Procedure definitions part in a derived type definition

        :class:`fparser.two.Fortran2003.Type_Bound_Procedure_Part` starts with
        the contains-stmt (:class:`fparser.two.Fortran2003.Contains_Stmt`) followed
        by (optionally) :class:`fparser.two.Fortran2003.Binding_Private_Stmt` and
        a sequence of :class:`fparser.two.Fortran2003.Proc_Binding_Stmt`
        """
        return tuple(self.visit(c, **kwargs) for c in o.children)

    def visit_Specific_Binding(self, o, **kwargs):
        """
        A specific binding for a type-bound procedure in a derived type

        :class:`fparser.two.Fortran2003.Specific_Binding` has five children:

        * interface name :class:`fparser.two.Fortran2003.Interface_Name`
        * binding attr list :class:`fparser.two.Fortran2003.Binding_Attr_List`
        * '::' (`str`) or `None`
        * name :class:`fparser.two.Fortran2003.Binding_Name`
        * procedure name :class:`fparser.two.Fortran2003.Procedure_Name`
        """
        scope = kwargs['scope']

        # Instantiate declared symbols
        symbols = as_tuple(self.visit(o.children[3], **kwargs))

        # Procedure we bind to this type
        interface = None
        if o.children[0]:
            # Procedure interface provided
            interface = self.visit(o.children[0], **kwargs)
            interface = AttachScopesMapper()(interface, scope=scope)
            bind_names = as_tuple(interface)
            func_names = [interface.name] * len(symbols)
            assert o.children[4] is None
        elif o.children[4]:
            bind_names = as_tuple(self.visit(o.children[4], **kwargs))
            bind_names = AttachScopesMapper()(bind_names, scope=scope)
            assert len(bind_names) == len(symbols)
            func_names = [i.name for i in bind_names]
        else:
            bind_names = None
            func_names = [s.name for s in symbols]

        # Look up the type of the procedure
        types = [scope.symbol_attrs.lookup(name) or SymbolAttributes(dtype=ProcedureType(name)) for name in func_names]

        # Any declared attributes
        attrs = self.visit(o.children[1], **kwargs) if o.children[1] else ()
        attrs = dict(attrs)
        types = [t.clone(**attrs) for t in types]

        # Store the bind_names
        if bind_names:
            types = [t.clone(bind_names=as_tuple(i)) for t, i in zip(types, bind_names)]

        # Update symbol table entries
        scope.symbol_attrs.update({s.name: s.type.clone(**t.__dict__) for s, t in zip(symbols, types)})

        symbols = tuple(var.rescope(scope=scope) for var in symbols)
        return ir.ProcedureDeclaration(symbols=symbols, interface=interface,
                                       source=kwargs.get('source'), label=kwargs.get('label'))

    def visit_Generic_Binding(self, o, **kwargs):
        """
        A generic binding for a type-bound procedure in a derived type

        :class:`fparser.two.Fortran2003.Generic_Binding` has three children:

        * :class:`fparser.two.Fortran2003.Access_Spec` or None (access specifier)
        * :class:`fparser.two.Fortran2003.Generic_Spec` (the local name of the binding)
        * :class:`fparser.two.Fortran2003.Binding_Name_List` (the names it binds to)
        """
        scope = kwargs['scope']
        name = self.visit(o.children[1], **kwargs)
        bind_names = self.visit(o.children[2], **kwargs)
        bind_names = AttachScopesMapper()(bind_names, scope=scope)
        _type = SymbolAttributes(ProcedureType(name=name.name, is_generic=True), bind_names=as_tuple(bind_names))
        if o.children[0] is not None:
            access_spec = self.visit(o.children[0], **kwargs)
            attrs = {access_spec[0]: access_spec[1]}
            _type = _type.clone(**attrs)
        scope.symbol_attrs[name.name] = _type
        name = name.rescope(scope=scope)
        return ir.ProcedureDeclaration(
            symbols=(name,), generic=True, source=kwargs.get('source'), label=kwargs.get('label')
        )

    def visit_Final_Binding(self, o, **kwargs):
        """
        A final binding for type-bound procedures in a derived type

        :class:`fparser.two.Fortran2003.Final_Binding` has two children:

        * keyword ``'FINAL'`` (`str`)
        * :class:`fparser.two.Fortran2003.Final_Subroutine_Name_List` (the list of routines)
        """
        scope = kwargs['scope']
        symbols = self.visit(o.children[1], **kwargs)
        symbols = tuple(var.rescope(scope=scope) for var in symbols)
        return ir.ProcedureDeclaration(
            symbols=symbols, final=True, source=kwargs.get('source'), label=kwargs.get('label')
        )

    visit_Binding_Name_List = visit_List
    visit_Final_Subroutine_Name_List = visit_List
    visit_Contains_Stmt = visit_Intrinsic_Stmt
    visit_Binding_Private_Stmt = visit_Intrinsic_Stmt
    visit_Private_Components_Stmt = visit_Intrinsic_Stmt
    visit_Sequence_Stmt = visit_Intrinsic_Stmt

    #
    # ASSOCIATE blocks
    #

    def visit_Associate_Construct(self, o, **kwargs):
        """
        The entire ASSOCIATE construct

        :class:`fparser.two.Fortran2003.Associate_Construct` has a variable
        number of children:

        * Any preceeding comments :class:`fparser.two.Fortran2003.Comment`
        * :class:`fparser.two.Fortran2003.Associate_Stmt` (the actual statement
          with the definition of associates)
        * the body of the ASSOCIATE construct
        * :class:`fparser.two.Fortran2003.End_Associate_Stmt`
        """
        # Find start and end of associate construct
        assoc_stmt = get_child(o, Fortran2003.Associate_Stmt)
        assoc_stmt_index = o.children.index(assoc_stmt)
        end_assoc_stmt = get_child(o, Fortran2003.End_Associate_Stmt)
        end_assoc_stmt_index = o.children.index(end_assoc_stmt)

        # Everything before the associate statement
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:assoc_stmt_index])

        # Extract source object for construct
        lines = (assoc_stmt.item.span[0], end_assoc_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)

        # Handle the associates
        associations = self.visit(assoc_stmt, **kwargs)

        # Create a scope for the associate
        parent_scope = kwargs['scope']
        associate = ir.Associate(associations=associations, body=(), parent=parent_scope,
                                 label=kwargs.get('label'), source=source)
        kwargs['scope'] = associate

        # Put associate expressions into the right scope and determine type of new symbols
        rescoped_associations = []
        for expr, name in associations:
            # Put symbols in associated expression into the right scope
            expr = AttachScopesMapper()(expr, scope=parent_scope)

            # Determine type of new names
            if isinstance(expr, (sym.TypedSymbol, sym.MetaSymbol)):
                # Use the type of the associated variable
                _type = expr.type.clone(parent=None)
                if isinstance(expr, sym.Array) and expr.dimensions is not None:
                    shape = ExpressionDimensionsMapper()(expr)
                    if shape == (sym.IntLiteral(1),):
                        # For a scalar expression, we remove the shape
                        shape = None
                    _type = _type.clone(shape=shape)
            else:
                # TODO: Handle data type and shape of complex expressions
                shape = ExpressionDimensionsMapper()(expr)
                if shape == (sym.IntLiteral(1),):
                    # For a scalar expression, we remove the shape
                    shape = None
                _type = SymbolAttributes(BasicType.DEFERRED, shape=shape)
            name = name.clone(scope=associate, type=_type)
            rescoped_associations += [(expr, name)]
        associations = as_tuple(rescoped_associations)

        # The body
        body = as_tuple(self.visit(c, **kwargs) for c in o.children[assoc_stmt_index+1:end_assoc_stmt_index])
        associate._update(associations=associations, body=body)

        # Everything past the END ASSOCIATE (should be empty)
        assert not o.children[end_assoc_stmt_index+1:]

        return (*pre, associate)

    def visit_Associate_Stmt(self, o, **kwargs):
        """
        The ASSOCIATE statement with the association list

        :class:`fparser.two.Fortran2003.Associate_Stmt` has two children:

        * The command `ASSOCIATE` (`str`)
        * The :class:`fparser.two.Fortran2003.Association_List` defining the
          associations
        """
        assert o.children[0].upper() == 'ASSOCIATE'
        return self.visit(o.children[1], **kwargs)

    visit_Association_List = visit_List

    def visit_Association(self, o, **kwargs):
        """
        A single association in an associate-stmt

        :class:`fparser.two.Fortran2003.Associate` has two children:

        * :class:`fparser.two.Fortran2003.Name` (the new assigned name)
        * the operator ``=>`` (`str`)
        * :class:`fparser.two.Fortran2003.Name` (the associated expression)
        """
        assert o.children[1] == '=>'
        associate_name = self.visit(o.children[0], **kwargs)
        selector = self.visit(o.children[2], **kwargs)
        return (selector, associate_name)  # (associate_name, selector)

    #
    # Interface block
    #

    def visit_Interface_Block(self, o, **kwargs):
        """
        An ``INTERFACE`` block

        :class:`fparser.two.Fortran2003.Interface_Block` has variable number of
        children:

        * Any preceeding comments :class:`fparser.two.Fortran2003.Comment`
        * :class:`fparser.two.Fortran2003.Interface_Stmt` (the actual statement
          that begins the construct)
        * the body, made up of :class:`fparser.two.Fortran2003.Subroutine_Body`,
          :class:`fparser.two.Fortran2003.Function_Body`,
          :class:`fparser.two.Fortran2003.Procedure_Stmt` and, potentially,
          any interleaving comments :class:`fparser.two.Fortran2003.Comment`
        * the closing :class:`fparser.two.Fortran2003.End_Interface_Stmt`
        """
        # Find start and end of construct
        interface_stmt = get_child(o, Fortran2003.Interface_Stmt)
        interface_stmt_index = o.children.index(interface_stmt)
        end_interface_stmt = get_child(o, Fortran2003.End_Interface_Stmt)
        end_interface_stmt_index = o.children.index(end_interface_stmt)

        # Everything before the construct
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:interface_stmt_index])

        # Extract source object for construct
        lines = (interface_stmt.item.span[0], end_interface_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)

        # The interface spec
        abstract = False
        spec = self.visit(interface_stmt, **kwargs)
        if spec == 'ABSTRACT':
            # This is an abstract interface
            abstract = True
            spec = None
        elif spec is not None:
            # This has a generic specification (and we might need to update symbol table)
            scope = kwargs['scope']
            if spec.name not in scope.symbol_attrs:
                scope.symbol_attrs[spec.name] = SymbolAttributes(ProcedureType(name=spec.name, is_generic=True))
            spec = spec.rescope(scope=scope)

        # Traverse the body and build the object
        body = as_tuple(self.visit(c, **kwargs) for c in o.children[interface_stmt_index+1:end_interface_stmt_index])
        interface = ir.Interface(body=body, abstract=abstract, spec=spec, label=kwargs.get('label'), source=source)

        # Everything past the END INTERFACE (should be empty)
        assert not o.children[end_interface_stmt_index+1:]

        return (*pre, interface)

    def visit_Interface_Stmt(self, o, **kwargs):
        """
        The specification of the interface

        :class:`fparser.two.Fortran2003.Interface_Stmt` has one child, which is either:

        * `None`, if no further specification exists
        * ``'ABSTRACT'`` (`str`) for an abstract interface
        * :class:`fparser.two.Fortran2003.Generic_Spec` for other specifications
        """
        if o.children[0] == 'ABSTRACT':
            return 'ABSTRACT'
        if o.children[0] is not None:
            return self.visit(o.children[0], **kwargs)
        return None

    def visit_Generic_Spec(self, o, **kwargs):
        """
        The generic-spec of an interface

        :class:`fparser.two.Fortran2003.Generic_Spec` has two children, which is either:

        * ``'OPERATOR'`` (`str`) followed by
        * :class:`fparser.two.Fortran2003.Defined_Operator`

        -or-

        * ``'ASSIGNMENT'`` (`str`) followed by
        * ``'='`` (`str`)
        """
        return sym.Variable(name=str(o), source=kwargs.get('source'))

    def visit_Procedure_Stmt(self, o, **kwargs):
        """
        Procedure statement

        :class:`fparser.two.Fortran2003.Procedure_Stmt` has 1 child:

        * :class:`fparser.two.Fortran2003.Procedure_Name_List`: the names of the procedures
        """
        module_proc = o.string.upper().startswith('MODULE')
        symbols = self.visit(o.children[0], **kwargs)
        symbols = AttachScopesMapper()(symbols, scope=kwargs['scope'])
        return ir.ProcedureDeclaration(
            symbols=symbols, module=module_proc,
            source=kwargs.get('source'), label=kwargs.get('label')
        )

    visit_Procedure_Name_List = visit_List
    visit_Procedure_Name = visit_Name

    def visit_Import_Stmt(self, o, **kwargs):
        """
        An import statement for named entities in an interface body

        :class:`fparser.two.Fortran2003.Import_Stmt` has two children:

        * The string ``'IMPORT'``
        * :class:`fparser.two.Fortran2003.Import_Name_List` with the names
          of imported entities
        """
        assert o.children[0] == 'IMPORT'
        symbols = self.visit(o.children[1], **kwargs)
        symbols = AttachScopesMapper()(symbols, scope=kwargs['scope'])
        return ir.Import(
            module=None, symbols=symbols, f_import=True, source=kwargs.get('source'), label=kwargs.get('label')
        )

    visit_Import_Name_List = visit_List
    visit_Import_Name = visit_Name

    #
    # Subroutine and Function definitions
    #

    def visit_Subroutine_Subprogram(self, o, **kwargs):
        """
        The entire block that comprises a ``SUBROUTINE`` definition, i.e.
        everything from the subroutine-stmt to the end-stmt

        :class:`fparser.two.Fortran2003.Subroutine_Subprogram` has variable number of children,
        where the internal nodes may be optional:

        * :class:`fparser.two.Fortran2003.Subroutine_Stmt` (the opening statement)
        * :class:`fparser.two.Fortran2003.Specification_Part` (variable declarations, module
          imports etc.); due to an fparser bug, this can appear multiple times interleaved with
          the execution-part
        * :class:`fparser.two.Fortran2003.Execution_Part` (the body of the routine)
        * :class:`fparser.two.Fortran2003.Internal_Subprogram_Part` (any member procedures
          declared inside the procedure)
        * :class:`fparser.two.Fortran2003.End_Subroutine_Stmt` (the final statement)
        """
        # Find start and end of construct
        subroutine_stmt = get_child(o, (Fortran2003.Subroutine_Stmt, Fortran2003.Function_Stmt))
        subroutine_stmt_index = o.children.index(subroutine_stmt)
        end_subroutine_stmt = get_child(o, (Fortran2003.End_Subroutine_Stmt, Fortran2003.End_Function_Stmt))
        end_subroutine_stmt_index = o.children.index(end_subroutine_stmt)

        # Everything before the construct
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:subroutine_stmt_index])

        # ...and there shouldn't be anything after the construct
        assert end_subroutine_stmt_index + 1 == len(o.children)

        # Instantiate the object
        (routine, return_type) = self.visit(subroutine_stmt, **kwargs)
        kwargs['scope'] = routine

        # Extract source object for construct
        lines = (subroutine_stmt.item.span[0], end_subroutine_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)

        # We make sure the subroutine objects for all member routines are
        # instantiated before parsing the actual spec and body of the parent routine.
        # This way, all procedure types should exist and any use of their symbol
        # (e.g. in a CallStatement) should have type.dtype.procedure initialized correctly
        contains_ast = get_child(o, Fortran2003.Internal_Subprogram_Part)
        if contains_ast is not None:
            member_asts = [
                c for c in contains_ast.children
                if isinstance(c, (Fortran2003.Subroutine_Subprogram, Fortran2003.Function_Subprogram))
            ]
            # Note that we overwrite this variable subsequently with the fully parsed subroutines
            # where the visit-method for the subroutine/function statement will pick out the existing
            # subroutine objects using the weakref pointers stored in the symbol table.
            # I know, it's not pretty but alternatively we could hand down this array as part of
            # kwargs but that feels like carrying around a lot of bulk, too.
            contains = [
                self.visit(get_child(c, (Fortran2003.Subroutine_Stmt, Fortran2003.Function_Stmt)), **kwargs)[0]
                for c in member_asts
            ]

        # Hack: Collect all spec and body parts and use all but the
        # last body as spec. Reason is that Fparser misinterprets statement
        # functions as array assignments and thus breaks off spec early
        part_asts = [
            c for c in o.children if isinstance(c, (Fortran2003.Specification_Part, Fortran2003.Execution_Part))
        ]
        if not part_asts:
            spec_asts = []
            body_ast = None
        elif isinstance(part_asts[-1], Fortran2003.Execution_Part):
            *spec_asts, body_ast = part_asts
        else:
            spec_asts = part_asts
            body_ast = None

        # Build the spec by parsing all relevant parts of the AST and appending them
        # to the same section object
        spec_parts = [self.visit(spec_ast, **kwargs) for spec_ast in spec_asts]
        spec_parts = flatten([part.body for part in spec_parts if part is not None])
        spec = ir.Section(body=as_tuple(spec_parts))
        spec = sanitize_ir(spec, FP, pp_registry=sanitize_registry[FP], pp_info=self.pp_info)

        # Now all declarations are well-defined and we can parse the member routines
        if contains_ast is not None:
            contains = self.visit(contains_ast, **kwargs)
        else:
            contains = None

        # Finally, take care of the body
        if body_ast is None:
            body = ir.Section(body=())
        else:
            body = self.visit(body_ast, **kwargs)
            body = sanitize_ir(body, FP, pp_registry=sanitize_registry[FP], pp_info=self.pp_info)

        # Another big hack: fparser allocates all comments before and after the
        # spec to the spec. We remove them from the beginning to get the docstring.
        comment_map = {}
        docs = []
        for node in spec.body:
            if not isinstance(node, (ir.Comment, ir.CommentBlock)):
                break
            docs.append(node)
            comment_map[node] = None

        # Now we move comments from the end to the beginning of the body as those
        # can potentially be pragmas.
        for node in reversed(spec.body):
            if not isinstance(node, (ir.Pragma, ir.Comment, ir.CommentBlock)):
                break
            body.prepend(node)
            comment_map[node] = None
        spec = Transformer(comment_map, invalidate_source=False).visit(spec)

        # To simplify things, we always declare the result-type of a function with
        # a declaration in the spec as this can capture every possible situation.
        # Therefore, if it has been declared as a prefix in the subroutine statement,
        # we now have to inject a declaration instead. To ensure we do this in the
        # right place in the spec to not violate the intrinsic order Fortran mandates,
        # we search for the first occurence of any VariableDeclaration or
        # ProcedureDeclaration and inject it before that one
        if return_type is not None:
            routine.symbol_attrs[routine.name] = return_type
            return_var = sym.Variable(name=routine.name, scope=routine)
            return_var_decl = ir.VariableDeclaration(symbols=(return_var,))

            decls = FindNodes((ir.VariableDeclaration, ir.ProcedureDeclaration)).visit(spec)
            if not decls:
                # No other declarations: add it to the end
                spec.append(return_var_decl)
            else:
                spec.insert(spec.body.index(decls[0]), return_var_decl)

        # Finally, call the subroutine constructor on the object again to register all
        # bits and pieces in place and rescope all symbols
        # pylint: disable=unnecessary-dunder-call
        routine.__init__(
            name=routine.name, args=routine._dummies,
            docstring=docs, spec=spec, body=body, contains=contains,
            ast=o, prefix=routine.prefix, bind=routine.bind, is_function=routine.is_function,
            rescope_symbols=True, source=source, parent=routine.parent, symbol_attrs=routine.symbol_attrs,
            incomplete=False
        )

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        routine.spec, routine.body = routine._infer_allocatable_shapes(routine.spec, routine.body)

        # Update array shapes with Loki dimension pragmas
        with pragmas_attached(routine, ir.VariableDeclaration):
            routine.spec = process_dimension_pragmas(routine.spec)

        # Inject statement function definitions
        inject_statement_functions(routine)

        if isinstance(o, Fortran2003.Subroutine_Body):
            # Return the subroutine object along with any clutter before it for interface declarations
            return (*pre, routine)

        return (*pre, routine)

    visit_Function_Subprogram = visit_Subroutine_Subprogram
    visit_Subroutine_Body = visit_Subroutine_Subprogram
    visit_Function_Body = visit_Subroutine_Body

    def visit_Subroutine_Stmt(self, o, **kwargs):
        """
        The ``SUBROUTINE`` statement

        :class:`fparser.two.Fortran2003.Subroutine_Stmt` has four children:

        * prefix :class:`fparser.two.Fortran2003.Prefix`
        * name :class:`fparser.two.Fortran2003.Subroutine_Name`
        * dummy argument list :class:`fparser.two.Fortran2003.Dummy_Arg_List`
        * language binding specs :class:`fparser.two.Fortran2003.Proc_Language_Binding_Spec`
        """
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel,cyclic-import

        # Parse the prefix
        prefix = ()
        return_type = None
        if o.children[0] is not None:
            prefix = self.visit(o.children[0], **kwargs)
            return_type = [i for i in prefix if not isinstance(i, str)]
            prefix = [i for i in prefix if isinstance(i, str)]
            assert len(return_type) in (0, 1)
            return_type = return_type[0] if return_type else None

        name = self.visit(o.children[1], **kwargs)
        name = name.name

        # Check if the Subroutine node has been created before by looking it up in the scope
        routine = None
        if kwargs['scope'] is not None and name in kwargs['scope'].symbol_attrs:
            proc_type = kwargs['scope'].symbol_attrs[name]  # Look-up only in current scope!
            if proc_type and proc_type.dtype.procedure != BasicType.DEFERRED:
                routine = proc_type.dtype.procedure
                if not routine._incomplete:
                    # We return the existing object right away, unless it exists from a
                    # previous incomplete parse for which we have to make sure we get a
                    # full parse first
                    return (routine, proc_type.dtype.return_type)

        # Build the dummy argument list
        if o.children[2] is None:
            args = ()
        else:
            dummy_arg_list = self.visit(o.children[2], **kwargs)
            args = tuple(str(arg) for arg in dummy_arg_list)

        # Parse language binding specs
        bind = None if o.children[3] is None else o.children[3].tostr()

        # Instantiate the object
        is_function = isinstance(o, Fortran2003.Function_Stmt)
        if routine is None:
            routine = Subroutine(
                name=name, args=args, prefix=prefix, bind=bind,
                is_function=is_function, parent=kwargs['scope']
            )
        else:
            routine.__init__(  # pylint: disable=unnecessary-dunder-call
                name=name, args=args, docstring=routine.docstring, spec=routine.spec, body=routine.body,
                contains=routine.contains, prefix=prefix, bind=bind, is_function=is_function,
                ast=routine._ast, source=routine._source, parent=routine.parent, symbol_attrs=routine.symbol_attrs,
                incomplete=routine._incomplete
            )

        return (routine, return_type)

    visit_Function_Stmt = visit_Subroutine_Stmt
    visit_Subroutine_Name = visit_Name
    visit_Function_Name = visit_Name
    visit_Dummy_Arg_List = visit_List

    def visit_Prefix(self, o, **kwargs):
        """
        The prefix of a subprogram definition

        :class:`fparser.two.Fortran2003.Prefix` has variable number of children
        that have the type

        * :class:`fparser.two.Fortran2003.Prefix_Spec` to declare attributes
        * :class:`fparser.two.Fortran2003.Declaration_Type_Spec` (or any of its
          variations) to declare the return type of a function
        """
        attrs = [self.visit(c, **kwargs) for c in o.children]
        return as_tuple(attrs)

    def visit_Prefix_Spec(self, o, **kwargs):
        """
        A prefix keyword in a subprogram definition

        :class:`fparser.two.Fortran2003.Prefix_Spec` has no children
        """
        return o.string

    #
    # Module definition
    #

    def visit_Module(self, o, **kwargs):
        """
        The definition of a Fortran module

        :class:`fparser.two.Fortran2003.Module` has up to four children:

        * The opening :class:`fparser.two.Fortran2003.Module_Stmt`
        * The specification part :class:`fparser.two.Fortran2003.Specification_Part`
        * The module subprogram part :class:`fparser.two.Fortran2003.Module_Subprogram_Part`
        * the closing :class:`fparser.two.Fortran2003.End_Module_Stmt`
        """
        # Find start and end of construct
        module_stmt = get_child(o, Fortran2003.Module_Stmt)
        module_stmt_index = o.children.index(module_stmt)
        end_module_stmt = get_child(o, Fortran2003.End_Module_Stmt)
        end_module_stmt_index = o.children.index(end_module_stmt)

        # Everything before the construct
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:module_stmt_index])

        # ...and there shouldn't be anything after the construct
        assert end_module_stmt_index + 1 == len(o.children)

        # Extract source object for construct
        lines = (module_stmt.item.span[0], end_module_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)

        # Instantiate the object
        module = self.visit(module_stmt, **kwargs)
        kwargs['scope'] = module

        # We make sure the subroutine objects for all member routines are
        # instantiated before parsing the actual spec of the module.
        # This way, all procedure types should exist and any use of their symbol
        # (e.g. in a derived type procedure binding etc) should be initialized correctly
        contains_ast = get_child(o, Fortran2003.Module_Subprogram_Part)
        if contains_ast is not None:
            member_asts = [
                c for c in contains_ast.children
                if isinstance(c, (Fortran2003.Subroutine_Subprogram, Fortran2003.Function_Subprogram))
            ]
            # Note that we overwrite this variable subsequently with the fully parsed subroutines
            # where the visit-method for the subroutine/function statement will pick out the existing
            # subroutine objects using the weakref pointers stored in the symbol table.
            # I know, it's not pretty but alternatively we could hand down this array as part of
            # kwargs but that feels like carrying around a lot of bulk, too.
            contains = [
                self.visit(get_child(c, (Fortran2003.Subroutine_Stmt, Fortran2003.Function_Stmt)), **kwargs)[0]
                for c in member_asts
            ]

        # Build the spec
        spec_ast = get_child(o, Fortran2003.Specification_Part)
        if spec_ast:
            spec = self.visit(spec_ast, **kwargs)
            spec = sanitize_ir(spec, FP, pp_registry=sanitize_registry[FP], pp_info=self.pp_info)

            # Another big hack: fparser allocates all comments before and after the
            # spec to the spec. We remove them from the beginning to get the docstring.
            comment_map = {}
            docs = []
            for node in spec.body:
                if not isinstance(node, (ir.Comment, ir.CommentBlock)):
                    break
                docs.append(node)
                comment_map[node] = None
            spec = Transformer(comment_map, invalidate_source=False).visit(spec)
        else:
            docs = []
            spec = None

        # Now that all declarations are well-defined we can parse the member routines
        if contains_ast is not None:
            contains = self.visit(contains_ast, **kwargs)
        else:
            contains = None

        # Finally, call the module constructor on the object again to register all
        # bits and pieces in place and rescope all symbols
        # pylint: disable=unnecessary-dunder-call
        module.__init__(
            name=module.name, docstring=docs, spec=spec, contains=contains,
            default_access_spec=module.default_access_spec, public_access_spec=module.public_access_spec,
            private_access_spec=module.private_access_spec, ast=o, rescope_symbols=True, source=source,
            parent=module.parent, symbol_attrs=module.symbol_attrs, incomplete=False
        )

        return (*pre, module)

    def visit_Module_Stmt(self, o, **kwargs):
        """
        The ``MODULE`` statement

        :class:`fparser.two.Fortran2003.Module_Stmt` has 2 children:
            * keyword `MODULE` (str)
            * name :class:`fparser.two.Fortran2003.Module_Name`
        """
        from loki.module import Module  # pylint: disable=import-outside-toplevel,cyclic-import

        name = self.visit(o.children[1], **kwargs)
        name = name.name

        # Check if the Module node has been created before by looking it up in the scope
        if kwargs['scope'] is not None and name in kwargs['scope'].symbol_attrs:
            module_type = kwargs['scope'].symbol_attrs[name]  # Look-up only in current scope!
            if module_type and module_type.dtype.module != BasicType.DEFERRED:
                return module_type.dtype.module

        return Module(name=name, parent=kwargs['scope'])

    visit_Module_Name = visit_Name


    #
    # Conditional
    #

    def visit_If_Construct(self, o, **kwargs):
        """
        The entire ``IF`` construct

        :class:`fparser.two.Fortran2003.If_Construct` has variable number of children:

        * Any preceeding comments :class:`fparser.two.Fortran2003.Comment`
        * :class:`fparser.two.Fortran2003.If_Then_Stmt` (the actual statement
          that begins the construct with the first condition)
        * the body of the conditional branch
        * Optionally, one or more :class:`fparser.two.Fortran2003.Else_If_Stmt`
          followed by their corresponding bodies
        * Optionally, a :class:`fparser.two.Fortran2003.Else_Stmt` followed by
          its body
        * :class:`fparser.two.Fortran2003.End_If_Stmt`
        """
        # Find start and end of construct
        if_then_stmt = get_child(o, Fortran2003.If_Then_Stmt)
        if_then_stmt_index = o.children.index(if_then_stmt)
        end_if_stmt = get_child(o, Fortran2003.End_If_Stmt)
        end_if_stmt_index = o.children.index(end_if_stmt)

        # Everything before the IF statement
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:if_then_stmt_index])

        # Find all branches
        else_if_stmts = tuple((i, c) for i, c in enumerate(o.children) if isinstance(c, Fortran2003.Else_If_Stmt))
        if else_if_stmts:
            else_if_stmt_index, else_if_stmts = zip(*else_if_stmts)
        else:
            else_if_stmt_index = ()
        else_stmt = get_child(o, Fortran2003.Else_Stmt)
        else_stmt_index = o.children.index(else_stmt) if else_stmt else end_if_stmt_index
        conditions = as_tuple(self.visit(c, **kwargs) for c in (if_then_stmt,) + else_if_stmts)
        bodies = [as_tuple(self.visit(c, **kwargs) for c in o.children[start+1:stop])
                  for start, stop in zip((if_then_stmt_index,) + else_if_stmt_index,
                                         else_if_stmt_index + (else_stmt_index,))]
        else_body = flatten([self.visit(c, **kwargs) for c in o.children[else_stmt_index+1:end_if_stmt_index]])

        # Extract source objects for branches
        sources, labels = [], []
        for conditional in (if_then_stmt,) + else_if_stmts:
            lines = (conditional.item.span[0], end_if_stmt.item.span[1])
            string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
            sources += [Source(lines=lines, string=string)]
            labels += [self.get_label(conditional)]

        # Build IR nodes backwards using else-if branch as else body
        node = ir.Conditional(condition=conditions[-1], body=bodies[-1], else_body=as_tuple(else_body),
                              inline=False, has_elseif=False, label=labels[-1], source=sources[-1])
        for idx in reversed(range(len(conditions)-1)):
            node = ir.Conditional(condition=conditions[idx], body=bodies[idx], else_body=as_tuple(node),
                                  inline=False, has_elseif=True, label=labels[idx], source=sources[idx])

        # Update with construct name
        name = if_then_stmt.get_start_name()
        node._update(name=name)

        # Everything past the END IF (should be empty)
        assert not o.children[end_if_stmt_index+1:]

        return (*pre, node)

    def visit_If_Then_Stmt(self, o, **kwargs):
        """
        The first conditional in a ``IF`` construct

        :class:`fparser.two.Fortran2003.If_Then_Stmt` has one child: the
        condition expression
        """
        return self.visit(o.children[0], **kwargs)

    visit_Else_If_Stmt = visit_If_Then_Stmt

    def visit_If_Stmt(self, o, **kwargs):
        """
        An inline ``IF`` statement with a single statement as body

        :class:`fparser.two.Fortran2003.If_Stmt` has two children:

        * the condition expression
        * the body
        """
        cond = self.visit(o.items[0], **kwargs)
        body = as_tuple(self.visit(o.items[1], **kwargs))
        return ir.Conditional(condition=cond, body=body, else_body=(), inline=True,
                              label=kwargs.get('label'), source=kwargs.get('source'))

    #
    # SELECT CASE constructs
    #

    def visit_Case_Construct(self, o, **kwargs):
        """
        The entire ``SELECT CASE`` construct

        :class:`fparser.two.Fortran2003.Case_Construct` has variable number of children:

        * Any preceeding comments :class:`fparser.two.Fortran2003.Comment`
        * :class:`fparser.two.Fortran2003.Select_Case_Stmt` (the actual statement
          with the selection expression)
        * the body of the case-construct, containing one or multiple
          :class:`fparser.two.Fortran2003.Case_Stmt` followed by their
          corresponding bodies
        * :class:`fparser.two.Fortran2003.End_Select_Stmt`
        """
        # Find start and end of case construct
        select_case_stmt = get_child(o, Fortran2003.Select_Case_Stmt)
        select_case_stmt_index = o.children.index(select_case_stmt)
        end_select_stmt = get_child(o, Fortran2003.End_Select_Stmt)
        end_select_stmt_index = o.children.index(end_select_stmt)

        # Everything before the SELECT CASE statement
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:select_case_stmt_index])

        # Extract source object for construct
        lines = (select_case_stmt.item.span[0], end_select_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)

        # Handle the SELECT CASE statement
        expr = self.visit(select_case_stmt, **kwargs)
        name = select_case_stmt.get_start_name()
        label = self.get_label(select_case_stmt)

        # Find all CASE statements and corresponding bodies
        case_stmts, case_stmt_index = zip(*[(c, i) for i, c in enumerate(o.children)
                                            if isinstance(c, Fortran2003.Case_Stmt)])

        # Retain any comments between `SELECT CASE` and the first `CASE` statement
        if case_stmt_index[0] > select_case_stmt_index + 1:
            # Our IR doesn't provide a means to store them in the right place, so
            # we'll just put them before the `SELECT CASE`
            pre += as_tuple(self.visit(c, **kwargs) for c in o.children[select_case_stmt_index+1:case_stmt_index[0]])

        values = as_tuple(self.visit(c, **kwargs) for c in case_stmts)
        bodies = tuple(
            as_tuple(self.visit(c, **kwargs) for c in o.children[start+1:stop])
            for start, stop in zip(case_stmt_index, case_stmt_index[1:] + (end_select_stmt_index,))
        )

        if 'DEFAULT' in values:
            default_index = values.index('DEFAULT')
            else_body = bodies[default_index]
            values = values[:default_index] + values[default_index+1:]
            bodies = bodies[:default_index] + bodies[default_index+1:]
        else:
            else_body = ()

        # Everything past the END ASSOCIATE (should be empty)
        assert not o.children[end_select_stmt_index+1:]

        case_construct = ir.MultiConditional(expr=expr, values=values, bodies=bodies, else_body=else_body,
                                             label=label, name=name, source=source)
        return (*pre, case_construct)

    def visit_Select_Case_Stmt(self, o, **kwargs):
        """
        A ``SELECT CASE`` statement for a case-construct

        :class:`fparser.two.Fortran2003.Select_Case_Stmt` has only one child:
        the selection expression.
        """
        return self.visit(o.children[0], **kwargs)

    def visit_Case_Stmt(self, o, **kwargs):
        """
        A ``CASE`` statement in a case-construct

        :class:`fparser.two.Fortran2003.Case_Stmt` has two children:

        * the selection expression
          :class:`fparser.two.Fortran2003.Case_Selector`.
        * the construct name
          :class:`fparser.two.Fortran2003.Case_Construct_Name` or `None`
        """
        return self.visit(o.children[0], **kwargs)

    def visit_Case_Selector(self, o, **kwargs):
        """
        The selector in a ``CASE`` statement

        :class:`fparser.two.Fortran2003.Case_Selector` has one child: the
        value-range-list :class:`fparser.two.Fortran2003.Case_Value_Range_List`
        or `None` for the ``DEFAULT`` case.
        """
        if o.children[0] is None:
            return 'DEFAULT'
        return self.visit(o.children[0], **kwargs)

    def visit_Case_Value_Range(self, o, **kwargs):
        """
        The range of values in a ``CASE`` statement

        :class:`fparser.two.Fortran2003.Case_Value_Range` has two children:

        * start :class:`fparser.two.Fortran2003.Case_Value` or `None`
        * stop :class:`fparser.two.Fortran2003.Case_Value` or `None`
        """
        start = self.visit(o.children[0], **kwargs) if o.children[0] is not None else None
        stop = self.visit(o.children[1], **kwargs) if o.children[1] is not None else None
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.RangeIndex((start, stop), source=source)

    visit_Case_Value_Range_List = visit_List

    #
    # Allocation statements
    #

    def visit_Allocate_Stmt(self, o, **kwargs):
        """
        A call to ``ALLOCATE``

        :class:`fparser.two.Fortran2003.Allocate_Stmt` has three children:

        * :class:`fparser.two.Fortran2003.Type_Spec` or `None`
        * :class:`fparser.two.Fortran2003.Allocation_List`
        * :class:`fparser.two.Fortran2003.Alloc_Opt_List` or `None`
        """
        if o.children[0] is not None:
            # We can't handle type spec at the moment
            self.warn_or_fail('type-spec in allocate-stmt not implemented')

        # Any allocation options. We can only deal with "source" at the moment
        alloc_opts = {}
        if o.children[2] is not None:
            alloc_opts = self.visit(o.children[2], **kwargs)
            # We need to filter out any options we can't handle currently (and which returned None)
            alloc_opts = [opt for opt in alloc_opts if opt is not None]
            alloc_opts = dict(alloc_opts)

        variables = self.visit(o.children[1], **kwargs)
        return ir.Allocation(
            variables=variables, data_source=alloc_opts.get('source'), status_var=alloc_opts.get('stat'),
            source=kwargs.get('source'), label=kwargs.get('label')
        )

    visit_Allocation_List = visit_List

    def visit_Allocation(self, o, **kwargs):
        """
        An allocation specification in an allocate-stmt

        :class:`fparser.two.Fortran2003.Allocation` has two children:

        * the name of the data object to be allocated:
          :class:`fparser.two.Fortran2003.Allocate_Object`
        * the shape of the object: :class:`fparser.two.Fortran2003.Allocate_Shape_Spec_List`
        """
        name = self.visit(o.children[0], **kwargs)
        shape = self.visit(o.children[1], **kwargs)
        name = name.clone(dimensions=shape, type=name.type.clone(shape=shape))
        return name

    visit_Allocate_Shape_Spec = visit_Explicit_Shape_Spec
    visit_Allocate_Shape_Spec_List = visit_List

    visit_Alloc_Opt_List = visit_List
    visit_Dealloc_Opt_List = visit_List
    visit_Allocate_Object_List = visit_List

    def visit_Alloc_Opt(self, o, **kwargs):
        """
        An allocation option in an allocate-stmt

        :class:`fparser.two.Fortran2003.Alloc_Opt` has two children:

        * the keyword (`str`)
        * the option value
        """
        keyword = o.children[0].lower()
        if keyword in ('source', 'stat'):
            return keyword, self.visit(o.children[1], **kwargs)
        # TODO: implement other alloc options
        self.warn_or_fail(f'Unsupported allocation option: {o.children[0]}')
        return None

    def visit_Deallocate_Stmt(self, o, **kwargs):
        """
        A call to ``DEALLOCATE``

        :class:`fparser.two.Fortran2003.Deallocate_Stmt` has two children:

        * the list of objects :class:`fparser.two.Fortran2003.Allocate_Object_List`
        * list of options :class:`fparser.two.Fortran2003.Dealloc_Opt_list`
        """
        variables = self.visit(o.children[0], **kwargs)

        dealloc_opts = {}
        if o.children[1] is not None:
            dealloc_opts = self.visit(o.children[1], **kwargs)
            # We need to filter out any options we can't handle currently (and which returned None)
            dealloc_opts = [opt for opt in dealloc_opts if opt is not None]
            dealloc_opts = dict(dealloc_opts)

        return ir.Deallocation(
            variables=variables, status_var=dealloc_opts.get('stat'),
            source=kwargs.get('source'), label=kwargs.get('label')
        )

    def visit_Dealloc_Opt(self, o, **kwargs):
        """
        A deallocation option in a deallocate-stmt

        :class:`fparser.two.Fortran2003.Dealloc_Opt` has two children:

        * the keyword (`str`)
        * the option value
        """
        keyword = o.children[0].lower()
        if keyword == 'stat':
            return keyword, self.visit(o.children[1], **kwargs)
        # TODO: implement other alloc options
        self.warn_or_fail(f'Unsupported deallocation option: {o.children[0]}')
        return None

    #
    # Subroutine and function calls
    #

    def visit_Call_Stmt(self, o, **kwargs):
        """
        A ``CALL`` statement

        :class:`fparser.two.Fortran2003.Call_Stmt` has two children:

        * the subroutine name :class:`fparser.two.Fortran2003.Procedure_Designator`
        * the argument list :class:`fparser.two.Fortran2003.Actual_Arg_Spec_List`
        """
        name = self.visit(o.children[0], **kwargs)
        if o.children[1] is not None:
            arguments = self.visit(o.children[1], **kwargs)
            kwarguments = tuple(arg for arg in arguments if isinstance(arg, tuple))
            arguments = tuple(arg for arg in arguments if not isinstance(arg, tuple))
        else:
            arguments, kwarguments = (), ()
        return ir.CallStatement(name=name, arguments=arguments, kwarguments=kwarguments,
                                label=kwargs.get('label'), source=kwargs.get('source'))

    def visit_Procedure_Designator(self, o, **kwargs):
        """
        The function or subroutine designator

        This appears only when a type-bound procedure is called (as otherwise Fparser
        hands through the relevant names directly).

        :class:`fparser.two.Fortran2003.Procedure_Designator` has three children:

        * Parent name :class:`fparser.two.Fortran2003.Data_Ref`
        * '%' (`str`)
        * procedure name :class:`fparser.two.Fortran2003.Binding_Name`
        """
        assert o.children[1] == '%'
        parent = self.visit(o.children[0], **kwargs)
        name = self.visit(o.children[2], **kwargs)
        name = name.clone(name=f'{parent.name}%{name.name}', parent=parent)
        return name

    visit_Actual_Arg_Spec_List = visit_List

    def visit_Actual_Arg_Spec(self, o, **kwargs):
        """
        A single argument in a subroutine call

        :class:`fparser.two.Fortran2003.Actual_Arg_Spec` has two children:

        * keyword :class:`fparser.two.Fortran2003.Keyword`
        * argument :class:`fparser.two.Fortran2003.Actual_Arg`
        """
        keyword = o.children[0].tostr() if o.children[0] is not None else None
        arg = self.visit(o.children[1], **kwargs)
        return (keyword, arg)

    def visit_Function_Reference(self, o, **kwargs):
        """
        An inline function call

        :class:`fparser.two.Fortran2003.Actual_Arg_Spec` has two children:

        * the function name :class:fparser.two.Fortran2003.ProcedureDesignator`
        * the argument list :class:`fparser.two.Fortran2003.Actual_Arg_Spec_List`
        """
        name = self.visit(o.children[0], **kwargs)
        if o.children[1] is not None:
            arguments = self.visit(o.children[1], **kwargs)
            kwarguments = tuple(arg for arg in arguments if isinstance(arg, tuple))
            arguments = tuple(arg for arg in arguments if not isinstance(arg, tuple))
        else:
            arguments, kwarguments = (), ()
        return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments,
                              source=kwargs.get('source'))

    def visit_Intrinsic_Function_Reference(self, o, **kwargs):
        name = self.visit(o.children[0], **kwargs)
        if o.children[1] is not None:
            arguments = self.visit(o.children[1], **kwargs)
            kwarguments = tuple(arg for arg in arguments if isinstance(arg, tuple))
            arguments = tuple(arg for arg in arguments if not isinstance(arg, tuple))
        else:
            arguments, kwarguments = (), ()

        if str(name).upper() in ('REAL', 'INT'):
            assert arguments
            expr = arguments[0]
            if kwarguments:
                assert len(arguments) == 1
                assert len(kwarguments) == 1 and kwarguments[0][0].lower() == 'kind'
                kind = kwarguments[0][1]
            else:
                kind = arguments[1] if len(arguments) > 1 else None
            return sym.Cast(name, expr, kind=kind, source=kwargs.get('source'))
        return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments,
                              source=kwargs.get('source'))

    visit_Intrinsic_Name = visit_Name

    def visit_Structure_Constructor(self, o, **kwargs):
        """
        Call to the constructor of a derived type

        :class:`fparser.two.Fortran2003.Structure_Constructor` has two children:

        * the structure name :class:`fparser.two.Fortran2003.Derived_Type_Spec`
        * the argument list :class:`fparser.two.Fortran2003.Component_Spec_List`
        """
        # Note: Fparser wrongfully interprets function calls as Structure_Constructor
        # sometimes. However, we represent constructor calls in the same way, so it
        # doesn't really matter for us.
        # This should go away once fparser has a basic symbol table, see
        # https://github.com/stfc/fparser/issues/201 for some details
        name = self.visit(o.children[0], **kwargs)
        assert isinstance(name, DerivedType)

        # `name` is a DerivedType but we represent a constructor call as InlineCall for
        # which we need ProcedureSymbol
        name = sym.Variable(name=name.name, source=self.get_source(o.children[0], kwargs.get('source')))

        if o.children[1] is not None:
            arguments = self.visit(o.children[1], **kwargs)
            kwarguments = tuple(arg for arg in arguments if isinstance(arg, tuple))
            arguments = tuple(arg for arg in arguments if not isinstance(arg, tuple))
        else:
            arguments, kwarguments = (), ()

        return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments, source=kwargs.get('source'))

    visit_Component_Spec = visit_Actual_Arg_Spec
    visit_Component_Spec_List = visit_List

    #
    # ENUM declaration
    #

    def visit_Enum_Def(self, o, **kwargs):
        """
        The definition of an ``ENUM``

        :class:`fparser.two.Fortran2003.Enum_Def` has variable number of children:

        * Any preceeding comments :class:`fparser.two.Fortran2003.Comment`
        * :class:`fparser.two.Fortran2003.Enum_Def_Stmt` (the statement indicating the
          beginning of the enum)
        * the body of the enum, containing one or multiple
          :class:`fparser.two.Fortran2003.Enumerator_Def_Stmt`
        * :class:`fparser.two.Fortran2003.End_Enum_Stmt`
        """
        # Find start end end of construct
        enum_def_stmt = get_child(o, Fortran2003.Enum_Def_Stmt)
        enum_def_stmt_index = o.children.index(enum_def_stmt)
        end_enum_stmt = get_child(o, Fortran2003.End_Enum_Stmt)
        end_enum_stmt_index = o.children.index(end_enum_stmt)

        # Everything before the construct
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:enum_def_stmt_index])

        # Take out any comments (and other stuff which shouldn't be there)
        # from inside the enum and put them behind it
        post = as_tuple(
            self.visit(c, **kwargs) for c in o.children[enum_def_stmt_index+1:end_enum_stmt_index]
            if not isinstance(c, Fortran2003.Enumerator_Def_Stmt)
        )

        # Find the constant definitions inside the enum
        symbols = flatten(
            self.visit(c, **kwargs) for c in o.children[enum_def_stmt_index+1:end_enum_stmt_index]
            if isinstance(c, Fortran2003.Enumerator_Def_Stmt)
        )

        # Update type information for symbols with deferred type
        # (applies to all constant that are defined without explicit value)
        symbols = tuple(
            s.clone(type=SymbolAttributes(BasicType.INTEGER)) if s.type.dtype is BasicType.DEFERRED else s
            for s in symbols
        )

        # Put symbols in the right scope (that should register their type in that scope's symbol table)
        symbols = tuple(s.rescope(scope=kwargs['scope']) for s in symbols)

        # Create the enum and make sure there's nothing else left to do
        enum = ir.Enumeration(symbols=symbols, source=kwargs['source'], label=kwargs['label'])
        assert end_enum_stmt_index + 1 == len(o.children)
        return (*pre, enum, *post)

    def visit_Enumerator_Def_Stmt(self, o, **kwargs):
        """
        A definition inside an ``ENUM``

        :class:`fparser.two.Fortran2003.Enumerator_Def_Stmt` has 2 children:

        * ``'ENUMERATOR'`` (str)
        * :class:`fparser.two.Fortran2003.Enumerator_List` (the constants)
        """
        return self.visit(o.children[1], **kwargs)

    visit_Enumerator_List = visit_List

    def visit_Enumerator(self, o, **kwargs):
        """
        A constant definition within an ``ENUM``'s definition stmt

        :class:`fparser.two.Fortran2003.Enumerator` has 3 children:

        * :class:`fparser.two.Fortran2003.Name` (the constant's name)
        * ``'='`` (str)
        * the constant's value given as some constant expression that
          must evaluate to an integer
        """
        assert o.children[1] == '='
        symbol = self.visit(o.children[0], **kwargs)
        initial = self.visit(o.children[2], **kwargs)
        _type = SymbolAttributes(BasicType.INTEGER, initial=initial)
        return symbol.clone(type=_type)

    #
    # WHERE construct
    #

    def visit_Where_Construct(self, o, **kwargs):
        """
        Fortran's masked array assignment construct

        :class:`fparser.two.Fortran2003.Where_Construct` has variable number of children:

        * Any preceeding comments :class:`fparser.two.Fortran2003.Comment`
        * :class:`fparser.two.Fortran2003.Where_Construct_Stmt` (the statement that marks
          the beginning of the construct)
        * body of the where-construct, usually an assignment
        * (optional) :class:`fparser.two.Fortran2003.Masked_Elsewhere_Stmt` (essentially
          an "else-if"), followed by its body; this can appear more than once
        * (optional) :class:`fparser.two.Fortran2003.Elsewhere_Stmt` (essentially
          an "else"), followed by its body
        * :class:`fparser.two.Fortran2003.End_Where_Stmt`
        """
        # Find start and end of construct
        where_stmt = get_child(o, Fortran2003.Where_Construct_Stmt)
        where_stmt_index = o.children.index(where_stmt)
        end_where_stmt = get_child(o, Fortran2003.End_Where_Stmt)
        end_where_stmt_index = o.children.index(end_where_stmt)

        # The banter before the construct...
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:where_stmt_index])

        # Extract source object for construct
        lines = (where_stmt.item.span[0], end_where_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)

        # Find all ELSEWHERE statements
        where_stmts, where_stmts_index = zip(*(
            [(where_stmt, where_stmt_index)] +
            [
                (c, i) for i, c in enumerate(o.children)
                if isinstance(c, (Fortran2003.Masked_Elsewhere_Stmt, Fortran2003.Elsewhere_Stmt))
            ]
        ))
        where_stmts_index = where_stmts_index + (end_where_stmt_index,)

        # Handle all cases
        conditions = [self.visit(c, **kwargs) for c in where_stmts]
        bodies = [
            as_tuple([self.visit(c, **kwargs) for c in o.children[start+1:stop]])
            for start, stop in zip(where_stmts_index[:-1], where_stmts_index[1:])
        ]

        # Extract the default case if any
        if conditions[-1] == 'DEFAULT':
            conditions = conditions[:-1]
            *bodies, default = bodies
        else:
            default = ()

        # Make sure there's nothing left to do
        assert not o.children[end_where_stmt_index+1:]

        masked_statement = ir.MaskedStatement(
            conditions=as_tuple(conditions), bodies=as_tuple(bodies),
            default=default, label=kwargs.get('label'), source=source
        )
        return (*pre, masked_statement)

    def visit_Where_Construct_Stmt(self, o, **kwargs):
        """
        The ``WHERE`` statement that marks the beginning of a where-construct

        :class:`fparser.two.Fortran2003.Where_Construct_Stmt` has 1 child:

        * the expression that marks the condition
        """
        return self.visit(o.children[0], **kwargs)

    def visit_Masked_Elsewhere_Stmt(self, o, **kwargs):
        """
        An ``ELSEWHERE`` statement with a condition in a where-construct

        :class:`fparser.two.Fortran2003.Masked_Elsewhere_Stmt` has 2 children:

        * the expression that marks the condition
        * the construct name or `None`
        """
        if o.children[1] is not None:
            self.warn_or_fail('where-construct-names not yet implemented')
        return self.visit(o.children[0], **kwargs)

    def visit_Elsewhere_Stmt(self, o, **kwargs):
        """
        An unconditional ``ELSEWHERE`` statement

        :class:`fparser.two.Fortran2003.Elsewhere_Stmt` has 2 children:

        * ``'ELSEWHERE'`` (str)
        * the construct name or `None`
        """
        if o.children[1] is not None:
            self.warn_or_fail('where-construct-names not yet implemented')
        assert o.children[0] == 'ELSEWHERE'
        return 'DEFAULT'

    def visit_Where_Stmt(self, o, **kwargs):
        """
        An inline ``WHERE`` assignment

        :class:`fparser.two.Fortran2003.Where_Stmt` has 2 children:

        * the expression that marks the condition
        * the assignment
        """
        condition = self.visit(o.children[0], **kwargs)
        body = as_tuple(self.visit(o.children[1], **kwargs))
        return ir.MaskedStatement(
            conditions=(condition, ), bodies=(body, ), default=(), inline=True,
            label=kwargs.get('label'), source=kwargs.get('source')
        )

    ### Below functions have not yet been revisited ###

    def visit_Base(self, o, **kwargs):
        """
        Universal default for ``Base`` FParser-AST nodes
        """
        name = getattr(o.__class__, 'name', o.__class__.__name__)
        self.warn_or_fail(f'No specific handler for node type {name}')
        children = tuple(self.visit(a, **kwargs) for c in o.items for a in as_tuple(c))
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_BlockBase(self, o, **kwargs):
        """
        Universal default for ``BlockBase`` FParser-AST nodes
        """
        self.warn_or_fail(f'No specific handler for node type {o.__class__.name}')
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_literal(self, o, _type, kind=None, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(str(o.items[0]))
            val = source.string
        else:
            val = o.items[0]
        if kind is not None:
            if kind.isdigit():
                kind = sym.Literal(value=int(kind), source=source)
            else:
                kind = AttachScopesMapper()(sym.Variable(name=kind, source=source), scope=kwargs['scope'])
            return sym.Literal(value=val, type=_type, kind=kind, source=source)
        return sym.Literal(value=val, type=_type, source=source)

    def visit_Char_Literal_Constant(self, o, **kwargs):
        return self.visit_literal(o, BasicType.CHARACTER, **kwargs)

    def visit_Int_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return self.visit_literal(o, BasicType.INTEGER, kind=kind, **kwargs)

    visit_Signed_Int_Literal_Constant = visit_Int_Literal_Constant

    def visit_Real_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return self.visit_literal(o, BasicType.REAL, kind=kind, **kwargs)

    visit_Signed_Real_Literal_Constant = visit_Real_Literal_Constant

    def visit_Logical_Literal_Constant(self, o, **kwargs):
        return self.visit_literal(o, BasicType.LOGICAL, **kwargs)

    def visit_Complex_Literal_Constant(self, o, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
            val = source.string
        else:
            val = o.string
        return sym.IntrinsicLiteral(value=val, source=source)

    visit_Binary_Constant = visit_Complex_Literal_Constant
    visit_Octal_Constant = visit_Complex_Literal_Constant
    visit_Hex_Constant = visit_Complex_Literal_Constant

    def visit_Include_Stmt(self, o, **kwargs):
        fname = o.items[0].tostr()
        return ir.Import(module=fname, f_include=True, source=kwargs.get('source'),
                         label=kwargs.get('label'))

    def visit_Implicit_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text=f'IMPLICIT {o.items[0]}', source=kwargs.get('source'),
                            label=kwargs.get('label'))

    def visit_Print_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text=f'PRINT {", ".join(str(i) for i in o.items)}',
                            source=kwargs.get('source'), label=kwargs.get('label'))

    # TODO: Deal with line-continuation pragmas!
    _re_pragma = re.compile(r'^\s*\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

    def visit_Comment(self, o, **kwargs):
        source = kwargs.get('source', None)
        match_pragma = self._re_pragma.search(o.tostr())
        if match_pragma:
            # Found pragma, generate this instead
            gd = match_pragma.groupdict()
            return ir.Pragma(keyword=gd['keyword'], content=gd['content'], source=source)
        return ir.Comment(text=o.tostr(), source=source)

    def visit_Data_Pointer_Object(self, o, **kwargs):
        v = self.visit(o.items[0], source=kwargs.get('source'))
        for i in o.items[1:-1]:
            if i == '%':
                continue
            # Careful not to propagate type or dims here
            v = self.visit(i, parent=v, source=kwargs.get('source'))
        # Attach types and dims to final leaf variable
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Proc_Component_Ref(self, o, **kwargs):
        '''This is the compound object for accessing procedure components of a variable.'''
        pname = o.items[0].tostr().lower()
        v = AttachScopesMapper()(sym.Variable(name=pname), scope=kwargs['scope'])
        for i in o.items[1:-1]:
            if i != '%':
                v = self.visit(i, parent=v, source=kwargs.get('source'))
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Block_Nonlabel_Do_Construct(self, o, **kwargs):
        do_stmt_types = (Fortran2003.Nonlabel_Do_Stmt, Fortran2003.Label_Do_Stmt)
        # In the banter before the loop, Pragmas are hidden...
        banter = []
        for ch in o.content:
            if isinstance(ch, do_stmt_types):
                do_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            do_stmt = get_child(o, do_stmt_types)
        # Extract source by looking at everything between DO and END DO statements
        end_do_stmt = rget_child(o, Fortran2003.End_Do_Stmt)
        has_end_do = True
        if end_do_stmt is None:
            # We may have a labeled loop with an explicit CONTINUE statement
            has_end_do = False
            end_do_stmt = rget_child(o, Fortran2003.Continue_Stmt)
            assert str(end_do_stmt.item.label) == do_stmt.label.string
        lines = (do_stmt.item.span[0], end_do_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)
        label = self.get_label(do_stmt)
        construct_name = do_stmt.item.name
        # Extract loop header and get stepping info
        variable, bounds = self.visit(do_stmt, **kwargs)
        # Extract and process the loop body
        body_nodes = node_sublist(o.content, do_stmt.__class__, Fortran2003.End_Do_Stmt)
        body = as_tuple(flatten(self.visit(node, **kwargs) for node in body_nodes))
        # Loop label for labeled do constructs
        loop_label = str(do_stmt.items[1]) if isinstance(do_stmt, Fortran2003.Label_Do_Stmt) else None
        # Select loop type
        if bounds:
            obj = ir.Loop(variable=variable, body=body, bounds=bounds, loop_label=loop_label,
                          label=label, name=construct_name, has_end_do=has_end_do, source=source)
        else:
            obj = ir.WhileLoop(condition=variable, body=body, loop_label=loop_label,
                               label=label, name=construct_name, has_end_do=has_end_do, source=source)
        return (*banter, obj, )

    visit_Block_Label_Do_Construct = visit_Block_Nonlabel_Do_Construct

    def visit_Nonlabel_Do_Stmt(self, o, **kwargs):
        variable, bounds = None, None
        loop_control = get_child(o, Fortran2003.Loop_Control)
        if loop_control:
            variable, bounds = self.visit(loop_control, **kwargs)
        return variable, bounds

    visit_Label_Do_Stmt = visit_Nonlabel_Do_Stmt

    def visit_Loop_Control(self, o, **kwargs):
        if o.items[0]:
            # Scalar logical expression
            return self.visit(o.items[0], **kwargs), None
        variable = self.visit(o.items[1][0], **kwargs)
        bounds = as_tuple(flatten(self.visit(a, **kwargs) for a in as_tuple(o.items[1][1])))
        source = kwargs.get('source')
        if source:
            variable_source = source.clone_with_string(o.string[:o.string.find('=')])
            variable = variable.clone(source=variable_source)
            source = source.clone_with_string(o.string[o.string.find('=')+1:])
        return variable, sym.LoopRange(bounds, source=source)

    def visit_Assignment_Stmt(self, o, **kwargs):
        ptr = isinstance(o, Fortran2003.Pointer_Assignment_Stmt)
        lhs = self.visit(o.items[0], **kwargs)
        rhs = self.visit(o.items[2], **kwargs)
        return ir.Assignment(lhs=lhs, rhs=rhs, ptr=ptr,
                             label=kwargs.get('label'), source=kwargs.get('source'))

    visit_Pointer_Assignment_Stmt = visit_Assignment_Stmt

    def create_operation(self, op, exprs, source):
        """
        Construct expressions from individual operations.
        """
        exprs = as_tuple(exprs)
        if op == '*':
            return sym.Product(exprs, source=source)
        if op == '/':
            return sym.Quotient(numerator=exprs[0], denominator=exprs[1], source=source)
        if op == '+':
            return sym.Sum(exprs, source=source)
        if op == '-':
            if len(exprs) > 1:
                # Binary minus
                return sym.Sum((exprs[0], sym.Product((-1, exprs[1]))), source=source)
            # Unary minus
            return sym.Product((-1, exprs[0]), source=source)
        if op == '**':
            return sym.Power(base=exprs[0], exponent=exprs[1], source=source)
        if op.lower() == '.and.':
            return sym.LogicalAnd(exprs, source=source)
        if op.lower() == '.or.':
            return sym.LogicalOr(exprs, source=source)
        if op.lower() in ('==', '.eq.'):
            return sym.Comparison(exprs[0], '==', exprs[1], source=source)
        if op.lower() in ('/=', '.ne.'):
            return sym.Comparison(exprs[0], '!=', exprs[1], source=source)
        if op.lower() in ('>', '.gt.'):
            return sym.Comparison(exprs[0], '>', exprs[1], source=source)
        if op.lower() in ('<', '.lt.'):
            return sym.Comparison(exprs[0], '<', exprs[1], source=source)
        if op.lower() in ('>=', '.ge.'):
            return sym.Comparison(exprs[0], '>=', exprs[1], source=source)
        if op.lower() in ('<=', '.le.'):
            return sym.Comparison(exprs[0], '<=', exprs[1], source=source)
        if op.lower() == '.not.':
            return sym.LogicalNot(exprs[0], source=source)
        if op.lower() == '.eqv.':
            return sym.LogicalOr((sym.LogicalAnd(exprs, source=source),
                                  sym.LogicalNot(sym.LogicalOr(exprs, source=source))), source=source)
        if op.lower() == '.neqv.':
            return sym.LogicalAnd((sym.LogicalNot(sym.LogicalAnd(exprs, source=source)),
                                   sym.LogicalOr(exprs, source=source)), source=source)
        if op == '//':
            return StringConcat(exprs, source=source)
        raise RuntimeError('FParser: Error parsing generic expression')

    def visit_Add_Operand(self, o, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        if len(o.items) > 2:
            # Binary operand
            exprs = [self.visit(o.items[0], **kwargs)]
            exprs += [self.visit(o.items[2], **kwargs)]
            return self.create_operation(op=o.items[1], exprs=exprs, source=source)
        # Unary operand
        exprs = [self.visit(o.items[1], **kwargs)]
        return self.create_operation(op=o.items[0], exprs=exprs, source=source)

    visit_Mult_Operand = visit_Add_Operand
    visit_And_Operand = visit_Add_Operand
    visit_Or_Operand = visit_Add_Operand
    visit_Equiv_Operand = visit_Add_Operand

    def visit_Level_2_Expr(self, o, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        e1 = self.visit(o.items[0], **kwargs)
        e2 = self.visit(o.items[2], **kwargs)
        return self.create_operation(op=o.items[1], exprs=(e1, e2), source=source)

    def visit_Level_2_Unary_Expr(self, o, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        exprs = as_tuple(self.visit(o.items[1], **kwargs))
        return self.create_operation(op=o.items[0], exprs=exprs, source=source)

    visit_Level_3_Expr = visit_Level_2_Expr
    visit_Level_4_Expr = visit_Level_2_Expr
    visit_Level_5_Expr = visit_Level_2_Expr

    def visit_Parenthesis(self, o, **kwargs):
        source = kwargs.get('source')
        expression = self.visit(o.items[1], **kwargs)
        if source:
            source = source.clone_with_string(o.string)
        if isinstance(expression, sym.Sum):
            expression = ParenthesisedAdd(expression.children, source=source)
        if isinstance(expression, sym.Product):
            expression = ParenthesisedMul(expression.children, source=source)
        if isinstance(expression, sym.Quotient):
            expression = ParenthesisedDiv(expression.numerator, expression.denominator, source=source)
        if isinstance(expression, sym.Power):
            expression = ParenthesisedPow(expression.base, expression.exponent, source=source)
        return expression

    visit_Format_Stmt = visit_Intrinsic_Stmt
    visit_Write_Stmt = visit_Intrinsic_Stmt
    visit_Goto_Stmt = visit_Intrinsic_Stmt
    visit_Return_Stmt = visit_Intrinsic_Stmt
    visit_Continue_Stmt = visit_Intrinsic_Stmt
    visit_Cycle_Stmt = visit_Intrinsic_Stmt
    visit_Exit_Stmt = visit_Intrinsic_Stmt
    visit_Save_Stmt = visit_Intrinsic_Stmt
    visit_Read_Stmt = visit_Intrinsic_Stmt
    visit_Open_Stmt = visit_Intrinsic_Stmt
    visit_Close_Stmt = visit_Intrinsic_Stmt
    visit_Inquire_Stmt = visit_Intrinsic_Stmt
    visit_Namelist_Stmt = visit_Intrinsic_Stmt
    visit_Parameter_Stmt = visit_Intrinsic_Stmt
    visit_Dimension_Stmt = visit_Intrinsic_Stmt
    visit_Equivalence_Stmt = visit_Intrinsic_Stmt
    visit_Common_Stmt = visit_Intrinsic_Stmt
    visit_Stop_Stmt = visit_Intrinsic_Stmt
    visit_Error_Stop_Stmt = visit_Intrinsic_Stmt
    visit_Backspace_Stmt = visit_Intrinsic_Stmt
    visit_Rewind_Stmt = visit_Intrinsic_Stmt
    visit_Entry_Stmt = visit_Intrinsic_Stmt

    def visit_Cpp_If_Stmt(self, o, **kwargs):
        return ir.PreprocessorDirective(text=o.tostr(), source=kwargs.get('source'))

    visit_Cpp_Elif_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Else_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Endif_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Macro_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Undef_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Line_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Warning_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Error_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Null_Stmt = visit_Cpp_If_Stmt

    def visit_Cpp_Include_Stmt(self, o, **kwargs):
        fname = o.items[0].tostr()
        return ir.Import(module=fname, c_import=True, source=kwargs.get('source'))

    def visit_Select_Type_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.Select_Type_Stmt):
                select_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            select_stmt = get_child(o, Fortran2003.Select_Type_Stmt)
        # Extract source by looking at everything between SELECT and END SELECT
        end_select_stmt = rget_child(o, Fortran2003.End_Select_Type_Stmt)
        lines = (select_stmt.item.span[0], end_select_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)
        label = self.get_label(select_stmt)
        # TODO: Treat this with a dedicated IR node (LOKI-33)
        return (*banter, ir.Intrinsic(text=string, label=label, source=source))

    def visit_Nullify_Stmt(self, o, **kwargs):
        if not o.items[1]:
            return ()
        variables = as_tuple(flatten(self.visit(v, **kwargs) for v in o.items[1].items))
        return ir.Nullify(variables=variables, label=kwargs.get('label'),
                          source=kwargs.get('source'))
