from collections import OrderedDict
import re

from fparser.two.parser import ParserFactory
from fparser.two.utils import get_child, BlockBase
try:
    from fparser.two.utils import walk
except ImportError:
    from fparser.two.utils import walk_ast as walk
from fparser.two import Fortran2003
from fparser.common.readfortran import FortranStringReader

from loki.visitors import GenericVisitor
from loki.frontend.source import Source
from loki.frontend.preprocessing import sanitize_registry
from loki.frontend.util import (
    inline_comments, cluster_comments, read_file, import_external_symbols, FP,
    combine_multiline_pragmas
)
import loki.ir as ir
import loki.expression.symbols as sym
from loki.expression.operations import (
    StringConcat, ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow)
from loki.expression import ExpressionDimensionsMapper, FindTypedSymbols, SubstituteExpressions
from loki.logging import DEBUG
from loki.tools import timeit, as_tuple, flatten, CaseInsensitiveDict
from loki.pragma_utils import attach_pragmas, process_dimension_pragmas, detach_pragmas
from loki.types import BasicType, DerivedType, ProcedureType, Scope, SymbolAttributes


__all__ = ['FParser2IR', 'parse_fparser_file', 'parse_fparser_source', 'parse_fparser_ast',
           'parse_fparser_expression']


@timeit(log_level=DEBUG)
def parse_fparser_file(filename):
    """
    Generate an internal IR from file via the fparser AST.
    """
    fcode = read_file(filename)
    return parse_fparser_source(source=fcode)


@timeit(log_level=DEBUG)
def parse_fparser_source(source):
    reader = FortranStringReader(source, ignore_comments=False)
    f2008_parser = ParserFactory().create(std='f2008')

    return f2008_parser(reader)


@timeit(log_level=DEBUG)
def parse_fparser_ast(ast, raw_source, pp_info=None, definitions=None, scope=None):
    """
    Generate an internal IR from file via the fparser AST.
    """

    # Parse the raw FParser language AST into our internal IR
    _ir = FParser2IR(raw_source=raw_source, definitions=definitions, scope=scope).visit(ast)

    # Apply postprocessing rules to re-insert information lost during preprocessing
    if pp_info is not None:
        for r_name, rule in sanitize_registry[FP].items():
            info = pp_info.get(r_name, None)
            _ir = rule.postprocess(_ir, info)

    # Perform some minor sanitation tasks
    _ir = inline_comments(_ir)
    _ir = cluster_comments(_ir)
    _ir = combine_multiline_pragmas(_ir)

    return _ir


def parse_fparser_expression(source, scope):
    """
    Parse an expression string into an expression tree.

    This exploits Fparser's internal parser structure that relies on recursively
    matching strings against a list of node types. Usually, this would start
    by matching against module, subroutine or program. Here, we shortcut this
    hierarchy by directly matching against a primary expression, thus this
    should be able to parse any syntactically correct Fortran expression.

    :param str source: the expression as a string.
    :param Scope scope: the scope to which symbol names inside the expression belong.

    :return: the expression tree.
    """
    _ = ParserFactory().create(std='f2008')
    # Wrap source in brackets to make sure it appears like a valid expression
    # for fparser, and strip that Parenthesis node from the ast immediately after
    ast = Fortran2003.Primary('(' + source + ')').children[1]
    _ir = parse_fparser_ast(ast, source, scope=scope)
    # TODO: use rescope visitor for this
    rescope_map = {v: v.clone(scope=scope) for v in FindTypedSymbols().visit(_ir)}
    _ir = SubstituteExpressions(rescope_map).visit(_ir)
    return _ir


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
    Extract the py:class:`Source` object for any py:class:`fparser.two.utils.BlockBase`
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
    # pylint: disable=no-self-use  # Stop warnings about visitor methods that could do without self
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    def __init__(self, raw_source, definitions=None, scope=None):
        super().__init__()
        self.raw_source = raw_source.splitlines(keepends=True)
        self.definitions = CaseInsensitiveDict((d.name, d) for d in as_tuple(definitions))
        self.scope = scope

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
        kwargs.setdefault('scope', self.scope)
        return super().visit(o, **kwargs)

    ### scratchpad: re-written visit functions ###

    def visit_List(self, o, **kwargs):
        """
        Universal routine for auto-generated *_List types in fparser

        `*_List` types have their items children
        """
        return tuple(self.visit(i, **kwargs) for i in o.children)

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

    visit_Execution_Part = visit_Specification_Part

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
        _type = kwargs['scope'].symbols.lookup(name.name)
        if _type and isinstance(_type.dtype, ProcedureType):
            name = name.clone(dimensions=None)
            call = sym.InlineCall(name, parameters=dimensions, kw_parameters=(), source=kwargs.get('source'))
            return call
        return name

    def visit_Data_Ref(self, o, **kwargs):
        """
        A fully qualified name for accessing a derived type or class member,
        composed from individual :any:`Part_Ref` as
        ``part-ref [% part-ref [% part-ref ...] ]``

        :class:`fparser.two.Fortran2003.Data_Ref` has variable number of children,
        depending on the number of part-ref.
        """
        var = self.visit(o.children[0], **kwargs)
        for c in o.children[1:]:
            parent = var
            var = self.visit(c, **kwargs)
            var = var.clone(name='{}%{}'.format(parent.name, var.name), parent=parent)
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
            raise NotImplementedError
        name = o.children[2].tostr()
        module = self.definitions.get(name)
        scope = kwargs['scope']
        if o.children[3] == '':
            # No ONLY list (import all)
            symbols = None
            if module is not None:
                for k, v in module.symbols.items():
                    scope.symbols[k] = v.clone(imported=True, module=module)
        elif o.children[3] == ', ONLY:':
            # ONLY list given (import only selected symbols)
            symbols = () if o.children[4] is None else self.visit(o.children[4], **kwargs)
            if module is None:
                scope.symbols.update({s.name: SymbolAttributes(BasicType.DEFERRED, imported=True) for s in symbols})
            else:
                for s in symbols:
                    scope.symbols[s.name] = module.symbols[s.name].clone(imported=True, module=module)
            symbols = tuple(s.clone(scope=scope) for s in symbols)
        elif o.children[3] == ',':
            # Rename list
            raise NotImplementedError
        else:
            raise NotImplementedError

        return ir.Import(module=name, symbols=symbols, source=kwargs.get('source'), label=kwargs.get('label'))

    visit_Only_List = visit_List

    #
    # Variable declarations
    #

    def visit_Type_Declaration_Stmt(self, o, **kwargs):
        """
        Declaration statement

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

        # EXTERNAL attribute means this is actually a function or subroutine
        external = _type.external is True
        if _type.external:
            return_type = _type.dtype if _type.dtype is not BasicType.DEFERRED else None
            _type = _type.clone(return_type=return_type)

        # Make sure KIND (which can be a name) is in the right scope
        scope = kwargs['scope']
        if _type.kind is not None and isinstance(_type.kind, sym.TypedSymbol):
            # TODO: put it in the right scope (Rescope Visitor)
            _type = _type.clone(kind=_type.kind.clone(scope=scope))

        # Update symbol table entries
        for var in variables:
            if external:
                type_kwargs = _type.__dict__.copy()
                type_kwargs['dtype'] = ProcedureType(var.name, is_function=_type.dtype is not None)
                scope.symbols[var.name] = var.type.clone(**type_kwargs)
            else:
                scope.symbols[var.name] = var.type.clone(**_type.__dict__)

        variables = tuple(v.clone(scope=scope) for v in variables)
        return ir.Declaration(variables=variables, dimensions=_type.shape, external=external,
                              source=kwargs.get('source'), label=kwargs.get('label'))

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
            if dtype in (BasicType.INTEGER, BasicType.REAL, BasicType.COMPLEX, BasicType.LOGICAL):
                return SymbolAttributes(dtype, kind=self.visit(o.children[1], **kwargs))
            if dtype is BasicType.CHARACTER:
                return SymbolAttributes(dtype, length=self.visit(o.children[1], **kwargs))
            raise NotImplementedError
        return SymbolAttributes(dtype)

    def visit_Kind_Selector(self, o, **kwargs):
        """
        A kind selector of an intrinsic type

        :class:`fparser.two.Fortran2003.Kind_Selector` has 3 children:
            * '(' (str)
            * :class:`fparser.two.Fortran2003.Scalar_Int_Initialization_Expr`
            * ')' (str)
        """
        assert o.children[0] == '(' and o.children[2] == ')'
        return self.visit(o.children[1], **kwargs)

    def visit_Length_Selector(self, o, **kwargs):
        """
        A length selector for intrinsic character type

        :class:`fparser.two.Fortran2003.Length_Selector` has 3 children:
            * '(' (str)
            * :class:`fparser.two.Fortran2003.Char_Length`
            * ')' (str)
        """
        if o.children[0] == '*':
            raise NotImplementedError
        assert o.children[0] == '(' and o.children[2] == ')'
        return self.visit(o.children[1], **kwargs)

    def visit_Declaration_Type_Spec(self, o, **kwargs):
        """
        A derived type specifier in a declaration

        :class:`fparser.two.Fortran2003.Declaration_Type_Spec` has 2 children:
            * keyword 'TYPE' or 'CLASS' (str)
            * :class:`fparser.two.Fortran2003.Derived_Type_Spec`
        """
        if o.children[0].upper() in ('TYPE', 'CLASS'):
            # TODO: record if `TYPE` or `CLASS` is used
            dtype = self.visit(o.children[1], **kwargs)

            # Look for a previous definition of this type
            _type = kwargs['scope'].symbols.lookup(dtype.name)
            if _type is None or _type.dtype is BasicType.DEFERRED:
                _type = SymbolAttributes(dtype)

            # Strip import annotations
            return _type.clone(imported=None, module=None)

        raise NotImplementedError

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

        :class:`fparser.two.Fortran2003.Attr_Spec` has no children
        """
        return (o.tostr().lower(), True)

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
            dimensions = self.visit(o.children[1], **kwargs)
            var = var.clone(dimensions=dimensions, type=var.type.clone(shape=dimensions))

        if o.children[2]:
            raise NotImplementedError

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
        raise NotImplementedError

    def visit_External_Stmt(self, o, **kwargs):
        """
        An ``EXTERNAL`` statement to specify the external attribute for a list of names

        :class:`fparser.two.Fortran2003.External_Stmt` has 2 children:
            * keyword 'EXTERNAL (`str`)
            * the list of names :class:`fparser.two.Fortran2003.External_Name_List`
        """
        assert o.children[0].upper() == 'EXTERNAL'

        # Compile the list of names...
        variables = self.visit(o.children[1], **kwargs)

        # ...and update their symbol table entry...
        scope = kwargs['scope']
        for var in variables:
            _type = scope.symbols.lookup(var.name)
            if _type is None:
                _type = SymbolAttributes(dtype=ProcedureType(var.name, is_function=False), external=True)
            else:
                _type = _type.clone(external=True)
                #return_type = None if _type.dtype is BasicType.DEFERRED else _type.dtype
                #dtype = ProcedureType(var.name, is_function=return_type is not None)
                #type_kwargs = {'dtype': dtype, 'external': True, 'return_type': None}
            scope.symbols[var.name] = _type

        variables = tuple(v.clone(scope=scope) for v in variables)
        declaration = ir.Declaration(variables=variables, external=True,
                                     source=kwargs.get('source'), label=kwargs.get('label'))
        return declaration

    visit_External_Name_List = visit_List

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
            raise NotImplementedError
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
        raise NotImplementedError

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
        raise NotImplementedError

    def visit_Substring_Range(self, o, **kwargs):
        raise NotImplementedError

    def visit_Stride(self, o, **kwargs):
        raise NotImplementedError

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

        # Name of the derived type
        name = self.visit(derived_type_stmt, **kwargs)

        # Initialize a symbol table for the typedef
        parent_scope = kwargs['scope']
        kwargs['scope'] = Scope(parent=parent_scope)
        body = [self.visit(c, **kwargs) for c in o.children[derived_type_stmt_index+1:end_type_stmt_index]]
        body = as_tuple(flatten(body))

        # TODO: type-bound procedures are currently stored flat as Intrinsic in the body.
        # These should become declarations and TypeDef should probably store them separately

        # Infer any additional shape information from `!$loki dimension` pragmas
        # Note that this needs to be done before we create `dtype` below, to allow
        # propagation of type info through multiple typedefs in the same module.
        body = attach_pragmas(body, ir.Declaration)
        body = process_dimension_pragmas(body)
        body = detach_pragmas(body, ir.Declaration)

        source = kwargs.get('source')
        label = kwargs.get('label')
        typedef = ir.TypeDef(name=name, body=body, scope=kwargs['scope'], source=source, label=label)

        # Make the typedef known in the parent scope
        parent_scope.symbols[name] = SymbolAttributes(DerivedType(name=name, typedef=typedef))
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
            raise NotImplementedError
        name = o.children[1].tostr()
        if o.children[2] is not None:
            raise NotImplementedError
        return name

    def visit_Component_Part(self, o, **kwargs):
        """
        Derived type definition components

        :class:`fparser.two.Fortran2003.Component_Part` has a list of
        :class:`fparser.two.Fortran2003.Data_Component_Def_Stmt` or
        :class:`fparser.two.Fortran2003.Proc_Component_Def_Stmt` as children
        """
        return tuple(self.visit(c, **kwargs) for c in o.children)

    visit_Data_Component_Def_Stmt = visit_Type_Declaration_Stmt
    visit_Component_Attr_Spec_List = visit_List
    visit_Component_Attr_Spec = visit_Attr_Spec
    visit_Dimension_Component_Attr_Spec = visit_Dimension_Attr_Spec
    visit_Component_Decl_List = visit_List
    visit_Component_Decl = visit_Entity_Decl

    def visit_Proc_Component_Def_Stmt(self, o, **kwargs):
        """
        A procedure declaration in a derived type definition
        """
        raise NotImplementedError

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
        return ir.Intrinsic(text=o.tostr(), label=kwargs.get('label'), source=kwargs.get('source'))

    visit_Contains_Stmt = visit_Specific_Binding
    visit_Binding_Private_Stmt = visit_Specific_Binding
    visit_Generic_Binding = visit_Specific_Binding
    visit_Final_Binding = visit_Specific_Binding

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
        scope = parent_scope  # TODO: actually create own scope
        kwargs['scope'] = scope

        # TODO: Apply some rescope-visitor here
        rescoped_associations = []
        for expr, name in associations:
            rescope_map = {var: var.clone(scope=parent_scope) for var in FindTypedSymbols().visit(expr)}
            expr = SubstituteExpressions(rescope_map).visit(expr)
            name = name.clone(scope=scope)
            rescoped_associations += [(expr, name)]
        associations = as_tuple(rescoped_associations)

        # Update symbol table for associates
        for expr, name in associations:
            if isinstance(expr, sym.TypedSymbol):
                # Use the type of the associated variable
                _type = parent_scope.symbols.lookup(expr.name)
                if isinstance(expr, sym.Array) and expr.dimensions is not None:
                    shape = ExpressionDimensionsMapper()(expr)
                    _type = _type.clone(shape=shape)
            else:
                # TODO: Handle data type and shape of complex expressions
                shape = ExpressionDimensionsMapper()(expr)
                _type = SymbolAttributes(BasicType.DEFERRED, shape=shape)
            scope.symbols[name.name] = _type

        # The body
        body = as_tuple(self.visit(c, **kwargs) for c in o.children[assoc_stmt_index+1:end_assoc_stmt_index])
        associate = ir.Associate(associations=associations, body=body, label=kwargs.get('label'), source=source)

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

        # The interface spec and body
        spec = self.visit(interface_stmt, **kwargs)
        body = as_tuple(self.visit(c, **kwargs) for c in o.children[interface_stmt_index+1:end_interface_stmt_index])
        interface = ir.Interface(spec=spec, body=body, label=kwargs.get('label'), source=source)

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
        if o.children[0] is None:
            return None
        if o.children[0] == 'ABSTRACT':
            return 'ABSTRACT'
        # We are currently capturing this simply as a string
        return o.children[0].tostr()

    def visit_Subroutine_Body(self, o, **kwargs):
        """
        A subroutine definition in an interface block

        :class:`fparser.two.Fortran2003.Subroutine_Body` has variable number of children:
            * Any preceeding comments :class:`fparser.two.Fortran2003.Comment`
            * :class:`fparser.two.Fortran2003.Subroutine_Stmt` (the actual statement
              that begins the construct)
            * the spec :class:`fparser.two.Fortran2003.Specification_Part`
            * the closing :class:`fparser.two.Fortran2003.End_Subroutine_Stmt`

        """
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel

        # Find start and end of construct
        subroutine_stmt = get_child(o, Fortran2003.Subroutine_Stmt)
        subroutine_stmt_index = o.children.index(subroutine_stmt)
        end_subroutine_stmt = get_child(o, Fortran2003.End_Subroutine_Stmt)
        end_subroutine_stmt_index = o.children.index(end_subroutine_stmt)

        # Everything before the construct
        pre = as_tuple(self.visit(c, **kwargs) for c in o.children[:subroutine_stmt_index])

        # Extract source object for construct
        lines = (subroutine_stmt.item.span[0], end_subroutine_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)

        # Create a scope
        parent_scope = kwargs['scope']
        scope = Scope(parent=parent_scope)
        kwargs['scope'] = scope

        # Name and dummy args
        name, args, bind = self.visit(subroutine_stmt, **kwargs)
        is_function = isinstance(subroutine_stmt, Fortran2003.Function_Stmt)

        # Spec
        spec_ast = get_child(o, Fortran2003.Specification_Part)
        spec_ast_index = o.children.index(spec_ast)
        spec = self.visit(spec_ast, **kwargs)

        # Make sure there is nothing else in there
        assert (subroutine_stmt_index, spec_ast_index, end_subroutine_stmt_index) == \
                as_tuple(range(subroutine_stmt_index, len(o.children)))

        # Note: the Subroutine constructor registers itself in the parent scope
        routine = Subroutine(name=name, args=args, spec=spec, ast=o, scope=scope, bind=bind,
                             is_function=is_function, source=source)
        return (*pre, routine)

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
        if o.children[0] is not None:
            # We can't handle prefix, yet
            raise NotImplementedError

        name = self.visit(o.children[1], **kwargs)
        name = name.name

        dummy_arg_list = self.visit(o.children[2], **kwargs)
        args = tuple(str(arg) for arg in dummy_arg_list)

        bind = None if o.children[3] is None else o.children[3].tostr()

        return name, args, bind

    visit_Function_Stmt = visit_Subroutine_Stmt
    visit_Subroutine_Name = visit_Name
    visit_Function_Name = visit_Name
    visit_Dummy_Arg_List = visit_List

    #
    # Conditional
    #

    def visit_If_Construct(self, o, **kwargs):
        """
        The entire ``IF`` construct

        :class:`fparser.two.Fortran2003.If_Construct has variable number of children:
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
        else_body = as_tuple(self.visit(c, **kwargs) for c in o.children[else_stmt_index+1:end_if_stmt_index])

        # Extract source objects for branches
        sources, labels = [], []
        for conditional in (if_then_stmt,) + else_if_stmts:
            lines = (conditional.item.span[0], end_if_stmt.item.span[1])
            string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
            sources += [Source(lines=lines, string=string)]
            labels += [self.get_label(conditional)]

        # Build IR nodes backwards using else-if branch as else body
        node = ir.Conditional(condition=conditions[-1], body=bodies[-1], else_body=else_body,
                              inline=False, has_elseif=False, label=labels[-1], source=sources[-1])
        for idx in reversed(range(len(conditions)-1)):
            node = ir.Conditional(condition=conditions[idx], body=bodies[idx], else_body=(node,),
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
        assert case_stmt_index[0] == select_case_stmt_index + 1

        values = as_tuple(self.visit(c, **kwargs) for c in case_stmts)
        bodies = [as_tuple(self.visit(c, **kwargs) for c in o.children[start+1:stop])
                  for start, stop in zip(case_stmt_index, case_stmt_index[1:] + (end_select_stmt_index,))]

        if 'DEFAULT' in values:
            default_index = values.index('DEFAULT')
            else_body = bodies[default_index]
            values = values[:default_index] + values[default_index+1:]
            bodies = bodies[:default_index] + bodies[default_index+1:]
        else:
            else_body = None

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
            raise NotImplementedError

        # Any allocation options. We can only deal with "source" at the moment
        alloc_opts = {}
        if o.children[2] is not None:
            alloc_opts = dict(self.visit(o.children[2], **kwargs))

        variables = self.visit(o.children[1], **kwargs)
        return ir.Allocation(variables=variables, data_source=alloc_opts.get('source'),
                             source=kwargs.get('source'), label=kwargs.get('label'))

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

    def visit_Alloc_Opt(self, o, **kwargs):
        """
        An allocation option in an allocate-stmt

        :class:`fparser.two.Fortran2003.Alloc_Opt has two children:
            * the keyword (`str`)
            * the option value
        """
        if o.children[0].lower() == 'source':
            return 'source', self.visit(o.children[1], **kwargs)
        raise NotImplementedError

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
        name = name.clone(name='{}%{}'.format(parent.name, name.name), parent=parent)
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
                assert len(kwarguments) == 1 and kwarguments[0][0] == 'kind'
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

    ### end scratchpad : re-written visit functions ###

    def visit_Base(self, o, **kwargs):
        """
        Universal default for ``Base`` FParser-AST nodes
        """
        import pdb; pdb.set_trace()
        children = tuple(self.visit(c, **kwargs) for c in o.items if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_BlockBase(self, o, **kwargs):
        """
        Universal default for ``BlockBase`` FParser-AST nodes
        """
        import pdb; pdb.set_trace()
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    #visit_Assumed_Shape_Spec_List = visit_List
    #visit_Deferred_Shape_Spec_List = visit_List
    #visit_Allocate_Shape_Spec_List = visit_List
    #visit_Ac_Value_List = visit_List
    #visit_Section_Subscript_List = visit_List

    def old_visit_Actual_Arg_Spec_List(self, o, **kwargs):
        """
        Needs special treatment to avoid flattening key-value-pair tuples.
        """
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def old_visit_Name(self, o, **kwargs):
        # This one is evil, as it is used flat in expressions,
        # forcing us to generate ``Variable`` objects, and in
        # declarations, where none of the metadata is available
        # at this low level!
        vname = o.tostr()

        # Careful! Mind the many ways in which this can get called with
        # outside information (either in kwargs or maps stored on self).
        dimensions = kwargs.get('dimensions')
        external = kwargs.get('external')
        dtype = kwargs.get('dtype')
        parent = kwargs.get('parent')
        shape = kwargs.get('shape')
        initial = kwargs.get('initial')
        scope = kwargs.get('scope', self.scope)

        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)

        if parent is not None:
            basename = vname
            vname = '%s%%%s' % (parent.name, vname)

        # Try to find the symbol in the symbol tables
        if dtype is None and scope is not None:
            dtype = self.scope.symbols.lookup(vname, recursive=True)

        # If a parent variable is given, try to infer type from the
        # derived type definition
        if parent is not None and dtype is None:
            if parent.type is not None and isinstance(parent.type.dtype, DerivedType):
                if parent.type.variables is not None and \
                        basename in parent.type.variables:
                    dtype = parent.type.variables[basename].type

        if shape is not None and dtype is not None and dtype.shape != shape:
            dtype = dtype.clone(shape=shape)

        if initial is not None :
            dtype = dtype.clone(initial=initial)

        if dimensions:
            dimensions = sym.ArraySubscript(dimensions)

        if external:
            # Fortran's EXTERNAL statement/attribute is evil, as it looks like a regular
            # variable declaration but declares a procedure symbol. Depending on whether
            # we have a data type for that symbol or not, we can deduce it to be a
            # function or subroutine
            if dtype is None:
                dtype = SymbolAttributes(dtype=ProcedureType(name=vname, is_function=False), external=external)
            else:
                dtype = dtype.clone(dtype=ProcedureType(name=vname, is_function=True),
                                    external=external, return_type=dtype.dtype)

        return sym.Variable(name=vname, dimensions=dimensions, type=dtype,
                            scope=scope, parent=parent, source=source)

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
                scope = kwargs.get('scope', self.scope)
                kind = sym.Variable(name=kind, scope=scope, source=source)
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

    def old_visit_Dimension_Attr_Spec(self, o, **kwargs):
        return self.visit(o.items[1], **kwargs)

    def old_visit_Component_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def old_visit_Intent_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def old_visit_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def old_visit_Specification_Part(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        return list(children)

    def old_visit_Use_Stmt(self, o, **kwargs):
        name = o.items[2].tostr()
        only_list = get_child(o, Fortran2003.Only_List)  # pylint: disable=no-member
        symbols = None
        if only_list:
            symbol_names = tuple(item.tostr() for item in only_list.items)
            module = self.definitions.get(name, None)
            symbols = import_external_symbols(module=module, symbol_names=symbol_names, scope=self.scope)
        return ir.Import(module=name, symbols=symbols, source=kwargs.get('source'),
                         label=kwargs.get('label'))

    def visit_Include_Stmt(self, o, **kwargs):
        fname = o.items[0].tostr()
        return ir.Import(module=fname, f_include=True, source=kwargs.get('source'),
                         label=kwargs.get('label'))

    def visit_Implicit_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text='IMPLICIT %s' % o.items[0], source=kwargs.get('source'),
                            label=kwargs.get('label'))

    def visit_Print_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text='PRINT %s' % (', '.join(str(i) for i in o.items)),
                            source=kwargs.get('source'), label=kwargs.get('label'))

    # TODO: Deal with line-continuation pragmas!
    _re_pragma = re.compile(r'\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

    def visit_Comment(self, o, **kwargs):
        source = kwargs.get('source', None)
        match_pragma = self._re_pragma.search(o.tostr())
        if match_pragma:
            # Found pragma, generate this instead
            gd = match_pragma.groupdict()
            return ir.Pragma(keyword=gd['keyword'], content=gd['content'], source=source)
        return ir.Comment(text=o.tostr(), source=source)

    def old_visit_Entity_Decl(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        dims = get_child(o, Fortran2003.Explicit_Shape_Spec_List)
        dims = get_child(o, Fortran2003.Assumed_Shape_Spec_List) if dims is None else dims
        if dims is not None:
            kwargs['dimensions'] = self.visit(dims)

        init = get_child(o, Fortran2003.Initialization)
        if init is not None:
            kwargs['initial'] = self.visit(init)

        # We know that this is a declaration, so the ``dimensions``
        # here also define the shape of the variable symbol within the
        # currently cached context.
        kwargs['shape'] = kwargs.get('dimensions', None)

        return self.visit(o.items[0], **kwargs)

    def old_visit_Component_Decl(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        dims = get_child(o, Fortran2003.Explicit_Shape_Spec_List)
        dims = get_child(o, Fortran2003.Assumed_Shape_Spec_List) if dims is None else dims
        dims = get_child(o, Fortran2003.Deferred_Shape_Spec_List) if dims is None else dims
        if dims is not None:
            dims = self.visit(dims)
            # We know that this is a declaration, so the ``dimensions``
            # here also define the shape of the variable symbol within the
            # currently cached context.
            kwargs['dimensions'] = dims
            kwargs['shape'] = dims

        return self.visit(o.items[0], **kwargs)

    def old_visit_Subscript_Triplet(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) if i is not None else None for i in o.items)
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.RangeIndex(children, source=source)

    #visit_Assumed_Shape_Spec = visit_Subscript_Triplet
    #visit_Deferred_Shape_Spec = visit_Subscript_Triplet

    def old_visit_Explicit_Shape_Spec(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) if i is not None else None for i in o.items)
        if children[0] is None:
            return children[1]
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.RangeIndex(children, source=source)

    #visit_Allocate_Shape_Spec = visit_Explicit_Shape_Spec

    def old_visit_Allocation(self, o, **kwargs):
        dimensions = self.visit(o.items[1])
        kwargs['dimensions'] = dimensions
        kwargs['shape'] = dimensions
        return self.visit(o.items[0], **kwargs)

    def old_visit_Allocate_Stmt(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        kw_args = {arg.items[0].lower(): self.visit(arg.items[1], **kwargs)
                   for arg in walk(o, Fortran2003.Alloc_Opt)}
        allocations = get_child(o, Fortran2003.Allocation_List)
        variables = tuple(self.visit(a, **kwargs) for a in allocations.items)
        return ir.Allocation(variables=variables, source=kwargs.get('source'),
                             data_source=kw_args.get('source'), label=kwargs.get('label'))

    def visit_Deallocate_Stmt(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        deallocations = get_child(o, Fortran2003.Allocate_Object_List)
        variables = tuple(self.visit(a, **kwargs) for a in deallocations.items)
        return ir.Deallocation(variables=variables, source=kwargs.get('source'),
                               label=kwargs.get('label'))

    def old_visit_Intrinsic_Type_Spec(self, o, **kwargs):
        dtype = o.items[0]
        kind = get_child(o, Fortran2003.Kind_Selector)
        if kind is not None:
            kind = self.visit(kind.items[1], **kwargs)
        length = get_child(o, Fortran2003.Length_Selector)
        if length is not None:
            length = length.items[1].tostr()
        return dtype, kind, length

    def old_visit_Intrinsic_Name(self, o, **kwargs):
        return o.tostr()

    def old_visit_Array_Constructor(self, o, **kwargs):
        values = self.visit(o.items[1], **kwargs)
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.LiteralList(values=values, source=source)

    def old_visit_Ac_Implied_Do(self, o, **kwargs):
        # TODO: Implement this properly!
        return o.tostr()

    def old_visit_Intrinsic_Function_Reference(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        # Do not recurse here to avoid treating function names as variables
        name = o.items[0].tostr()  # self.visit(o.items[0], **kwargs)

        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)

        scope = kwargs.get('scope', self.scope)
        fct_symbol = sym.ProcedureSymbol(name, scope=scope, source=source)

        if name.upper() in ('REAL', 'INT'):
            args = walk(o.items, (Fortran2003.Actual_Arg_Spec_List,))[0]
            expr = self.visit(args.items[0])
            if len(args.items) > 1:
                # Do not recurse here to avoid treating kind names as variables
                kind = walk(o.items, (Fortran2003.Actual_Arg_Spec,))
                # If kind is not specified as named argument, simply take the second argument
                kind = kind[0].items[1] if kind else args.items[1]
                kind = self.visit(kind, **kwargs)
            else:
                kind = None
            return sym.Cast(fct_symbol, expr, kind=kind, source=source)

        args = self.visit(o.items[1], **kwargs) if o.items[1] else None
        if args:
            kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
            arguments = tuple(a for a in args if not isinstance(a, tuple))
        else:
            arguments = None
            kwarguments = None
        return sym.InlineCall(fct_symbol, parameters=arguments,
                              kw_parameters=kwarguments, source=source)

    #visit_Function_Reference = visit_Intrinsic_Function_Reference

    def old_visit_Actual_Arg_Spec(self, o, **kwargs):
        key = o.items[0].tostr()
        value = self.visit(o.items[1], **kwargs)
        return (key, value)

    def old_visit_Data_Ref(self, o, **kwargs):
        v = self.visit(o.items[0], source=kwargs.get('source'))
        for i in o.items[1:-1]:
            # Careful not to propagate type or dims here
            v = self.visit(i, parent=v, source=kwargs.get('source'))
        # Attach types and dims to final leaf variable
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Data_Pointer_Object(self, o, **kwargs):
        v = self.visit(o.items[0], source=kwargs.get('source'))
        for i in o.items[1:-1]:
            if i == '%':
                continue
            # Careful not to propagate type or dims here
            v = self.visit(i, parent=v, source=kwargs.get('source'))
        # Attach types and dims to final leaf variable
        return self.visit(o.items[-1], parent=v, **kwargs)

    def old_visit_Part_Ref(self, o, **kwargs):
        # WARNING: Due to fparser's lack of a symbol table, it is not always possible to
        # distinguish between array subscript and function call. This employs a heuristic
        # identifying only intrinsic function calls and calls with keyword parameters as
        # a function call.
        name = o.items[0].tostr()
        parent = kwargs.get('parent', None)
        if parent:
            name = '%s%%%s' % (parent, name)
        args = as_tuple(self.visit(o.items[1])) if o.items[1] else None
        if args:
            kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
            arguments = as_tuple(a for a in args if not isinstance(a, tuple))
        else:
            arguments = None
            kwarguments = None

        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)

        fct_type = self.scope.symbols.lookup(name)
        if fct_type and isinstance(fct_type.dtype, ProcedureType):
            # We know this function from out own type table
            fct_symbol = sym.ProcedureSymbol(name, type=fct_type, scope=self.scope, source=source)
            return sym.InlineCall(fct_symbol, parameters=arguments, kw_parameters=kwarguments,
                                  source=source)

        if name.lower() in Fortran2003.Intrinsic_Name.function_names or kwarguments:
            # This is (presumably) a function call
            fct_type = ProcedureType('name')
            fct_symbol = sym.ProcedureSymbol(name, type=fct_type, scope=self.scope, source=source)
            return sym.InlineCall(fct_symbol, parameters=arguments, kw_parameters=kwarguments,
                                  source=source)

        # This is an array access and the arguments define the dimension.
        kwargs['dimensions'] = args
        # Recurse down to visit_Name
        return self.visit(o.items[0], **kwargs)

    def old_visit_Structure_Constructor(self, o, **kwargs):
        # TODO: fparser wrongfully parses calls to functions without arguments as this type.
        # This means this routine also produces inline calls for actual inline calls...
        name = get_child(o, Fortran2003.Type_Name).tostr()
        component_specs = get_child(o, Fortran2003.Component_Spec_List)  # pylint: disable=no-member
        if component_specs:
            args = as_tuple(self.visit(component_specs, **kwargs))
            kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
            arguments = as_tuple(a for a in args if not isinstance(a, tuple))
        else:
            arguments = None
            kwarguments = None

        source = kwargs.get('source')
        fct_type = self.scope.symbols.lookup(name)
        fct_symbol = sym.Variable(name=name, type=fct_type, scope=self.scope, source=source)
        return sym.InlineCall(fct_symbol, parameters=arguments, kw_parameters=kwarguments,
                              source=source)

    def visit_Proc_Component_Ref(self, o, **kwargs):
        '''This is the compound object for accessing procedure components of a variable.'''
        pname = o.items[0].tostr().lower()
        v = sym.Variable(name=pname, scope=self.scope)
        for i in o.items[1:-1]:
            if i != '%':
                v = self.visit(i, parent=v, source=kwargs.get('source'))
        return self.visit(o.items[-1], parent=v, **kwargs)

    def old_visit_Array_Section(self, o, **kwargs):
        kwargs['dimensions'] = as_tuple(self.visit(o.items[1]))
        return self.visit(o.items[0], **kwargs)

    #visit_Substring_Range = visit_Subscript_Triplet

    def old_visit_Type_Declaration_Stmt(self, o, **kwargs):
        """
        Declaration statement in the spec of a module/routine. This function is also called
        for declarations of members of a derived type.
        """
        # Super-hacky, this fecking DIMENSION keyword will be my undoing one day!
        dimensions = [self.visit(a, **kwargs)
                      for a in walk(o.items, (Fortran2003.Dimension_Component_Attr_Spec,
                                              Fortran2003.Dimension_Attr_Spec))]
        if dimensions:
            if isinstance(o, Fortran2003.Data_Component_Def_Stmt):
                dimensions = dimensions[0][1]
            else:
                dimensions = dimensions[0]
        else:
            dimensions = None

        # First, pick out parameters, including explicit DIMENSIONs and EXTERNAL
        attrs = as_tuple(str(self.visit(a)).lower().strip()
                         for a in walk(o.items, (
                             Fortran2003.Attr_Spec, Fortran2003.Component_Attr_Spec,
                             Fortran2003.Intent_Attr_Spec)))
        intent = None
        if 'intent(in)' in attrs:
            intent = 'in'
        elif 'intent(inout)' in attrs:
            intent = 'inout'
        elif 'intent(out)' in attrs:
            intent = 'out'

        external = 'external' in attrs

        # All the declared attributes that we retain
        type_attrs = {
            'intent': intent,
            'parameter': 'parameter' in attrs,
            'optional': 'optional' in attrs,
            'allocatable': 'allocatable' in attrs,
            'pointer': 'pointer' in attrs,
            'contiguous': 'contiguous' in attrs,
            'target': 'target' in attrs,
            'shape': dimensions,
        }

        # Next, figure out the type we're declaring
        stype = None
        basetype_ast = get_child(o, Fortran2003.Intrinsic_Type_Spec)
        if basetype_ast is not None:
            dtype, kind, length = self.visit(basetype_ast)
            stype = SymbolAttributes(BasicType.from_fortran_type(dtype), kind=kind, length=length, **type_attrs)

        derived_type_ast = get_child(o, Fortran2003.Declaration_Type_Spec)
        if derived_type_ast is not None:
            typename = derived_type_ast.items[1].tostr().lower()
            dtype = self.scope.symbols.lookup(typename, recursive=True)
            if dtype is None or dtype.dtype is BasicType.DEFERRED:
                dtype = DerivedType(name=typename, typedef=BasicType.DEFERRED)
            else:
                dtype = dtype.dtype
            stype = SymbolAttributes(dtype, **type_attrs)

        assert stype is not None

        # Now create the actual variables declared in this statement
        # (and provide them with the type and dimension information)
        kwargs['dimensions'] = dimensions
        kwargs['external'] = external
        kwargs['dtype'] = stype
        variables = as_tuple(self.visit(o.items[2], **kwargs))
        return ir.Declaration(variables=variables, dimensions=dimensions, external=external,
                              source=kwargs.get('source'), label=kwargs.get('label'))

    def old_visit_External_Stmt(self, o, **kwargs):
        # pylint: disable=no-member
        kwargs['external'] = True
        variables = as_tuple(self.visit(get_child(o, Fortran2003.External_Name_List), **kwargs))
        return ir.Declaration(variables=variables, external=True, source=kwargs.get('source'),
                              label=kwargs.get('label'))

    def old_visit_Derived_Type_Def(self, o, **kwargs):
        name = get_child(o, Fortran2003.Derived_Type_Stmt).items[1].tostr().lower()
        source = kwargs.get('source')

        # Initialize a local scope for typedef objects
        typedef_scope = Scope(parent=self.scope)

        # Create declarations and update the parent typedef
        component_nodes = (Fortran2003.Component_Part, Fortran2003.Comment)
        body_kwargs = kwargs.copy()
        body_kwargs['scope'] = typedef_scope
        body = flatten([self.visit(i, **body_kwargs)
                        for i in walk(o.content, component_nodes)])
        # Infer any additional shape information from `!$loki dimension` pragmas
        # Note that this needs to be done before we create `dtype` below, to allow
        # propagation of type info through multiple typedefs in the same module.
        body = attach_pragmas(body, ir.Declaration)
        body = process_dimension_pragmas(body)
        body = detach_pragmas(body, ir.Declaration)
        typedef = ir.TypeDef(name=name, body=body, scope=typedef_scope,
                             source=source, label=kwargs.get('label'))

        # Now make the typedef known in its scope's type table
        self.scope.symbols[name] = SymbolAttributes(DerivedType(name=name, typedef=typedef))

        return typedef

    def old_visit_Component_Part(self, o, **kwargs):
        return as_tuple(flatten(self.visit(a, **kwargs) for a in o.content))

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

    def old_visit_If_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.If_Then_Stmt):
                if_then_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            if_then_stmt = get_child(o, Fortran2003.If_Then_Stmt)
        # Identifiers
        end_if_stmt = rget_child(o, Fortran2003.End_If_Stmt)
        construct_name = if_then_stmt.item.name
        # Start with the condition that is always there
        conditions = [self.visit(if_then_stmt, **kwargs)]
        labels = [self.get_label(if_then_stmt)]
        # Walk throught the if construct and collect statements for the if branch
        # Pick up any ELSE IF along the way and collect their statements as well
        bodies, source = [], []
        body = []
        for child in node_sublist(o.content, Fortran2003.If_Then_Stmt, Fortran2003.Else_Stmt):
            if isinstance(child, Fortran2003.End_If_Stmt):
                # Skip this explicitly (only there in case it has a construct name)
                continue
            node = self.visit(child, **kwargs)
            if isinstance(child, Fortran2003.Else_If_Stmt):
                bodies.append(as_tuple(flatten(body)))
                source.append(self.get_block_source(if_then_stmt, end_if_stmt))
                conditions.append(node)
                labels.append(self.get_label(child))
                if_then_stmt = child
                body = []
            else:
                body.append(node)
        bodies.append(as_tuple(flatten(body)))
        source.append(self.get_block_source(if_then_stmt, end_if_stmt))
        assert len(conditions) == len(bodies) and len(source) == len(bodies)
        # Take care of the else branch
        else_ast = node_sublist(o.content, Fortran2003.Else_Stmt, Fortran2003.End_If_Stmt)
        else_body = as_tuple(flatten(self.visit(a, **kwargs) for a in as_tuple(else_ast)))
        # Now build IR nodes backwards, using else-if branch as else body
        node = ir.Conditional(condition=conditions[-1], body=bodies[-1], else_body=else_body, inline=False,
                              has_elseif=False, label=labels[-1], source=source[-1])
        for idx in reversed(range(len(conditions)-1)):
            name = construct_name if idx == 0 else None
            node = ir.Conditional(condition=conditions[idx], body=bodies[idx], else_body=(node,), inline=False,
                                  has_elseif=True, label=labels[idx], name=name, source=source[idx])
        return (*banter, node)

    def old_visit_If_Then_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    #visit_Else_If_Stmt = visit_If_Then_Stmt

    def visit_If_Stmt(self, o, **kwargs):
        cond = self.visit(o.items[0], **kwargs)
        body = as_tuple(self.visit(o.items[1], **kwargs))
        return ir.Conditional(condition=cond, body=body, else_body=(), inline=True,
                              label=kwargs.get('label'), source=kwargs.get('source'))

    def old_visit_Call_Stmt(self, o, **kwargs):
        name = o.items[0].tostr()
        # FParser inserts annoying spaces, so remove them!
        name = name.replace(' ', '')
        args = self.visit(o.items[1], **kwargs) if o.items[1] else None
        if args:
            kw_args = tuple(arg for arg in args if isinstance(arg, tuple))
            args = tuple(arg for arg in args if not isinstance(arg, tuple))
        else:
            args = ()
            kw_args = ()
        return ir.CallStatement(name=name, arguments=args, kwarguments=kw_args,
                                label=kwargs.get('label'), source=kwargs.get('source'))

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
        if isinstance(expression, sym.Power):
            expression = ParenthesisedPow(expression.base, expression.exponent, source=source)
        return expression

    def old_visit_Associate_Construct(self, o, **kwargs):
        children = [self.visit(c, **kwargs) for c in o.content]
        children = as_tuple(flatten(c for c in children if c is not None))
        # Search for the ASSOCIATE statement and add all following items as its body
        assoc_index = [isinstance(ch, ir.Associate) for ch in children].index(True)
        # Extract source for the entire scope
        lines = (children[assoc_index].source.lines[0], children[-1].source.lines[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        children[assoc_index]._update(body=children[assoc_index + 1:],
                                      source=Source(lines=lines, string=string))
        return children[:assoc_index + 1]

    def old_visit_Associate_Stmt(self, o, **kwargs):
        associations = OrderedDict()
        for assoc in get_child(o, Fortran2003.Association_List).items:  # pylint: disable=no-member
            var = self.visit(assoc.items[2], **kwargs)
            if isinstance(var, sym.Array):
                shape = ExpressionDimensionsMapper()(var)
            else:
                shape = None
            dtype = var.type.clone(name=None, parent=None, shape=shape)
            associations[var] = self.visit(assoc.items[0], dtype=dtype, **kwargs)
        return ir.Associate(associations=associations, label=kwargs.get('label'),
                            source=kwargs.get('source'))

    def visit_Intrinsic_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text=o.tostr(), label=kwargs.get('label'), source=kwargs.get('source'))

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
    visit_Access_Stmt = visit_Intrinsic_Stmt
    visit_Namelist_Stmt = visit_Intrinsic_Stmt
    visit_Parameter_Stmt = visit_Intrinsic_Stmt
    visit_Dimension_Stmt = visit_Intrinsic_Stmt
    visit_Final_Binding = visit_Intrinsic_Stmt
    visit_Procedure_Stmt = visit_Intrinsic_Stmt
    visit_Equivalence_Stmt = visit_Intrinsic_Stmt
    visit_Common_Stmt = visit_Intrinsic_Stmt
    visit_Stop_Stmt = visit_Intrinsic_Stmt
    visit_Backspace_Stmt = visit_Intrinsic_Stmt
    visit_Rewind_Stmt = visit_Intrinsic_Stmt

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

    def visit_Where_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.Where_Construct_Stmt):
                break
            banter += [self.visit(ch, **kwargs)]
        # The mask condition
        condition = self.visit(get_child(o, Fortran2003.Where_Construct_Stmt), **kwargs)
        default_ast = node_sublist(o.children, Fortran2003.Elsewhere_Stmt,
                                   Fortran2003.End_Where_Stmt)
        if default_ast:
            body_ast = node_sublist(o.children, Fortran2003.Where_Construct_Stmt,
                                    Fortran2003.Elsewhere_Stmt)
        else:
            body_ast = node_sublist(o.children, Fortran2003.Where_Construct_Stmt,
                                    Fortran2003.End_Where_Stmt)
        body = as_tuple(self.visit(ch, **kwargs) for ch in body_ast)
        default = as_tuple(self.visit(ch, **kwargs) for ch in default_ast)
        return (*banter, ir.MaskedStatement(condition, body, default, label=kwargs.get('label'),
                                            source=kwargs.get('source')))

    def visit_Where_Construct_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def visit_Where_Stmt(self, o, **kwargs):
        condition = self.visit(o.items[0], **kwargs)
        body = as_tuple(self.visit(o.items[1], **kwargs))
        default = ()
        return ir.MaskedStatement(condition, body, default, label=kwargs.get('label'),
                                  source=kwargs.get('source'))

    def old_visit_Case_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.Select_Case_Stmt):
                select_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            select_stmt = get_child(o, Fortran2003.Select_Case_Stmt)
        # Extract source by looking at everything between SELECT and END SELECT
        end_select_stmt = rget_child(o, Fortran2003.End_Select_Stmt)
        lines = (select_stmt.item.span[0], end_select_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)
        label = self.get_label(select_stmt)
        construct_name = select_stmt.item.name

        # The SELECT argument
        expr = self.visit(select_stmt, **kwargs)
        body_ast = node_sublist(o.children, Fortran2003.Select_Case_Stmt,
                                Fortran2003.End_Select_Stmt)
        values = []
        bodies = []
        body = []
        is_else_body = False
        else_body = ()
        for child in body_ast:
            node = self.visit(child, **kwargs)
            if isinstance(child, Fortran2003.Case_Stmt):
                if is_else_body:
                    else_body = as_tuple(body)
                    is_else_body = False
                elif values:  # Avoid appending empty body before first Case_Stmt
                    bodies.append(as_tuple(body))
                body = []
                if node is None:  # default case
                    is_else_body = True
                else:
                    values.append(node)
            else:
                body.append(node)
        if is_else_body:
            else_body = body
        else:
            bodies.append(as_tuple(body))
        assert len(values) == len(bodies)
        return (*banter, ir.MultiConditional(expr, values, bodies, else_body, label=label,
                                             name=construct_name, source=source))

    def old_visit_Select_Case_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def old_visit_Case_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    #visit_Case_Value_Range = visit_Subscript_Triplet

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

    def old_visit_Data_Stmt_Set(self, o, **kwargs):
        # TODO: actually parse the statements
        # pylint: disable=no-member
        variable = self.visit(get_child(o, Fortran2003.Data_Stmt_Object_List), **kwargs)
        values = as_tuple(self.visit(get_child(o, Fortran2003.Data_Stmt_Value_List), **kwargs))
        return ir.DataDeclaration(variable=variable, values=values, label=kwargs.get('label'),
                                  source=kwargs.get('source'))

    def old_visit_Data_Stmt_Value(self, o, **kwargs):
        exprs = as_tuple(flatten(self.visit(c) for c in o.items))
        return self.create_operation('*', exprs, source=kwargs.get('source'))

    def visit_Nullify_Stmt(self, o, **kwargs):
        if not o.items[1]:
            return ()
        variables = as_tuple(flatten(self.visit(v, **kwargs) for v in o.items[1].items))
        return ir.Nullify(variables=variables, label=kwargs.get('label'),
                          source=kwargs.get('source'))

    def old_visit_Interface_Block(self, o, **kwargs):
        spec = get_child(o, Fortran2003.Interface_Stmt).items[0]
        if spec:
            spec = spec if isinstance(spec, str) else spec.tostr()
        body_ast = node_sublist(o.children, Fortran2003.Interface_Stmt,
                                Fortran2003.End_Interface_Stmt)
        body = as_tuple(flatten(self.visit(ch, **kwargs) for ch in body_ast))
        return ir.Interface(spec=spec, body=body, label=kwargs.get('label'),
                            source=kwargs.get('source'))

    def old_visit_Function_Stmt(self, o, **kwargs):
        # TODO: not implemented
        return ir.Intrinsic(text=o.tostr(), label=kwargs.get('label'), source=kwargs.get('source'))

    #visit_Subroutine_Stmt = visit_Function_Stmt
