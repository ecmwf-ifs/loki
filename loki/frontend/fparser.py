import codecs
from collections import OrderedDict
import re
from pathlib import Path

from fparser.two.parser import ParserFactory
from fparser.two.utils import get_child
try:
    from fparser.two.utils import walk
except ImportError:
    from fparser.two.utils import walk_ast as walk
from fparser.two import Fortran2003
from fparser.common.readfortran import FortranStringReader

from loki.visitors import GenericVisitor
from loki.frontend.source import Source
from loki.frontend.util import (
    inline_comments, cluster_comments, inline_pragmas, process_dimension_pragmas
)
import loki.ir as ir
import loki.expression.symbol_types as sym
from loki.expression.operations import (
    StringConcat, ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow)
from loki.expression import ExpressionDimensionsMapper
from loki.logging import DEBUG, warning
from loki.tools import timeit, as_tuple, flatten
from loki.types import DataType, SymbolType


__all__ = ['FParser2IR', 'parse_fparser_file', 'parse_fparser_source', 'parse_fparser_ast']


_regex_ifndef = re.compile(r'#\s*if\b\s+[!]\s*defined\b\s*\(?([A-Za-z_]+)\)?')


@timeit(log_level=DEBUG)
def parse_fparser_file(filename):
    """
    Generate an internal IR from file via the fparser AST.
    """
    filepath = Path(filename)
    try:
        with filepath.open('r') as f:
            fcode = f.read()
    except UnicodeDecodeError as excinfo:
        warning('Skipping bad character in input file "%s": %s',
                str(filepath), str(excinfo))
        kwargs = {'mode': 'r', 'encoding': 'utf-8', 'errors': 'ignore'}
        with codecs.open(filepath, **kwargs) as f:
            fcode = f.read()

    return parse_fparser_source(source=fcode)


@timeit(log_level=DEBUG)
def parse_fparser_source(source):

    # Comment out ``@PROCESS`` instructions
    fcode = source.replace('@PROCESS', '! @PROCESS')

    # Replace ``#if !defined(...)`` by ``#ifndef ...`` due to fparser removing
    # everything that looks like an in-line comment (i.e., anything from the
    # letter '!' onwards).
    fcode = _regex_ifndef.sub(r'#ifndef \1', fcode)

    reader = FortranStringReader(fcode, ignore_comments=False)
    f2008_parser = ParserFactory().create(std='f2008')

    return f2008_parser(reader)


@timeit(log_level=DEBUG)
def parse_fparser_ast(ast, raw_source, typedefs=None, scope=None):
    """
    Generate an internal IR from file via the fparser AST.
    """

    # Parse the raw FParser language AST into our internal IR
    _ir = FParser2IR(raw_source=raw_source, typedefs=typedefs, scope=scope).visit(ast)

    # Perform soime minor sanitation tasks
    _ir = inline_comments(_ir)
    _ir = cluster_comments(_ir)
    _ir = inline_pragmas(_ir)

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


class FParser2IR(GenericVisitor):
    # pylint: disable=no-self-use  # Stop warnings about visitor methods that could do without self
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    def __init__(self, raw_source, typedefs=None, scope=None):
        super(FParser2IR, self).__init__()
        self.raw_source = raw_source.splitlines(keepends=True)
        self.typedefs = typedefs
        self.scope = scope

    def get_source(self, o, source):
        """
        Helper method that builds the source object for the node.
        """
        if not isinstance(o, str) and o.item is not None:
            label = getattr(o.item, 'label', None)
            lines = (o.item.span[0], o.item.span[1])
            string = ''.join(self.raw_source[lines[0] - 1:lines[1]])
            source = Source(lines=lines, string=string, label=label)
        return source

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        source = self.get_source(o, kwargs.pop('source', None))
        return super(FParser2IR, self).visit(o, source=source, **kwargs)

    def visit_Base(self, o, **kwargs):
        """
        Universal default for ``Base`` FParser-AST nodes
        """
        children = tuple(self.visit(c, **kwargs) for c in o.items if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_BlockBase(self, o, **kwargs):
        """
        Universal default for ``BlockBase`` FParser-AST nodes
        """
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_List(self, o, **kwargs):
        """
        Universal routine for auto-generated *_List types in fparser.
        """
        return as_tuple(flatten(self.visit(i, **kwargs) for i in o.items))

    visit_Attr_Spec_List = visit_List
    visit_Component_Attr_Spec_List = visit_List
    visit_Entity_Decl_List = visit_List
    visit_Component_Decl_list = visit_List
    visit_Explicit_Shape_Spec_List = visit_List
    visit_Assumed_Shape_Spec_List = visit_List
    visit_Deferred_Shape_Spec_List = visit_List
    visit_Allocate_Shape_Spec_List = visit_List
    visit_Ac_Value_List = visit_List
    visit_Section_Subscript_List = visit_List

    def visit_Actual_Arg_Spec_List(self, o, **kwargs):
        """
        Needs special treatment to avoid flattening key-value-pair tuples.
        """
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Name(self, o, **kwargs):
        # This one is evil, as it is used flat in expressions,
        # forcing us to generate ``Variable`` objects, and in
        # declarations, where none of the metadata is available
        # at this low level!
        vname = o.tostr().lower()

        # Careful! Mind the many ways in which this can get called with
        # outside information (either in kwargs or maps stored on self).
        dimensions = kwargs.get('dimensions', None)
        dtype = kwargs.get('dtype', None)
        parent = kwargs.get('parent', None)
        shape = kwargs.get('shape', None)
        initial = kwargs.get('initial', None)
        scope = kwargs.get('scope', self.scope)
        source = kwargs.get('source', None)

        if parent is not None:
            basename = vname
            vname = '%s%%%s' % (parent.name, vname)

        # Try to find the symbol in the symbol tables
        if dtype is None and scope is not None:
            dtype = self.scope.symbols.lookup(vname, recursive=True)

        # If a parent variable is given, try to infer type from the
        # derived type definition
        if parent is not None and dtype is None:
            if parent.type is not None and parent.type.dtype == DataType.DERIVED_TYPE:
                if parent.type.variables is not None and \
                        basename in parent.type.variables:
                    dtype = parent.type.variables[basename].type

        if shape is not None and dtype is not None and dtype.shape != shape:
            dtype = dtype.clone(shape=shape)

        if dimensions:
            dimensions = sym.ArraySubscript(dimensions)

        return sym.Variable(name=vname, dimensions=dimensions, type=dtype, scope=scope.symbols,
                            parent=parent, initial=initial, source=source)

    def visit_literal(self, val, _type, kind=None, **kwargs):
        if kind is not None:
            return sym.Literal(value=val, type=_type, kind=kind, source=kwargs.get('source'))
        return sym.Literal(value=val, type=_type, source=kwargs.get('source'))

    def visit_Char_Literal_Constant(self, o, **kwargs):
        return self.visit_literal(str(o.items[0]), DataType.CHARACTER, **kwargs)

    def visit_Int_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return self.visit_literal(int(o.items[0]), DataType.INTEGER, kind=kind, **kwargs)

    visit_Signed_Int_Literal_Constant = visit_Int_Literal_Constant

    def visit_Real_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return self.visit_literal(o.items[0], DataType.REAL, kind=kind, **kwargs)

    visit_Signed_Real_Literal_Constant = visit_Real_Literal_Constant

    def visit_Logical_Literal_Constant(self, o, **kwargs):
        return self.visit_literal(o.items[0], DataType.LOGICAL, **kwargs)

    def visit_Complex_Literal_Constant(self, o, **kwargs):
        return sym.IntrinsicLiteral(value=o.string, source=kwargs.get('source'))

    visit_Binary_Constant = visit_Complex_Literal_Constant
    visit_Octal_Constant = visit_Complex_Literal_Constant
    visit_Hex_Constant = visit_Complex_Literal_Constant

    def visit_Dimension_Attr_Spec(self, o, **kwargs):
        return self.visit(o.items[1], **kwargs)

    def visit_Component_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def visit_Intent_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def visit_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def visit_Specification_Part(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        return list(children)

    def visit_Use_Stmt(self, o, **kwargs):
        name = o.items[2].tostr()
        # TODO: This is probably not good
        # symbols = as_tuple(self.visit(s, **kwargs) for s in o.items[4].items)
        if o.items[4]:
            symbols = as_tuple(s.tostr() for s in o.items[4].items)
        else:
            symbols = None
        return ir.Import(module=name, symbols=symbols, source=kwargs.get('source'))

    def visit_Include_Stmt(self, o, **kwargs):
        fname = o.items[0].tostr()
        return ir.Import(module=fname, f_include=True, source=kwargs.get('source'))

    def visit_Implicit_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text='IMPLICIT %s' % o.items[0], source=kwargs.get('source', None))

    def visit_Print_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text='PRINT %s' % (', '.join(str(i) for i in o.items)),
                            source=kwargs.get('source', None))

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

    def visit_Entity_Decl(self, o, **kwargs):
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

    def visit_Component_Decl(self, o, **kwargs):
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

    def visit_Subscript_Triplet(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) if i is not None else None for i in o.items)
        return sym.RangeIndex(children)

    visit_Assumed_Shape_Spec = visit_Subscript_Triplet
    visit_Deferred_Shape_Spec = visit_Subscript_Triplet

    def visit_Explicit_Shape_Spec(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) if i is not None else None for i in o.items)
        if children[0] is None:
            return children[1]
        return sym.RangeIndex(children)

    visit_Allocate_Shape_Spec = visit_Explicit_Shape_Spec

    def visit_Allocation(self, o, **kwargs):
        dimensions = self.visit(o.items[1])
        kwargs['dimensions'] = dimensions
        kwargs['shape'] = dimensions
        return self.visit(o.items[0], **kwargs)

    def visit_Allocate_Stmt(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        source = kwargs.get('source', None)
        kw_args = {arg.items[0].lower(): self.visit(arg.items[1], **kwargs)
                   for arg in walk(o, Fortran2003.Alloc_Opt)}
        allocations = get_child(o, Fortran2003.Allocation_List)
        variables = tuple(self.visit(a, **kwargs) for a in allocations.items)
        return ir.Allocation(variables=variables, source=source, data_source=kw_args.get('source'))

    def visit_Deallocate_Stmt(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        source = kwargs.get('source', None)
        deallocations = get_child(o, Fortran2003.Allocate_Object_List)
        variables = tuple(self.visit(a, **kwargs) for a in deallocations.items)
        return ir.Deallocation(variables=variables, source=source)

    def visit_Intrinsic_Type_Spec(self, o, **kwargs):
        dtype = o.items[0]
        kind = get_child(o, Fortran2003.Kind_Selector)
        if kind is not None:
            kind = kind.items[1].tostr()
        length = get_child(o, Fortran2003.Length_Selector)
        if length is not None:
            length = length.items[1].tostr()
        return dtype, kind, length

    def visit_Intrinsic_Name(self, o, **kwargs):
        return o.tostr()

    def visit_Initialization(self, o, **kwargs):
        return self.visit(o.items[1], **kwargs)

    def visit_Array_Constructor(self, o, **kwargs):
        values = self.visit(o.items[1], **kwargs)
        return sym.LiteralList(values=values)

    def visit_Ac_Implied_Do(self, o, **kwargs):
        # TODO: Implement this properly!
        return o.tostr()

    def visit_Intrinsic_Function_Reference(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        # Do not recurse here to avoid treating function names as variables
        name = o.items[0].tostr()  # self.visit(o.items[0], **kwargs)

        if name.upper() in ('REAL', 'INT'):
            args = walk(o.items, (Fortran2003.Actual_Arg_Spec_List,))[0]
            expr = self.visit(args.items[0])
            if len(args.items) > 1:
                # Do not recurse here to avoid treating kind names as variables
                kind = walk(o.items, (Fortran2003.Actual_Arg_Spec,))
                # If kind is not specified as named argument, simply take the second
                # argument and convert it to a string
                kind = kind[0].items[1].tostr() if kind else args.items[1].tostr()
            else:
                kind = None
            return sym.Cast(name, expr, kind=kind)

        args = self.visit(o.items[1], **kwargs) if o.items[1] else None
        if args:
            kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
            arguments = tuple(a for a in args if not isinstance(a, tuple))
        else:
            arguments = None
            kwarguments = None
        return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments)

    visit_Function_Reference = visit_Intrinsic_Function_Reference

    def visit_Actual_Arg_Spec(self, o, **kwargs):
        key = o.items[0].tostr()
        value = self.visit(o.items[1], **kwargs)
        return (key, value)

    def visit_Data_Ref(self, o, **kwargs):
        v = self.visit(o.items[0], source=kwargs.get('source', None))
        for i in o.items[1:-1]:
            # Careful not to propagate type or dims here
            v = self.visit(i, parent=v, source=kwargs.get('source', None))
        # Attach types and dims to final leaf variable
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Data_Pointer_Object(self, o, **kwargs):
        v = self.visit(o.items[0], source=kwargs.get('source', None))
        for i in o.items[1:-1]:
            if i == '%':
                continue
            # Careful not to propagate type or dims here
            v = self.visit(i, parent=v, source=kwargs.get('source', None))
        # Attach types and dims to final leaf variable
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Part_Ref(self, o, **kwargs):
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

        if name.lower() in Fortran2003.Intrinsic_Name.function_names or kwarguments:
            # This is (presumably) a function call
            return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments)

        # This is an array access and the arguments define the dimension.
        kwargs['dimensions'] = args
        # Recurse down to visit_Name
        return self.visit(o.items[0], **kwargs)

    def visit_Proc_Component_Ref(self, o, **kwargs):
        '''This is the compound object for accessing procedure components of a variable.'''
        pname = o.items[0].tostr().lower()
        v = sym.Variable(name=pname, scope=self.scope.symbols)
        for i in o.items[1:-1]:
            if i != '%':
                v = self.visit(i, parent=v, source=kwargs.get('source'))
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Array_Section(self, o, **kwargs):
        kwargs['dimensions'] = as_tuple(self.visit(o.items[1]))
        return self.visit(o.items[0], **kwargs)

    visit_Substring_Range = visit_Subscript_Triplet

    def visit_Type_Declaration_Stmt(self, o, **kwargs):
        """
        Declaration statement in the spec of a module/routine. This function is also called
        for declarations of members of a derived type.
        """
        source = kwargs.get('source', None)

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

        # First, pick out parameters, including explicit DIMENSIONs
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

        # Next, figure out the type we're declaring
        dtype = None
        basetype_ast = get_child(o, Fortran2003.Intrinsic_Type_Spec)
        if basetype_ast is not None:
            dtype, kind, length = self.visit(basetype_ast)
            dtype = SymbolType(DataType.from_fortran_type(dtype), kind=kind, intent=intent,
                               parameter='parameter' in attrs, optional='optional' in attrs,
                               allocatable='allocatable' in attrs, pointer='pointer' in attrs,
                               contiguous='contiguous' in attrs, shape=dimensions, length=length)

        derived_type_ast = get_child(o, Fortran2003.Declaration_Type_Spec)
        if derived_type_ast is not None:
            typename = derived_type_ast.items[1].tostr().lower()
            dtype = self.scope.types.lookup(typename, recursive=True)
            if dtype is not None:
                # Add declaration attributes to the data type from the typedef
                dtype = dtype.clone(intent=intent, allocatable='allocatable' in attrs,
                                    pointer='pointer' in attrs, optional='optional' in attrs,
                                    parameter='parameter' in attrs, target='target' in attrs,
                                    contiguous='contiguous' in attrs, shape=dimensions)
            else:
                # TODO: Insert variable information from stored TypeDef!
                if self.typedefs is not None and typename in self.typedefs:
                    variables = OrderedDict([(v.basename, v) for v in self.typedefs[typename].variables])
                else:
                    variables = None
                dtype = SymbolType(DataType.DERIVED_TYPE, name=typename, variables=variables,
                                   intent=intent, allocatable='allocatable' in attrs,
                                   pointer='pointer' in attrs, optional='optional' in attrs,
                                   parameter='parameter' in attrs, target='target' in attrs,
                                   contiguous='contiguous' in attrs, shape=dimensions)

        # Now create the actual variables declared in this statement
        # (and provide them with the type and dimension information)
        kwargs['dimensions'] = dimensions
        kwargs['dtype'] = dtype
        variables = as_tuple(self.visit(o.items[2], **kwargs))
        return ir.Declaration(variables=variables, dimensions=dimensions, source=source)

    def visit_Derived_Type_Def(self, o, **kwargs):
        name = get_child(o, Fortran2003.Derived_Type_Stmt).items[1].tostr().lower()
        source = kwargs.get('source', None)
        # Visit comments (and pragmas)
        comments = [self.visit(i, **kwargs) for i in walk(o.content, (Fortran2003.Comment,))]
        pragmas = [c for c in comments if isinstance(c, ir.Pragma)]
        comments = [c for c in comments if not isinstance(c, ir.Pragma)]
        # Create the typedef with all the information we have so far (we need its symbol table
        # for the next step)
        typedef = ir.TypeDef(name=name, declarations=[], pragmas=pragmas, comments=comments,
                             source=source)
        # Create declarations and update the parent typedef
        declarations = flatten([self.visit(i, scope=typedef, **kwargs)
                                for i in walk(o.content, (Fortran2003.Component_Part,))])
        typedef._update(declarations=declarations, symbols=typedef.symbols)
        # Infer any additional shape information from `!$loki dimension` pragmas
        process_dimension_pragmas(typedef)
        # Now create a SymbolType instance to make the typedef known in its scope's type table
        variables = OrderedDict([(v.basename, v) for v in typedef.variables])
        dtype = SymbolType(DataType.DERIVED_TYPE, name=name, variables=variables, source=source)
        self.scope.types[name] = dtype
        return typedef

    def visit_Component_Part(self, o, **kwargs):
        return as_tuple(flatten(self.visit(a, **kwargs) for a in o.content))

    # Declaration of members of a derived type (i.e., part of the definition of the derived type.
    visit_Data_Component_Def_Stmt = visit_Type_Declaration_Stmt

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
        if end_do_stmt is None:
            # We may have a labeled loop with an explicit CONTINUE statement
            end_do_stmt = rget_child(o, Fortran2003.Continue_Stmt)
            assert str(end_do_stmt.item.label) == do_stmt.label.string
        lines = (do_stmt.item.span[0], end_do_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]])
        source = Source(lines=lines, string=string, label=do_stmt.item.name)
        # Extract loop header and get stepping info
        variable, bounds = self.visit(do_stmt, **kwargs)
        # Extract and process the loop body
        body_nodes = node_sublist(o.content, do_stmt.__class__, Fortran2003.End_Do_Stmt)
        body = as_tuple(flatten(self.visit(node, **kwargs) for node in body_nodes))
        # Loop label for labeled do constructs
        label = str(do_stmt.items[1]) if isinstance(do_stmt, Fortran2003.Label_Do_Stmt) else None
        # Select loop type
        if bounds:
            obj = ir.Loop(variable=variable, body=body, bounds=bounds, label=label, source=source)
        else:
            obj = ir.WhileLoop(condition=variable, body=body, label=label, source=source)
        return (*banter, obj, )

    visit_Block_Label_Do_Construct = visit_Block_Nonlabel_Do_Construct

    def visit_Nonlabel_Do_Stmt(self, o, **kwargs):
        variable, bounds = None, None
        loop_control = get_child(o, Fortran2003.Loop_Control)
        if loop_control:
            variable, bounds = self.visit(loop_control, **kwargs)
        return variable, bounds

    visit_Label_Do_Stmt = visit_Nonlabel_Do_Stmt

    def visit_If_Construct(self, o, **kwargs):
        # The banter before the loop...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.If_Then_Stmt):
                if_then_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            if_then_stmt = get_child(o, Fortran2003.If_Then_Stmt)
        # Extract source by looking at everything between IF and END IF statements
        end_if_stmt = rget_child(o, Fortran2003.End_If_Stmt)
        lines = (if_then_stmt.item.span[0], end_if_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]])
        source = Source(lines=lines, string=string, label=if_then_stmt.item.name)
        # Start with the condition that is always there
        conditions = [self.visit(if_then_stmt, **kwargs)]
        # Walk throught the if construct and collect statements for the if branch
        # Pick up any ELSE IF along the way and collect their statements as well
        bodies = []
        body = []
        for child in node_sublist(o.content, Fortran2003.If_Then_Stmt, Fortran2003.Else_Stmt):
            node = self.visit(child, **kwargs)
            if isinstance(child, Fortran2003.Else_If_Stmt):
                bodies.append(as_tuple(body))
                body = []
                conditions.append(node)
            else:
                body.append(node)
        bodies.append(as_tuple(body))
        assert len(conditions) == len(bodies)
        else_ast = node_sublist(o.content, Fortran2003.Else_Stmt, Fortran2003.End_If_Stmt)
        else_body = as_tuple(flatten(self.visit(a, **kwargs) for a in as_tuple(else_ast)))
        return (*banter, ir.Conditional(conditions=conditions, bodies=bodies,
                                        else_body=else_body, inline=False, source=source))

    def visit_If_Then_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    visit_Else_If_Stmt = visit_If_Then_Stmt

    def visit_If_Stmt(self, o, **kwargs):
        source = kwargs.get('source', None)
        cond = as_tuple(self.visit(o.items[0], **kwargs))
        body = as_tuple(self.visit(o.items[1], **kwargs))
        return ir.Conditional(conditions=cond, bodies=body, else_body=(), inline=True, source=source)

    def visit_Call_Stmt(self, o, **kwargs):
        name = o.items[0].tostr()
        args = self.visit(o.items[1], **kwargs) if o.items[1] else None
        source = kwargs.get('source', None)
        if args:
            kw_args = tuple(arg for arg in args if isinstance(arg, tuple))
            args = tuple(arg for arg in args if not isinstance(arg, tuple))
        else:
            args = ()
            kw_args = ()
        return ir.CallStatement(name=name, arguments=args, kwarguments=kw_args, source=source)

    def visit_Loop_Control(self, o, **kwargs):
        if o.items[0]:
            # Scalar logical expression
            return self.visit(o.items[0], **kwargs), None
        variable = self.visit(o.items[1][0], **kwargs)
        bounds = as_tuple(flatten(self.visit(a, **kwargs) for a in as_tuple(o.items[1][1])))
        return variable, sym.LoopRange(bounds)

    def visit_Assignment_Stmt(self, o, **kwargs):
        ptr = isinstance(o, Fortran2003.Pointer_Assignment_Stmt)
        source = kwargs.get('source', None)
        target = self.visit(o.items[0], **kwargs)
        expr = self.visit(o.items[2], **kwargs)
        return ir.Statement(target=target, expr=expr, ptr=ptr, source=source)

    visit_Pointer_Assignment_Stmt = visit_Assignment_Stmt

    def visit_operation(self, op, exprs, **kwargs):
        """
        Construct expressions from individual operations.
        """
        source = kwargs.get('source')
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
        if op == '==' or op.lower() == '.eq.':
            return sym.Comparison(exprs[0], '==', exprs[1], source=source)
        if op == '/=' or op.lower() == '.ne.':
            return sym.Comparison(exprs[0], '!=', exprs[1], source=source)
        if op == '>' or op.lower() == '.gt.':
            return sym.Comparison(exprs[0], '>', exprs[1], source=source)
        if op == '<' or op.lower() == '.lt.':
            return sym.Comparison(exprs[0], '<', exprs[1], source=source)
        if op == '>=' or op.lower() == '.ge.':
            return sym.Comparison(exprs[0], '>=', exprs[1], source=source)
        if op == '<=' or op.lower() == '.le.':
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
        if len(o.items) > 2:
            # Binary operand
            exprs = [self.visit(o.items[0], **kwargs)]
            exprs += [self.visit(o.items[2], **kwargs)]
            return self.visit_operation(op=o.items[1], exprs=exprs, **kwargs)
        # Unary operand
        exprs = [self.visit(o.items[1], **kwargs)]
        return self.visit_operation(op=o.items[0], exprs=exprs, **kwargs)

    visit_Mult_Operand = visit_Add_Operand
    visit_And_Operand = visit_Add_Operand
    visit_Or_Operand = visit_Add_Operand
    visit_Equiv_Operand = visit_Add_Operand

    def visit_Level_2_Expr(self, o, **kwargs):
        e1 = self.visit(o.items[0], **kwargs)
        e2 = self.visit(o.items[2], **kwargs)
        return self.visit_operation(op=o.items[1], exprs=(e1, e2), **kwargs)

    def visit_Level_2_Unary_Expr(self, o, **kwargs):
        exprs = as_tuple(self.visit(o.items[1], **kwargs))
        return self.visit_operation(op=o.items[0], exprs=exprs, **kwargs)

    visit_Level_3_Expr = visit_Level_2_Expr
    visit_Level_4_Expr = visit_Level_2_Expr
    visit_Level_5_Expr = visit_Level_2_Expr

    def visit_Parenthesis(self, o, **kwargs):
        expression = self.visit(o.items[1], **kwargs)
        if isinstance(expression, sym.Sum):
            expression = ParenthesisedAdd(expression.children)
        if isinstance(expression, sym.Product):
            expression = ParenthesisedMul(expression.children)
        if isinstance(expression, sym.Power):
            expression = ParenthesisedPow(expression.base, expression.exponent)
        return expression

    def visit_Associate_Construct(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        # Search for the ASSOCIATE statement and add all following items as its body
        assoc_index = [isinstance(ch, ir.Scope) for ch in children].index(True)
        children[assoc_index].body = children[assoc_index + 1:]
        return children[:assoc_index + 1]

    def visit_Associate_Stmt(self, o, **kwargs):
        associations = OrderedDict()
        for assoc in o.items[1].items:
            var = self.visit(assoc.items[2], **kwargs)
            if isinstance(var, sym.Array):
                shape = ExpressionDimensionsMapper()(var)
            else:
                shape = None
            dtype = var.type.clone(name=None, parent=None, shape=shape)
            associations[var] = self.visit(assoc.items[0], dtype=dtype, **kwargs)
        return ir.Scope(associations=associations)

    def visit_Intrinsic_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text=o.tostr(), source=kwargs.get('source'))

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
    visit_External_Stmt = visit_Intrinsic_Stmt
    visit_Common_Stmt = visit_Intrinsic_Stmt
    visit_Stop_Stmt = visit_Intrinsic_Stmt

    def visit_Cpp_If_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text=o.tostr(), source=kwargs.get('source'))

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
        body = as_tuple(flatten(self.visit(ch) for ch in body_ast))
        default = as_tuple(flatten(self.visit(ch) for ch in default_ast))
        source = kwargs.get('source', None)
        return (*banter, ir.MaskedStatement(condition, body, default, source=source))

    def visit_Where_Construct_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def visit_Where_Stmt(self, o, **kwargs):
        source = kwargs.get('source', None)
        condition = self.visit(o.items[0], **kwargs)
        body = as_tuple(self.visit(o.items[1], **kwargs))
        default = ()
        return ir.MaskedStatement(condition, body, default, source=source)

    def visit_Case_Construct(self, o, **kwargs):
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
        string = ''.join(self.raw_source[lines[0]-1:lines[1]])
        source = Source(lines=lines, string=string, label=select_stmt.item.name)
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
        return (*banter, ir.MultiConditional(expr, values, bodies, else_body, source=source))

    def visit_Select_Case_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def visit_Case_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    visit_Case_Value_Range = visit_Subscript_Triplet

    def visit_Data_Stmt_Set(self, o, **kwargs):
        # TODO: actually parse the statements
        # pylint: disable=no-member
        variable = self.visit(get_child(o, Fortran2003.Data_Stmt_Object_List), **kwargs)
        values = as_tuple(self.visit(get_child(o, Fortran2003.Data_Stmt_Value_List), **kwargs))
        return ir.DataDeclaration(variable=variable, values=values, source=kwargs.get('source'))

    def visit_Data_Stmt_Value(self, o, **kwargs):
        exprs = as_tuple(flatten(self.visit(c) for c in o.items))
        return self.visit_operation('*', exprs, **kwargs)

    def visit_Nullify_Stmt(self, o, **kwargs):
        if not o.items[1]:
            return ()
        source = kwargs.get('source', None)
        variables = as_tuple(flatten(self.visit(v, **kwargs) for v in o.items[1].items))
        return ir.Nullify(variables=variables, source=source)

    def visit_Interface_Block(self, o, **kwargs):
        spec = get_child(o, Fortran2003.Interface_Stmt).items[0]
        if spec:
            spec = spec if isinstance(spec, str) else spec.tostr()
        body_ast = node_sublist(o.children, Fortran2003.Interface_Stmt,
                                Fortran2003.End_Interface_Stmt)
        body = as_tuple(flatten(self.visit(ch, **kwargs) for ch in body_ast))
        source = kwargs.get('source', None)
        return ir.Interface(spec=spec, body=body, source=source)

    def visit_Function_Stmt(self, o, **kwargs):
        # TODO: not implemented
        return ir.Intrinsic(text=o.tostr(), source=kwargs.get('source'))

    visit_Subroutine_Stmt = visit_Function_Stmt
