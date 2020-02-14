import codecs
from collections import OrderedDict
import re

from fparser.two.parser import ParserFactory
from fparser.two.utils import get_child
try:
    from fparser.two.utils import walk
except ImportError:
    from fparser.two.utils import walk_ast as walk
from fparser.two import Fortran2003
import fparser.two.Fortran2003 as fp
from fparser.common.readfortran import FortranStringReader
from pymbolic.primitives import (Sum, Product, Quotient, Power, Comparison, LogicalNot,
                                 LogicalAnd, LogicalOr)
from pathlib import Path

from loki.visitors import GenericVisitor
from loki.frontend.source import Source
from loki.frontend.util import inline_comments, cluster_comments, inline_pragmas
from loki.ir import (
    Comment, Declaration, Statement, Loop, Conditional, Allocation, Deallocation,
    TypeDef, Import, Intrinsic, CallStatement, Scope, Pragma, MaskedStatement, MultiConditional
)
from loki.expression import (Variable, Literal, InlineCall, Array, RangeIndex, LiteralList, Cast,
                             ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow, StringConcat,
                             ExpressionDimensionsMapper)
from loki.logging import DEBUG, warning
from loki.tools import timeit, as_tuple, flatten
from loki.types import DataType, SymbolType

__all__ = ['FParser2IR', 'parse_fparser_file', 'parse_fparser_ast']


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


class FParser2IR(GenericVisitor):

    def __init__(self, typedefs=None, scope=None):
        super(FParser2IR, self).__init__()
        self.typedefs = typedefs
        self.scope = scope

    def visit(self, o, **kwargs):
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        source = kwargs.pop('source', None)
        if not isinstance(o, str) and o.item is not None:
            label = getattr(o.item, 'label', None)
            string = getattr(o.item, 'line', None)
            source = Source(lines=o.item.span, string=string, label=label)
        return super(FParser2IR, self).visit(o, source=source, **kwargs)

    def visit_Base(self, o, **kwargs):
        """
        Universal default for ``Base`` FParser-AST nodes
        """
        children = tuple(self.visit(c, **kwargs) for c in o.items if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        else:
            return children if len(children) > 0 else None

    def visit_BlockBase(self, o, **kwargs):
        """
        Universal default for ``BlockBase`` FParser-AST nodes
        """
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        else:
            return children if len(children) > 0 else None

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
                if parent.type.variables is not None:
                    dtype = parent.type.variables[basename]

        if shape is not None and dtype is not None and dtype.shape != shape:
            dtype = dtype.clone(shape=shape)

        return Variable(name=vname, dimensions=dimensions, type=dtype, scope=scope.symbols,
                        parent=parent, initial=initial, source=source)

    def visit_Char_Literal_Constant(self, o, **kwargs):
        return Literal(value=str(o.items[0]), type=DataType.CHARACTER)

    def visit_Int_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return Literal(value=int(o.items[0]), type=DataType.INTEGER, kind=kind)

    def visit_Signed_Int_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return Literal(value=int(o.items[0]), type=DataType.INTEGER, kind=kind)

    def visit_Real_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return Literal(value=o.items[0], type=DataType.REAL, kind=kind)

    def visit_Logical_Literal_Constant(self, o, **kwargs):
        return Literal(value=o.items[0], type=DataType.LOGICAL)

    def visit_Attr_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Component_Attr_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

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
        return Import(module=name, symbols=symbols)

    def visit_Include_Stmt(self, o, **kwargs):
        fname = o.items[0].tostr()
        return Import(module=fname, source=kwargs.get('source'))

    def visit_Implicit_Stmt(self, o, **kwargs):
        return Intrinsic(text='IMPLICIT %s' % o.items[0],
                         source=kwargs.get('source', None))

    def visit_Print_Stmt(self, o, **kwargs):
        return Intrinsic(text='PRINT %s' % (', '.join(str(i) for i in o.items)),
                         source=kwargs.get('source', None))

    # TODO: Deal with line-continuation pragmas!
    _re_pragma = re.compile('\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

    def visit_Comment(self, o, **kwargs):
        source = kwargs.get('source', None)
        match_pragma = self._re_pragma.search(o.tostr())
        if match_pragma:
            # Found pragma, generate this instead
            gd = match_pragma.groupdict()
            return Pragma(keyword=gd['keyword'], content=gd['content'], source=source)
        else:
            return Comment(text=o.tostr(), source=source)

    def visit_Entity_Decl(self, o, **kwargs):
        dims = get_child(o, fp.Explicit_Shape_Spec_List)
        dims = get_child(o, fp.Assumed_Shape_Spec_List) if dims is None else dims
        if dims is not None:
            kwargs['dimensions'] = self.visit(dims)

        init = get_child(o, fp.Initialization)
        if init is not None:
            kwargs['initial'] = self.visit(init)

        # We know that this is a declaration, so the ``dimensions``
        # here also define the shape of the variable symbol within the
        # currently cached context.
        kwargs['shape'] = kwargs.get('dimensions', None)

        return self.visit(o.items[0], **kwargs)

    def visit_Component_Decl(self, o, **kwargs):
        dims = get_child(o, fp.Explicit_Shape_Spec_List)
        dims = get_child(o, fp.Assumed_Shape_Spec_List) if dims is None else dims
        dims = get_child(o, fp.Deferred_Shape_Spec_List) if dims is None else dims
        if dims is not None:
            dims = self.visit(dims)
            # We know that this is a declaration, so the ``dimensions``
            # here also define the shape of the variable symbol within the
            # currently cached context.
            kwargs['dimensions'] = dims
            kwargs['shape'] = dims

        return self.visit(o.items[0], **kwargs)

    def visit_Entity_Decl_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in as_tuple(o.items))

    def visit_Component_Decl_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in as_tuple(o.items))

    def visit_Explicit_Shape_Spec(self, o, **kwargs):
        lower = None if o.items[0] is None else self.visit(o.items[0])
        upper = None if o.items[1] is None else self.visit(o.items[1])
        return RangeIndex(lower=lower, upper=upper, step=None)

    def visit_Explicit_Shape_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Assumed_Shape_Spec(self, o, **kwargs):
        lower = None if o.items[0] is None else self.visit(o.items[0])
        upper = None if o.items[1] is None else self.visit(o.items[1])
        return RangeIndex(lower=lower, upper=upper, step=None)

    def visit_Assumed_Shape_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Deferred_Shape_Spec(self, o, **kwargs):
        lower = None if o.items[0] is None else self.visit(o.items[0])
        upper = None if o.items[1] is None else self.visit(o.items[1])
        return RangeIndex(lower=lower, upper=upper, step=None)

    def visit_Deferred_Shape_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Allocation(self, o, **kwargs):
        dimensions = self.visit(o.items[1])
        kwargs['dimensions'] = dimensions
        kwargs['shape'] = dimensions
        return self.visit(o.items[0], **kwargs)

    def visit_Allocate_Shape_Spec(self, o, **kwargs):
        lower = None if o.items[0] is None else self.visit(o.items[0])
        upper = None if o.items[1] is None else self.visit(o.items[1])
        return RangeIndex(lower=lower, upper=upper, step=None)

    def visit_Allocate_Shape_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Allocate_Stmt(self, o, **kwargs):
        allocations = get_child(o, fp.Allocation_List)
        variables = as_tuple(self.visit(a, **kwargs) for a in allocations.items)
        return Allocation(variables=variables)

    def visit_Deallocate_Stmt(self, o, **kwargs):
        deallocations = get_child(o, fp.Allocate_Object_List)
        variables = as_tuple(self.visit(a, **kwargs) for a in deallocations.items)
        return Deallocation(variable=variables)

    def visit_Intrinsic_Type_Spec(self, o, **kwargs):
        dtype = o.items[0]
        kind = get_child(o, fp.Kind_Selector)
        if kind is not None:
            kind = kind.items[1].tostr()
        length = get_child(o, fp.Length_Selector)
        if length is not None:
            length = length.items[1].tostr()
        return dtype, kind, length

    def visit_Intrinsic_Name(self, o, **kwargs):
        return o.tostr()

    def visit_Initialization(self, o, **kwargs):
        return self.visit(o.items[1], **kwargs)

    def visit_Array_Constructor(self, o, **kwargs):
        values = self.visit(o.items[1], **kwargs)
        return LiteralList(values=values)

    def visit_Ac_Value_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Intrinsic_Function_Reference(self, o, **kwargs):
        # Do not recurse here to avoid treating function names as variables
        name = o.items[0].tostr()  # self.visit(o.items[0], **kwargs)
        if name.upper() in ('REAL', 'INT'):
            args = walk(o.items, (fp.Actual_Arg_Spec_List,))[0]
            expr = self.visit(args.items[0])
            if len(args.items) > 1:
                # Do not recurse here to avoid treating kind names as variables
                kind = walk(o.items, (fp.Actual_Arg_Spec,))
                # If kind is not specified as named argument, simply take the second
                # argument and convert it to a string
                kind = kind[0].items[1].tostr() if kind else args.items[1].tostr()
            else:
                kind = None
            return Cast(name, expr, kind=kind)
        else:
            args = self.visit(o.items[1], **kwargs) if o.items[1] else None
            if args:
                kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
                arguments = as_tuple(a for a in args if not isinstance(a, tuple))
            else:
                arguments = None
                kwarguments = None
            return InlineCall(name, parameters=arguments, kw_parameters=kwarguments)

    def visit_Section_Subscript_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Subscript_Triplet(self, o, **kwargs):
        lower = None if o.items[0] is None else self.visit(o.items[0])
        upper = None if o.items[1] is None else self.visit(o.items[1])
        step = None if o.items[2] is None else self.visit(o.items[2])
        return RangeIndex(lower=lower, upper=upper, step=step)

    def visit_Actual_Arg_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Actual_Arg_Spec(self, o, **kwargs):
        key = o.items[0].tostr()
        value = self.visit(o.items[1], **kwargs)
        return (key, value)

    def visit_Data_Ref(self, o, **kwargs):
        pname = o.items[0].tostr().lower()
        v = Variable(name=pname, scope=self.scope.symbols)
        for i in o.items[1:-1]:
            # Careful not to propagate type or dims here
            v = self.visit(i, parent=v, source=kwargs.get('source', None))
        # Attach types and dims to final leaf variable
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Part_Ref(self, o, **kwargs):
        name = o.items[0].tostr()
        args = as_tuple(self.visit(o.items[1])) if o.items[1] else None
        if name.lower() in ['min', 'max', 'exp', 'sqrt', 'abs', 'log',
                            'selected_real_kind', 'allocated', 'present']:
            if args:
                kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
                arguments = as_tuple(a for a in args if not isinstance(a, tuple))
            else:
                arguments = None
                kwarguments = None
            return InlineCall(name, parameters=arguments, kw_parameters=kwarguments)
        else:
            # This is an array access and the arguments define the dimension.
            kwargs['dimensions'] = args
            # Recurse down to visit_Name
            return self.visit(o.items[0], **kwargs)

    def visit_Proc_Component_Ref(self, o, **kwargs):
        '''This is the compound object for accessing procedure components of a variable.'''
        pname = o.items[0].tostr().lower()
        v = Variable(name=pname, scope=self.scope.symbols)
        for i in o.items[1:-1]:
            if i != '%':
                v = self.visit(i, parent=v, source=kwargs.get('source'))
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Array_Section(self, o, **kwargs):
        kwargs['dimensions'] = as_tuple(self.visit(o.items[1]))
        return self.visit(o.items[0], **kwargs)

    def visit_Substring_Range(self, o, **kwargs):
        lower = None if o.items[0] is None else self.visit(o.items[0])
        upper = None if o.items[1] is None else self.visit(o.items[1])
        return RangeIndex(lower=lower, upper=upper)

    def visit_Type_Declaration_Stmt(self, o, **kwargs):
        """
        Declaration statement in the spec of a module/routine. This function is also called
        for declarations of members of a derived type.
        """
        source = kwargs.get('source', None)

        # Super-hacky, this fecking DIMENSION keyword will be my undoing one day!
        dimensions = [self.visit(a, **kwargs)
                      for a in walk(o.items, (fp.Dimension_Component_Attr_Spec,
                                              fp.Dimension_Attr_Spec))]
        if len(dimensions) > 0:
            if isinstance(o, fp.Data_Component_Def_Stmt):
                dimensions = dimensions[0][1]
            else:
                dimensions = dimensions[0]
        else:
            dimensions = None

        # First, pick out parameters, including explicit DIMENSIONs
        attrs = as_tuple(str(self.visit(a)).lower().strip()
                         for a in walk(o.items, (fp.Attr_Spec, fp.Component_Attr_Spec,
                                                 fp.Intent_Attr_Spec)))
        intent = None
        if 'intent(in)' in attrs:
            intent = 'in'
        elif 'intent(inout)' in attrs:
            intent = 'inout'
        elif 'intent(out)' in attrs:
            intent = 'out'

        # Next, figure out the type we're declaring
        dtype = None
        basetype_ast = get_child(o, fp.Intrinsic_Type_Spec)
        if basetype_ast is not None:
            dtype, kind, length = self.visit(basetype_ast)
            dtype = SymbolType(DataType.from_fortran_type(dtype), kind=kind, intent=intent,
                               parameter='parameter' in attrs, optional='optional' in attrs,
                               allocatable='allocatable' in attrs, pointer='pointer' in attrs,
                               shape=dimensions, length=length)

        derived_type_ast = get_child(o, fp.Declaration_Type_Spec)
        if derived_type_ast is not None:
            typename = derived_type_ast.items[1].tostr().lower()
            dtype = self.scope.types.lookup(typename, recursive=True)
            if dtype is not None:
                # Add declaration attributes to the data type from the typedef
                dtype = dtype.clone(intent=intent, allocatable='allocatable' in attrs,
                                    pointer='pointer' in attrs, optional='optional' in attrs,
                                    parameter='parameter' in attrs, target='target' in attrs,
                                    shape=dimensions)
            else:
                # TODO: Insert variable information from stored TypeDef!
                if self.typedefs is not None and typename in self.typedefs:
                    variables = OrderedDict([(v.basename, v)
                                             for v in self.typedefs[typename].variables])
                else:
                    variables = None
                dtype = SymbolType(DataType.DERIVED_TYPE, name=typename, variables=variables,
                                   intent=intent, allocatable='allocatable' in attrs,
                                   pointer='pointer' in attrs, optional='optional' in attrs,
                                   parameter='parameter' in attrs, target='target' in attrs,
                                   shape=dimensions)

        # Now create the actual variables declared in this statement
        # (and provide them with the type and dimension information)
        kwargs['dimensions'] = dimensions
        kwargs['dtype'] = dtype
        variables = flatten(self.visit(o.items[2], **kwargs))

        return Declaration(variables=variables, type=dtype, dimensions=dimensions, source=source)

    def visit_Derived_Type_Def(self, o, **kwargs):
        name = get_child(o, fp.Derived_Type_Stmt).items[1].tostr().lower()
        source = kwargs.get('source', None)
        # Visit comments (and pragmas)
        comments = [self.visit(i, **kwargs) for i in walk(o.content, (fp.Comment,))]
        pragmas = [c for c in comments if isinstance(c, Pragma)]
        comments = [c for c in comments if not isinstance(c, Pragma)]
        # Create the parent type with all the information we have so far
        typedef = TypeDef(name=name, declarations=[], pragmas=pragmas, comments=comments,
                          source=source)
        dtype = SymbolType(DataType.DERIVED_TYPE, name=name, variables=OrderedDict(),
                           source=source)
        # Create declarations and update the parent type with the children from the declarations
        declarations = flatten([self.visit(i, scope=typedef, **kwargs)
                                for i in walk(o.content, (fp.Component_Part,))])
        typedef._update(declarations=declarations, symbols=typedef.symbols)
        for decl in typedef.declarations:
            dtype.variables.update([(v.basename, v) for v in decl.variables])
        # Make type known in its scope's types table
        self.scope.types[name] = dtype
        return typedef

    def visit_Component_Part(self, o, **kwargs):
        return as_tuple(self.visit(a, **kwargs) for a in o.content)

    # Declaration of members of a derived type (i.e., part of the definition of the derived type.
    visit_Data_Component_Def_Stmt = visit_Type_Declaration_Stmt

    def visit_Block_Nonlabel_Do_Construct(self, o, **kwargs):
        # In the banter before the loop, Pragmas are hidden...
        banter = []
        for ch in o.content:
            if isinstance(ch, fp.Nonlabel_Do_Stmt):
                break
            banter += [self.visit(ch, **kwargs)]
        # Extract loop header and get stepping info
        variable, bounds = self.visit(ch, **kwargs)
        # TODO: Will need to handle labeled ones too at some point
        if len(bounds) == 2:
            # Ensure we always have a step size
            bounds += (None,)

        # Extract and process the loop body
        body_nodes = node_sublist(o.content, fp.Nonlabel_Do_Stmt, fp.End_Do_Stmt)
        body = as_tuple(self.visit(node) for node in body_nodes)

        return (*banter, Loop(variable=variable, body=body, bounds=bounds), )

    def visit_Nonlabel_Do_Stmt(self, o, **kwargs):
        variable, bounds = self.visit(o.items[1])
        return variable, bounds

    def visit_If_Construct(self, o, **kwargs):
        # The banter before the loop...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.If_Then_Stmt):
                break
            banter += [self.visit(ch, **kwargs)]
        # Start with the condition that is always there
        conditions = [self.visit(get_child(o, Fortran2003.If_Then_Stmt), **kwargs)]
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
        else_body = as_tuple(self.visit(a) for a in as_tuple(else_ast))
        source = kwargs.get('source', None)
        return (*banter, Conditional(conditions=conditions, bodies=bodies,
                                     else_body=else_body, inline=False, source=source))

    def visit_If_Then_Stmt(self, o, **kwargs):
        return self.visit(o.items[0])

    def visit_If_Stmt(self, o, **kwargs):
        source = kwargs.get('source', None)
        conditions = as_tuple(self.visit(o.items[0]))
        body = as_tuple(self.visit(o.items[1]))
        return Conditional(conditions=conditions, bodies=body, else_body=(),
                           inline=True, source=source)

    def visit_Call_Stmt(self, o, **kwargs):
        name = o.items[0].tostr()
        args = self.visit(o.items[1]) if o.items[1] else None
        source = kwargs.get('source', None)
        if args:
            kwarguments = as_tuple(a for a in args if isinstance(a, tuple))
            arguments = as_tuple(a for a in args if not isinstance(a, tuple))
        else:
            arguments = None
            kwarguments = None
        return CallStatement(name=name, arguments=arguments, kwarguments=kwarguments, source=source)

    def visit_Loop_Control(self, o, **kwargs):
        variable = self.visit(o.items[1][0])
        bounds = as_tuple(self.visit(a) for a in as_tuple(o.items[1][1]))
        return variable, bounds

    def visit_Assignment_Stmt(self, o, **kwargs):
        target = self.visit(o.items[0], **kwargs)
        expr = self.visit(o.items[2], **kwargs)
        source = kwargs.get('source', None)
        return Statement(target=target, expr=expr, source=source)

    def visit_operation(self, op, exprs):
        """
        Construct expressions from individual operations.
        """
        exprs = as_tuple(exprs)
        if op == '*':
            return Product(exprs)
        elif op == '/':
            return Quotient(numerator=exprs[0], denominator=exprs[1])
        elif op == '+':
            return Sum(exprs)
        elif op == '-':
            if len(exprs) > 1:
                # Binary minus
                return Sum((exprs[0], Product((-1, exprs[1]))))
            else:
                # Unary minus
                return Product((-1, exprs[0]))
        elif op == '**':
            return Power(base=exprs[0], exponent=exprs[1])
        elif op.lower() == '.and.':
            return LogicalAnd(exprs)
        elif op.lower() == '.or.':
            return LogicalOr(exprs)
        elif op == '==' or op.lower() == '.eq.':
            return Comparison(exprs[0], '==', exprs[1])
        elif op == '/=' or op.lower() == '.ne.':
            return Comparison(exprs[0], '!=', exprs[1])
        elif op == '>' or op.lower() == '.gt.':
            return Comparison(exprs[0], '>', exprs[1])
        elif op == '<' or op.lower() == '.lt.':
            return Comparison(exprs[0], '<', exprs[1])
        elif op == '>=' or op.lower() == '.ge.':
            return Comparison(exprs[0], '>=', exprs[1])
        elif op == '<=' or op.lower() == '.le.':
            return Comparison(exprs[0], '<=', exprs[1])
        elif op.lower() == '.not.':
            return LogicalNot(exprs[0])
        elif op.lower() == '.eqv.':
            return LogicalOr((LogicalAnd(exprs), LogicalNot(LogicalOr(exprs))))
        elif op.lower() == '.neqv.':
            return LogicalAnd((LogicalNot(LogicalAnd(exprs)), LogicalOr(exprs)))
        elif op == '//':
            return StringConcat(exprs)
        else:
            raise RuntimeError('FParser: Error parsing generic expression')

    def visit_Add_Operand(self, o, **kwargs):
        if len(o.items) > 2:
            exprs = [self.visit(o.items[0], **kwargs)]
            exprs += [self.visit(o.items[2], **kwargs)]
            return self.visit_operation(op=o.items[1], exprs=exprs)
        else:
            exprs = [self.visit(o.items[1], **kwargs)]
            return self.visit_operation(op=o.items[0], exprs=exprs)

    visit_Mult_Operand = visit_Add_Operand
    visit_And_Operand = visit_Add_Operand
    visit_Or_Operand = visit_Add_Operand
    visit_Equiv_Operand = visit_Add_Operand

    def visit_Level_2_Expr(self, o, **kwargs):
        e1 = self.visit(o.items[0], **kwargs)
        e2 = self.visit(o.items[2], **kwargs)
        return self.visit_operation(op=o.items[1], exprs=(e1, e2))

    def visit_Level_2_Unary_Expr(self, o, **kwargs):
        exprs = as_tuple(self.visit(o.items[1], **kwargs))
        return self.visit_operation(op=o.items[0], exprs=exprs)

    visit_Level_3_Expr = visit_Level_2_Expr
    visit_Level_4_Expr = visit_Level_2_Expr
    visit_Level_5_Expr = visit_Level_2_Expr

    def visit_Parenthesis(self, o, **kwargs):
        expression = self.visit(o.items[1], **kwargs)
        if isinstance(expression, Sum):
            expression = ParenthesisedAdd(expression.children)
        if isinstance(expression, Product):
            expression = ParenthesisedMul(expression.children)
        if isinstance(expression, Power):
            expression = ParenthesisedPow(expression.base, expression.exponent)
        return expression

    def visit_Associate_Construct(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        # Search for the ASSOCIATE statement and add all following items as its body
        assoc_index = [isinstance(ch, Scope) for ch in children].index(True)
        children[assoc_index].body = children[assoc_index + 1:]
        return children[:assoc_index + 1]

    def visit_Associate_Stmt(self, o, **kwargs):
        associations = OrderedDict()
        for assoc in o.items[1].items:
            var = self.visit(assoc.items[2], **kwargs)
            if isinstance(var, Array):
                shape = ExpressionDimensionsMapper()(var)
            else:
                shape = None
            dtype = var.type.clone(name=None, parent=None, shape=shape)
            associations[var] = self.visit(assoc.items[0], dtype=dtype, **kwargs)
        return Scope(associations=associations)

    def visit_Format_Stmt(self, o, **kwargs):
        return Intrinsic(text=o.tofortran(), source=kwargs.get('source'))

    def visit_Write_Stmt(self, o, **kwargs):
        return Intrinsic(text=o.tostr(), source=kwargs.get('source'))

    def visit_Cpp_If_Stmt(self, o, **kwargs):
        return Intrinsic(text=o.tostr(), source=kwargs.get('source'))

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
        return Import(module=fname, c_import=True, source=kwargs.get('source'))

    def visit_Goto_Stmt(self, o, **kwargs):
        return Intrinsic(text=o.tostr(), source=kwargs.get('source'))

    visit_Return_Stmt = visit_Goto_Stmt
    visit_Continue_Stmt = visit_Goto_Stmt

    def visit_Read_Stmt(self, o, **kwargs):
        return Intrinsic(text=o.tostr(), source=kwargs.get('source'))

    def visit_Open_Stmt(self, o, **kwargs):
        return Intrinsic(text=o.tostr(), source=kwargs.get('source'))

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
        body = as_tuple(self.visit(ch) for ch in body_ast)
        default = as_tuple(self.visit(ch) for ch in default_ast)
        source = kwargs.get('source', None)
        return (*banter, MaskedStatement(condition, body, default, source=source))

    def visit_Where_Construct_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def visit_Where_Stmt(self, o, **kwargs):
        condition = self.visit(o.items[0], **kwargs)
        body = as_tuple(self.visit(o.items[1], **kwargs))
        default = ()
        source = kwargs.get('source', None)
        return MaskedStatement(condition, body, default, source=source)

    def visit_Case_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.Select_Case_Stmt):
                break
            banter += [self.visit(ch, **kwargs)]
        # The SELECT argument
        expr = self.visit(get_child(o, Fortran2003.Select_Case_Stmt), **kwargs)
        body_ast = node_sublist(o.children, Fortran2003.Select_Case_Stmt,
                                Fortran2003.End_Select_Stmt)
        values = []
        bodies = []
        body = []
        for child in body_ast:
            node = self.visit(child, **kwargs)
            if isinstance(child, Fortran2003.Case_Stmt):
                if values:  # Avoid appending empty body before first Case_Stmt
                    bodies.append(as_tuple(body))
                body = []
                values.append(node)
            else:
                body.append(node)
        bodies.append(as_tuple(body))
        assert len(values) == len(bodies)
        source = kwargs.get('source', None)
        return (*banter, MultiConditional(expr, values, bodies, source=source))

    def visit_Select_Case_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def visit_Case_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def visit_Case_Value_Range(self, o, **kwargs):
        lower = None if o.items[0] is None else self.visit(o.items[0])
        upper = None if o.items[1] is None else self.visit(o.items[1])
        return RangeIndex(lower=lower, upper=upper, step=None)


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

    # Comment out ``@PROCESS`` instructions
    fcode = fcode.replace('@PROCESS', '! @PROCESS')

    reader = FortranStringReader(fcode, ignore_comments=False)
    f2008_parser = ParserFactory().create(std='f2008')

    return f2008_parser(reader)


@timeit(log_level=DEBUG)
def parse_fparser_ast(ast, typedefs=None, scope=None):
    """
    Generate an internal IR from file via the fparser AST.
    """

    # Parse the raw FParser language AST into our internal IR
    ir = FParser2IR(typedefs=typedefs, scope=scope).visit(ast)

    # Perform soime minor sanitation tasks
    ir = inline_comments(ir)
    ir = cluster_comments(ir)
    ir = inline_pragmas(ir)

    return ir
