from fparser.two.parser import ParserFactory
from fparser.two.utils import get_child, walk_ast
from fparser.two.Fortran2003 import *
from fparser.common.readfortran import FortranFileReader
from sympy import Add, Mul, Pow, Equality, Unequality, Not, And, Or

from loki.visitors import GenericVisitor
from loki.frontend.source import Source
from loki.frontend.util import inline_comments, cluster_comments, inline_pragmas
from loki.ir import Comment, Declaration, Statement, Loop
from loki.types import DataType, BaseType
from loki.expression import Variable, Literal, InlineCall, Array, RangeIndex
from loki.expression.operations import ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow
from loki.logging import info, error, DEBUG
from loki.tools import timeit, as_tuple, flatten

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

    def __init__(self, cache=None):
        super(FParser2IR, self).__init__()

        # Use provided symbol cache for variable generation
        self._cache = cache

    def Variable(self, *args, **kwargs):
        """
        Instantiate cached variable symbols from local symbol cache.
        """
        if self._cache is None:
            return Variable(*args, **kwargs)
        else:
            return self._cache.Variable(*args, **kwargs)

    def visit(self, o, **kwargs):
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        source = kwargs.pop('source', None)
        if o.item is not None:
            file = o.item.reader.file.name
            source = Source(lines=o.item.span, file=file)
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
        # declarations, where nonde of the metadata is available
        # at this low level!
        vname = o.tostr()
        dimensions = kwargs.get('dimensions', None)
        return self.Variable(name=vname, dimensions=dimensions)

    def visit_Int_Literal_Constant(self, o, **kwargs):
        return Literal(value=int(o.items[0]), kind=o.items[1])

    def visit_Real_Literal_Constant(self, o, **kwargs):
        return Literal(value=float(o.items[0]), kind=o.items[1])

    def visit_Logical_Literal_Constant(self, o, **kwargs):
        return Literal(value=o.items[0], type=DataType.BOOL)

    def visit_Attr_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i) for i in o.items)
        
    def visit_Dimension_Attr_Spec(self, o, **kwargs):
        return self.visit(o.items[1])

    def visit_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def visit_Specification_Part(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        return list(children)

    def visit_Comment(self, o, **kwargs):
        return Comment(text=o.tostr())

    def visit_Entity_Decl(self, o, **kwargs):
        # Don't recurse here, as the node is a ``Name`` and will
        # generate a pre-cached ``Variable`` object otherwise!
        vname = o.items[0].tostr()
        dtype = kwargs.get('dtype', None)

        dims = get_child(o, Explicit_Shape_Spec_List)
        dimensions = self.visit(dims) if dims is not None else kwargs.get('dimensions', None)

        init = get_child(o, Initialization)
        initial = self.visit(init) if init is not None else None

        return self.Variable(name=vname, dtype=dtype, dimensions=dimensions, initial=initial)

    def visit_Entity_Decl_List(self, o, **kwargs):
        return as_tuple(self.visit(i, **kwargs) for i in as_tuple(o.items))

    def visit_Explicit_Shape_Spec(self, o, **kwargs):
        return self.visit(o.items[1])

    def visit_Explicit_Shape_Spec_List(self, o, **kwargs):
        return as_tuple(self.visit(i) for i in o.items)

    def visit_Intrinsic_Type_Spec(self, o, **kwargs):
        dtype = o.items[0]
        kind = o.items[1].items[1].tostr() if o.items[1] is not None else None
        return dtype, kind

    def visit_Intrinsic_Name(self, o, **kwargs):
        return o.tostr()

    def visit_Initialization(self, o, **kwargs):
        return self.visit(o.items[1])

    def visit_Intrinsic_Function_Reference(self, o, **kwargs):
        name = self.visit(o.items[0])
        arguments = self.visit(o.items[1])
        return InlineCall(name=name, arguments=arguments)

    def visit_Section_Subscript_List(self, o, **kwargs):
        return as_tuple(self.visit(i) for i in o.items)

    visit_Actual_Arg_Spec_List = visit_Section_Subscript_List

    def visit_Part_Ref(self, o, **kwargs):
        name = o.items[0].tostr()
        args = as_tuple(self.visit(o.items[1]))
        if name.lower() in ['min', 'max', 'exp', 'sqrt', 'abs', 'log',
                             'selected_real_kind', 'allocated', 'present']:
            kwarguments = as_tuple(a for a in args if isinstance(a, tuple))
            arguments = as_tuple(a for a in args if not isinstance(a, tuple))
            return InlineCall(name=name, arguments=arguments, kwarguments=kwarguments)
        else:
            return self.Variable(name=name, dimensions=args)

    def visit_Array_Section(self, o, **kwargs):
        dimensions = as_tuple(self.visit(o.items[1]))
        return self.visit(o.items[0], dimensions=dimensions)

    def visit_Substring_Range(self, o, **kwargs):
        return RangeIndex(lower=o.items[0], upper=o.items[1])

    def visit_Type_Declaration_Stmt(self, o, **kwargs):
        basetype_node = get_child(o, Intrinsic_Type_Spec)
        dtype, kind = self.visit(basetype_node)
        attrs = as_tuple(self.visit(o.items[1])) if o.items[1] is not None else ()
        # Super-hacky, this fecking DIMENSION keyword will be my undoing one day!
        dimensions = [a for a in attrs if isinstance(a, tuple)]
        dimensions = None if len(dimensions) == 0 else dimensions[0]
        attrs = tuple(str(a).lower().strip() for a in attrs if isinstance(a, str))
        intent = None
        if 'intent(in)' in attrs:
            intent = 'in'
        elif 'intent(inout)' in attrs:
            intent = 'inout'
        elif 'intent(out)' in attrs:
            intent = 'out'
        base_type = BaseType(dtype, kind=kind, intent=intent,
                             parameter='parameter' in attrs)
        variables = self.visit(o.items[2], dtype=base_type, dimensions=dimensions)
        return Declaration(variables=flatten(variables), type=base_type, dimensions=dimensions)

    def visit_Block_Nonlabel_Do_Construct(self, o, **kwargs):
        # Extract loop header and get stepping info
        # TODO: Will need to handle labeled ones too at some point
        dostmt = get_child(o, Nonlabel_Do_Stmt)
        variable, (lower, upper) = self.visit(dostmt)
        step = None  # TOOD: Need to handle this at some point!
        bounds = lower, upper, step

        # Extract and process the loop body
        body_nodes = node_sublist(o.content, Nonlabel_Do_Stmt, End_Do_Stmt)
        body = as_tuple(self.visit(node) for node in body_nodes)

        return Loop(variable=variable, body=body, bounds=bounds)

    def visit_Nonlabel_Do_Stmt(self, o, **kwargs):
        variable, bounds = self.visit(o.items[1])
        return variable, bounds

    def visit_Loop_Control(self, o, **kwargs):
        variable = self.visit(o.items[1][0])
        bounds = as_tuple(self.visit(a) for a in as_tuple(o.items[1][1]))
        return variable, bounds

    def visit_Assignment_Stmt(self, o, **kwargs):
        target = self.visit(o.items[0])
        expr = self.visit(o.items[2])
        return Statement(target=target, expr=expr)

    def visit_operation(self, op, exprs):
        """
        Construct expressions from individual operations, suppressing SymPy simplifications.
        """

        def booleanize(expr):
            """
            Super-hacky helper function to force boolean array when needed
            """
            if isinstance(expr, Array) and not expr.is_Boolean:
                return expr.clone(type=BaseType(name='logical'), cache=self._cache)
            else:
                return expr

        if op == '*':
            return Mul(exprs[0], exprs[1], evaluate=False)
        elif op == '/':
            return Mul(exprs[0], Pow(exprs[1], -1, evaluate=False), evaluate=False)
        elif op == '+':
            return Add(exprs[0], exprs[1], evaluate=False)
        elif op == '-':
            return Add(exprs[0], Mul(-1, exprs[1], evaluate=False), evaluate=False)
        elif op == '**':
            return Pow(exprs[0], exprs[1], evaluate=False)
        elif op.lower() == '.and.':
            e1 = booleanize(exprs[0])
            e2 = booleanize(exprs[1])
            return And(e1, e2, evaluate=False)
        elif op.lower() == '.or.':
            e1 = booleanize(exprs[0])
            e2 = booleanize(exprs[1])
            return Or(e1, e2, evaluate=False)
        elif op == '==' or op.lower() == '.eq.':
            return Equality(exprs[0], exprs[1], evaluate=False)
        elif op == '/=' or op.lower() == '.ne.':
            return Unequality(exprs[0], exprs[1], evaluate=False)
        elif op.lower() == '.not.':
            e1 = booleanize(exprs[0])
            return Not(e1, evaluate=False)
        else:
            raise RuntimeError('FParser: Error parsing generic expression')

    def visit_Add_Operand(self, o, **kwargs):
        if len(o.items) > 2:
            exprs = [self.visit(o.items[0])]
            exprs += [self.visit(o.items[2])]
            return self.visit_operation(op=o.items[1], exprs=exprs)
        else:
            exprs = [self.visit(o.items[1])]
            return self.visit_operation(op=o.items[0], exprs=exprs)

    visit_Mult_Operand = visit_Add_Operand
    visit_And_Operand = visit_Add_Operand
    visit_Or_Operand = visit_Add_Operand
    visit_Equiv_Operand = visit_Add_Operand

    def visit_Level_2_Expr(self, o, **kwargs):
        e1 = self.visit(o.items[0])
        e2 = self.visit(o.items[2])
        return self.visit_operation(op=o.items[1], exprs=[e1, e2])

    visit_Level_4_Expr = visit_Level_2_Expr

    def visit_Parenthesis(self, o, **kwargs):
        expression = self.visit(o.items[1])
        if expression.is_Add:
            expression = ParenthesisedAdd(*expression.args, evaluate=False)
        if expression.is_Mul:
            expression = ParenthesisedMul(*expression.args, evaluate=False)
        if expression.is_Pow:
            expression = ParenthesisedPow(*expression.args, evaluate=False)
        return expression

    

@timeit(log_level=DEBUG)
def parse_fparser_file(filename):
    """
    Generate an internal IR from file via the fparser AST.
    """
    reader = FortranFileReader(filename, ignore_comments=False)
    # raw_source = reader.source.read()
    f2008_parser = ParserFactory().create(std='f2008')
    return f2008_parser(reader)#, raw_source


@timeit(log_level=DEBUG)
def parse_fparser_ast(ast, cache=None):
    """
    Generate an internal IR from file via the fparser AST.
    """

    # Parse the raw FParser language AST into our internal IR
    ir = FParser2IR(cache=cache).visit(ast)

    # Perform soime minor sanitation tasks
    ir = inline_comments(ir)
    ir = cluster_comments(ir)
    ir = inline_pragmas(ir)

    return ir
