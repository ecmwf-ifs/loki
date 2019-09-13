from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader
from sympy import Add, Mul, Pow, Equality, Unequality

from loki.visitors import GenericVisitor
from loki.frontend.source import Source
from loki.frontend.util import inline_comments, cluster_comments, inline_pragmas
from loki.ir import Comment, Declaration, Statement
from loki.types import DataType, BaseType
from loki.expression import Variable, Literal, InlineCall
from loki.expression.operations import ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow
from loki.logging import info, error, DEBUG
from loki.tools import timeit, as_tuple, flatten

__all__ = ['FParser2IR', 'parse_fparser_file', 'parse_fparser_ast']


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
        # return o.tostr()
        dtype = kwargs.get('dtype', None)
        initial = kwargs.get('initial', None)
        vname = o.tostr()
        return self.Variable(name=vname, dtype=dtype, initial=initial)

    def visit_Int_Literal_Constant(self, o, **kwargs):
        return Literal(value=o.items[0], kind=o.items[1])

    def visit_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def visit_Specification_Part(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        return list(children)

    def visit_Comment(self, o, **kwargs):
        return Comment(text=o.tostr())

    def visit_Entity_Decl(self, o, **kwargs):
        dtype = kwargs.get('dtype', None)
        initial = None if o.items[3] is None else self.visit(o.items[3])
        v = self.visit(o.items[0], dtype=dtype, initial=initial)
        return v

    def visit_Intrinsic_Type_Spec(self, o, **kwargs):
        dtype = o.items[0]
        kind = o.items[1].items[1].tostr() if o.items[1] is not None else None
        return dtype, kind

    def visit_Initialization(self, o, **kwargs):
        return self.visit(o.items[1])

    def visit_Section_Subscript_List(self, o, **kwargs):
        return as_tuple(self.visit(i) for i in o.items)

    def visit_Part_Ref(self, o, **kwargs):
        name = o.items[0].tostr()
        args = self.visit(o.items[1])
        return InlineCall(name=name, arguments=args)

    def visit_Type_Declaration_Stmt(self, o, **kwargs):
        dtype, kind = self.visit(o.items[0])
        attrs = tuple(self.visit(a) for a in as_tuple(o.items[1]))
        attrs = tuple(str(a).lower().strip() for a in attrs)
        intent = None
        if 'intent(in)' in attrs:
            intent = 'in'
        elif 'intent(inout)' in attrs:
            intent = 'inout'
        elif 'intent(out)' in attrs:
            intent = 'out'
        base_type = BaseType(dtype, kind=kind, intent=intent,
                             parameter='parameter' in attrs)
        variables = tuple(self.visit(v, dtype=base_type) for v in as_tuple(o.items[2]))
        return Declaration(variables=flatten(variables), type=base_type)

    def visit_Assignment_Stmt(self, o, **kwargs):
        target = self.visit(o.items[0])
        expr = self.visit(o.items[2])
        return Statement(target=target, expr=expr)

    def generic_expr(self, op, exprs):
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
        else:
            raise RuntimeError('FParser: Error parsing generic expression')

    def visit_Add_Operand(self, o, **kwargs):
        e1 = self.visit(o.items[0])
        e2 = self.visit(o.items[2])
        return self.generic_expr(op=o.items[1], exprs=[e1, e2])

    def visit_Mult_Operand(self, o, **kwargs):
        e1 = self.visit(o.items[0])
        e2 = self.visit(o.items[2])
        return self.generic_expr(op=o.items[1], exprs=[e1, e2])

    def visit_Level_2_Expr(self, o, **kwargs):
        e1 = self.visit(o.items[0])
        e2 = self.visit(o.items[2])
        return self.generic_expr(op=o.items[1], exprs=[e1, e2])

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
