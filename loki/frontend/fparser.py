from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader

from loki.visitors import GenericVisitor
from loki.frontend.source import Source
from loki.frontend.util import inline_comments, cluster_comments, inline_pragmas
from loki.ir import Comment, Declaration
from loki.types import DataType, BaseType
from loki.expression import Variable, Literal, InlineCall
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
            # from IPython import embed; embed()
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
        return o.tostr()

    def visit_Int_Literal_Constant(self, o, **kwargs):
        return Literal(value=o.items[0], kind=o.items[1])

    visit_Attr_Spec = visit_Name

    def visit_Specification_Part(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        return children

    def visit_Comment(self, o, **kwargs):
        return Comment(text=o.tostr())

    def visit_Entity_Decl(self, o, **kwargs):
        dtype = kwargs.get('dtype', None)
        vname = o.items[0].tostr()
        initial = None if o.items[3] is None else self.visit(o.items[3])
        return self.Variable(name=vname, type=dtype, initial=initial)

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
        attrs = tuple(str(a).lower() for a in attrs)
        base_type = BaseType(dtype, kind=kind, parameter='parameter' in attrs)
        variables = tuple(self.visit(v, dtype=base_type) for v in as_tuple(o.items[2]))
        return Declaration(variables=flatten(variables), type=base_type)
    

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
