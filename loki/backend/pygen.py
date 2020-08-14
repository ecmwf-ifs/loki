from itertools import zip_longest
from pymbolic.mapper.stringifier import PREC_NONE

from loki.expression import LokiStringifyMapper, symbol_types as sym, retrieve_expressions
from loki.visitors import Stringifier
from loki.types import DataType


__all__ = ['pygen', 'PyCodegen', 'PyCodeMapper']

def dace_type(_type):
    if _type.dtype == DataType.LOGICAL:
        return 'dace.bool'
    if _type.dtype == DataType.INTEGER:
        return 'dace.int32'
    if _type.dtype == DataType.REAL:
        if str(_type.kind) in ('real32',):
            return 'dace.float32'
        return 'dace.float64'
    raise ValueError(str(_type))


class PyCodeMapper(LokiStringifyMapper):
    # pylint: disable=abstract-method, unused-argument

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return 'True' if bool(expr.value) else 'False'

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        return str(expr.value)

    map_int_literal = map_float_literal

    def map_scalar(self, expr, enclosing_prec, *args, **kwargs):
        return expr.name

    def map_array(self, expr, enclosing_prec, *args, **kwargs):
        dims = ''
        if expr.dimensions:
            dims = self.rec(expr.dimensions, PREC_NONE, *args, **kwargs)
        return self.format('%s%s', expr.name, dims)

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        index_str = ''
        for index in expr.index_tuple:
            d = self.format(self.rec(index, PREC_NONE, *args, **kwargs))
            if d:
                index_str += self.format('[%s]', d)
        return index_str

    def map_string_concat(self, expr, enclosing_prec, *args, **kwargs):
        return ' + '.join(self.rec(c, enclosing_prec, *args, **kwargs) for c in expr.children)


class PyCodegen(Stringifier):
    """
    Tree visitor to generate standardized Python code from IR
    (optionally using the reduced and annotated syntax understood by DaCe).
    """

    # pylint: disable=no-self-use

    def __init__(self, with_dace=False, depth=0, indent='  ', linewidth=100):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont='\n{}  '.format, symgen=PyCodeMapper())
        self.with_dace = with_dace

    # Handler for outer objects

    def visit_SourceFile(self, o, **kwargs):
        """
        Format as
          ...modules...
          ...subroutines...
        """
        modules = self.visit_all(o.modules, **kwargs)
        subroutines = self.visit_all(o.subroutines, **kwargs)
        return self.join_lines(*modules, *subroutines)

    def visit_Module(self, o, **kwargs):
        raise NotImplementedError()

    def visit_Subroutine(self, o, **kwargs):
        """
        Format as:
            ...imports...
            def <name>(<args>):
                ...spec without imports and only declarations with initial values...
                ...body...
        """
        # Some boilerplate imports...
        standard_imports = []
        if self.with_dace:
            standard_imports += ['dace']
        header = [self.format_line('import ', name) for name in standard_imports]

        # ...and imports from the spec
        # TODO

        # Generate header with argument signature
        if self.with_dace:
            symbols = set()
            for arg in o.arguments:
                if isinstance(arg, sym.Array):
                    shape_vars = retrieve_expressions(arg.shape, lambda e: isinstance(e, sym.Scalar))
                    symbols |= set(v.name.lower() for v in shape_vars)
            arguments = ['{}: {}'.format(arg.name.lower(), self.visit(arg.type, **kwargs))
                         for arg in o.arguments if arg.name.lower() not in symbols]
            header += [self.format_line('{name} = dace.symbol("{name}")'.format(name=s))
                       for s in symbols]
            header += [self.format_line('@dace.program')]
        else:
            arguments = [arg.name.lower() for arg in o.arguments]
        header += [self.format_line('def ', o.name.lower(), '(', self.join_items(arguments), '):')]

        # ...and generate the spec without imports and only declarations with initial value
        self.depth += 1
        body = [self.visit(o.spec, **kwargs)]

        # Fill the body and close everything off
        body += [self.visit(o.body, **kwargs)]
        self.depth -= 1

        return self.join_lines(*header, *body)

    # Handler for IR nodes

    def visit_Intrinsic(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Format intrinsic nodes.
        """
        return self.format_line(str(o.text).lstrip())

    def visit_Comment(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Format comments.
        """
        text = o.text or o.source.string
        text = str(text).lstrip().replace('!', '#', 1)
        return self.format_line(text, no_wrap=True)

    def visit_CommentBlock(self, o, **kwargs):
        """
        Format comment blocks.
        """
        comments = self.visit_all(o.comments, **kwargs)
        return self.join_lines(*comments)

    def visit_Declaration(self, o, **kwargs):
        """
        Format declaration as
          <name> = <initial>
        and skip any without an initial value
        """
        decls = []
        if o.comment:
            decls += [self.visit(o.comment, **kwargs)]
        decls += [self.format_line(v.name.lower(), ' = ', self.visit(v.initial, **kwargs))
                  for v in o.variables if v.initial is not None]
        return self.join_lines(*decls)

    def visit_Import(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Skip imports
        """
        return None

    def visit_Loop(self, o, **kwargs):
        """
        Format loop with explicit range as
          for <var> in range(<start>, <end> + <incr>, <incr>):
            ...body...
        """
        var = self.visit(o.variable, **kwargs)
        start = self.visit(o.bounds.start, **kwargs)
        end = self.visit(o.bounds.stop, **kwargs)
        if self.with_dace:
            if o.bounds.step:
                incr = self.visit(o.bounds.step, **kwargs)
                cntrl = 'dace.map[{start}:{end}+{inc}:{inc}]'.format(start=start, end=end, inc=incr)
            else:
                cntrl = 'dace.map[{start}:{end}+1]'.format(start=start, end=end)
        else:
            if o.bounds.step:
                incr = self.visit(o.bounds.step, **kwargs)
                cntrl = 'range({start}, {end} + {inc}, {inc})'.format(start=start, end=end, inc=incr)
            else:
                cntrl = 'range({start}, {end} + 1)'.format(start=start, end=end)
        header = self.format_line('for ', var, ' in ', cntrl, ':')
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body)

    def visit_WhileLoop(self, o, **kwargs):
        """
        Format loop as:
          while <condition>:
            ...body...
        """
        if o.condition is not None:
            condition = self.visit(o.condition, **kwargs)
        else:
            condition = 'True'
        header = self.format_line('while ', condition, ':')
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body)

    def visit_Conditional(self, o, **kwargs):
        """
        Format conditional as
        if <condition>:
          ...body...
        [elif <condition>:]
          [...body...]
        [else:]
          [...body...]
        """
        conditions = self.visit_all(o.conditions, **kwargs)
        conditions = [self.format_line(kw, ' ', cond, ':')
                      for kw, cond in zip_longest(['if'], conditions, fillvalue='elif')]
        if o.else_body:
            conditions.append(self.format_line('else:'))
        self.depth += 1
        bodies = self.visit_all(*o.bodies, o.else_body, **kwargs)
        self.depth -= 1
        branches = [item for branch in zip(conditions, bodies) for item in branch]
        return self.join_lines(*branches)

    def visit_Statement(self, o, **kwargs):
        """
        Format statement as
          <target> = <expr> [<comment>]
        """
        target = self.visit(o.target, **kwargs)
        expr = self.visit(o.expr, **kwargs)
        comment = None
        if o.comment:
            comment = '  {}'.format(self.visit(o.comment, **kwargs))
        return self.format_line(target, ' = ', expr, comment=comment)

    def visit_Section(self, o, **kwargs):
        """
        Format the section's body.
        """
        return self.visit(o.body, **kwargs)

    def visit_CallStatement(self, o, **kwargs):
        """
        Format call statement as
          <name>(<args>)
        """
        args = self.visit_all(o.arguments, **kwargs)
        kw_args = ['{}={}'.format(kw, self.visit(arg, **kwargs)) for kw, arg in o.kwarguments]
        return self.format_line(o.name, '(', self.join_items(args + kw_args), ')')

    def visit_SymbolType(self, o, **kwargs):
        if not self.with_dace:
            return None
        dtype = dace_type(o)
        shape = ''
        if o.shape is not None:
            for dim in o.shape:
                d = self.visit(dim, **kwargs)
                if d:
                    shape += '[{}]'.format(d)
        return '{}{}'.format(dtype, shape)


def pygen(ir, with_dace=False):
    """
    Generate standardized Python 3 code from one or many IR objects/trees
    (optionally using the reduced and annotated syntax understood by DaCe).
    """
    return PyCodegen(with_dace=with_dace).visit(ir)
