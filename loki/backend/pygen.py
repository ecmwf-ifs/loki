from itertools import zip_longest
from pymbolic.mapper.stringifier import PREC_NONE, PREC_CALL

from loki.expression import symbols as sym, LokiStringifyMapper
from loki.visitors import Stringifier
from loki.types import BasicType, SymbolType


__all__ = ['pygen', 'PyCodegen', 'PyCodeMapper']


def numpy_type(_type):
    if _type.shape is not None:
        return 'np.ndarray'
    if _type.dtype == BasicType.LOGICAL:
        return 'np.bool'
    if _type.dtype == BasicType.INTEGER:
        return 'np.int32'
    if _type.dtype == BasicType.REAL:
        if str(_type.kind) in ('real32',):
            return 'np.float32'
        return 'np.float64'
    raise ValueError(str(_type))


class PyCodeMapper(LokiStringifyMapper):
    """
    Generate Python representation of expression trees using numpy syntax.
    """
    # pylint: disable=abstract-method, unused-argument

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return 'True' if bool(expr.value) else 'False'

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        return str(expr.value)

    map_int_literal = map_float_literal

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        _type = SymbolType(BasicType.from_fortran_type(expr.name), kind=expr.kind)
        expression = self.parenthesize_if_needed(
            self.join_rec('', expr.parameters, PREC_NONE, *args, **kwargs),
            PREC_CALL, PREC_NONE)
        return self.parenthesize_if_needed(
            self.format('%s(%s)', numpy_type(_type), expression), enclosing_prec, PREC_CALL)

    def map_scalar(self, expr, enclosing_prec, *args, **kwargs):
        return expr.name

    def map_array(self, expr, enclosing_prec, *args, **kwargs):
        dims = ''
        if expr.dimensions:
            dims = self.rec(expr.dimensions, PREC_NONE, *args, **kwargs)
        return self.format('%s%s', expr.name, dims)

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        dims = [self.format(self.rec(d, PREC_NONE, *args, **kwargs)) for d in expr.index_tuple]
        dims = [d for d in dims if d]
        if not dims:
            index_str = ''
        else:
            index_str = '[{}]'.format(', '.join(dims))
        return index_str

    def map_string_concat(self, expr, enclosing_prec, *args, **kwargs):
        return ' + '.join(self.rec(c, enclosing_prec, *args, **kwargs) for c in expr.children)


class PyCodegen(Stringifier):
    """
    Tree visitor to generate standard Python code (with Numpy) from IR.
    """

    # pylint: disable=no-self-use

    def __init__(self, depth=0, indent='  ', linewidth=100):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont='\n{}  '.format, symgen=PyCodeMapper())

    # Handler for outer objects

    def visit_Sourcefile(self, o, **kwargs):
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
        standard_imports = ['numpy as np']
        header = [self.format_line('import ', name) for name in standard_imports]

        # ...and imports from the spec
        # TODO

        # Generate header with argument signature
        # Note: we skip scalar out arguments and add a return statement for those below
        inout_args = [arg for arg in o.arguments
                      if isinstance(arg, sym.Scalar) and arg.type.intent.lower() == 'inout']
        out_args = [arg for arg in o.arguments
                    if isinstance(arg, sym.Scalar) and arg.type.intent.lower() == 'out']
        arguments = ['{}: {}'.format(arg.name.lower(), self.visit(arg.type, **kwargs))
                     for arg in o.arguments if arg not in out_args]
        header += [self.format_line('def ', o.name.lower(), '(', self.join_items(arguments), '):')]

        # ...and generate the spec without imports and only declarations for variables that
        # either are local arrays or are assigned an initial value
        self.depth += 1
        body = [self.visit(o.spec, **kwargs)]

        # Fill the body
        body += [self.visit(o.body, **kwargs)]

        # Add return statement for scalar out arguments and close everything off
        ret_args = [arg for arg in o.arguments if arg in inout_args + out_args]
        body += [self.format_line('return ', self.join_items(self.visit_all(ret_args, **kwargs)))]
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
        and skip any arguments or scalars without an initial value
        """
        decls = []
        if o.comment:
            decls += [self.visit(o.comment, **kwargs)]
        local_arrays = [v for v in o.variables if isinstance(v, sym.Array) and not v.type.intent]
        decls += [self.format_line(v.name.lower(), ' = np.ndarray(order="F", shape=(',
                                   self.join_items(self.visit_all(v.shape, **kwargs)), ',))')
                  for v in local_arrays]
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

    def visit_Assignment(self, o, **kwargs):
        """
        Format statement as
          <target> = <expr> [<comment>]
        """
        lhs = self.visit(o.lhs, **kwargs)
        rhs = self.visit(o.rhs, **kwargs)
        comment = None
        if o.comment:
            comment = '  {}'.format(self.visit(o.comment, **kwargs))
        return self.format_line(lhs, ' = ', rhs, comment=comment)

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
        return numpy_type(o)


def pygen(ir):
    """
    Generate standard Python 3 code (that uses Numpy) from one or many IR objects/trees.
    """
    return PyCodegen().visit(ir)
