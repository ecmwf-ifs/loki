from loki.expression import symbols as sym, retrieve_expressions
from loki.backend import PyCodegen
from loki.types import BasicType

__all__ = ['dacegen', 'DaceCodegen']


def dace_type(_type):
    if _type.dtype == BasicType.LOGICAL:
        return 'dace.bool'
    if _type.dtype == BasicType.INTEGER:
        return 'dace.int32'
    if _type.dtype == BasicType.REAL:
        if str(_type.kind) in ('real32',):
            return 'dace.float32'
        return 'dace.float64'
    raise ValueError(str(_type))


class DaceCodegen(PyCodegen):
    """
    Tree visitor that extends `PyCodegen` with Dace-specific language variations.
    """
    # pylint: disable=no-self-use

    def __init__(self, depth=0, indent='  ', linewidth=100):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth)

    # Handler for outer objects

    def visit_Subroutine(self, o, **kwargs):
        """
        Format as:
            ...imports...
            def <name>(<args>):
                ...spec without imports and only declarations with initial values...
                ...body...
        """
        # Some boilerplate imports...
        standard_imports = ['dace', 'numpy as np']
        header = [self.format_line('import ', name) for name in standard_imports]

        # ...and imports from the spec
        # TODO

        # Generate header with argument signature
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
        header += [self.format_line('def ', o.name.lower(), '(', self.join_items(arguments), '):')]

        # ...and generate the spec without imports and only declarations with initial value
        self.depth += 1
        body = [self.visit(o.spec, **kwargs)]

        # Fill the body and close everything off
        body += [self.visit(o.body, **kwargs)]
        self.depth -= 1

        return self.join_lines(*header, *body)

    # Handler for IR nodes

    def visit_Loop(self, o, **kwargs):
        """
        Format loop with explicit range as
          for <var> in range(<start>, <end> + <incr>, <incr>):
            ...body...
        """
        is_dataflow_loop = (o.pragma is not None and o.pragma.keyword == 'loki' and
                            o.pragma.content.startswith('dataflow'))
        if not is_dataflow_loop:
            return super().visit_Loop( o, **kwargs)

        var = self.visit(o.variable, **kwargs)
        start = self.visit(o.bounds.start, **kwargs)
        end = self.visit(o.bounds.stop, **kwargs)
        if o.bounds.step:
            incr = self.visit(o.bounds.step, **kwargs)
            cntrl = 'dace.map[{start}:{end}+{inc}:{inc}]'.format(start=start, end=end, inc=incr)
        else:
            cntrl = 'dace.map[{start}:{end}+1]'.format(start=start, end=end)
        header = self.format_line('for ', var, ' in ', cntrl, ':')
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body)

    def visit_SymbolType(self, o, **kwargs):
        dtype = dace_type(o)
        shape = ''
        if o.shape is not None:
            dims = [self.visit(dim, **kwargs) for dim in o.shape]
            shape = '[{}]'.format(', '.join(d for d in dims if d))
        return '{}{}'.format(dtype, shape)



def dacegen(ir):
    """
    Generate standard Python 3 code with Dace-specializations (and Numpy) from one
    or many IR objects/trees.
    """
    return DaceCodegen().visit(ir)
