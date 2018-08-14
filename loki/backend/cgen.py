from loki.visitors import Visitor

__all__ = ['cgen', 'CCodegen', 'cexprgen', 'CExprCodegen']


class CCodegen(Visitor):
    """
    Tree visitor to generate standardized C code from IR.
    """

    def __init__(self, depth=0, linewidth=90, chunking=4):
        super(CCodegen, self).__init__()
        self.linewidth = linewidth
        self.chunking = chunking
        self._depth = depth

    @classmethod
    def default_retval(cls):
        return ""

    @property
    def indent(self):
        return '  ' * self._depth

    def visit_Node(self, o):
        return self.indent + '// <%s>' % o.__class__.__name__

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    visit_list = visit_tuple

    def visit_Subroutine(self, o):
        body = self.visit(o.ir)

        header = 'int %s(%s)\n{\n' % (o.name, o._argnames)
        footer = '\nreturn 0;\n}'
        return header + body + footer


def cgen(ir):
    """
    Generate standardized C code from one or many IR objects/trees.
    """
    return CCodegen().visit(ir)


class CExprCodegen(Visitor):
    """
    Tree visitor to generate a single C assignment expression from a
    tree of sub-expressions.

    :param linewidth: Maximum width to after which to insert linebreaks.
    :param op_spaces: Flag indicating whether to use spaces around operators.
    """

    def __init__(self, linewidth=90, indent='', op_spaces=False):
        super(CExprCodegen, self).__init__()
        self.linewidth = linewidth
        self.indent = indent
        self.op_spaces = op_spaces


def cexprgen(expr, linewidth=90, indent='', op_spaces=False):
    """
    Generate C expression code from a tree of sub-expressions.
    """
    return CExprCodegen(linewidth=linewidth, indent=indent,
                        op_spaces=op_spaces).visit(expr, line='')
