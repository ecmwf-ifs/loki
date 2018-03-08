from ecir.visitors import Visitor
from ecir.tools import chunks

__all__ = ['fgen', 'FortranCodegen']


class FortranCodegen(Visitor):
    """
    Tree visitor to generate standardized Fortran code from IR.
    """

    def __init__(self, depth=0, chunking=6, conservative=True):
        super(FortranCodegen, self).__init__()
        self.conservative = conservative
        self.chunking = chunking
        self._depth = 0

    @classmethod
    def default_retval(cls):
        return ""

    @property
    def indent(self):
        return '  ' * self._depth

    def segment(self, arguments, chunking=None):
        chunking = chunking or self.chunking
        delim = ', &\n%s & ' % self.indent
        args = list(chunks(list(arguments), chunking))
        return delim.join(', '.join(c) for c in args)

    def visit(self, o):
        if self.conservative and hasattr(o, '_source') and o._source is not None:
            # Re-use original source associated with node
            return o._source
        else:
            return super(FortranCodegen, self).visit(o)

    def visit_Node(self, o):
        return self.indent + '! <%s>' % o.__class__.__name__

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    visit_list = visit_tuple

    def visit_Subroutine(self, o):
        arguments = self.segment([a.name for a in o.arguments])
        body = self.visit(o.ir)
        header = 'SUBROUTINE %s &\n & (%s)\n' % (o.name, arguments)
        return header + body + '\nEND SUBROUTINE %s\n' % o.name

    def visit_Comment(self, o):
        return self.indent + o._source

    def visit_Pragma(self, o):
        return self.indent + o._source

    def visit_CommentBlock(self, o):
        comments = [self.visit(c) for c in o.comments]
        return '\n'.join(comments)

    def visit_Declaration(self, o):
        type = self.visit(o.variables[0].type)
        return self.indent + '%s :: %s' % (type, ', '.join(str(v) for v in o.variables))

    def visit_Import(self, o):
        return 'USE %s, ONLY: %s' % (o.module, ', '.join(o.symbols))

    def visit_Loop(self, o):
        pragma = (self.visit(o.pragma) + '\n') if o.pragma else ''
        self._depth += 1
        body = self.visit(o.body)
        self._depth -= 1
        header = '%s=%s, %s' % (o.variable, o.bounds[0], o.bounds[1])
        return pragma + self.indent + 'DO %s\n%s\n%sEND DO' % (header, body, self.indent)

    def visit_Conditional(self, o):
        self._depth += 1
        bodies = [self.visit(b) for b in o.bodies]
        else_body = self.visit(o.else_body)
        self._depth -= 1
        headers = ['IF (%s) THEN' % str(c) for c in  o.conditions]
        main_branch = ('\n%sELSE'%self.indent).join('%s\n%s' % (h, b) for h, b in zip(headers, bodies))
        else_branch = '\n%sELSE\n%s' % (self.indent, else_body) if o.else_body else ''
        return self.indent + main_branch + '%s\n%sEND IF' % (else_branch, self.indent)

    def visit_Statement(self, o):
        target = self.visit(o.target)
        expr = self.visit(o.expr)
        return self.indent + '%s = %s' % (target, expr)

    def visit_Scope(self, o):
        associates = ['%s=>%s' % (v, a) for a, v in o.associations.items()]
        associates = self.segment(associates, chunking=3)
        body = self.visit(o.body)
        return 'ASSOCIATE(%s)\n%s\nEND ASSOCIATE' % (associates, body)

    def visit_Call(self, o):
        if len(o.arguments) > 6:
            self._depth += 2
            signature = self.segment(self.visit(a) for a in o.arguments)
            self._depth -= 2
        else:
            signature = ', '.join(str(a) for a in o.arguments)
        return self.indent + 'CALL %s(%s)' % (o.name, signature)

    def visit_Expression(self, o):
        # TODO: Expressions are currently purely treated as strings
        return str(o.expr)

    def visit_Variable(self, o):
        dims = '(%s)' % ','.join([str(d) for d in o.dimensions]) if len(o.dimensions) > 0 else ''
        return '%s%s' % (o.name, dims)

    def visit_Type(self, o):
        return '%s%s%s%s%s%s' % (o.name, '(KIND=%s)' % o.kind if o.kind else '',
                                 ', INTENT(%s)' % o.intent.upper() if o.intent else '',
                                 ', ALLOCATE' if o.allocatable else '',
                                 ', POINTER' if o.pointer else '',
                                 ', OPTIONAL' if o.optional else '')


def fgen(ir, depth=0, chunking=6, conservative=True):
    """
    Generate standardized Fortran code from one or many IR objects/trees.
    """
    return FortranCodegen(depth=depth, chunking=chunking,
                          conservative=conservative).visit(ir)
