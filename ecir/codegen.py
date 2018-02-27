from ecir.visitors import Visitor

__all__ = ['fgen', 'FortranCodegen']


class FortranCodegen(Visitor):
    """
    Tree visitor to generate standardized Fortran code from IR.
    """

    def __init__(self, conservative=True, depth=0):
        super(FortranCodegen, self).__init__()
        self.conservative = conservative
        self._depth = 0

    @classmethod
    def default_retval(cls):
        return ""

    @property
    def indent(self):
        return '  ' * self._depth

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

    def visit_Comment(self, o):
        return self.indent + o._source

    def visit_CommentBlock(self, o):
        comments = [self.visit(c) for c in o.comments]
        return '\n'.join(comments)

    def visit_Declaration(self, o):
        type = self.visit(o.variables[0].type)
        return self.indent + '%s :: %s' % (type, ', '.join(str(v) for v in o.variables))

    def visit_Import(self, o):
        return 'USE %s, ONLY: %s' % (o.module, ', '.join(o.symbols))

    def visit_Variable(self, o):
        dims = '(%s)' % ','.join([str(d) for d in o.dimensions]) if len(o.dimensions) > 0 else ''
        return '%s(%s)' % (o.name, dims)


    def visit_Type(self, o):
        return '%s%s%s%s%s%s' % (o.name, '(KIND=%s)' % o.kind if o.kind else '',
                                 ', INTENT(%s)' % o.intent.upper() if o.intent else '',
                                 ', ALLOCATE' if o.allocatable else '',
                                 ', POINTER' if o.pointer else '',
                                 ', OPTIONAL' if o.optional else '')

def fgen(ir, depth=0, conservative=True):
    """
    Generate standardized Fortran code from one or many IR objects/trees.
    """
    return FortranCodegen(depth=depth, conservative=conservative).visit(ir)
