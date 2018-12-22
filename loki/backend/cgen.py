from sympy.printing.ccode import C99CodePrinter
from functools import partial
from sympy.codegen.ast import (real, float32, float64)

from loki.tools import chunks, as_tuple
from loki.visitors import Visitor, FindNodes
from loki.types import DataType, DerivedType
from loki.ir import TypeDef, Declaration, Import
from loki.expression import Operation, Literal, indexify, FindVariables

__all__ = ['cgen', 'CCodegen', 'csymgen']


class CExpressionPrinter(C99CodePrinter):
    """
    Custom CodePrinter extension for forcing our specific flavour
    of C expression printing.
    """

    def _print_Indexed(self, expr):
        """
        Print an Indexed as a C-like multidimensional array.

        Examples
        --------
        V[x,y,z] -> V[x][y][z]
        """
        output = self._print(expr.base.label) \
                 + ''.join(['[' + self._print(x) + ']' for x in expr.indices])

        return output

    def _print_Scalar(self, expr):
        if expr.parent is None:
            return super(CExpressionPrinter, self)._print_Symbol(expr)
        else:
            # TODO: Words cannot express my disguust here...
            return '%s->%s' % (expr.parent, expr.name.split('%')[1])


def csymgen(expr, assign_to=None, **kwargs):
    settings = {
        'contract': False,
    }
    settings.update(**kwargs)
    return CExpressionPrinter(settings).doprint(expr, assign_to)


class CCodegen(Visitor):
    """
    Tree visitor to generate standardized C code from IR.
    """

    def __init__(self, depth=0, linewidth=90, chunking=6):
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

    def segment(self, arguments, chunking=None):
        chunking = chunking or self.chunking
        delim = ',\n%s  ' % self.indent
        args = list(chunks(list(arguments), chunking))
        return delim.join(', '.join(c) for c in args)

    def visit_Node(self, o):
        return self.indent + '// <%s>' % o.__class__.__name__

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    visit_list = visit_tuple

    def visit_Module(self, o):
        # Assuming this will be put in header files...
        spec = self.visit(o.spec)
        routines = self.visit(o.routines)
        return spec + '\n\n' + routines

    def visit_Subroutine(self, o):
        # Re-generate variable declarations
        o._externalize(c_backend=True)

        # Generate header with argument signature
        aptr = []
        for a in o.arguments:
            # TODO: Oh dear, the pointer derivation is beyond hacky; clean up!
            if a.is_Array > 0:
                aptr += ['* restrict v_']
            elif isinstance(a.type, DerivedType):
                aptr += ['*']
            elif a.type.pointer:
                aptr += ['*']
            else:
                aptr += ['']
        arguments = ['%s %s%s' % (self.visit(a.type), p, a.name.lower())
                     for a, p in zip(o.arguments, aptr)]
        arguments = self.segment(arguments)
        header = 'int %s(%s)\n{\n' % (o.name, arguments)

        self._depth += 1

        # Generate the array casts for pointer arguments
        casts = '%s/* Array casts for pointer arguments */\n' % self.indent
        for a in o.arguments:
            if a.is_Array > 0:
                dtype = self.visit(a.type)
                # str(d).lower() is a bad hack to ensure caps-alignment
                outer_dims = ''.join('[%s]' % str(d).lower() for d in a.dimensions[1:])
                casts += self.indent + '%s (*%s)%s = (%s (*)%s) v_%s;\n' % (
                    dtype, a.name.lower(), outer_dims, dtype, outer_dims, a.name.lower())

        spec = self.visit(o.spec)
        body = self.visit(o.body)
        footer = '\n%sreturn 0;\n}' % self.indent
        self._depth -= 1

        # And finally some boilerplate imports...
        imports = '#include <stdio.h>\n'  # For manual debugging
        imports += '#include <stdbool.h>\n'
        imports += '#include <float.h>\n'
        imports += '#include <math.h>\n'
        imports += self.visit(FindNodes(Import).visit(o.spec))

        return imports + '\n\n' + header + casts + spec + '\n' +  body + footer

    def visit_Section(self, o):
        return self.visit(o.body) + '\n'

    def visit_Import(self, o):
        return ('#include "%s"' % o.module) if o.c_import else ''

    def visit_Declaration(self, o):
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        type = self.visit(o.type)
        vstr = [csymgen(indexify(v)) for v in o.variables]
        vptr = [('*' if v.type.pointer or v.type.allocatable else '') for v in o.variables]
        vinit = ['' if v.initial is None else (' = %s' % csymgen(v.initial)) for v in o.variables]
        variables = self.segment('%s%s%s' % (p, v, i) for v, p, i in zip(vstr, vptr, vinit))
        return self.indent + '%s %s;' % (type, variables) + comment

    def visit_BaseType(self, o):
        return o.dtype.ctype

    def visit_DerivedType(self, o):
        return 'struct %s' % o.name

    def visit_TypeDef(self, o):
        self._depth += 1
        decls = self.visit(o.declarations)
        self._depth += 1
        return 'struct %s {\n%s\n} ;' % (o.name, decls)

    def visit_Comment(self, o):
        text = o._source.string if o.text is None else o.text
        return self.indent + text.replace('!', '//')

    def visit_CommentBlock(self, o):
        comments = [self.visit(c) for c in o.comments]
        return '\n'.join(comments)

    def visit_Loop(self, o):
        self._depth += 1
        body = self.visit(o.body)
        self._depth -= 1
        increment = ('++' if o.bounds[2] is None else '+=%s' % o.bounds[2])
        lvar = csymgen(o.variable)
        lower = csymgen(o.bounds[0])
        upper = csymgen(o.bounds[1])
        criteria = '<=' if o.bounds[2] is None or eval(str(o.bounds[2])) > 0 else '>='
        header = 'for (%s=%s; %s%s%s; %s%s)' % (lvar, lower, lvar, criteria, upper, lvar, increment)
        return self.indent + '%s {\n%s\n%s}\n' % (header, body, self.indent)

    def visit_Statement(self, o):
        target = indexify(o.target)
        expr = indexify(o.expr)

        type_aliases = {}
        if o.target.type.dtype == DataType.FLOAT32:
            type_aliases[real] = float32

        # Collect pointer variables for dereferencing
        # TODO: Cache sets of symbols on statements
        dereference = [v for v in FindVariables().visit(o)
                       if hasattr(v, 'type') and v.type.pointer]

        stmt = csymgen(expr, assign_to=target,
                       type_aliases=type_aliases,
                       dereference=dereference)
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        return self.indent + stmt + comment

    def visit_Conditional(self, o):
        self._depth += 1
        bodies = [self.visit(b) for b in o.bodies]
        else_body = self.visit(o.else_body)
        self._depth -= 1
        if len(bodies) > 1:
            raise NotImplementedError('Multi-body cnoditionals not yet supported')

        cond = csymgen(o.conditions[0])
        main_branch = 'if (%s)\n%s{\n%s\n' % (cond, self.indent, bodies[0])
        else_branch = '%s} else {\n%s\n' % (self.indent, else_body) if o.else_body else ''
        return self.indent + main_branch + else_branch + '%s}\n' % self.indent

    def visit_Intrinsic(self, o):
        return o.text

def cgen(ir):
    """
    Generate standardized C code from one or many IR objects/trees.
    """
    return CCodegen().visit(ir)
