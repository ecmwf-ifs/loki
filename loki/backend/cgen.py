from loki.tools import chunks, as_tuple
from loki.visitors import Visitor
from loki.types import DataType, DerivedType
from loki.ir import TypeDef, Declaration

__all__ = ['cgen', 'CCodegen', 'cexprgen', 'CExprCodegen']


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

    def visit_Subroutine(self, o):
        # Re-generate variable declarations
        o._externalize(c_backend=True)

        # Generate and dump C struct definitions for derived types
        typedefs = []
        for a in o.arguments:
            if isinstance(a.type, DerivedType):
                decls = as_tuple(Declaration(variables=(v, ), type=v.type)
                                     for _, v in a.type.variables.items())
                typedefs += [TypeDef(name=a.type.name, declarations=decls)]
        c_structs = self.visit(typedefs)
        c_structs += '\n\n'

        # Generate header with argument signature
        aptr = []
        for a in o.arguments:
            # TODO: Oh dear, the pointer derivation is beyond hacky; clean up!
            if a.dimensions is not None and len(a.dimensions) > 0:
                aptr += ['*v_']
            elif isinstance(a.type, DerivedType):
                aptr += ['*']
            else:
                aptr += ['']
        arguments = ['%s %s%s' % (self.visit(a.type), p, a.name)
                     for a, p in zip(o.arguments, aptr)]
        arguments = self.segment(arguments)
        header = 'int %s(%s)\n{' % (o.name, arguments)

        self._depth += 1

        # Generate the array casts for pointer arguments
        casts = '\n%s/* Array casts for pointer arguments */\n' % self.indent
        for a in o.arguments:
            if a.dimensions is not None and len(a.dimensions) > 0:
                dtype = self.visit(a.type)
                # str(d).lower() is a bad hack to ensure caps-alignment
                outer_dims = ''.join('[%s]' % str(d).lower() for d in a.dimensions[:-1])
                casts += self.indent + '%s (*%s)%s = (%s (*)%s) v_%s;\n' % (
                    dtype, a.name, outer_dims, dtype, outer_dims, a.name)

        body = self.visit(o.ir)
        footer = '\n%sreturn 0;\n}' % self.indent
        self._depth -= 1

        # And finally some boilerplate imports...
        imports = '#include <stdio.h>\n'  # For manual debugging
        imports += '#include <stdbool.h>\n'
        imports += '\n\n'

        return imports + c_structs + header + casts + body + footer

    def visit_Section(self, o):
        return self.visit(o.body) + '\n'

    def visit_Declaration(self, o):
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        type = self.visit(o.type)
        vstr = [cexprgen(v) for v in o.variables]
        vptr = [('*' if v.dimensions is not None and len(v.dimensions) > 0 else '')
                for v in o.variables]
        variables = self.segment('%s%s' % (p, v) for v, p in zip(vstr, vptr))
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
        increment = ('++' if o.bounds.step is None else '+=%s' % o.bounds.step)
        header = 'for (%s=%s-1; %s<%s; %s%s)' % (o.variable, o.bounds.lower,
                                            o.variable, o.bounds.upper,
                                            o.variable, increment)
        return self.indent + '%s {\n%s\n%s}\n' % (header, body, self.indent)

    def visit_Statement(self, o):
        stmt = cexprgen(o, linewidth=self.linewidth, indent=self.indent)
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        return self.indent + stmt + comment


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

    def __init__(self, linewidth=90, indent='', op_spaces=False, parenthesise=True):
        super(CExprCodegen, self).__init__()
        self.linewidth = linewidth
        self.indent = indent
        self.op_spaces = op_spaces
        self.parenthesise = parenthesise

        # We ignore outer indents and count from 0
        self._width = 0

    def append(self, line, txt):
        """Insert linebreaks when requested width is hit."""
        if self._width + len(txt) > self.linewidth:
            self._width = len(txt)
            line += '&\n%s& ' % self.indent + txt
        else:
            self._width += len(txt)
            line += txt
        return line

    @classmethod
    def default_retval(cls):
        return ""

    def visit(self, o, line):
        """
        Overriding base `.visit()` to auto-count width and enforce
        line breaks.
        """
        meth = self.lookup_method(o)
        return meth(o, line)

    def visit_str(self, o, line):
        return self.append(line, str(o))

    visit_Expression = visit_str

    def visit_Statement(self, o, line):
        line = self.visit(o.target, line=line)
        line = self.append(line, ' => ' if o.ptr else ' = ')
        line = self.visit(o.expr, line=line)
        line = self.append(line, ';')
        return line

    def visit_Variable(self, o, line):
        if o.ref is not None:
            # TODO: Super-hacky; we always assume pointer-to-struct arguments
            line = self.visit(o.ref, line=line)
            line = self.append(line, '->')
        line = self.append(line, o.name)
        if o.dimensions is not None and len(o.dimensions) > 0:
            for d in o.dimensions:
                line = self.append(line, '[')
                line = self.visit(d, line=line)
                line = self.append(line, ']')
        return line

    def visit_Operation(self, o, line):
        if len(o.ops) == 1 and len(o.operands) == 1:
            # Special case: a unary operator
            if o.parenthesis or self.parenthesise:
                line = self.append(line, '(')
            line = self.append(line, o.ops[0])
            line = self.visit(o.operands[0], line=line)
            if o.parenthesis or self.parenthesise:
                line = self.append(line, ')')
            return line

        if o.parenthesis or self.parenthesise:
            line = self.append(line, '(')
        line = self.visit(o.operands[0], line=line)
        for op, operand in zip(o.ops, o.operands[1:]):
            s_op = (' %s ' % op) if self.op_spaces else str(op)
            line = self.append(line, s_op)
            line = self.visit(operand, line=line)
        if o.parenthesis or self.parenthesise:
            line = self.append(line, ')')
        return line

    def visit_Literal(self, o, line):
        if o.type is not None and o.type is DataType.BOOL:
            bmap = {'.true.': 'true', '.false.': 'false'}
            value = bmap[o.value.lower()]
        else:
            value = o.value
        return self.append(line, value)


def cexprgen(expr, linewidth=90, indent='', op_spaces=False):
    """
    Generate C expression code from a tree of sub-expressions.
    """
    return CExprCodegen(linewidth=linewidth, indent=indent,
                        op_spaces=op_spaces).visit(expr, line='')
