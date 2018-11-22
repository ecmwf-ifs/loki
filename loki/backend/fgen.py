from sympy.printing import fcode
from sympy.codegen.fnodes import ArrayConstructor
from functools import partial
from collections import Iterable

from loki.visitors import Visitor
from loki.tools import chunks, flatten, as_tuple
from loki.types import BaseType
from loki.ir import Statement
from loki.expression import indexify

__all__ = ['fgen', 'FortranCodegen', 'fexprgen', 'FExprCodegen']


# TODO: Make configurable
fsymgen = partial(fcode, standard=95, source_format='free', contract=False)


class FortranCodegen(Visitor):
    """
    Tree visitor to generate standardized Fortran code from IR.
    """

    def __init__(self, depth=0, linewidth=90, chunking=4, conservative=True):
        super(FortranCodegen, self).__init__()
        self.linewidth = linewidth
        self.conservative = conservative
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
        delim = ', &\n%s & ' % self.indent
        args = list(chunks(list(arguments), chunking))
        return delim.join(', '.join(c) for c in args)

    def visit(self, o):
        if self.conservative and hasattr(o, '_source') and o._source is not None:
            # Re-use original source associated with node
            return o._source.string
        else:
            return super(FortranCodegen, self).visit(o)

    def visit_Node(self, o):
        return self.indent + '! <%s>' % o.__class__.__name__

    def visit_Intrinsic(self, o):
        return o.text

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    visit_list = visit_tuple

    def visit_Module(self, o):
        self._depth += 1
        body = self.visit(o.routines)
        spec = self.visit(o.spec)
        self._depth -= 1
        header = 'MODULE %s \n\n' % o.name
        contains = '\ncontains\n\n'
        footer = '\nEND MODULE %s\n' % o.name
        return header + spec + contains + body + footer

    def visit_Subroutine(self, o):
        # Make sure declarations are re-inserted
        o._externalize()

        ftype = 'FUNCTION' if o.is_function else 'SUBROUTINE'
        arguments = self.segment([a.name for a in o.arguments])
        argument = ' &\n & (%s)' % arguments if len(o.arguments) > 0 else '()'
        bind_c = ' &\n%s & bind(c, name=\'%s\')' % (self.indent, o.bind) if o.bind else ''
        header = '%s %s%s%s\n' % (ftype, o.name, argument, bind_c)
        self._depth += 1
        docstring = '%s\n\n' % self.visit(o.docstring) if o.docstring else ''
        spec = '%s\n\n' % self.visit(o.spec) if o.spec else ''
        body = self.visit(o.body) if o.body else ''
        self._depth -= 1
        footer = '\n%sEND %s %s\n' % (self.indent, ftype, o.name)
        if o.members is not None:
            members = '\n\n'.join(self.visit(s) for s in o.members)
            contains = '\nCONTAINS\n\n'
        else:
            members = ''
            contains = ''
        return self.indent + header + docstring + spec + body + contains + members + footer

    def visit_InterfaceBlock(self, o):
        arguments = self.segment([a.name for a in o.arguments])
        argument = ' &\n & (%s)\n' % arguments if len(o.arguments) > 0 else '\n'
        header = 'INTERFACE\nSUBROUTINE %s%s' % (o.name, argument)
        footer = '\nEND SUBROUTINE %s\nEND INTERFACE\n' % o.name
        imports = self.visit(o.imports)
        declarations = self.visit(o.declarations)
        return header + imports + '\n' + declarations + footer

    def visit_Comment(self, o):
        text = o._source.string if o.text is None else o.text
        return self.indent + text

    def visit_Pragma(self, o):
        if o.content is not None:
            return '!$%s %s' % (o.keyword, o.content)
        else:
            return o._source.string

    def visit_CommentBlock(self, o):
        comments = [self.visit(c) for c in o.comments]
        return '\n'.join(comments)

    def visit_Declaration(self, o):
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        type = self.visit(o.type)
        variables = self.segment([self.visit(v) for v in o.variables])
        if o.dimensions is None:
            dimensions = ''
        else:
            dimensions = ', DIMENSION(%s)' % ','.join(str(d) for d in o.dimensions)
        return self.indent + '%s%s :: %s' % (type, dimensions, variables) + comment

    def visit_DataDeclaration(self, o):
        values = self.segment([str(v) for v in o.values], chunking=8)
        return self.indent + 'DATA %s/%s/' % (o.variable, values)

    def visit_Import(self, o):
        if o.c_import:
            return '#include "%s"' % o.module
        else:
            only = (', ONLY: %s' % self.segment(o.symbols)) if len(o.symbols) > 0 else ''
            return self.indent + 'USE %s%s' % (o.module, only)

    def visit_Interface(self, o):
        self._depth += 1
        body = self.visit(o.body)
        self._depth -= 1
        return self.indent + 'INTERFACE\n%s\n%sEND INTERFACE\n' % (body, self.indent)

    def visit_Loop(self, o):
        pragma = (self.visit(o.pragma) + '\n') if o.pragma else ''
        pragma_post = ('\n' + self.visit(o.pragma_post)) if o.pragma_post else ''
        self._depth += 1
        body = self.visit(o.body)
        self._depth -= 1
        header = '%s=%s, %s%s' % (o.variable, o.bounds[0], o.bounds[1],
                                  ', %s' % o.bounds[2] if o.bounds[2] is not None else '')
        return pragma + self.indent + 'DO %s\n%s\n%sEND DO%s' % (header, body, self.indent, pragma_post)

    def visit_WhileLoop(self, o):
        condition = fexprgen(o.condition, op_spaces=True)
        self._depth += 1
        body = self.visit(o.body)
        self._depth -= 1
        header = 'DO WHILE (%s)\n' % condition
        footer = '\n' + self.indent + 'END DO'
        return self.indent + header + body + footer

    def visit_Conditional(self, o):
        if o.inline:
            assert len(o.conditions) == 1 and len(flatten(o.bodies)) == 1
            indent_depth = self._depth
            self._depth = 0  # Surpress indentation
            body = self.visit(flatten(o.bodies)[0])
            self._depth = indent_depth
            cond = fexprgen(o.conditions[0], op_spaces=True)
            return self.indent + 'IF (%s) %s' % (cond, body)
        else:
            self._depth += 1
            bodies = [self.visit(b) for b in o.bodies]
            else_body = self.visit(o.else_body)
            self._depth -= 1
            headers = ['IF (%s) THEN' % fexprgen(c, op_spaces=True) for c in o.conditions]
            main_branch = ('\n%sELSE' % self.indent).join('%s\n%s' % (h, b) for h, b in zip(headers, bodies))
            else_branch = '\n%sELSE\n%s' % (self.indent, else_body) if o.else_body else ''
            return self.indent + main_branch + '%s\n%sEND IF' % (else_branch, self.indent)

    def visit_MultiConditional(self, o):
        expr = fexprgen(o.expr)
        values = ['DEFAULT' if v is None else '(%s)' % fexprgen(v) for v in o.values]
        self._depth += 1
        bodies = [self.visit(b) for b in o.bodies]
        self._depth -= 1
        header = self.indent + 'SELECT CASE (%s)\n' % expr
        footer = self.indent + 'END SELECT'
        cases = [self.indent + 'CASE %s\n' % v + b for v, b in zip(values, bodies)]
        return header + '\n'.join(cases) + '\n' + footer

    def visit_Statement(self, o):
        target = indexify(o.target)
        stmt = fsymgen(o.expr, assign_to=target)
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        return self.indent + stmt + comment

    def visit_MaskedStatement(self, o):
        condition = fexprgen(o.condition)
        self._depth += 1
        body = self.visit(o.body)
        default = self.visit(o.default)
        self._depth -= 1
        header = self.indent + 'WHERE (%s)\n' % condition
        footer = '\n' + self.indent + 'END WHERE'
        default = '\n%sELSEWHERE\n' % self.indent + default if len(o.default) > 0 else ''
        return header + body + default + footer

    def visit_Section(self, o):
        return self.visit(o.body)

    def visit_Scope(self, o):
        associates = ['%s=>%s' % (v, str(a)) for a, v in o.associations.items()]
        associates = self.segment(associates, chunking=3)
        body = self.visit(o.body)
        return 'ASSOCIATE(%s)\n%s\nEND ASSOCIATE' % (associates, body)

    def visit_Call(self, o):
        if o.kwarguments is not None:
            kwargs = tuple('%s=%s' % (k, v) for k, v in o.kwarguments)
            args = as_tuple(o.arguments) + kwargs
        else:
            args = o.arguments
        if len(args) > self.chunking:
            self._depth += 2
            # TODO: Temporary hack to force cloudsc_driver into the Fortran
            # line limit. The linewidth chaeck should be made smarter to
            # adjust the chunking to the linewidth, like expressions do.
            signature = self.segment([str(a) for a in args], chunking=3)
            self._depth -= 2
        else:
            signature = ', '.join(str(a) for a in args)
        return self.indent + 'CALL %s(%s)' % (o.name, signature)

    def visit_Allocation(self, o):
        source = '' if o.data_source is None else ', source=%s' % self.visit(o.data_source)
        variables = ','.join(str(v) for v in o.variables)
        return self.indent + 'ALLOCATE(%s%s)' % (variables, source)

    def visit_Deallocation(self, o):
        return self.indent + 'DEALLOCATE(%s)' % o.variable

    def visit_Nullify(self, o):
        return self.indent + 'NULLIFY(%s)' % o.variable

    def visit_Expression(self, o):
        # TODO: Expressions are currently purely treated as strings
        return str(o.expr)

    def visit_Scalar(self, o):
        if o.initial is not None:
            # TODO: This can cause re-creation of "Array" symbols,
            # which ultimately ends in disaster...
            if isinstance(o.initial, Iterable):
                value = ArrayConstructor(elements=o.initial)
            else:
                value = o.initial
            return fsymgen(value, assign_to=indexify(o))
        else:
            return fsymgen(o)

    visit_Array = visit_Scalar

    def visit_BaseType(self, o):
        tname = o.name if o.name.upper() in BaseType._base_types else 'TYPE(%s)' % o.name
        return '%s%s%s%s%s%s%s%s%s%s' % (
            tname,
            '(KIND=%s)' % o.kind if o.kind else '',
            ', ALLOCATABLE' if o.allocatable else '',
            ', POINTER' if o.pointer else '',
            ', VALUE' if o.value else '',
            ', OPTIONAL' if o.optional else '',
            ', PARAMETER' if o.parameter else '',
            ', TARGET' if o.target else '',
            ', CONTIGUOUS' if o.contiguous else '',
            ', INTENT(%s)' % o.intent.upper() if o.intent else '',
        )

    def visit_TypeDef(self, o):
        bind_c = ', bind(c) ::' if o.bind_c else ''
        self._depth += 2
        declarations = self.visit(o.declarations)
        self._depth -= 2
        header = self.indent + 'TYPE%s %s\n' % (bind_c, o.name)
        footer = '\n%sEND TYPE %s' % (self.indent, o.name)
        return header + declarations + footer


def fgen(ir, depth=0, chunking=4, conservative=False):
    """
    Generate standardized Fortran code from one or many IR objects/trees.
    """
    return FortranCodegen(depth=depth, chunking=chunking,
                          conservative=conservative).visit(ir)


class FExprCodegen(Visitor):
    """
    Tree visitor to generate a single Fortran assignment expression
    from a tree of sub-expressions.

    :param linewidth: Maximum width to after which to insert linebreaks.
    :param op_spaces: Flag indicating whether to use spaces around operators.
    """

    def __init__(self, linewidth=90, indent='', op_spaces=False,
                 parenthesise=True):
        super(FExprCodegen, self).__init__()
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

    def visit_tuple(self, o, line):
        for i, e in enumerate(o):
            line = self.visit(e, line=line)
            if i < len(o)-1:
                line = self.append(line, ', ')
        return line

    visit_list = visit_tuple

    def visit_Variable(self, o, line):
        if o.ref is not None:
            line = self.visit(o.ref, line=line)
            line = self.append(line, '%')
        line = self.append(line, o.name)
        if o.dimensions is not None and len(o.dimensions) > 0:
            line = self.append(line, '(')
            line = self.visit(o.dimensions, line=line)
            line = self.append(line, ')')
        return line

    def visit_Statement(self, o, line):
        line = self.visit(o.target, line=line)
        line = self.append(line, ' => ' if o.ptr else ' = ')
        line = self.visit(o.expr, line=line)
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
        value = str(o)
        return self.append(line, value)

    def visit_LiteralList(self, o, line):
        line = self.append(line, '(/')
        line = self.visit(o.values, line=line)
        line = self.append(line, '/)')
        return line

    def visit_InlineCall(self, o, line):
        line = self.append(line, '%s(' % o.name)
        if len(o.arguments) > 0:
            line = self.visit(o.arguments[0], line=line)
            for arg in o.arguments[1:]:
                line = self.append(line, ', ')
                line = self.visit(arg, line=line)
            for kw, arg in as_tuple(o.kwarguments):
                line = self.append(line, ', ')
                line = self.append(line, '%s=' % kw)
                line = self.visit(arg, line=line)
        return self.append(line, ')')

    def visit_Cast(self, o, line):
        line = self.append(line, '%s(' % o.type.name)
        line = self.visit(o._expr, line=line)
        line = self.append(line, ', kind=')
        line = self.visit(o.type.kind, line=line)
        return self.append(line, ')')


def fexprgen(expr, linewidth=90, indent='', op_spaces=False, parenthesise=True):
    """
    Generate Fortran expression code from a tree of sub-expressions.
    """
    return FExprCodegen(linewidth=linewidth, indent=indent, op_spaces=op_spaces,
                        parenthesise=parenthesise).visit(expr, line='')
