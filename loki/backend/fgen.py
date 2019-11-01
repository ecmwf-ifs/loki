from textwrap import wrap
from pymbolic.primitives import Expression, Sum, is_zero, Product
from pymbolic.mapper.stringifier import (PREC_UNARY, PREC_LOGICAL_AND, PREC_LOGICAL_OR,
                                         PREC_COMPARISON, PREC_SUM, PREC_PRODUCT)

from loki.visitors import Visitor
from loki.tools import chunks, flatten, as_tuple, is_iterable
from loki.types import BaseType, DataType
from loki.expression import LokiStringifyMapper
from loki.expression.symbol_types import StringLiteral, Scalar

__all__ = ['fgen', 'FortranCodegen', 'FCodeMapper']


class FCodeMapper(LokiStringifyMapper):
    """
    A :class:`StringifyMapper`-derived visitor for Pymbolic expression trees that converts an
    expression to a string adhering to the Fortran standard.
    """

    COMPARISON_OP_TO_FORTRAN = {
        "==": r"==",
        "!=": r"/=",
        "<=": r"<=",
        ">=": r">=",
        "<": r"<",
        ">": r">",
    }

    def __init__(self, constant_mapper=repr):
        super(FCodeMapper, self).__init__(constant_mapper)

    def map_logic_literal(self, expr, *args, **kwargs):
        return '.true.' if expr.value else '.false.'

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        result = '%s_%s' % (self(expr.value), expr._kind) if expr._kind else self(expr.value)
        if not (result.startswith("(") and result.endswith(")")) \
                and ("-" in result or "+" in result) and (enclosing_prec > PREC_SUM):
            return self.parenthesize(result)
        else:
            return result

    map_int_literal = map_float_literal

    def map_logical_not(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            ".not." + self.rec(expr.child, PREC_UNARY, *args, **kwargs),
            enclosing_prec, PREC_UNARY)

    def map_logical_and(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.join_rec(" .and. ", expr.children, PREC_LOGICAL_AND, *args, **kwargs),
            enclosing_prec, PREC_LOGICAL_AND)

    def map_logical_or(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.join_rec(" .or. ", expr.children, PREC_LOGICAL_OR, *args, **kwargs),
            enclosing_prec, PREC_LOGICAL_OR)

    def map_comparison(self, expr, enclosing_prec, *args, **kwargs):
        """
        This translates the C-style notation for comparison operators used internally in Pymbolic
        to the corresponding Fortran comparison operators.
        """
        return self.parenthesize_if_needed(
            self.format("%s %s %s", self.rec(expr.left, PREC_COMPARISON, *args, **kwargs),
                        self.COMPARISON_OP_TO_FORTRAN[expr.operator],
                        self.rec(expr.right, PREC_COMPARISON, *args, **kwargs)),
            enclosing_prec, PREC_COMPARISON)

    def map_sum(self, expr, enclosing_prec, *args, **kwargs):
        """
        Since substraction and unary minus are mapped to multiplication with (-1), we are here
        looking for such cases and determine the matching operator for the output.
        """
        def get_op_prec_expr(expr):
            if isinstance(expr, Product) and len(expr.children) and is_zero(expr.children[0]+1):
                if len(expr.children) == 2:
                    # only the minus sign and the other child
                    return '-', PREC_PRODUCT, expr.children[1]
                else:
                    return '-', PREC_PRODUCT, Product(expr.children[1:])
            else:
                return '+', PREC_SUM, expr

        terms = []
        for ch in expr.children:
            op, prec, expr = get_op_prec_expr(ch)
            terms += [op, self.rec(expr, prec, *args, **kwargs)] 

        # Remove leading '+'
        if terms[0] == '-':
            terms[1] = '%s%s' % (terms[0], terms[1])
        terms = terms[1:]

        return self.parenthesize_if_needed(self.join(' ', terms), enclosing_prec, PREC_SUM)

    def map_literal_list(self, expr, *args, **kwargs):
        return '(/' + ','.join(str(c) for c in expr.elements) + '/)'

    def map_foreign(self, expr, *args, **kwargs):
        try:
            return super().map_foreign(expr, *args, **kwargs)
        except ValueError:
            return '! Not supported: %s\n' % str(expr)


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
        self._fsymgen = FCodeMapper()

    @classmethod
    def default_retval(cls):
        return ""

    @property
    def indent(self):
        return '  ' * self._depth

    @property
    def fsymgen(self):
        return self._fsymgen

    def chunk(self, arg):
        return ('&\n%s &' % self.indent).join(wrap(arg, width=60, drop_whitespace=False,
                                                   break_long_words=False))

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
        return str(o.text)

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
        condition = self.fsymgen(o.condition)
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
            cond = self.fsymgen(o.conditions[0])
            return self.indent + 'IF (%s) %s' % (cond, body)
        else:
            self._depth += 1
            bodies = [self.visit(b) for b in o.bodies]
            else_body = self.visit(o.else_body)
            self._depth -= 1
            headers = ['IF (%s) THEN' % self.fsymgen(c) for c in o.conditions]
            main_branch = ('\n%sELSE' % self.indent).join('%s\n%s' % (h, b) for h, b in zip(headers, bodies))
            else_branch = '\n%sELSE\n%s' % (self.indent, else_body) if o.else_body else ''
            return self.indent + main_branch + '%s\n%sEND IF' % (else_branch, self.indent)

    def visit_MultiConditional(self, o):
        expr = self.fsymgen(o.expr)
        values = ['DEFAULT' if v is None else (self.fsymgen(v) if isinstance(v, tuple)
                  else '(%s)' % self.fsymgen(v)) for v in o.values]
        self._depth += 1
        bodies = [self.visit(b) for b in o.bodies]
        self._depth -= 1
        header = self.indent + 'SELECT CASE (%s)\n' % expr
        footer = self.indent + 'END SELECT'
        cases = [self.indent + 'CASE %s\n' % v + b for v, b in zip(values, bodies)]
        return header + '\n'.join(cases) + '\n' + footer

    def visit_Statement(self, o):
        stmt = '%s = %s' % (self.fsymgen(o.target), self.fsymgen(o.expr))
        if o.ptr:
            # Manually force pointer assignment notation
            # ... Hack me baby, one more time ...
            stmt = stmt.replace(' = ', ' => ')
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        return self.indent + self.chunk(stmt) + comment

    def visit_MaskedStatement(self, o):
        condition = self.fsymgen(o.condition)
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
            kwargs = tuple(String('%s=%s' % (k, v)) for k, v in o.kwarguments)
            args = as_tuple(o.arguments) + kwargs
        else:
            args = o.arguments
        if len(args) > self.chunking:
            self._depth += 2
            # TODO: Temporary hack to force cloudsc_driver into the Fortran
            # line limit. The linewidth chaeck should be made smarter to
            # adjust the chunking to the linewidth, like expressions do.
            signature = self.segment([self.fsymgen(a) if isinstance(a, Expression)
                                      else a for a in args], chunking=3)
            self._depth -= 2
        else:
            signature = ', '.join(str(a) for a in args)
        return self.indent + 'CALL %s(%s)' % (o.name, signature)

    def visit_Allocation(self, o):
        source = '' if o.data_source is None else ', source=%s' % self.visit(o.data_source)
        variables = ','.join(v.name if isinstance(v, str) else str(v)
                             for v in o.variables)
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
            if is_iterable(o.initial):
                value = ArrayConstructor(elements=o.initial)
            else:
                value = o.initial
            # TODO: This is super-hacky! We need to find
            # a rigorous way to do this, but various corner
            # cases around opinter assignments break the
            # shape verification in sympy.
            return '%s = %s' % (o, self.fsymgen(value))
        else:
            return self.fsymgen(o)

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
