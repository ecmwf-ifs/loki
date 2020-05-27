from textwrap import wrap
from pymbolic.primitives import is_zero, Product
from pymbolic.mapper.stringifier import (PREC_UNARY, PREC_LOGICAL_AND, PREC_LOGICAL_OR,
                                         PREC_COMPARISON, PREC_SUM, PREC_PRODUCT, PREC_NONE)

from loki.visitors import Visitor
from loki.tools import chunks, flatten, as_tuple
from loki.expression import LokiStringifyMapper
from loki.types import DataType

__all__ = ['fgen', 'fexprgen', 'FortranCodegen', 'FCodeMapper']


class FCodeMapper(LokiStringifyMapper):
    """
    A :class:`StringifyMapper`-derived visitor for Pymbolic expression trees that converts an
    expression to a string adhering to the Fortran standard.
    """
    # pylint: disable=abstract-method

    COMPARISON_OP_TO_FORTRAN = {
        "==": r"==",
        "!=": r"/=",
        "<=": r"<=",
        ">=": r">=",
        "<": r"<",
        ">": r">",
    }

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return '.true.' if expr.value else '.false.'

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            return '%s_%s' % (str(expr.value), str(expr.kind))

        result = str(expr.value)
        if not (result.startswith("(") and result.endswith(")")) \
                and ("-" in result or "+" in result) and (enclosing_prec > PREC_SUM):
            return self.parenthesize(result)
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
            if isinstance(expr, Product) and expr.children and is_zero(expr.children[0]+1):
                if len(expr.children) == 2:
                    # only the minus sign and the other child
                    return '-', PREC_PRODUCT, expr.children[1]
                return '-', PREC_PRODUCT, Product(expr.children[1:])
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

    def map_literal_list(self, expr, enclosing_prec, *args, **kwargs):
        return '(/' + ','.join(str(c) for c in expr.elements) + '/)'

    def map_foreign(self, expr, *args, **kwargs):
        try:
            return super().map_foreign(expr, *args, **kwargs)
        except ValueError:
            return '! Not supported: %s\n' % str(expr)

    def map_loop_range(self, expr, enclosing_prec, *args, **kwargs):
        children = [self.rec(child, PREC_NONE, *args, **kwargs) if child is not None else ''
                    for child in expr.children]
        if expr.step is None:
            children = children[:-1]
        return self.parenthesize_if_needed(self.join(',', children), enclosing_prec, PREC_NONE)


class FortranCodegen(Visitor):
    """
    Tree visitor to generate standardized Fortran code from IR.
    """
    # pylint: disable=no-self-use, unused-argument

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

    def visit(self, o, *args, **kwargs):
        if self.conservative and hasattr(o, '_source') and o._source is not None:
            # Re-use original source associated with node
            return o._source.string

        if hasattr(o, '_source') and o._source is not None and o._source.label:
            label = '{} '.format(o._source.label)
        else:
            label = ''
        return '%s%s' % (label, super(FortranCodegen, self).visit(o, **kwargs))

    def visit_Expression(self, o, **kwargs):
        return self.fsymgen(o)

    def visit_str(self, o, **kwargs):
        return o

    def visit_Node(self, o, **kwargs):
        return self.indent + '! <%s>' % o.__class__.__name__

    def visit_Intrinsic(self, o, **kwargs):
        return str(o.text)

    def visit_tuple(self, o, **kwargs):
        return '\n'.join([self.visit(i, **kwargs) for i in o])

    visit_list = visit_tuple

    def visit_Module(self, o, **kwargs):
        self._depth += 1
        body = self.visit(o.routines, **kwargs)
        spec = self.visit(o.spec, **kwargs)
        self._depth -= 1
        header = 'MODULE %s \n\n' % o.name
        contains = '\ncontains\n\n'
        footer = '\nEND MODULE %s\n' % o.name
        return header + spec + contains + body + footer

    def visit_Subroutine(self, o, **kwargs):
        ftype = 'FUNCTION' if o.is_function else 'SUBROUTINE'
        arguments = self.segment([a.name for a in o.arguments])
        argument = ' &\n & (%s)' % arguments if len(o.arguments) > 0 else '()'
        bind_c = ' &\n%s & bind(c, name=\'%s\')' % (self.indent, o.bind) if o.bind else ''
        header = '%s %s%s%s\n' % (ftype, o.name, argument, bind_c)
        self._depth += 1
        docstring = '%s\n\n' % self.visit(o.docstring, **kwargs) if o.docstring else ''
        spec = '%s\n\n' % self.visit(o.spec, **kwargs) if o.spec else ''
        body = self.visit(o.body, **kwargs) if o.body else ''
        self._depth -= 1
        footer = '\n%sEND %s %s\n' % (self.indent, ftype, o.name)
        if o.members is not None:
            members = '\n\n'.join(self.visit(s, **kwargs) for s in o.members)
            contains = '\nCONTAINS\n\n'
        else:
            members = ''
            contains = ''
        return self.indent + header + docstring + spec + body + contains + members + footer

    def visit_Comment(self, o, **kwargs):
        text = o._source.string if o.text is None else o.text
        return self.indent + text

    def visit_Pragma(self, o, **kwargs):
        if o.content is not None:
            return '!$%s %s' % (o.keyword, o.content)
        return o._source.string

    def visit_CommentBlock(self, o, **kwargs):
        comments = [self.visit(c, **kwargs) for c in o.comments]
        return '\n'.join(comments)

    def visit_Declaration(self, o, **kwargs):
        assert len(o.variables) > 0
        types = [v.type for v in o.variables]
        # Ensure all variable types are equal, except for shape and dimension
        # TODO: Should extend to deeper recursion of `variables` if
        # the symbol has a known derived type
        ignore = ['shape', 'dimensions', 'variables', 'source']
        assert all(t.compare(types[0], ignore=ignore) for t in types)
        dtype = self.visit(types[0])

        comment = '  %s' % self.visit(o.comment, **kwargs) if o.comment is not None else ''
        if o.dimensions is None:
            dimensions = ''
        else:
            dimensions = ', DIMENSION(%s)' % ','.join(str(d) for d in o.dimensions)
        variables = []
        for v in o.variables:
            # This is a bit dubious, but necessary, as we otherwise pick up
            # array dimensions from the internal representation of the variable.
            stmt = self.visit(v, **kwargs) if o.dimensions is None else v.basename
            if v.initial is not None:
                stmt += ' = %s' % self.visit(v.initial, **kwargs)
            # Hack the pointer assignment (very ugly):
            if v.type.pointer:
                stmt = stmt.replace(' = ', ' => ')
            variables += [stmt]
        variables = self.segment(variables)
        return self.indent + '%s%s :: %s' % (dtype, dimensions, variables) + comment

    def visit_DataDeclaration(self, o, **kwargs):
        values = self.segment([str(v) for v in o.values], chunking=8)
        return self.indent + 'DATA %s/%s/' % (o.variable, values)

    def visit_Import(self, o, **kwargs):
        if o.c_import:
            return '#include "%s"' % o.module
        if o.f_include:
            return 'include "%s"' % o.module
        only = (', ONLY: %s' % self.segment(o.symbols)) if len(o.symbols) > 0 else ''
        return self.indent + 'USE %s%s' % (o.module, only)

    def visit_Interface(self, o, **kwargs):
        self._depth += 1
        spec = ' %s' % o.spec if o.spec else ''
        header = 'INTERFACE%s\n' % spec
        body = ('\n%s' % self.indent).join(self.visit(ch, **kwargs) for ch in o.body)
        self._depth -= 1
        return self.indent + header + body + '\n%sEND INTERFACE\n' % self.indent

    def visit_Loop(self, o, **kwargs):
        pragma = (self.visit(o.pragma, **kwargs) + '\n') if o.pragma else ''
        pragma_post = ('\n' + self.visit(o.pragma_post, **kwargs)) if o.pragma_post else ''
        self._depth += 1
        body = self.visit(o.body, **kwargs)
        self._depth -= 1
        header = ' %s=%s' % (self.visit(o.variable, **kwargs), self.visit(o.bounds, **kwargs))
        return pragma + self.indent + \
            'DO%s\n%s\n%sEND DO%s' % (header, body, self.indent, pragma_post)

    def visit_WhileLoop(self, o, **kwargs):
        condition = self.visit(o.condition, **kwargs)
        self._depth += 1
        body = self.visit(o.body, **kwargs)
        self._depth -= 1
        header = 'DO WHILE (%s)\n' % condition
        footer = '\n' + self.indent + 'END DO'
        return self.indent + header + body + footer

    def visit_Conditional(self, o, **kwargs):
        if o.inline:
            # No indentation and only a single body node
            assert len(o.conditions) == 1 and len(flatten(o.bodies)) == 1
            self._depth, indent_depth = 0, self._depth  # Suppress indentation
            body = self.visit(flatten(o.bodies)[0], **kwargs)
            self._depth = indent_depth
            cond = self.visit(o.conditions[0], **kwargs)
            return self.indent + 'IF (%s) %s' % (cond, body)

        self._depth += 1
        bodies = [self.visit(b, **kwargs) for b in o.bodies]
        else_body = self.visit(o.else_body, **kwargs)
        self._depth -= 1
        headers = ['IF (%s) THEN' % self.visit(c, **kwargs) for c in o.conditions]
        main_branch = ('\n%sELSE' % self.indent).join(
            '%s\n%s' % (h, b) for h, b in zip(headers, bodies))
        else_branch = '\n%sELSE\n%s' % (self.indent, else_body) if o.else_body else ''
        return self.indent + main_branch + '%s\n%sEND IF' % (else_branch, self.indent)

    def visit_MultiConditional(self, o, **kwargs):
        expr = self.visit(o.expr, **kwargs)
        values = ['(%s)' % ', '.join(self.visit(e, **kwargs) for e in as_tuple(v))
                  for v in o.values]
        if o.else_body:
            values += ['DEFAULT']
        self._depth += 1
        bodies = [self.visit(b, **kwargs) for b in o.bodies]
        if o.else_body:
            bodies += [self.visit(o.else_body, **kwargs)]
        self._depth -= 1
        assert len(values) == len(bodies)
        header = self.indent + 'SELECT CASE (%s)\n' % expr
        footer = self.indent + 'END SELECT'
        cases = [self.indent + 'CASE %s\n' % v + b for v, b in zip(values, bodies)]
        return header + '\n'.join(cases) + '\n' + footer

    def visit_Statement(self, o, **kwargs):
        stmt = '%s = %s' % (self.visit(o.target, **kwargs), self.visit(o.expr, **kwargs))
        if o.ptr:
            # Manually force pointer assignment notation
            # ... Hack me baby, one more time ...
            stmt = stmt.replace(' = ', ' => ')
        comment = '  %s' % self.visit(o.comment, **kwargs) if o.comment is not None else ''
        return self.indent + self.chunk(stmt) + comment

    def visit_MaskedStatement(self, o, **kwargs):
        condition = self.visit(o.condition, **kwargs)
        self._depth += 1
        body = self.visit(o.body, **kwargs)
        default = self.visit(o.default, **kwargs)
        self._depth -= 1
        header = self.indent + 'WHERE (%s)\n' % condition
        footer = '\n' + self.indent + 'END WHERE'
        default = '\n%sELSEWHERE\n' % self.indent + default if len(o.default) > 0 else ''
        return header + body + default + footer

    def visit_Section(self, o, **kwargs):
        return self.visit(o.body, **kwargs)

    def visit_Scope(self, o, **kwargs):
        associates = ['%s=>%s' % (self.visit(v, **kwargs), self.visit(a, **kwargs))
                      for a, v in o.associations.items()]
        associates = self.segment(associates, chunking=3)
        body = self.visit(o.body, **kwargs)
        return 'ASSOCIATE(%s)\n%s\nEND ASSOCIATE' % (associates, body)

    def visit_CallStatement(self, o, **kwargs):
        if o.kwarguments is not None:
            kwarguments = tuple('%s=%s' % (k, self.visit(v, **kwargs)) for k, v in o.kwarguments)
            args = as_tuple(o.arguments) + kwarguments
        else:
            args = o.arguments
        if len(args) > self.chunking:
            self._depth += 2
            # TODO: Temporary hack to force cloudsc_driver into the Fortran
            # line limit. The linewidth check should be made smarter to
            # adjust the chunking to the linewidth, like expressions do.
            signature = self.segment([self.visit(a, **kwargs) for a in args], chunking=3)
            self._depth -= 2
        else:
            signature = ', '.join(self.visit(a, **kwargs) for a in args)
        return self.indent + 'CALL %s(%s)' % (o.name, signature)

    def visit_Allocation(self, o, **kwargs):
        source = ''
        if o.data_source is not None:
            source = ', source=%s' % self.visit(o.data_source, **kwargs)
        variables = ','.join(v if isinstance(v, str) else self.visit(v, **kwargs)
                             for v in o.variables)
        return self.indent + 'ALLOCATE(%s%s)' % (variables, source)

    def visit_Deallocation(self, o, **kwargs):
        variables = ','.join(v if isinstance(v, str) else self.visit(v, **kwargs)
                             for v in o.variables)
        return self.indent + 'DEALLOCATE(%s)' % variables

    def visit_Nullify(self, o, **kwargs):
        variables = ', '.join(self.visit(v, **kwargs) for v in o.variables)
        return self.indent + 'NULLIFY(%s)' % variables

    def visit_SymbolType(self, o, **kwargs):
        if o.dtype == DataType.DERIVED_TYPE:
            tname = 'TYPE(%s)' % o.name
        else:
            type_map = {DataType.LOGICAL: 'LOGICAL', DataType.INTEGER: 'INTEGER',
                        DataType.REAL: 'REAL', DataType.CHARACTER: 'CHARACTER',
                        DataType.COMPLEX: 'COMPLEX'}
            tname = type_map[o.dtype]
        return '%s%s%s%s%s%s%s%s%s%s%s' % (
            tname,
            '(LEN=%s)' % o.length if o.length else '',
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

    def visit_TypeDef(self, o, **kwargs):
        bind_c = ', bind(c) ::' if o.bind_c else ''
        self._depth += 2
        declarations = self.visit(o.declarations, **kwargs)
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

"""
Expose the expression generator for testing purposes.
"""
fexprgen = FCodeMapper()
