from pymbolic.mapper.stringifier import (PREC_SUM, PREC_PRODUCT, PREC_UNARY, PREC_LOGICAL_OR,
                                         PREC_LOGICAL_AND, PREC_NONE, PREC_CALL)

from loki.tools import chunks
from loki.visitors import Visitor, FindNodes, Transformer
from loki.types import DataType, DerivedType
from loki.ir import Import
from loki.expression import LokiStringifyMapper, Array

__all__ = ['cgen', 'CCodegen', 'CCodeMapper']


class CCodeMapper(LokiStringifyMapper):

    def __init__(self, constant_mapper=repr):
        super(CCodeMapper, self).__init__(constant_mapper)

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return super().map_logic_literal(expr, enclosing_prec, *args, **kwargs).lower()

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        result = ('(%s) %s' % (DataType.from_type_kind('real', expr._kind).ctype, self(expr.value))
                  if expr._kind else self(expr.value))
        if not (result.startswith("(") and result.endswith(")")) \
                and ("-" in result or "+" in result) and (enclosing_prec > PREC_SUM):
            return self.parenthesize(result)
        else:
            return result

    def map_int_literal(self, expr, enclosing_prec, *args, **kwargs):
        result = ('(%s) %s' % (DataType.from_type_kind('integer', expr._kind).ctype, self(expr.value))
                  if expr._kind else self(expr.value))
        if not (result.startswith("(") and result.endswith(")")) \
                and ("-" in result or "+" in result) and (enclosing_prec > PREC_SUM):
            return self.parenthesize(result)
        else:
            return result

    def map_string_literal(self, expr, *args, **kwargs):
        return '"%s"' % expr.value

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        dtype = DataType.from_type_kind(expr.name, expr.kind).ctype
        expression = self.parenthesize_if_needed(
            self.join_rec('', expr.parameters, PREC_NONE, *args, **kwargs),
            PREC_CALL, PREC_NONE)
        return self.parenthesize_if_needed(self.format('(%s) %s', dtype, expression),
                                           enclosing_prec, PREC_CALL)

    def map_scalar(self, expr, *args, **kwargs):
        # TODO: Properly resolve pointers to the parent to replace '->' by '.'
        parent = self.rec(expr.parent, *args, **kwargs) + '->' if expr.parent else ''
        # TODO: Big hack, this is completely agnostic to whether value or address is to be assigned
        ptr = '*' if expr.type and expr.type.pointer else ''
        return '%s%s%s' % (ptr, parent, expr.name)

    def map_array(self, expr, *args, **kwargs):
        parent = '(' + self.rec(expr.parent, *args, **kwargs) + ').' if expr.parent else ''
        dims = ['[%s]' % self.rec(d, *args, **kwargs) for d in expr.dimensions]
        dims = ''.join(dims)
        return '%s%s%s' % (parent, expr.name, dims)

    def map_logical_not(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            "!" + self.rec(expr.child, PREC_UNARY, *args, **kwargs),
            enclosing_prec, PREC_UNARY)

    def map_logical_or(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.join_rec(" || ", expr.children, PREC_LOGICAL_OR, *args, **kwargs),
            enclosing_prec, PREC_LOGICAL_OR)

    def map_logical_and(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.join_rec(" && ", expr.children, PREC_LOGICAL_AND, *args, **kwargs),
            enclosing_prec, PREC_LOGICAL_AND)

    def map_range_index(self, expr, *args, **kwargs):
        return self.rec(expr.upper, *args, **kwargs) if expr.upper else ''

    def map_sum(self, expr, enclosing_prec, *args, **kwargs):
        """
        Since substraction and unary minus are mapped to multiplication with (-1), we are here
        looking for such cases and determine the matching operator for the output.
        """
        def get_neg_product(expr):
            from pymbolic.primitives import is_zero, Product

            if isinstance(expr, Product) and len(expr.children) and is_zero(expr.children[0]+1):
                if len(expr.children) == 2:
                    # only the minus sign and the other child
                    return expr.children[1]
                else:
                    return Product(expr.children[1:])
            else:
                return None

        terms = []
        is_neg_term = []
        for ch in expr.children:
            neg_prod = get_neg_product(ch)
            is_neg_term.append(neg_prod is not None)
            if neg_prod is not None:
                terms.append(self.rec(neg_prod, PREC_PRODUCT, *args, **kwargs))
            else:
                terms.append(self.rec(ch, PREC_SUM, *args, **kwargs))

        result = ['%s%s' % ('-' if is_neg_term[0] else '', terms[0])]
        result += [' %s %s' % ('-' if is_neg else '+', term)
                   for is_neg, term in zip(is_neg_term[1:], terms[1:])]

        return self.parenthesize_if_needed(''.join(result), enclosing_prec, PREC_SUM)

    def map_power(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.format('pow(%s, %s)', self.rec(expr.base, PREC_NONE, *args, **kwargs),
                        self.rec(expr.exponent, PREC_NONE, *args, **kwargs)),
            enclosing_prec, PREC_NONE)


class CCodegen(Visitor):
    """
    Tree visitor to generate standardized C code from IR.
    """

    def __init__(self, depth=0, linewidth=90, chunking=6):
        super(CCodegen, self).__init__()
        self.linewidth = linewidth
        self.chunking = chunking
        self._depth = depth
        self._csymgen = CCodeMapper()

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
            if isinstance(a, Array) > 0:
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
            if isinstance(a, Array):
                dtype = self.visit(a.type)
                # str(d).lower() is a bad hack to ensure caps-alignment
                outer_dims = ''.join('[%s]' % self._csymgen(d).lower() for d in a.dimensions[1:])
                casts += self.indent + '%s (*%s)%s = (%s (*)%s) v_%s;\n' % (
                    dtype, a.name.lower(), outer_dims, dtype, outer_dims, a.name.lower())

        # Some boilerplate imports...
        imports = '#include <stdio.h>\n'  # For manual debugging
        imports += '#include <stdbool.h>\n'
        imports += '#include <float.h>\n'
        imports += '#include <math.h>\n'
        imports += self.visit(FindNodes(Import).visit(o.spec))

        # ...remove imports from the spec...
        import_map = {imprt: None for imprt in FindNodes(Import).visit(o.spec)}
        o.spec = Transformer(import_map).visit(o.spec)

        # ...and generate the rest of spec and body
        spec = self.visit(o.spec)
        body = self.visit(o.body)
        footer = '\n%sreturn 0;\n}' % self.indent
        self._depth -= 1

        return imports + '\n\n' + header + casts + spec + '\n' + body + footer

    def visit_Section(self, o):
        return self.visit(o.body) + '\n'

    def visit_Import(self, o):
        return ('#include "%s"' % o.module) if o.c_import else ''

    def visit_Declaration(self, o):
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        type = self.visit(o.type)
        vstr = [self._csymgen(v) for v in o.variables]
        vptr = [('*' if v.type.pointer or v.type.allocatable else '') for v in o.variables]
        vinit = ['' if v.initial is None else (' = %s' % self._csymgen(v.initial)) for v in o.variables]
        variables = self.segment('%s%s%s' % (p, v, i) for v, p, i in zip(vstr, vptr, vinit))
        return self.indent + '%s %s;' % (type, variables) + comment

    def visit_BaseType(self, o):
        return o.dtype.ctype

    def visit_DerivedType(self, o):
        return 'struct %s' % o.name

    def visit_TypeDef(self, o):
        self._depth += 1
        decls = self.visit(o.declarations)
        self._depth -= 1
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
        lvar = self._csymgen(o.variable)
        lower = self._csymgen(o.bounds[0])
        upper = self._csymgen(o.bounds[1])
        criteria = '<=' if o.bounds[2] is None or eval(str(o.bounds[2])) > 0 else '>='
        header = 'for (%s=%s; %s%s%s; %s%s)' % (lvar, lower, lvar, criteria, upper, lvar, increment)
        return self.indent + '%s {\n%s\n%s}\n' % (header, body, self.indent)

    def visit_Statement(self, o):
        stmt = '%s = %s;' % (self._csymgen(o.target), self._csymgen(o.expr))
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        return self.indent + stmt + comment

    def visit_Conditional(self, o):
        self._depth += 1
        bodies = [self.visit(b) for b in o.bodies]
        else_body = self.visit(o.else_body)
        self._depth -= 1
        if len(bodies) > 1:
            raise NotImplementedError('Multi-body conditionals not yet supported')

        cond = self._csymgen(o.conditions[0])
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
