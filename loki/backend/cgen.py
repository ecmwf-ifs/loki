from pymbolic.mapper.stringifier import (PREC_SUM, PREC_PRODUCT, PREC_UNARY, PREC_LOGICAL_OR,
                                         PREC_LOGICAL_AND, PREC_NONE, PREC_CALL)

from loki.tools import chunks
from loki.visitors import Visitor, FindNodes, Transformer
from loki.ir import Import
from loki.expression import LokiStringifyMapper, Array
from loki.types import DataType, SymbolType

__all__ = ['cgen', 'CCodegen', 'CCodeMapper']


def c_intrinsic_type(_type):
    if _type.dtype == DataType.LOGICAL:
        return 'int'
    if _type.dtype == DataType.INTEGER:
        return 'int'
    if _type.dtype == DataType.REAL:
        if str(_type.kind) in ['real32']:
            return 'float'
        return 'double'
    raise ValueError(str(_type))


class CCodeMapper(LokiStringifyMapper):
    # pylint: disable=abstract-method, unused-argument

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return super().map_logic_literal(expr, enclosing_prec, *args, **kwargs).lower()

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            _type = SymbolType(DataType.REAL, kind=expr.kind)
            result = '(%s) %s' % (c_intrinsic_type(_type), str(expr.value))
        else:
            result = str(expr.value)
        if not (result.startswith("(") and result.endswith(")")) \
                and ("-" in result or "+" in result) and (enclosing_prec > PREC_SUM):
            return self.parenthesize(result)
        return result

    def map_int_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            _type = SymbolType(DataType.INTEGER, kind=expr.kind)
            result = '(%s) %s' % (c_intrinsic_type(_type), str(expr.value))
        else:
            result = str(expr.value)
        if not (result.startswith("(") and result.endswith(")")) \
                and ("-" in result or "+" in result) and (enclosing_prec > PREC_SUM):
            return self.parenthesize(result)
        return result

    def map_string_literal(self, expr, enclosing_prec, *args, **kwargs):
        return '"%s"' % expr.value

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        _type = SymbolType(DataType.from_fortran_type(expr.name), kind=expr.kind)
        expression = self.parenthesize_if_needed(
            self.join_rec('', expr.parameters, PREC_NONE, *args, **kwargs),
            PREC_CALL, PREC_NONE)
        return self.parenthesize_if_needed(
            self.format('(%s) %s', c_intrinsic_type(_type), expression), enclosing_prec, PREC_CALL)

    def map_scalar(self, expr, enclosing_prec, *args, **kwargs):
        # TODO: Big hack, this is completely agnostic to whether value or address is to be assigned
        ptr = '*' if expr.type and expr.type.pointer else ''
        if expr.parent is not None:
            parent = self.parenthesize(self.rec(expr.parent, enclosing_prec, *args, **kwargs))
            return self.format('%s%s.%s', ptr, parent, expr.basename)
        return self.format('%s%s', ptr, expr.name)

    def map_array(self, expr, enclosing_prec, *args, **kwargs):
        dims = [self.rec(d, enclosing_prec, *args, **kwargs) for d in expr.dimensions]
        dims = ''.join(['[%s]' % d for d in dims if len(d) > 0])
        if expr.parent is not None:
            parent = self.parenthesize(self.rec(expr.parent, enclosing_prec, *args, **kwargs))
            return self.format('%s.%s%s', parent, expr.basename, dims)
        return self.format('%s%s', expr.basename, dims)

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

    def map_range_index(self, expr, enclosing_prec, *args, **kwargs):
        return self.rec(expr.upper, enclosing_prec, *args, **kwargs) if expr.upper else ''

    def map_sum(self, expr, enclosing_prec, *args, **kwargs):
        def get_neg_product(expr):
            """
            Since substraction and unary minus are mapped to multiplication with (-1), we are here
            looking for such cases and determine the matching operator for the output.
            """
            # pylint: disable=import-outside-toplevel
            from pymbolic.primitives import is_zero, Product

            if isinstance(expr, Product) and expr.children and is_zero(expr.children[0]+1):
                if len(expr.children) == 2:
                    # only the minus sign and the other child
                    return expr.children[1]
                return Product(expr.children[1:])
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
    # pylint: disable=no-self-use

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

    @property
    def csymgen(self):
        return self._csymgen

    def segment(self, arguments, chunking=None):
        chunking = chunking or self.chunking
        delim = ',\n%s  ' % self.indent
        args = list(chunks(list(arguments), chunking))
        return delim.join(', '.join(c) for c in args)

    def visit_Expression(self, o, **kwargs):
        return self.csymgen(o)

    def visit_str(self, o, **kwargs):
        return o

    def visit_Node(self, o, **kwargs):
        return self.indent + '// <%s>' % o.__class__.__name__

    def visit_tuple(self, o, **kwargs):
        return '\n'.join([self.visit(i, **kwargs) for i in o])

    visit_list = visit_tuple

    def visit_Module(self, o, **kwargs):
        # Assuming this will be put in header files...
        spec = self.visit(o.spec, **kwargs)
        routines = self.visit(o.routines, **kwargs)
        return spec + '\n\n' + routines

    def visit_Subroutine(self, o, **kwargs):
        # Re-generate variable declarations
        o._externalize(c_backend=True)

        # Generate header with argument signature
        aptr = []
        for a in o.arguments:
            # TODO: Oh dear, the pointer derivation is beyond hacky; clean up!
            if isinstance(a, Array) > 0:
                aptr += ['* restrict v_']
            elif a.type.dtype == DataType.DERIVED_TYPE:
                aptr += ['*']
            elif a.type.pointer:
                aptr += ['*']
            else:
                aptr += ['']
        arguments = ['%s %s%s' % (self.visit(a.type, **kwargs), p, a.name.lower())
                     for a, p in zip(o.arguments, aptr)]
        arguments = self.segment(arguments)
        header = 'int %s(%s)\n{\n' % (o.name, arguments)

        self._depth += 1

        # Generate the array casts for pointer arguments
        casts = '%s/* Array casts for pointer arguments */\n' % self.indent
        for a in o.arguments:
            if isinstance(a, Array):
                dtype = self.visit(a.type, **kwargs)
                # str(d).lower() is a bad hack to ensure caps-alignment
                outer_dims = ''.join('[%s]' % self.visit(d, **kwargs).lower() for d in a.dimensions[1:])
                casts += self.indent + '%s (*%s)%s = (%s (*)%s) v_%s;\n' % (
                    dtype, a.name.lower(), outer_dims, dtype, outer_dims, a.name.lower())

        # Some boilerplate imports...
        imports = '#include <stdio.h>\n'  # For manual debugging
        imports += '#include <stdbool.h>\n'
        imports += '#include <float.h>\n'
        imports += '#include <math.h>\n'
        imports += self.visit(FindNodes(Import).visit(o.spec), **kwargs)

        # ...remove imports from the spec...
        import_map = {imprt: None for imprt in FindNodes(Import).visit(o.spec, **kwargs)}
        o.spec = Transformer(import_map).visit(o.spec, **kwargs)

        # ...and generate the rest of spec and body
        spec = self.visit(o.spec, **kwargs)
        body = self.visit(o.body, **kwargs)
        footer = '\n%sreturn 0;\n}' % self.indent
        self._depth -= 1

        return imports + '\n\n' + header + casts + spec + '\n' + body + footer

    def visit_Section(self, o, **kwargs):
        return self.visit(o.body, **kwargs) + '\n'

    def visit_Import(self, o, **kwargs):  # pylint: disable=unused-argument
        return ('#include "%s"' % o.module) if o.c_import else ''

    def visit_Declaration(self, o, **kwargs):
        comment = '  %s' % self.visit(o.comment, **kwargs) if o.comment is not None else ''
        _type = self.visit(o.type, **kwargs)
        variables = self.segment(self.visit(v, **kwargs) for v in o.variables)
        return self.indent + '%s %s;' % (_type, variables) + comment

    def visit_Scalar(self, o, **kwargs):
        var = self.csymgen(o)
        if o.type.pointer or o.type.allocatable:
            var = '*' + var
        if o.initial:
            var += ' = %s' % self.visit(o.initial)
        return var

    visit_Array = visit_Scalar

    def visit_SymbolType(self, o, **kwargs):  # pylint: disable=unused-argument
        if o.dtype == DataType.DERIVED_TYPE:
            return 'struct %s' % o.name
        return c_intrinsic_type(o)

    def visit_TypeDef(self, o, **kwargs):
        self._depth += 1
        decls = self.visit(o.declarations, **kwargs)
        self._depth -= 1
        return 'struct %s {\n%s\n} ;' % (o.name.lower(), decls)

    def visit_Comment(self, o, **kwargs):  # pylint: disable=unused-argument
        text = o._source.string if o.text is None else o.text
        return self.indent + text.replace('!', '//')

    def visit_CommentBlock(self, o, **kwargs):
        comments = [self.visit(c, **kwargs) for c in o.comments]
        return '\n'.join(comments)

    def visit_Loop(self, o, **kwargs):
        self._depth += 1
        body = self.visit(o.body, **kwargs)
        self._depth -= 1
        increment = '++' if o.bounds.step is None else '+=%s' % self.visit(o.bounds.step, **kwargs)
        lvar = self.visit(o.variable, **kwargs)
        lower = self.visit(o.bounds.start, **kwargs)
        upper = self.visit(o.bounds.stop, **kwargs)
        criteria = '<=' if o.bounds.step is None or eval(str(o.bounds.step)) > 0 else '>='
        header = 'for (%s=%s; %s%s%s; %s%s)' % (lvar, lower, lvar, criteria, upper, lvar, increment)
        return self.indent + '%s {\n%s\n%s}\n' % (header, body, self.indent)

    def visit_WhileLoop(self, o, **kwargs):
        condition = self.visit(o.condition, **kwargs)
        self._depth += 1
        body = self.visit(o.body, **kwargs)
        self._depth -= 1
        header = 'while (%s)' % condition
        return self.indent + '%s {\n%s\n%s}\n' % (header, body, self.indent)

    def visit_Statement(self, o, **kwargs):
        stmt = '%s = %s;' % (self.csymgen(o.target), self.visit(o.expr, **kwargs))
        comment = '  %s' % self.visit(o.comment, **kwargs) if o.comment is not None else ''
        return self.indent + stmt + comment

    def visit_Conditional(self, o, **kwargs):
        self._depth += 1
        bodies = [self.visit(b, **kwargs) for b in o.bodies]
        else_body = self.visit(o.else_body, **kwargs)
        self._depth -= 1
        if len(bodies) > 1:
            raise NotImplementedError('Multi-body conditionals not yet supported')

        cond = self.visit(o.conditions[0], **kwargs)
        main_branch = 'if (%s)\n%s{\n%s\n' % (cond, self.indent, bodies[0])
        else_branch = '%s} else {\n%s\n' % (self.indent, else_body) if o.else_body else ''
        return self.indent + main_branch + else_branch + '%s}\n' % self.indent

    def visit_Intrinsic(self, o, **kwargs):  # pylint: disable=unused-argument
        return o.text


def cgen(ir):
    """
    Generate standardized C code from one or many IR objects/trees.
    """
    return CCodegen().visit(ir)
