from pymbolic.mapper.stringifier import (PREC_SUM, PREC_PRODUCT, PREC_UNARY, PREC_LOGICAL_OR,
                                         PREC_LOGICAL_AND, PREC_NONE, PREC_CALL)

from loki.visitors import Stringifier, FindNodes
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
        dims = ''
        if expr.dimensions:
            dims = self.rec(expr.dimensions, enclosing_prec, *args, **kwargs)
        if expr.parent is not None:
            parent = self.parenthesize(self.rec(expr.parent, enclosing_prec, *args, **kwargs))
            return self.format('%s.%s%s', parent, expr.basename, dims)
        return self.format('%s%s', expr.basename, dims)

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        index_str = ''
        for index in expr.index_tuple:
            d = self.format(self.rec(index, PREC_NONE, *args, **kwargs))
            if d:
                index_str += self.format('[%s]', d)
        return index_str

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

    def map_power(self, expr, enclosing_prec, *args, **kwargs):
        return self.parenthesize_if_needed(
            self.format('pow(%s, %s)', self.rec(expr.base, PREC_NONE, *args, **kwargs),
                        self.rec(expr.exponent, PREC_NONE, *args, **kwargs)),
            enclosing_prec, PREC_NONE)


class CCodegen(Stringifier):
    """
    Tree visitor to generate standardized C code from IR.
    """

    def __init__(self, depth=0, indent='  ', linewidth=90):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont=lambda indent: '\n{}  '.format(indent),  # pylint: disable=unnecessary-lambda
                         symgen=CCodeMapper())

    # Handler for outer objects

    def visit_SourceFile(self, o, **kwargs):
        """
        Format as
          ...modules...
          ...subroutines...
        """
        modules = self.visit_all(o.modules, **kwargs)
        subroutines = self.visit_all(o.subroutines, **kwargs)
        return self.join_lines(*modules, *subroutines)

    def visit_Module(self, o, **kwargs):
        # Assuming this will be put in header files...
        spec = self.visit(o.spec, **kwargs)
        routines = self.visit(o.routines, **kwargs)
        return self.join_lines(spec, routines)

    def visit_Subroutine(self, o, **kwargs):
        """
        Format as:

          ...imports...
          int <name>(<args>) {
            ...spec without imports and argument declarations...
            ...body...
          }
        """
        # Some boilerplate imports...
        standard_imports = ['stdio.h', 'stdbool.h', 'float.h', 'math.h']
        header = [self.format_line('#include <', name, '>') for name in standard_imports]

        # ...and imports from the spec
        spec_imports = FindNodes(Import).visit(o.spec)
        header += [self.visit(spec_imports, **kwargs)]

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
        header += [self.format_line('int ', o.name, '(', self.join_items(arguments), ') {')]

        self.depth += 1

        # ...and generate the spec without imports and argument declarations
        body = [self.visit(o.spec, skip_imports=True, skip_argument_declarations=True, **kwargs)]

        # Generate the array casts for pointer arguments
        if any(isinstance(a, Array) for a in o.arguments):
            body += [self.format_line('/* Array casts for pointer arguments */')]
            for a in o.arguments:
                if isinstance(a, Array):
                    dtype = self.visit(a.type, **kwargs)
                    # str(d).lower() is a bad hack to ensure caps-alignment
                    outer_dims = ''.join('[%s]' % self.visit(d, **kwargs).lower()
                                         for d in a.dimensions.index_tuple[1:])
                    body += [self.format_line(dtype, ' (*', a.name.lower(), ')', outer_dims, ' = (',
                                              dtype, ' (*)', outer_dims, ') v_', a.name.lower(), ';')]

        # Fill the body
        body += [self.visit(o.body, **kwargs)]
        body += [self.format_line('return 0;')]

        # Close everything off
        self.depth -= 1
        footer = [self.format_line('}')]

        return self.join_lines(*header, *body, *footer)

    # Handler for AST base nodes

    def visit_Node(self, o, **kwargs):
        """
        Format non-supported nodes as
          // <repr(Node)>
        """
        return self.format_line('// <', repr(o), '>')

    # Handler for IR nodes

    def visit_Intrinsic(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Format intrinsic nodes.
        """
        return self.format_line(str(o.text).lstrip())

    def visit_Comment(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Format comments.
        """
        text = o.text or o.source.string
        text = str(text).lstrip().replace('!', '//', 1)
        return self.format_line(text, no_wrap=True)

    def visit_CommentBlock(self, o, **kwargs):
        """
        Format comment blocks.
        """
        comments = self.visit_all(o.comments, **kwargs)
        return self.join_lines(*comments)

    def visit_Declaration(self, o, **kwargs):
        """
        Format declaration as
          <type> <name> [= <initial>]
        """
        types = [v.type for v in o.variables]
        # Ensure all variable types are equal, except for shape and dimension
        ignore = ['shape', 'dimensions', 'source']
        assert all(t.compare(types[0], ignore=ignore) for t in types)
        dtype = self.visit(types[0], **kwargs)
        assert len(o.variables) > 0
        variables = []
        for v in o.variables:
            if kwargs.get('skip_argument_declarations') and v.type.intent:
                continue
            var = self.visit(v, **kwargs)
            initial = ''
            if v.initial is not None:
                initial = ' = {}'.format(self.visit(v.initial, **kwargs))
            if v.type.pointer or v.type.allocatable:
                var = '*' + var
            variables += ['{}{}'.format(var, initial)]
        if not variables:
            return None
        comment = None
        if o.comment:
            comment = str(self.visit(o.comment, **kwargs))
        return self.format_line(dtype, ' ', self.join_items(variables), ';', comment=comment)

    def visit_Import(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Format C imports as
          #include "<name>"
        """
        if not kwargs.get('skip_imports') and o.c_import:
            return self.format_line('#include "', str(o.module), '"')
        return None

    def visit_Loop(self, o, **kwargs):
        """
        Format loop with explicit range as
          for (<var>=<start>; <var><criteria><end>; <var> += <incr>) {
            ...body...
          }
        """
        control = 'for ({var} = {start}; {var} {crit} {end}; {var} += {incr})'.format(
            var=self.visit(o.variable, **kwargs), start=self.visit(o.bounds.start, **kwargs),
            end=self.visit(o.bounds.stop, **kwargs),
            crit='<=' if not o.bounds.step or eval(str(o.bounds.step)) > 0 else '>=',
            incr=self.visit(o.bounds.step, **kwargs) if o.bounds.step else 1)
        header = self.format_line(control, ' {')
        footer = self.format_line('}')
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body, footer)

    def visit_WhileLoop(self, o, **kwargs):
        """
        Format loop as
          while (<condition>) {
            ...body...
          }
        """
        if o.condition is not None:
            condition = self.visit(o.condition, **kwargs)
        else:
            condition = '1'
        header = self.format_line('while (', condition, ') {')
        footer = self.format_line('}')
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body, footer)

    def visit_Conditional(self, o, **kwargs):
        """
        Format conditional as
          if (<condition>) {
            ...body...
          } [ else { ]
            [...body...]
          [}]
        """
        condition = self.visit(o.conditions[0], **kwargs)
        header = self.format_line('if (', condition, ') {')
        else_header = None
        if o.else_body:
            else_header = self.format_line('} else {')
        footer = '}'
        self.depth += 1
        bodies = self.visit_all(o.bodies, **kwargs)
        else_body = self.visit(o.else_body, **kwargs)
        self.depth -= 1
        if len(bodies) > 1:
            raise NotImplementedError('Multi-body conditionals not yet supported')
        return self.join_lines(header, *bodies, else_header, else_body, footer)

    def visit_Statement(self, o, **kwargs):
        """
        Format statement as
          <target> = <expr> [<comment>]
        """
        target = self.visit(o.target, **kwargs)
        expr = self.visit(o.expr, **kwargs)
        comment = None
        if o.comment:
            comment = '  {}'.format(self.visit(o.comment, **kwargs))
        return self.format_line(target, ' = ', expr, ';', comment=comment)

    def visit_Section(self, o, **kwargs):
        """
        Format the section's body.
        """
        return self.visit(o.body, **kwargs)

    def visit_CallStatement(self, o, **kwargs):
        """
        Format call statement as
          <name>(<args>)
        """
        args = self.visit_all(o.arguments, **kwargs)
        assert not o.kwarguments
        return self.format_line(o.name, '(', self.join_items(args), ');')

    def visit_SymbolType(self, o, **kwargs):  # pylint: disable=unused-argument
        if o.dtype == DataType.DERIVED_TYPE:
            return 'struct %s' % o.name
        return c_intrinsic_type(o)

    def visit_TypeDef(self, o, **kwargs):
        header = self.format_line('struct ', o.name.lower(), ' {')
        footer = self.format_line('};')
        self.depth += 1
        decls = self.visit(o.declarations, **kwargs)
        self.depth -= 1
        return self.join_lines(header, decls, footer)


def cgen(ir):
    """
    Generate standardized C code from one or many IR objects/trees.
    """
    return CCodegen().visit(ir)
