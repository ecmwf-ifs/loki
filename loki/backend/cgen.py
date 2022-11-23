# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from operator import gt
from pymbolic.mapper.stringifier import (
    PREC_UNARY, PREC_LOGICAL_OR, PREC_LOGICAL_AND, PREC_NONE, PREC_CALL
)

from loki.visitors import Stringifier, FindNodes
from loki.ir import Import
from loki.expression import LokiStringifyMapper, Array, symbolic_op, Literal
from loki.types import BasicType, SymbolAttributes, DerivedType

__all__ = ['cgen', 'CCodegen', 'CCodeMapper']


def c_intrinsic_type(_type):
    if _type.dtype == BasicType.LOGICAL:
        return 'int'
    if _type.dtype == BasicType.INTEGER:
        return 'int'
    if _type.dtype == BasicType.REAL:
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
            _type = SymbolAttributes(BasicType.REAL, kind=expr.kind)
            return f'({c_intrinsic_type(_type)}) {str(expr.value)}'
        return str(expr.value)

    def map_int_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            _type = SymbolAttributes(BasicType.INTEGER, kind=expr.kind)
            return f'({c_intrinsic_type(_type)}) {str(expr.value)}'
        return str(expr.value)

    def map_string_literal(self, expr, enclosing_prec, *args, **kwargs):
        return f'"{expr.value}"'

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        _type = SymbolAttributes(BasicType.from_fortran_type(expr.name), kind=expr.kind)
        expression = self.parenthesize_if_needed(
            self.join_rec('', expr.parameters, PREC_NONE, *args, **kwargs),
            PREC_CALL, PREC_NONE)
        return self.parenthesize_if_needed(
            self.format('(%s) %s', c_intrinsic_type(_type), expression), enclosing_prec, PREC_CALL)

    def map_variable_symbol(self, expr, enclosing_prec, *args, **kwargs):
        # TODO: Big hack, this is completely agnostic to whether value or address is to be assigned
        ptr = '*' if expr.type and expr.type.pointer else ''
        if expr.parent is not None:
            parent = self.parenthesize(self.rec(expr.parent, PREC_NONE, *args, **kwargs))
            return self.format('%s%s.%s', ptr, parent, expr.basename)
        return self.format('%s%s', ptr, expr.name)

    def map_meta_symbol(self, expr, enclosing_prec, *args, **kwargs):
        return self.rec(expr._symbol, enclosing_prec, *args, **kwargs)

    map_scalar = map_meta_symbol
    map_array = map_meta_symbol

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        name_str = self.rec(expr.aggregate, PREC_NONE, *args, **kwargs)
        if expr.aggregate.type.pointer and name_str.startswith('*'):
            # Strip the pointer '*' because subscript dereference
            name_str = name_str[1:]
        index_str = ''
        for index in expr.index_tuple:
            d = self.format(self.rec(index, PREC_NONE, *args, **kwargs))
            if d:
                index_str += self.format('[%s]', d)
        return self.format('%s%s', name_str, index_str)

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
                         line_cont='\n{}  '.format, symgen=CCodeMapper())

    # Handler for outer objects

    def visit_Sourcefile(self, o, **kwargs):
        """
        Format as
          ...modules...
          ...subroutines...
        """
        return self.visit(o.ir, **kwargs)

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
            elif isinstance(a.type.dtype, DerivedType):
                aptr += ['*']
            elif a.type.pointer:
                aptr += ['*']
            else:
                aptr += ['']
        arguments = [f'{self.visit(a.type, **kwargs)} {p}{a.name.lower()}'
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
                    outer_dims = ''.join(f'[{self.visit(d, **kwargs).lower()}]'
                                         for d in a.dimensions[1:])
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

    def visit_VariableDeclaration(self, o, **kwargs):
        """
        Format declaration as
          <type> <name> [= <initial>]
        """
        types = [v.type for v in o.symbols]
        # Ensure all variable types are equal, except for shape and dimension
        ignore = ['shape', 'dimensions', 'source']
        assert all(t.compare(types[0], ignore=ignore) for t in types)
        dtype = self.visit(types[0], **kwargs)
        assert len(o.symbols) > 0
        variables = []
        for v in o.symbols:
            if kwargs.get('skip_argument_declarations') and v.type.intent:
                continue
            var = self.visit(v, **kwargs)
            initial = ''
            if v.initial is not None:
                initial = f' = {self.visit(v.initial, **kwargs)}'
            if v.type.pointer or v.type.allocatable:
                var = '*' + var
            variables += [f'{var}{initial}']
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
            crit='<=' if not o.bounds.step or symbolic_op(o.bounds.step, gt, Literal(0)) else '>=',
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
          [ } else if (<condition>) { ]
            [...body...]
          [ } else { ]
            [...body...]
          }
        """
        is_elseif = kwargs.pop('is_elseif', False)
        if is_elseif:
            header = self.format_line('} else if (', self.visit(o.condition, **kwargs), ') {')
        else:
            header = self.format_line('if (', self.visit(o.condition, **kwargs), ') {')
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        if o.has_elseif:
            self.depth -= 1
            else_body = [self.visit(o.else_body, is_elseif=True, **kwargs)]
        else:
            else_body = [self.visit(o.else_body, **kwargs)]
            self.depth -= 1
            if o.else_body:
                else_body = [self.format_line('} else {')] + else_body
            else_body += [self.format_line('}')]
        return self.join_lines(header, body, *else_body)

    def visit_Assignment(self, o, **kwargs):
        """
        Format statement as
          <target> = <expr> [<comment>]
        """
        lhs = self.visit(o.lhs, **kwargs)
        rhs = self.visit(o.rhs, **kwargs)
        comment = None
        if o.comment:
            comment = f'  {self.visit(o.comment, **kwargs)}'
        return self.format_line(lhs, ' = ', rhs, ';', comment=comment)

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

    def visit_SymbolAttributes(self, o, **kwargs):  # pylint: disable=unused-argument
        if isinstance(o.dtype, DerivedType):
            return f'struct {o.dtype.name}'
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
