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

from loki.logging import warning
from loki.tools import as_tuple
from loki.ir import (
        Import, Stringifier, FindNodes,
        FindVariables, FindRealLiterals
)
from loki.expression import (
        LokiStringifyMapper, Array, symbolic_op, Literal,
        symbols as sym
)
from loki.types import BasicType, SymbolAttributes, DerivedType

__all__ = ['cgen', 'CCodegen', 'CCodeMapper', 'IntrinsicTypeC']


class IntrinsicTypeC:
    """
    Mapping Fortran type to corresponding C type.
    """

    # pylint: disable=unused-argument

    def __call__(self, _type, *args, **kwargs):
        return self.get_str_from_symbol_attr(_type, *args, **kwargs)

    def get_str_from_symbol_attr(self, _type, *args, **kwargs):
        if _type.dtype == BasicType.LOGICAL:
            return 'int'
        if _type.dtype == BasicType.INTEGER:
            return 'int'
        if _type.dtype == BasicType.REAL:
            if str(_type.kind) in ['real32']:
                return 'float'
            return 'double'
        return str(_type)
        # raise ValueError(str(_type))

c_intrinsic_type = IntrinsicTypeC()

class CCodeMapper(LokiStringifyMapper):
    """
    A :class:`StringifyMapper`-derived visitor for Pymbolic expression trees that converts an
    expression to a string adhering to standardized C.
    """

    # pylint: disable=abstract-method, unused-argument

    def __init__(self, intrinsic_type_mapper, *args, **kwargs):
        super().__init__()
        self.intrinsic_type_mapper = intrinsic_type_mapper

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return super().map_logic_literal(expr, enclosing_prec, *args, **kwargs).lower()

    def map_float_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            _type = SymbolAttributes(BasicType.REAL, kind=expr.kind)
            return f'({self.intrinsic_type_mapper(_type)}) {str(expr.value)}'
        return str(expr.value)

    def map_int_literal(self, expr, enclosing_prec, *args, **kwargs):
        if expr.kind is not None:
            _type = SymbolAttributes(BasicType.INTEGER, kind=expr.kind)
            return f'({self.intrinsic_type_mapper(_type)}) {str(expr.value)}'
        return str(expr.value)

    def map_string_literal(self, expr, enclosing_prec, *args, **kwargs):
        return f'"{expr.value}"'

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        _type = SymbolAttributes(BasicType.from_fortran_type(expr.name), kind=expr.kind)
        expression = self.parenthesize_if_needed(
            self.join_rec('', expr.parameters, PREC_NONE, *args, **kwargs),
            PREC_CALL, PREC_NONE)
        return self.parenthesize_if_needed(
            self.format('(%s) %s', self.intrinsic_type_mapper(_type), expression), enclosing_prec, PREC_CALL)

    def map_variable_symbol(self, expr, enclosing_prec, *args, **kwargs):
        if expr.parent is not None:
            parent = self.rec(expr.parent, PREC_NONE, *args, **kwargs)
            return self.format('%s.%s', parent, expr.basename)
        return self.format('%s', expr.name)

    def map_meta_symbol(self, expr, enclosing_prec, *args, **kwargs):
        return self.rec(expr._symbol, enclosing_prec, *args, **kwargs)

    map_scalar = map_meta_symbol
    map_array = map_meta_symbol

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        name_str = self.rec(expr.aggregate, PREC_NONE, *args, **kwargs)
        if expr.aggregate.type is not None:
            if expr.aggregate.type.pointer and name_str.startswith('*'):
                # Strip the pointer '*' because subscript dereference
                name_str = name_str[1:]
            index_str = ''
            for index in expr.index_tuple:
                d = self.format(self.rec(index, PREC_NONE, *args, **kwargs))
                if d:
                    index_str += self.format('[%s]', d)
            return self.format('%s%s', name_str, index_str)
        return self.format('%s', name_str)

    map_string_subscript = map_array_subscript

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

    def map_c_reference(self, expr, enclosing_prec, *args, **kwargs):
        return self.format(' (&%s)', self.rec(expr.expression, PREC_NONE, *args, **kwargs))

    def map_c_dereference(self, expr, enclosing_prec, *args, **kwargs):
        return self.format(' (*%s)', self.rec(expr.expression, PREC_NONE, *args, **kwargs))

    def map_inline_call(self, expr, enclosing_prec, *args, **kwargs):
        if expr.function.name.lower() == 'mod':
            parameters = [self.rec(param, PREC_NONE, *args, **kwargs) for param in expr.parameters]
            # TODO: this check is not quite correct, as it should evaluate the
            #  expression(s) of both arguments/parameters and choose the integer version of modulo ('%')
            #  instead of the floating-point version ('fmod')
            #  whenever the mentioned evaluations result in being of kind 'integer' ...
            #  as an example: 'celing(3.1415)' got an floating point value in it, however it evaluates/returns
            #  an integer, in that case the wrong modulo function/operation is chosen
            if any(var.type.dtype != BasicType.INTEGER for var in FindVariables().visit(expr.parameters)) or\
                    FindRealLiterals().visit(expr.parameters):
                return f'fmod({parameters[0]}, {parameters[1]})'
            return f'({parameters[0]})%({parameters[1]})'

        if expr.function.name.lower() == 'present':
            return self.format('true /*ATTENTION: present({%s})*/', expr.parameters[0].name)

        return super().map_inline_call(expr, enclosing_prec, *args, **kwargs)


class CCodegen(Stringifier):
    """
    Tree visitor to generate standardized C code from IR.
    """
    # pylint: disable=abstract-method, unused-argument

    standard_imports = ['stdio.h', 'stdbool.h', 'float.h', 'math.h']

    def __init__(self, depth=0, indent='  ', linewidth=90, **kwargs):
        symgen = kwargs.get('symgen', CCodeMapper(c_intrinsic_type))
        line_cont = kwargs.get('line_cont', '\n{}  '.format)
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont=line_cont, symgen=symgen)

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

    def _subroutine_header(self, o, **kwargs):
        """
        Helper function/header for :func:`~loki.backend.CCodegen.visit_Subroutine`.
        """
        # Some boilerplate imports...
        header = [self.format_line('#include <', name, '>') for name in self.standard_imports]
        # ...and imports from the spec
        spec_imports = FindNodes(Import).visit(o.spec)
        header += [self.visit(spec_imports, **kwargs)]
        return header

    def _subroutine_argument_keyword(self, a):
        return ''

    def _subroutine_argument_pass_by(self, a):
        if isinstance(a, Array):
            return '* restrict '
        if isinstance(a.type.dtype, DerivedType):
            return '*'
        if a.type.pointer:
            return '*'
        if a.type.optional:
            return '*'
        return ''

    def _subroutine_optional_args(self, a):
        if a.type.optional:
            warning(f'Argument "{a}" is optional! No support for optional arguments in {self.__class__.__name__}.')
        return ''

    def _subroutine_declaration(self, o, **kwargs):
        """
        Helper function/function declaration part for :func:`~loki.backend.CCodegen.visit_Subroutine`.
        """
        # pass_by, var_keywords = self._subroutine_arguments(o, **kwargs)
        # arguments = [f'{k}{self.visit(a.type, **kwargs)} {p}{a.name}'
        #              for a, p, k in zip(o.arguments, pass_by, var_keywords)]
        arguments = [
            (f'{self._subroutine_argument_keyword(a)}{self.visit(a.type, **kwargs)} '
            f'{self._subroutine_argument_pass_by(a)}{a.name}{self._subroutine_optional_args(a)}')
            for a in o.arguments
        ]
        opt_header = kwargs.get('header', False)
        end = ' {' if not opt_header else ';'
        # check whether to return something and define function return type accordingly
        if o.is_function:
            return_type = self.symgen.intrinsic_type_mapper(o.return_type)
        else:
            return_type = 'void'
        declaration = [self.format_line(f'{return_type} ', o.name, '(', self.join_items(arguments), ')', end)]
        return declaration

    def _subroutine_body(self, o, **kwargs):
        """
        Helper function/body for :func:`~loki.backend.CCodegen.visit_Subroutine`.
        """
        self.depth += 1

        # ...and generate the spec without imports and argument declarations
        body = [self.visit(o.spec, skip_imports=True, skip_argument_declarations=True, **kwargs)]

        # Fill the body
        body += [self.visit(o.body, **kwargs)]

        # if something to be returned, add 'return <var>' statement
        if o.result_name is not None:
            body += [self.format_line(f'return {o.result_name.lower()};')]

        # Close everything off
        self.depth -= 1
        return body

    def _subroutine_footer(self, o, **kwargs):
        """
        Helper function/footer for :func:`~loki.backend.CCodegen.visit_Subroutine`.
        """
        footer = [self.format_line('}')]
        return footer

    def visit_Interface(self, o, **kwargs):
        return None

    def visit_Subroutine(self, o, **kwargs):
        """
        Format as:

          ...imports...
          <return_type> <name>(<args>) {
            ...spec without imports and argument declarations...
            ...body...
          }
        """
        opt_header = kwargs.get('header', False)
        opt_guards = kwargs.get('guards', False)
        opt_guard_name = kwargs.get('guard_name', None)

        header = self._subroutine_header(o, **kwargs)
        declaration = self._subroutine_declaration(o, **kwargs)
        body = self._subroutine_body(o, **kwargs) if not opt_header else []
        footer = self._subroutine_footer(o, **kwargs) if not opt_header else []

        if opt_guards:
            guard_name = f'{o.name.upper()}_H' if opt_guard_name is None else opt_guard_name
            header = [self.format_line(f'#ifndef {guard_name}'), self.format_line(f'#define {guard_name}\n\n')] + header
            footer += ['\n#endif']

        return self.join_lines(*header, '\n', *declaration, *body, *footer)

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
        text = o.text
        if text is None and o.source:
            text = o.source.string
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
            try:
                if v.initial is not None:
                    initial = f' = {self.visit(v.initial, **kwargs)}'
            except Exception as e:
                print(f"e: {e}")
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
        return self.format_line(str(o.name), '(', self.join_items(args), ');')

    def visit_SymbolAttributes(self, o, **kwargs):  # pylint: disable=unused-argument
        if isinstance(o.dtype, DerivedType):
            return f'struct {o.dtype.name}'
        return self.symgen.intrinsic_type_mapper(o)

    def visit_TypeDef(self, o, **kwargs):
        """
        Format type definition/struct as
          struct <name> {
            ...declarations...
          };
        """
        header = self.format_line('struct ', o.name.lower(), ' {')
        footer = self.format_line('};')
        self.depth += 1
        decls = self.visit(o.declarations, **kwargs)
        self.depth -= 1
        return self.join_lines(header, decls, footer)

    def visit_MultiConditional(self, o, **kwargs):
        """
        Format as
          switch case (<expr>) {
          case <value>:
          {
            ...body...
          }
          [case <value>:]
          {
            [...body...]
          }
          [default:] {
            [...body...]
          }
          }

        E.g., the following

        select case (in)
            case (:2)
                out = 1
            case (4, 5, 7:9)
                out = 2
            case (6)
                out = 3
            case default
                out = 4
        end select

        becomes

        switch (in) {
            case 0:
            case 1:
            case 2:
            {
              out = 1;
              break;
            }
            case 4:
            case 5:
            case 7:
            case 8:
            case 9:
            {
              out = 2;
              break;
            }
            case 6:
            {
              out = 3;
              break;
            }
            default:
            {
              out = 4;
              breal;
            }
        }
        """
        header = self.format_line('switch (', self.visit(o.expr, **kwargs), ') {')
        cases = []
        end_cases = []
        for value in o.values:
            sub_cases = []
            for val in value:
                if not isinstance(val, sym.RangeIndex):
                    sub_cases.append(self.visit(val, **kwargs))
                else:
                    assert (val.lower is None or isinstance(val.lower, sym.IntLiteral))\
                            and isinstance(val.upper, sym.IntLiteral)
                    lower = val.lower.value if val.lower is not None else 0
                    sub_cases.extend([str(v) for v in list(range(lower, val.upper.value + 1))])
            case = ()
            for sub_case in sub_cases:
                case += (self.format_line('case ', self.join_items(as_tuple(sub_case)), ':'),)
            cases.append(self.join_lines(*case, self.format_line('{')))
            end_cases.append(self.join_lines(self.format_line('break;'), self.format_line('}')))
        if o.else_body:
            cases.append(self.join_lines(self.format_line('default: '), self.format_line('{')))
            end_cases.append(self.join_lines(self.format_line('break;'), self.format_line('}')))
        footer = self.format_line('}')
        self.depth += 1
        bodies = self.visit_all(*o.bodies, o.else_body, **kwargs)
        self.depth -= 1
        branches = [item for branch in zip(cases, bodies, end_cases) for item in branch]
        return self.join_lines(header, *branches, footer)


def cgen(ir, **kwargs):
    """
    Generate standardized C code from one or many IR objects/trees.
    """
    return CCodegen().visit(ir, **kwargs)
