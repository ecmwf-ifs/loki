# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pymbolic.mapper.stringifier import PREC_NONE, PREC_CALL, PREC_COMPARISON

from loki.expression.mappers import LokiStringifyMapper
from loki.ir import Import
from loki.types import BasicType, DerivedType
from loki.visitors import Stringifier, FindNodes

__all__ = ['maxjgen', 'MaxjCodegen', 'MaxjCodeMapper']


def maxj_local_type(_type):
    if _type.dtype == BasicType.DEFERRED:
        return _type.name
    if _type.dtype == BasicType.LOGICAL:
        return 'boolean'
    if _type.dtype == BasicType.INTEGER:
        return 'int'
    if _type.dtype == BasicType.REAL:
        if str(_type.kind) in ['real32']:
            return 'float'
        return 'double'
    raise ValueError(str(_type))


def maxj_dfevar_type(_type):
    if _type.dtype == BasicType.LOGICAL:
        return 'dfeBool()'
    if _type.dtype == BasicType.INTEGER:
        return 'dfeUInt(32)'  # TODO: Distinguish between signed and unsigned
    if _type.dtype == BasicType.REAL:
        if str(_type.kind) in ['real32']:
            return 'dfeFloat(8, 24)'
        return 'dfeFloat(11, 53)'
    raise ValueError(str(_type))


class MaxjCodeMapper(LokiStringifyMapper):
    # pylint: disable=abstract-method, unused-argument

    def map_logic_literal(self, expr, enclosing_prec, *args, **kwargs):
        return super().map_logic_literal(expr, enclosing_prec, *args, **kwargs).lower()

    def map_string_literal(self, expr, enclosing_prec, *args, **kwargs):
        return f'"{expr.value}"'

    def map_variable_symbol(self, expr, enclosing_prec, *args, **kwargs):
        # TODO: Big hack, this is completely agnostic to whether value or address is to be assigned
        ptr = '*' if expr.type and expr.type.pointer else ''
        if expr.parent is not None:
            parent = self.parenthesize(self.rec(expr.parent, enclosing_prec, *args, **kwargs))
            return self.format('%s%s.%s', ptr, parent, expr.basename)
        return self.format('%s%s', ptr, expr.name)

    def map_meta_symbol(self, expr, enclosing_prec, *args, **kwargs):
        return self.rec(expr._symbol, enclosing_prec, *args, **kwargs)

    map_scalar = map_meta_symbol
    map_array = map_meta_symbol

    def map_array_subscript(self, expr, enclosing_prec, *args, **kwargs):
        name_str = self.rec(expr.aggregate, PREC_NONE, *args, **kwargs)
        index_str = ''
        for index in expr.index_tuple:
            d = self.format(self.rec(index, PREC_NONE, *args, **kwargs))
            if d:
                index_str += self.format('[%s]', d)
        return self.format('%s%s', name_str, index_str)

    def map_range_index(self, expr, enclosing_prec, *args, **kwargs):
        return self.rec(expr.upper, enclosing_prec, *args, **kwargs) if expr.upper else ''

    def map_cast(self, expr, enclosing_prec, *args, **kwargs):
        name = self.rec(expr.function, PREC_CALL, *args, **kwargs)
        expression = self.rec(expr.parameters[0], PREC_NONE, *args, **kwargs)
        kind = f'{maxj_dfevar_type(expr.kind)}, ' if expr.kind else ''
        return self.format('%s(%s%s)', name, kind, expression)

    def map_comparison(self, expr, enclosing_prec, *args, **kwargs):
        if expr.operator in ('==', '!='):
            return self.parenthesize_if_needed(
                self.format("%s.%s(%s)", self.rec(expr.left, PREC_CALL, *args, **kwargs),
                            {'==': 'eq', '!=': 'neq'}[expr.operator],
                            self.rec(expr.right, PREC_NONE, *args, **kwargs)),
                enclosing_prec, PREC_COMPARISON)
        return super().map_comparison(expr, enclosing_prec, *args, **kwargs)


class MaxjCodegen(Stringifier):
    """
    Tree visitor to generate Maxeler maxj kernel code from IR.
    """

    def __init__(self, depth=0, indent='  ', linewidth=90):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont='\n{}  '.format, symgen=MaxjCodeMapper())

    # Handler for outer objects

    def visit_Sourcefile(self, o, **kwargs):
        """
        Format as

        .. code-block::

            ...modules...
            ...subroutines...
        """
        return self.visit(o.ir, **kwargs)

    def visit_Module(self, o, **kwargs):
        """
        Format modules for a kernel as:

        .. code-block::

            package <name without Kernel>;
            ...imports...
            class <name> extends Kernel {
                ...spec without imports...
                ...routines...
            }

        Format modules for the manager as:

        .. code-block::

            package <name without Manager>;
            ...imports...
            public interface <name> extends ManagerPCIe, ManagerKernel {
                ...spec without imports...
                ...routines...
            }

        or for the platform-specific instantiation:

        .. code-block::

            package <name without ManagerMAX5C>;
            ...imports...
            public class <name> extends MAX5CManager implements <name without MAX5C> {
                ...spec without imports...
                ...routines...
            }
        """
        # Figure out what kind of module we have here
        if o.name.endswith('ManagerMAX5C'):
            is_manager = True
            is_interface = False
            package_name = o.name[:-len('ManagerMAX5C')]
        elif o.name.endswith('Manager'):
            is_manager = True
            is_interface = True
            package_name = o.name[:-len('Manager')]
        elif o.name.endswith('Kernel'):
            is_manager = False
            package_name = o.name[:-len('Kernel')]
        else:
            raise ValueError('Module is neither Manager nor Kernel')

        # Declare package
        header = [self.format_line('package ', package_name, ';')]

        # Some imports
        # TODO: include here imports defined by routines
        imports = FindNodes(Import).visit(o.spec)
        header += self.visit_all(imports, **kwargs)

        # Class signature
        if is_manager:
            if is_interface:
                header += [self.format_line(
                    'public interface ', o.name, ' extends ManagerPCIe, ManagerKernel {')]
            else:
                header += [self.format_line(
                    'public class ', o.name, ' extends MAX5CManager implements ', o.name[:-5], ' {')]
        else:
            header += [self.format_line('class ', o.name, ' extends Kernel {')]
        self.depth += 1

        # Rest of the spec
        body = [self.visit(o.spec, skip_imports=True, **kwargs)]

        # Create subroutines
        body += self.visit_all(o.subroutines, **kwargs)

        # Footer
        self.depth -= 1
        footer = [self.format_line('}')]

        return self.join_lines(*header, *body, *footer)

    def visit_Subroutine(self, o, **kwargs):
        """
        Format as:

        .. code-block::

            <name>(<args>) {
                ...spec without arg declarations...
                ...body...
            }
        """
        # Constructor signature
        args = [f'{self.visit(arg.type, **kwargs)} {self.visit(arg, **kwargs)}'
                for arg in o.arguments]
        header = [self.format_line(o.name, '(', self.join_items(args), ') {')]
        self.depth += 1

        # Generate body
        body = [self.visit(o.spec, skip_imports=True, **kwargs)]
        body += [self.visit(o.body, **kwargs)]

        # Closing brackets
        self.depth -= 1
        footer = [self.format_line('}')]

        return self.join_lines(*header, *body, *footer)

    # Handler for AST base nodes

    def visit_Node(self, o, **kwargs):
        """
        Format non-supported nodes as

        .. code-block::

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

        .. code-block::

            <type> <name> = <initial>
        """
        comment = None
        if o.comment:
            comment = str(self.visit(o.comment, **kwargs))

        def format_declaration(var):
            var_type = self.visit(var.type, **kwargs)
            var_name = self.visit(var.clone(dimensions=None), **kwargs)
            if var.initial:
                initial = self.visit(var.initial, **kwargs)
                return self.format_line(var_type, ' ', var_name, ' = ', initial, ';')
            return self.format_line(var_type, ' ', var_name, ';')

        declarations = [format_declaration(var) for var in o.symbols
                        if not var.type.intent or var.type.dfevar]
        return self.join_lines(comment, *declarations)

    def visit_Loop(self, o, **kwargs):
        """
        Format loop with explicit range as

        .. code-block::

            for (<var> = <start>; <var> <= <end>; <var> += <incr>) {
                ...body...
            }
        """
        control = 'for ({var} = {start}; {var} <= {end}; {var} += {incr})'.format(
            var=self.visit(o.variable, **kwargs), start=self.visit(o.bounds.start, **kwargs),
            end=self.visit(o.bounds.stop, **kwargs),
            incr=self.visit(o.bounds.step, **kwargs) if o.bounds.step else 1)
        header = self.format_line(control, ' {')
        footer = self.format_line('}')
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body, footer)

    def visit_Assignment(self, o, **kwargs):
        """
        Format statement as

        .. code-block::

            <target> = <expr>

        or

        .. code-block::

            <dfe_target> <== <expr>
        """
        lhs = self.visit(o.lhs, **kwargs)
        rhs = self.visit(o.rhs, **kwargs)
        comment = ''
        if o.comment:
            comment = f'  {self.visit(o.comment, **kwargs)}'
        if o.lhs.type.dfevar and o.lhs.type.shape:
            return self.format_line(lhs, ' <== ', rhs, ';', comment=comment)
        return self.format_line(lhs, ' = ', rhs, ';', comment=comment)

    def visit_ConditionalAssignment(self, o, **kwargs):
        """
        Format conditional statement as

        .. code-block::

            <target> = <condition> ? <expr> : <else_expr>
        """
        lhs = self.visit(o.lhs, **kwargs)
        condition = self.visit(o.condition, **kwargs)
        rhs = self.visit(o.rhs, **kwargs)
        else_rhs = self.visit(o.else_rhs, **kwargs)
        return self.format_line(lhs, ' = ', condition, ' ? ', rhs, ' : ', else_rhs, ';')

    def visit_Section(self, o, **kwargs):
        """
        Format the section's body.
        """
        return self.visit(o.body, **kwargs)

    def visit_CallStatement(self, o, **kwargs):
        """
        Format call statement as

        .. code-block::

            <name>(<args>)
        """
        name = self.visit(o.name, **kwargs)
        args = self.visit_all(o.arguments, **kwargs)
        assert not o.kwarguments
        return self.format_line(name, '(', self.join_items(args), ');')

    def visit_Import(self, o, **kwargs):
        """
        Format imports as

        .. code-block::

            import <name>;
        """
        if kwargs.get('skip_imports') is True:
            return None
        assert not o.symbols
        return self.format_line('import ', o.module, ';')

    def visit_SymbolAttributes(self, o, **kwargs):  # pylint: disable=unused-argument
        if isinstance(o.dtype, DerivedType):
            return o.dtype.name
        #    return 'DFEStructType {}'.format(o.name)
        if o.dfevar:
            if o.shape:
                return f'DFEVector<{self.visit(o.clone(shape=o.shape[:-1]), **kwargs)}>'
            return 'DFEVar'
        return maxj_local_type(o)

    def visit_TypeDef(self, o):
        self.depth += 1
        decls = self.visit(o.declarations)
        self.depth -= 1
        return f'DFEStructType {o.name} {{\n{decls}\n}} ;'


def maxjgen(ir):
    """
    Generate Maxeler maxj kernel code from one or many IR objects/trees.
    """
    return MaxjCodegen().visit(ir)
