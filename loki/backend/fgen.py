from pymbolic.mapper.stringifier import (
    PREC_UNARY, PREC_LOGICAL_AND, PREC_LOGICAL_OR, PREC_COMPARISON, PREC_NONE
)

from loki.visitors import Stringifier
from loki.tools import as_tuple, JoinableStringList
from loki.expression import LokiStringifyMapper
from loki.types import BasicType, DerivedType, ProcedureType
from loki.pragma_utils import get_pragma_parameters


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
            return f'{str(expr.value)}_{str(expr.kind)}'
        return str(expr.value)

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

    def map_literal_list(self, expr, enclosing_prec, *args, **kwargs):
        return '(/' + ','.join(str(c) for c in expr.elements) + '/)'

    def map_foreign(self, expr, *args, **kwargs):
        try:
            return super().map_foreign(expr, *args, **kwargs)
        except ValueError:
            return f'! Not supported: {str(expr)}\n'

    def map_loop_range(self, expr, enclosing_prec, *args, **kwargs):
        children = [self.rec(child, PREC_NONE, *args, **kwargs) if child is not None else ''
                    for child in expr.children]
        # Do not unnecessarily print `:1` stepping for loops
        if expr.step is None or str(expr.step) == '1':
            children = children[:-1]
        return self.parenthesize_if_needed(self.join(',', children), enclosing_prec, PREC_NONE)


class FortranCodegen(Stringifier):
    """
    Tree visitor to generate standardized Fortran code from IR.
    """
    # pylint: disable=no-self-use, unused-argument

    def __init__(self, depth=0, indent='  ', linewidth=90, conservative=True):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth,
                         line_cont=' &\n{}& '.format, symgen=FCodeMapper())
        self.conservative = conservative

    def apply_label(self, line, label):
        """
        Apply a label to the given (formatted) line by replacing indentation with the label.

        :param str line: the formatted line.
        :param label: the label to apply.
        :type label: str or NoneType

        :return: the line with the label applied if given, else the original line.
        :rtype: str
        """
        if label is not None:
            # Replace indentation by label
            indent = max(1, len(line) - len(line.lstrip()) - 1)
            line = f'{label:{indent}} {line.lstrip()}'
        return line

    def visit(self, o, *args, **kwargs):
        """
        Overwrite standard visit routine to inject original source in conservative mode.
        """
        if self.conservative and hasattr(o, 'source') and getattr(o.source, 'string', None) is not None:
            # Re-use original source associated with node
            return o.source.string
        return super().visit(o, *args, **kwargs)

    # Handler for outer objects

    def visit_Sourcefile(self, o, **kwargs):
        """
        Format as
          ...modules...
          ...subroutines...
        """
        modules = self.visit_all(o.modules, **kwargs)
        subroutines = self.visit_all(o.subroutines, **kwargs)
        return self.join_lines(*modules, *subroutines)

    def visit_Module(self, o, **kwargs):
        """
        Format as
          MODULE <name>
            ...spec...
          CONTAINS
            ...routines...
          END MODULE
        """
        header = self.format_line('MODULE ', o.name)
        contains = self.format_line('CONTAINS')
        footer = self.format_line('END MODULE ', o.name)

        self.depth += 1
        spec = self.visit(o.spec, **kwargs)
        routines = self.visit(o.routines, **kwargs)
        self.depth -= 1
        return self.join_lines(header, spec, contains, routines, footer)

    def visit_Subroutine(self, o, **kwargs):
        """
        Format as
          <ftype> <name> ([<args>]) [BIND(c, name=<name>)]
            ...docstring...
            ...spec...
            ...body...
          [CONTAINS]
            [...member...]
          END <ftype> <name>
        """
        ftype = 'FUNCTION' if o.is_function else 'SUBROUTINE'
        arguments = self.join_items(o.argnames)
        bind_c = f' BIND(c, name=\'{o.bind}\')' if o.bind else ''
        header = self.format_line(ftype, ' ', o.name, ' (', arguments, ')', bind_c)
        contains = self.format_line('CONTAINS')
        footer = self.format_line('END ', ftype, ' ', o.name)

        self.depth += 1
        docstring = self.visit(o.docstring, **kwargs)
        spec = self.visit(o.spec, **kwargs)
        body = self.visit(o.body, **kwargs)
        members = self.visit(o.members, **kwargs)
        self.depth -= 1
        if members:
            return self.join_lines(header, docstring, spec, body, contains, members, footer)
        return self.join_lines(header, docstring, spec, body, footer)

    # Handler for AST base nodes

    def visit_Node(self, o, **kwargs):
        """
        Format non-supported nodes as
          ! <repr(Node)>
        """
        return self.format_line('! <', repr(o), '>')

    def visit_tuple(self, o, **kwargs):
        """
        Recurse for each item in the tuple and return as separate lines.
        Insert labels if existing.
        """
        lines = []
        for item in o:
            line = self.visit(item, **kwargs)
            line = self.apply_label(line, getattr(item, 'label', None))
            lines.append(line)
        return self.join_lines(*lines)

    visit_list = visit_tuple

    def visit_str(self, o, **kwargs):
        return o

    # Handler for IR nodes

    def visit_Intrinsic(self, o, **kwargs):
        """
        Format intrinsic nodes.
        """
        return self.format_line(str(o.text).lstrip())

    def visit_Comment(self, o, **kwargs):
        """
        Format comments.
        """
        text = o.text
        if not text:
            text = o.source.string if o.source else ''
        return self.format_line(str(text).lstrip(), no_wrap=True)

    def visit_Pragma(self, o, **kwargs):
        """
        Format pragmas.
        """
        if o.content is not None:
            # Deconstruct and re-assemble pragma from parameters
            line_cont = f' &\n!${o.keyword} & '
            items = [f'!${o.keyword}']
            for k, v in get_pragma_parameters(o, only_loki_pragmas=False).items():
                items += [k + '(' if v else k]
                if v:
                    # Need to additionally filter all old line continuations
                    items += list(v.replace('&', '').strip().split())
                    items += [')']

            # Ensure '!$<keyword> &' line continuation in final string
            return str(JoinableStringList(items, sep=' ', width=self.linewidth,
                                          cont=line_cont, separable=True))
        return o.source.string

    def visit_CommentBlock(self, o, **kwargs):
        """
        Format comment blocks.
        """
        comments = self.visit_all(o.comments, **kwargs)
        return self.join_lines(*comments)

    def visit_PreprocessorDirective(self, o, **kwargs):
        """
        Format preprocessor directives.
        """
        return self.format_line(str(o.text).lstrip(), no_wrap=True, no_indent=True)

    def visit_VariableDeclaration(self, o, **kwargs):
        """
        Format declaration as
          [<type>] [, DIMENSION(...)] [, EXTERNAL] :: var [= initial] [, var [= initial] ] ...
        """
        attributes = []
        assert len(o.variables) > 0
        types = [v.type for v in o.variables]

        # Ensure all variable types are equal, except for shape and dimension
        # TODO: Should extend to deeper recursion of `variables` if
        # the symbol has a known derived type
        ignore = ['shape', 'dimensions', 'variables', 'source', 'initial']
        if o.external or isinstance(types[0].dtype, ProcedureType):
            # TODO: We can't fully compare forward declarations of functions or statement functions,
            # yet but we can make at least sure other declared attributes are compatible and that all
            # have the same return type
            ignore += ['dtype']
            assert all(t.dtype.return_type == types[0].dtype.return_type or
                       t.dtype.return_type.compare(types[0].dtype.return_type, ignore=ignore) for t in types)

        assert all(t.compare(types[0], ignore=ignore) for t in types)

        dtype = self.visit(types[0], dimensions=o.dimensions, **kwargs)
        if str(dtype):
            attributes += [dtype]
        if o.external:
            attributes += ['EXTERNAL']
        variables = []
        for v in o.variables:
            # This is a bit dubious, but necessary, as we otherwise pick up
            # array dimensions from the internal representation of the variable.
            var = self.visit(v, **kwargs) if o.dimensions is None else v.basename
            initial = ''
            if v.type.initial is not None:
                op = '=>' if v.type.pointer else '='
                initial = f' {op} {self.visit(v.type.initial, **kwargs)}'
            variables += [f'{var}{initial}']
        comment = None
        if o.comment:
            comment = str(self.visit(o.comment, **kwargs))
        return self.format_line(self.join_items(attributes), ' :: ', self.join_items(variables),
                                comment=comment)

    def visit_DataDeclaration(self, o, **kwargs):
        """
        Format as
          DATA <var> /<values>/
        """
        values = self.visit_all(o.values, **kwargs)
        return self.format_line('DATA ', o.variable, '/', values, '/')

    def visit_StatementFunction(self, o, **kwargs):
        """
        Format as
          <variable>(<arguments>) = <rhs>
        """
        name = self.visit(o.variable, **kwargs)
        arguments = self.visit_all(o.arguments, **kwargs)
        rhs = self.visit(o.rhs, **kwargs)
        return self.format_line(name, '(', self.join_items(arguments), ') = ', rhs)

    def visit_Import(self, o, **kwargs):
        """
        Format imports according to their type as
          #include "..."
        or
          include "..."
        or
          USE <module> [, ONLY: <symbols>]
        or
          USE <module> [, <rename-list>]
        """
        if o.c_import:
            return f'#include "{o.module}"'
        if o.f_include:
            return self.format_line('include "', o.module, '"')
        if o.rename_list:
            rename_list = [f'{self.visit(local, **kwargs)} => {use}' for use, local in o.rename_list]
            return self.format_line('USE ', o.module, ', ', self.join_items(rename_list))
        if o.symbols:
            symbols = []
            for s in o.symbols:
                if s.type.use_name:
                    symbols += [f'{self.visit(s, **kwargs)} => {s.type.use_name}']
                else:
                    symbols += [self.visit(s, **kwargs)]
            return self.format_line('USE ', o.module, ', ONLY: ', self.join_items(symbols))
        return self.format_line('USE ', o.module)

    def visit_Interface(self, o, **kwargs):
        """
        Format interface node as
          INTERFACE [<spec>]
            ...body...
          END INTERFACE
        """
        spec = f' {o.spec}' if o.spec else ''
        header = self.format_line('INTERFACE', spec)
        footer = self.format_line('END INTERFACE', spec)
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body, footer)

    def visit_Loop(self, o, **kwargs):
        """
        Format loop with explicit range as
          [name:] DO [label] <var>=<loop range>
            ...body...
          END DO [name]
        """
        pragma = self.visit(o.pragma, **kwargs)
        pragma_post = self.visit(o.pragma_post, **kwargs)
        control = f'{self.visit(o.variable, **kwargs)}={self.visit(o.bounds, **kwargs)}'
        header_name = f'{o.name}: ' if o.name else ''
        label = f'{o.loop_label} ' if o.loop_label else ''
        header = self.format_line(header_name, 'DO ', label, control)
        if o.has_end_do:
            footer_name = f' {o.name}' if o.name else ''
            footer = self.format_line('END DO', footer_name)
            footer = self.apply_label(footer, o.loop_label)
        else:
            footer = None
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(pragma, header, body, footer, pragma_post)

    def visit_WhileLoop(self, o, **kwargs):
        """
        Format loop as
          [name:] DO [label] [WHILE (<condition>)]
            ...body...
          END DO [name]
        """
        pragma = self.visit(o.pragma, **kwargs)
        pragma_post = self.visit(o.pragma_post, **kwargs)
        control = ''
        if o.condition is not None:
            control = f' WHILE ({self.visit(o.condition, **kwargs)})'
        header_name = f'{o.name}: ' if o.name else ''
        label = f' {o.loop_label}' if o.loop_label else ''
        header = self.format_line(header_name, 'DO', label, control)
        if o.has_end_do:
            footer_name = f' {o.name}' if o.name else ''
            footer = self.format_line('END DO', footer_name)
            footer = self.apply_label(footer, o.loop_label)
        else:
            footer = None
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(pragma, header, body, footer, pragma_post)

    def visit_Conditional(self, o, **kwargs):
        """
        Format conditional as
          IF (<condition>) <single-statement body>
        or
          [name:] IF (<condition>) THEN
            ...body...
          [ELSE IF (<condition>) THEN [name]]
            [...body...]
          [ELSE [name]]
            [...body...]
          END IF [name]
        """
        if o.inline:
            # No indentation and only a single body node
            cond = self.visit(o.condition, **kwargs)
            body = self.visit(o.body, **kwargs)
            return self.format_line('IF (', cond, ') ', body)

        name = kwargs.pop('name', f' {o.name}' if o.name else '')
        is_elseif = kwargs.pop('is_elseif', False)

        if is_elseif:
            header = self.format_line('ELSE IF', ' (', self.visit(o.condition, **kwargs), ') THEN', name)
        else:
            header = f'{name[1:]}: IF' if name else 'IF'
            header = self.format_line(header, ' (', self.visit(o.condition, **kwargs), ') THEN')

        self.depth += 1
        body = self.visit(o.body, **kwargs)
        if o.has_elseif:
            self.depth -= 1
            else_body = [self.visit(o.else_body, is_elseif=True, name=name, **kwargs)]
        else:
            else_body = [self.visit(o.else_body, **kwargs)]
            self.depth -= 1
            if o.else_body:
                else_body = [self.format_line('ELSE', name)] + else_body
            else_body += [self.format_line('END IF', name)]

        return self.join_lines(header, body, *else_body)

    def visit_MultiConditional(self, o, **kwargs):
        """
        Format as
          [name:] SELECT CASE (<expr>)
          CASE (<value>) [name]
            ...body...
          [CASE (<value>) [name]]
            [...body...]
          [CASE DEFAULT [name]]
            [...body...]
          END SELECT [name]
        """
        header_name = f'{o.name}: ' if o.name else ''
        header = self.format_line(header_name, 'SELECT CASE (', self.visit(o.expr, **kwargs), ')')
        cases = []
        name = f' {o.name}' if o.name else ''
        for value in o.values:
            case = self.visit_all(as_tuple(value), **kwargs)
            cases.append(self.format_line('CASE (', self.join_items(case), ')', name))
        if o.else_body:
            cases.append(self.format_line('CASE DEFAULT', name))
        footer = self.format_line('END SELECT', name)
        self.depth += 1
        bodies = self.visit_all(*o.bodies, o.else_body, **kwargs)
        self.depth -= 1
        branches = [item for branch in zip(cases, bodies) for item in branch]
        return self.join_lines(header, *branches, footer)

    def visit_Assignment(self, o, **kwargs):
        """
        Format statement as
          <lhs> = <rhs>
        or
          <pointer> => <rhs>
        """
        lhs = self.visit(o.lhs, **kwargs)
        rhs = self.visit(o.rhs, **kwargs)
        comment = None
        if o.comment:
            comment = f'  {self.visit(o.comment, **kwargs)}'
        if o.ptr:
            return self.format_line(lhs, ' => ', rhs, comment=comment)
        return self.format_line(lhs, ' = ', rhs, comment=comment)

    def visit_MaskedStatement(self, o, **kwargs):
        """
        Format masked assignment as
          WHERE (<condition>)
            ...body...
          [ELSEWHERE]
            [...body...]
          END WHERE
        """
        header = self.format_line('WHERE (', self.visit(o.condition, **kwargs), ')')
        footer = self.format_line('END WHERE')
        default_header = self.format_line('ELSEWHERE')
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        default = self.visit(o.default, **kwargs)
        self.depth -= 1
        if o.default:
            return self.join_lines(header, body, default_header, default, footer)
        return self.join_lines(header, body, footer)

    def visit_Section(self, o, **kwargs):
        """
        Format the section's body.
        """
        return self.visit(o.body, **kwargs)

    def visit_Associate(self, o, **kwargs):
        """
        Format scope as
          ASSOCIATE (<associates>)
            ...body...
          END ASSOCIATE
        """
        assocs = [f'{self.visit(a[1], **kwargs)}=>{self.visit(a[0], **kwargs)}' for a in o.associations]
        header = self.format_line('ASSOCIATE (', self.join_items(assocs), ')')
        footer = self.format_line('END ASSOCIATE')
        body = self.visit(o.body, **kwargs)
        return self.join_lines(header, body, footer)

    def visit_CallStatement(self, o, **kwargs):
        """
        Format call statement as
          CALL <name>(<args>)
        """
        pragma = self.visit(o.pragma, **kwargs)
        name = self.visit(o.name, **kwargs)
        args = self.visit_all(o.arguments, **kwargs)
        if o.kwarguments:
            args += tuple(f'{self.visit(arg[0], **kwargs)}={self.visit(arg[1], **kwargs)}' for arg in o.kwarguments)
        call = self.format_line('CALL ', name, '(', self.join_items(args), ')')
        return self.join_lines(pragma, call)

    def visit_Allocation(self, o, **kwargs):
        """
        Format allocation statement as
          ALLOCATE(<variables> [, SOURCE=<source>])
        """
        items = self.visit_all(o.variables, **kwargs)
        if o.data_source is not None:
            items += (f'SOURCE={self.visit(o.data_source, **kwargs)}', )
        if o.status_var is not None:
            items += (f'STAT={self.visit(o.status_var, **kwargs)}', )
        return self.format_line('ALLOCATE (', self.join_items(items), ')')

    def visit_Deallocation(self, o, **kwargs):
        """
        Format de-allocation statement as
          DEALLOCATE(<variables>)
        """
        items = self.visit_all(o.variables, **kwargs)
        if o.status_var is not None:
            items += (f'STAT={self.visit(o.status_var, **kwargs)}', )
        return self.format_line('DEALLOCATE (', self.join_items(items), ')')

    def visit_Nullify(self, o, **kwargs):
        """
        Format pointer nullification as
          NULLIFY(<variables>)
        """
        items = self.visit_all(o.variables, **kwargs)
        return self.format_line('NULLIFY (', self.join_items(items), ')')

    def visit_SymbolAttributes(self, o, **kwargs):
        """
        Format declaration attributes as
          <typename>[(<spec>)] [, <attributes>]
        """
        dimensions = kwargs.pop('dimensions', None)
        attributes = []
        type_map = {BasicType.LOGICAL: 'LOGICAL', BasicType.INTEGER: 'INTEGER',
                    BasicType.REAL: 'REAL', BasicType.CHARACTER: 'CHARACTER',
                    BasicType.COMPLEX: 'COMPLEX', BasicType.DEFERRED: ''}
        if isinstance(o.dtype, ProcedureType):
            if o.dtype.is_function:
                typename = self.visit(o.dtype.return_type, **kwargs)
            else:
                typename = ''
        elif isinstance(o.dtype, DerivedType):
            typename = f'TYPE({o.dtype.name})'
        else:
            typename = type_map[o.dtype]
        if o.length:
            typename += f'(LEN={self.visit(o.length, **kwargs)})'
        if o.kind:
            typename += f'(KIND={self.visit(o.kind, **kwargs)})'
        if typename:
            attributes += [typename]

        if dimensions:
            attributes += [f'DIMENSION({", ".join(self.visit_all(dimensions, **kwargs))})']
        if o.allocatable:
            attributes += ['ALLOCATABLE']
        if o.pointer:
            attributes += ['POINTER']
        if o.value:
            attributes += ['VALUE']
        if o.optional:
            attributes += ['OPTIONAL']
        if o.parameter:
            attributes += ['PARAMETER']
        if o.target:
            attributes += ['TARGET']
        if o.contiguous:
            attributes += ['CONTIGUOUS']
        if o.intent:
            attributes += [f'INTENT({o.intent.upper()})']
        if o.private:
            attributes += ['PRIVATE']
        if o.public:
            attributes += ['PUBLIC']
        return self.join_items(attributes)

    def visit_TypeDef(self, o, **kwargs):
        """
        Format type definition as
          TYPE [, BIND(c) ::] <name>
            ...declarations...
          END TYPE <name>
        """
        bind_c = ', BIND(c) ::' if o.bind_c else ''
        header = self.format_line('TYPE', bind_c, ' ', o.name)
        footer = self.format_line('END TYPE ', o.name)
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body, footer)

    def visit_DerivedType(self, o, **kwargs):
        return o.name

    def visit_ProcedureType(self, o, **kwargs):
        return o.name

def fgen(ir, depth=0, conservative=False, linewidth=132):
    """
    Generate standardized Fortran code from one or many IR objects/trees.
    """
    return FortranCodegen(depth=depth, linewidth=linewidth, conservative=conservative).visit(ir)


"""
Expose the expression generator for testing purposes.
"""
fexprgen = FCodeMapper()
