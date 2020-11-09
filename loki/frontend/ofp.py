from collections import OrderedDict, deque
from collections.abc import Iterable
from pathlib import Path
import re

import open_fortran_parser

from loki.frontend.source import extract_source
from loki.frontend.preprocessing import blacklist
from loki.frontend.util import (
    inline_comments, cluster_comments, inline_pragmas, inline_labels,
    process_dimension_pragmas, import_external_symbols, OFP
)
from loki.visitors import GenericVisitor
import loki.ir as ir
import loki.expression.symbols as sym
from loki.expression.operations import (
    ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow, StringConcat)
from loki.expression import ExpressionDimensionsMapper
from loki.tools import as_tuple, timeit, disk_cached, flatten, gettempdir, filehash, CaseInsensitiveDict
from loki.logging import info, DEBUG
from loki.types import BasicType, SymbolType, DerivedType, ProcedureType, Scope


__all__ = ['parse_ofp_file', 'parse_ofp_source', 'parse_ofp_ast']


@timeit(log_level=DEBUG)
@disk_cached(argname='filename', suffix='ofpast')
def parse_ofp_file(filename):
    """
    Read and parse a source file using the Open Fortran Parser (OFP).

    Note: The parsing is cached on disk in ``<filename>.cache``.
    """
    filepath = Path(filename)
    info("[Frontend.OFP] Parsing %s" % filepath.name)
    return open_fortran_parser.parse(filepath, raise_on_error=True)


@timeit(log_level=DEBUG)
def parse_ofp_source(source, xmods=None):  # pylint: disable=unused-argument
    """
    Read and parse a source string using the Open Fortran Parser (OFP).
    """
    filepath = gettempdir()/filehash(source, prefix='ofp-', suffix='.f90')
    with filepath.open('w') as f:
        f.write(source)

    return parse_ofp_file(filename=filepath)


@timeit(log_level=DEBUG)
def parse_ofp_ast(ast, pp_info=None, raw_source=None, definitions=None, scope=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    # Parse the raw OMNI language AST
    _ir = OFP2IR(definitions=definitions, raw_source=raw_source, scope=scope).visit(ast)

    # Apply postprocessing rules to re-insert information lost during preprocessing
    for r_name, rule in blacklist[OFP].items():
        _info = pp_info[r_name] if pp_info is not None and r_name in pp_info else None
        _ir = rule.postprocess(_ir, _info)

    # Perform some minor sanitation tasks
    _ir = inline_comments(_ir)
    _ir = inline_pragmas(_ir)
    _ir = process_dimension_pragmas(_ir)
    _ir = cluster_comments(_ir)
    _ir = inline_labels(_ir)

    return _ir


def match_tag_sequence(sequence, patterns):
    """
    Return a list of tuples with all AST nodes (xml objects) that
    match a tag pattern in the list of :param patterns:.
    """
    new_sequence = []
    tags = [i.tag for i in sequence]
    for i, _ in enumerate(zip(sequence, tags)):
        for pattern in patterns:
            if tuple(tags[i:i+len(pattern)]) == pattern:
                new_sequence.append(tuple(sequence[i:i+len(pattern)]))
    return new_sequence


class OFP2IR(GenericVisitor):
    # pylint: disable=no-self-use  # Stop warnings about visitor methods that could do without self
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    def __init__(self, raw_source, definitions=None, scope=None):
        super().__init__()

        self._raw_source = raw_source
        self.definitions = CaseInsensitiveDict((d.name, d) for d in as_tuple(definitions))
        self.scope = scope

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        if isinstance(instance, Iterable):
            return super().lookup_method(instance)

        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        return super().lookup_method(instance)

    def get_label(self, o):
        """
        Helper routine to extract the label from one of the many places it could be.
        """
        if o is None or not hasattr(o, 'attrib'):
            return None
        if 'lbl' in o.attrib:
            return o.attrib['lbl']
        if 'label' in o.attrib:
            return o.attrib['label']
        return self.get_label(o.find('label'))

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        if isinstance(o, Iterable):
            return super().visit(o, **kwargs)

        kwargs['label'] = self.get_label(o)

        try:
            kwargs['source'] = extract_source(o.attrib, self._raw_source, label=kwargs['label'])
        except KeyError:
            pass
        return super().visit(o, **kwargs)

    def visit_tuple(self, o, label=None, source=None):
        return as_tuple(flatten(self.visit(c) for c in o))

    visit_list = visit_tuple

    def visit_Element(self, o, label=None, source=None):
        """
        Universal default for XML element types
        """
        children = tuple(self.visit(c) for c in o)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    visit_body = visit_Element

    def visit_loop(self, o, label=None, source=None):
        body = as_tuple(self.visit(o.find('body')))
        # Store full lines with loop body for easy replacement
        source = extract_source(o.attrib, self._raw_source, full_lines=True)
        # Extract loop label if any
        loop_label = o.find('do-stmt').attrib['digitString'] or None
        construct_name = o.find('do-stmt').attrib['id'] or None
        label = self.get_label(o.find('do-stmt'))
        has_end_do = o.find('end-do-stmt') is not None
        term_stmt = o.find('do-term-action-stmt')
        if not has_end_do and term_stmt is not None:
            # Yay, a special case for loops with label
            has_end_do = term_stmt.get('endKeyword') == 'end'

        if o.find('header/index-variable') is None:
            if o.find('do-stmt').attrib['hasLoopControl'] == 'false':
                # We are processing an unbounded do loop
                condition = None
            else:
                # We are processing a while loop
                condition = self.visit(o.find('header'))
            return ir.WhileLoop(condition=condition, body=body, loop_label=loop_label,
                                label=label, name=construct_name, has_end_do=has_end_do,
                                source=source)

        # We are processing a regular for/do loop with bounds
        vname = o.find('header/index-variable').attrib['name']
        variable = sym.Variable(name=vname, scope=self.scope, source=source)
        lower = self.visit(o.find('header/index-variable/lower-bound'))
        upper = self.visit(o.find('header/index-variable/upper-bound'))
        step = None
        if o.find('header/index-variable/step') is not None:
            step = self.visit(o.find('header/index-variable/step'))
        bounds = sym.LoopRange((lower, upper, step), source=source)
        return ir.Loop(variable=variable, body=body, bounds=bounds, loop_label=loop_label,
                       label=label, name=construct_name, has_end_do=has_end_do, source=source)

    def visit_if(self, o, label=None, source=None):
        conditions = tuple(self.visit(h) for h in o.findall('header'))
        bodies = tuple([self.visit(b)] for b in o.findall('body'))
        ncond = len(conditions)
        else_body = bodies[-1] if len(bodies) > ncond else None
        inline = o.find('if-then-stmt') is None
        construct_name = None if inline else o.find('if-then-stmt').attrib['id'] or None
        if not inline:
            label = self.get_label(o.find('if-then-stmt'))
        return ir.Conditional(conditions=conditions, bodies=bodies[:ncond], else_body=else_body,
                              inline=inline, label=label, name=construct_name, source=source)

    def visit_select(self, o, label=None, source=None):
        expr = self.visit(o.find('header'))
        cases = [self.visit(case) for case in o.findall('body/case')]
        values, bodies = zip(*cases)
        if None in values:
            else_index = values.index(None)
            values, bodies = list(values), list(bodies)
            values.pop(else_index)
            else_body = as_tuple(bodies.pop(else_index))
        else:
            else_body = ()
        construct_name = o.find('select-case-stmt').attrib['id'] or None
        label = self.get_label(o.find('select-case-stmt'))
        return ir.MultiConditional(expr=expr, values=as_tuple(values), bodies=as_tuple(bodies),
                                   else_body=else_body, label=label, name=construct_name, source=source)

    def visit_case(self, o, label=None, source=None):
        value = self.visit(o.find('header'))
        if isinstance(value, tuple) and len(value) > int(o.find('header/value-ranges').attrib['count']):
            value = sym.RangeIndex(value, source=source)
        body = self.visit(o.find('body'))
        return value, body

    # TODO: Deal with line-continuation pragmas!
    _re_pragma = re.compile(r'\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

    def visit_comment(self, o, label=None, source=None):
        match_pragma = self._re_pragma.search(source.string)
        if match_pragma:
            # Found pragma, generate this instead
            gd = match_pragma.groupdict()
            return ir.Pragma(keyword=gd['keyword'], content=gd['content'], source=source)
        return ir.Comment(text=o.attrib['text'], label=label, source=source)

    def visit_statement(self, o, label=None, source=None):
        # TODO: Hacky pre-emption for special-case statements
        if o.find('name/nullify-stmt') is not None:
            variable = self.visit(o.find('name'))
            return ir.Nullify(variables=as_tuple(variable), label=label, source=source)
        if o.find('cycle') is not None:
            return self.visit(o.find('cycle'))
        if o.find('where-construct-stmt') is not None:
            # Parse a WHERE statement(s)...
            children = [self.visit(c) for c in o]
            children = [c for c in children if c is not None]

            stmts = []
            while 'ENDWHERE_CONSTRUCT' in children:
                iend = children.index('ENDWHERE_CONSTRUCT')
                w_children = children[:iend]

                condition = w_children[0]
                if 'ELSEWHERE_CONSTRUCT' in w_children:
                    iw = w_children.index('ELSEWHERE_CONSTRUCT')
                    body = w_children[1:iw]
                    default = w_children[iw:]
                else:
                    body = w_children[1:]
                    default = ()

                stmts += [ir.MaskedStatement(condition=condition, body=body, default=default,
                                             label=label, source=source)]
                children = children[iend+1:]

            # TODO: Deal with alternative conditions (multiple ELSEWHERE)
            return as_tuple(stmts)
        if o.find('goto-stmt') is not None:
            target_label = o.find('goto-stmt').attrib['target_label']
            return ir.Intrinsic(text='go to %s' % target_label, label=label, source=source)
        return self.visit_Element(o, label=label, source=source)

    def visit_elsewhere_stmt(self, o, label=None, source=None):
        # Only used as a marker above
        return 'ELSEWHERE_CONSTRUCT'

    def visit_end_where_stmt(self, o, label=None, source=None):
        # Only used as a marker above
        return 'ENDWHERE_CONSTRUCT'

    def visit_assignment(self, o, label=None, source=None):
        lhs = self.visit(o.find('target'))
        rhs = self.visit(o.find('value'))
        return ir.Assignment(lhs=lhs, rhs=rhs, label=label, source=source)

    def visit_pointer_assignment(self, o, label=None, source=None):
        lhs = self.visit(o.find('target'))
        rhs = self.visit(o.find('value'))
        return ir.Assignment(lhs=lhs, rhs=rhs, ptr=True, label=label, source=source)

    def visit_specification(self, o, label=None, source=None):
        body = tuple(self.visit(c) for c in o)
        body = tuple(c for c in body if c is not None)
        # Wrap spec area into a separate Scope
        return ir.Section(body=body, label=label, source=source)

    def visit_declaration(self, o, label=None, source=None):
        if len(o.attrib) == 0:
            return None  # Empty element, skip
        if o.find('save-stmt') is not None:
            return ir.Intrinsic(text=source.string.strip(), label=label, source=source)
        if o.find('implicit-stmt') is not None:
            return ir.Intrinsic(text=source.string.strip(), label=label, source=source)
        if o.find('access-spec') is not None:
            # PUBLIC or PRIVATE declarations
            return ir.Intrinsic(text=source.string.strip(), label=label, source=source)
        if o.attrib['type'] == 'variable':
            if o.find('end-type-stmt') is not None:
                # We are dealing with a derived type
                name = o.find('end-type-stmt').attrib['id']

                # Initialize a local scope for typedef objects
                typedef_scope = Scope(parent=self.scope)

                # This is still ugly, but better than before! In order to
                # process certain tag combinations (groups) into declaration
                # objects, we first group them in place, while also allowing
                # comments/pragmas through here. We then explicitly process
                # them into the intended nodes in the order we found them.
                grouped_elems = match_tag_sequence(o, [
                    ('type', 'attributes', 'components'),
                    ('type', 'components'),
                    ('comment', ),
                ])

                body = []
                for group in grouped_elems:
                    if len(group) == 1:
                        # Process indidividual comments/pragmas
                        body.append(self.visit(group[0]))

                    elif len(group) == 2:
                        # Process declarations without attributes
                        decl = self.create_typedef_declaration(t=group[0], comps=group[1],
                                                               scope=typedef_scope, source=source)
                        body.append(decl)

                    elif len(group) == 3:
                        # Process declarations with attributes
                        decl = self.create_typedef_declaration(t=group[0], attr=group[1], comps=group[2],
                                                               scope=typedef_scope, source=source)
                        body.append(decl)

                    else:
                        raise RuntimeError("OFP: Unknown tag grouping in TypeDef declaration processing")

                # Infer any additional shape information from `!$loki dimension` pragmas
                body = inline_pragmas(body)
                body = process_dimension_pragmas(body)
                typedef = ir.TypeDef(name=name, body=as_tuple(body), scope=typedef_scope,
                                     label=label, source=source)

                # Now make the typedef known in its scope's type table
                self.scope.types[name] = DerivedType(name=name, typedef=typedef)

                return typedef

            # We are dealing with a single declaration, so we retrieve
            # all the declaration-level information first.
            typename = o.find('type').attrib['name']
            kind = o.find('type/kind/name')
            if kind is not None:
                if kind.attrib['id'].isnumeric():
                    kind = sym.Literal(value=kind.attrib['id'])
                else:
                    kind = sym.Variable(name=kind.attrib['id'], scope=self.scope)
            intent = o.find('intent').attrib['type'] if o.find('intent') else None
            allocatable = o.find('attribute-allocatable') is not None
            pointer = o.find('attribute-pointer') is not None
            parameter = o.find('attribute-parameter') is not None
            optional = o.find('attribute-optional') is not None
            target = o.find('attribute-target') is not None
            external = o.find('attribute-external') is not None
            dims = o.find('dimensions')
            dimensions = None if dims is None else as_tuple(self.visit(dims))

            if o.find('type').attrib['type'] == 'intrinsic':
                # Create a basic variable type
                # TODO: Character length attribute
                stype = SymbolType(BasicType.from_fortran_type(typename), kind=kind,
                                   intent=intent, allocatable=allocatable, pointer=pointer,
                                   optional=optional, parameter=parameter, shape=dimensions,
                                   target=target, source=source)
            else:
                # Create the local variant of the derived type
                dtype = self.scope.types.lookup(typename, recursive=True)
                if dtype is None:
                    dtype = DerivedType(name=typename, typedef=BasicType.DEFERRED)

                stype = SymbolType(dtype, intent=intent, allocatable=allocatable,
                                   pointer=pointer, optional=optional, parameter=parameter,
                                   target=target, source=source)

            variables = [self.visit(v, type=stype, dimensions=dimensions, external=external)
                         for v in o.findall('variables/variable')]
            variables = [v for v in variables if v is not None]
            return ir.Declaration(variables=variables, dimensions=dimensions, external=external,
                                  label=label, source=source)
        if o.attrib['type'] == 'external':
            variables = [self.visit(v) for v in o.findall('names/name')]
            for v in variables:
                v.type.external = True
            return ir.Declaration(variables=variables, external=True, label=label, source=source)
        if o.attrib['type'] in ('implicit', 'intrinsic', 'parameter'):
            return ir.Intrinsic(text=source.string.strip(), label=label, source=source)
        if o.attrib['type'] == 'data':
            # Data declaration blocks
            declarations = []
            for variables, values in zip(o.findall('variables'), o.findall('values')):
                # TODO: actually parse the statements
                variable = source.string[5:source.string.index('/')].strip()
                # variable = self.visit(variables)
                # Lists of literal values are again nested, so extract
                # them recursively.
                lit = values.find('literal')  # We explicitly recurse on l
                vals = []
                while lit.find('literal') is not None:
                    vals += [self.visit(lit)]
                    lit = lit.find('literal')
                vals += [self.visit(lit)]
                declarations += [ir.DataDeclaration(variable=variable, values=vals, label=label, source=source)]
            return as_tuple(declarations)

        raise NotImplementedError('Unknown declaration type encountered: %s' % o.attrib['type'])

    def visit_associate(self, o, label=None, source=None):
        associations = OrderedDict()
        for a in o.findall('header/keyword-arguments/keyword-argument'):
            var = self.visit(a.find('name'))
            if isinstance(var, sym.Array):
                shape = ExpressionDimensionsMapper()(var)
            else:
                shape = None
            _type = var.type.clone(name=None, parent=None, shape=shape)
            assoc_name = a.find('association').attrib['associate-name']
            associations[var] = sym.Variable(name=assoc_name, type=_type, scope=self.scope,
                                             source=source)
        body = self.visit(o.find('body'))
        return ir.Associate(body=as_tuple(body), associations=associations, label=label, source=source)

    def visit_allocate(self, o, label=None, source=None):
        variables = as_tuple(self.visit(v) for v in o.findall('expressions/expression/name'))
        kw_args = {arg.attrib['name'].lower(): self.visit(arg)
                   for arg in o.findall('keyword-arguments/keyword-argument')}
        return ir.Allocation(variables=variables, label=label, source=source, data_source=kw_args.get('source'))

    def visit_deallocate(self, o, label=None, source=None):
        variables = as_tuple(self.visit(v) for v in o.findall('expressions/expression/name'))
        return ir.Deallocation(variables=variables, label=label, source=source)

    def visit_use(self, o, label=None, source=None):
        name = o.attrib['name']
        symbol_names = [n.attrib['id'] for n in o.findall('only/name')]
        symbols = None
        if len(symbol_names) > 0:
            module = self.definitions.get(name, None)
            symbols = import_external_symbols(module=module, symbol_names=symbol_names, scope=self.scope)
        return ir.Import(module=name, symbols=symbols, label=label, source=source)

    def visit_directive(self, o, label=None, source=None):
        if '#include' in o.attrib['text']:
            # Straight pipe-through node for header includes (#include ...)
            match = re.search(r'#include\s[\'"](?P<module>.*)[\'"]', o.attrib['text'])
            module = match.groupdict()['module']
            return ir.Import(module=module, c_import=True, source=source)
        return ir.PreprocessorDirective(text=source.string.strip(), source=source)

    def visit_exit(self, o, label=None, source=None):
        stmt_tag = '{}-stmt'.format(o.tag)
        stmt = self.visit(o.find(stmt_tag))
        if o.find('label') is not None:
            stmt._update(label=o.find('label').attrib['lbl'])
        return stmt

    visit_return = visit_exit
    visit_continue = visit_exit
    visit_cycle = visit_exit
    visit_format = visit_exit
    visit_print = visit_exit
    visit_open = visit_exit
    visit_close = visit_exit
    visit_write = visit_exit
    visit_read = visit_exit

    def create_intrinsic_from_source(self, o, attrib_name, label=None, source=None):
        cstart = source.string.lower().find(o.attrib[attrib_name].lower())
        assert cstart != -1
        return ir.Intrinsic(text=source.string[cstart:].strip(), label=label, source=source)

    def visit_exit_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'exitKeyword', label=label, source=source)

    def visit_return_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'keyword', label=label, source=source)

    def visit_continue_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'continueKeyword', label=label, source=source)

    def visit_cycle_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'cycleKeyword', label=label, source=source)

    def visit_format_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'formatKeyword', label=label, source=source)

    def visit_print_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'printKeyword', label=label, source=source)

    def visit_open_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'openKeyword', label=label, source=source)

    def visit_close_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'closeKeyword', label=label, source=source)

    def visit_write_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'writeKeyword', label=label, source=source)

    def visit_read_stmt(self, o, label=None, source=None):
        return self.create_intrinsic_from_source(o, 'readKeyword', label=label, source=source)

    def visit_call(self, o, label=None, source=None):
        # Need to re-think this: the 'name' node already creates
        # a 'Variable', which in this case is wrong...
        name = o.find('name').attrib['id']
        args = tuple(self.visit(i) for i in o.findall('name/subscripts/subscript'))
        kwargs = list([self.visit(i) for i in o.findall('name/subscripts/argument')])
        return ir.CallStatement(name=name, arguments=args, kwarguments=kwargs, label=label, source=source)

    def visit_argument(self, o, label=None, source=None):
        key = o.attrib['name']
        val = self.visit(list(o)[0])
        return key, val

    def visit_label(self, o, label=None, source=None):
        assert label is not None
        return ir.Comment('__STATEMENT_LABEL__', label=label, source=source)

    # Expression parsing below; maye move to its own parser..?

    def visit_name(self, o, label=None, source=None):

        def generate_variable(vname, indices, kwargs, parent, source):
            if vname.upper() == 'RESHAPE':
                # return reshape(indices[0], shape=indices[1])
                raise NotImplementedError()
            if vname.upper() in ['MIN', 'MAX', 'EXP', 'SQRT', 'ABS', 'LOG', 'MOD',
                                 'SELECTED_REAL_KIND', 'ALLOCATED', 'PRESENT',
                                 'SIGN', 'EPSILON']:
                fct_symbol = sym.ProcedureSymbol(vname, scope=self.scope, source=source)
                return sym.InlineCall(fct_symbol, parameters=indices, source=source)
            if vname.upper() in ['REAL', 'INT']:
                kind = kwargs.get('kind', indices[1] if len(indices) > 1 else None)
                return sym.Cast(vname, expression=indices[0], kind=kind, source=source)
            if indices is not None and len(indices) == 0:
                # HACK: We (most likely) found a call out to a C routine
                fct_symbol = sym.ProcedureSymbol(o.attrib['id'], scope=self.scope, source=source)
                return sym.InlineCall(fct_symbol, parameters=indices, source=source)

            if parent is not None:
                basename = vname
                vname = '%s%%%s' % (parent.name, vname)

            _type = self.scope.symbols.lookup(vname, recursive=True)
            if _type and isinstance(_type.dtype, ProcedureType):
                    fct_symbol = sym.ProcedureSymbol(vname, type=_type, scope=self.scope, source=source)
                    return sym.InlineCall(fct_symbol, parameters=indices, source=source)

            # No previous type declaration known for this symbol,
            # see if it's a function call to a known procedure
            if not _type or _type.dtype == BasicType.DEFERRED:
                if self.scope.types.lookup(vname):
                    fct_type = SymbolType(self.scope.types.lookup(vname))
                    fct_symbol = sym.ProcedureSymbol(vname, type=fct_type, scope=self.scope, source=source)
                    return sym.InlineCall(fct_symbol, parameters=indices, source=source)

            # If the (possibly external) struct definitions exist
            # try to derive the type from it.
            if _type is None and parent is not None and parent.type is not None:
                if isinstance(parent.type.dtype, DerivedType) \
                   and parent.type.dtype.typedef is not BasicType.DEFERRED:
                    _type = parent.type.dtype.typedef.variables.get(basename)

            if indices:
                indices = sym.ArraySubscript(indices, source=source)

            var = sym.Variable(name=vname, dimensions=indices, parent=parent,
                               type=_type, scope=self.scope, source=source)
            return var

        # Creating compound variables is a bit tricky, so let's first
        # process all our children and shove them into a deque
        _children = deque(self.visit(c) for c in o)
        _children = deque(c for c in _children if c is not None)

        # Hack: find kwargs for Casts
        kwargs = {i.attrib['name']: i.find('name').attrib['id']
                  for i in o.findall('subscripts/argument')}

        # Now we nest variables, dimensions and sub-variables by
        # walking through our queue of nested symbols
        variable = None
        while len(_children) > 0:
            # Indices sit on the left of their base symbol
            if len(_children) > 0 and isinstance(_children[0], tuple):
                indices = _children.popleft()
            else:
                indices = None

            item = _children.popleft()
            variable = generate_variable(vname=item, indices=indices, kwargs=kwargs,
                                         parent=variable, source=source)
        return variable

    def visit_generic_name_list_part(self, o, source=None, **kwargs):
        return o.attrib['id']

    def visit_variable(self, o, source=None, **kwargs):
        if 'id' not in o.attrib and 'name' not in o.attrib:
            return None
        name = o.attrib['id'] if 'id' in o.attrib else o.attrib['name']
        if o.find('dimensions') is not None:
            dimensions = tuple(self.visit(d) for d in o.find('dimensions'))
            dimensions = tuple(d for d in dimensions if d is not None)
        else:
            dimensions = kwargs.get('dimensions', None)
        _type = kwargs.get('type', None)
        initial = None if o.find('initial-value') is None else self.visit(o.find('initial-value'))
        if _type is not None:
            _type = _type.clone(shape=dimensions, initial=initial)
        if dimensions:
            dimensions = sym.ArraySubscript(dimensions, source=source)
        external = kwargs.get('external')
        if external:
            _type.external = external
        return sym.Variable(name=name, scope=self.scope, dimensions=dimensions,
                            type=_type, source=source)

    def visit_part_ref(self, o, label=None, source=None):
        # Return a pure string, as part of a variable name
        return o.attrib['id']

    def visit_literal(self, o, label=None, source=None):
        boz_literal = o.find('boz-literal-constant')
        if boz_literal is not None:
            return sym.IntrinsicLiteral(boz_literal.attrib['constant'], source=source)

        kwargs = {'source': source}
        value = o.attrib['value']
        _type = o.attrib['type'] if 'type' in o.attrib else None
        if _type is not None:
            tmap = {'bool': BasicType.LOGICAL, 'int': BasicType.INTEGER,
                    'real': BasicType.REAL, 'char': BasicType.CHARACTER}
            _type = tmap[_type] if _type in tmap else BasicType.from_fortran_type(_type)
            kwargs['type'] = _type
        kind_param = o.find('kind-param')
        if kind_param is not None:
            kwargs['kind'] = kind_param.attrib['kind']
        return sym.Literal(value, **kwargs)

    def visit_subscripts(self, o, label=None, source=None):
        return tuple(self.visit(c) for c in o
                     if c.tag in ['subscript', 'name'])

    def visit_subscript(self, o, label=None, source=None):
        # TODO: Drop this entire routine, but beware the base-case!
        if o.find('range'):
            lower, upper, step = None, None, None
            if o.find('range/lower-bound') is not None:
                lower = self.visit(o.find('range/lower-bound'))
            if o.find('range/upper-bound') is not None:
                upper = self.visit(o.find('range/upper-bound'))
            if o.find('range/step') is not None:
                step = self.visit(o.find('range/step'))
            return sym.RangeIndex((lower, upper, step), source=source)
        if 'type' in o.attrib and o.attrib['type'] == "upper-bound-assumed-shape":
            lower = self.visit(o[0])
            return sym.RangeIndex((lower, None, None), source=source)
        if o.find('name'):
            return self.visit(o.find('name'))
        if o.find('literal'):
            return self.visit(o.find('literal'))
        if o.find('operation'):
            return self.visit(o.find('operation'))
        if o.find('array-constructor-values'):
            return self.visit(o.find('array-constructor-values'))
        return sym.RangeIndex((None, None, None), source=source)

    visit_dimension = visit_subscript

    def visit_array_constructor_values(self, o, label=None, source=None):
        values = [self.visit(v) for v in o.findall('value')]
        values = [v for v in values if v is not None]  # Filter empy values
        return sym.LiteralList(values=values, source=source)

    def visit_operation(self, o, label=None, source=None):
        """
        Construct expressions from individual operations, using left-recursion.
        """
        ops = [self.visit(op) for op in o.findall('operator')]
        ops = [str(op).lower() for op in ops if op is not None]  # Filter empty ops
        exprs = [self.visit(c) for c in o.findall('operand')]
        exprs = [e for e in exprs if e is not None]  # Filter empty operands

        # Left-recurse on the list of operations and expressions
        exprs = deque(exprs)
        expression = exprs.popleft()
        for op in ops:

            if op == '+':
                expression = sym.Sum((expression, exprs.popleft()), source=source)
            elif op == '-':
                if len(exprs) > 0:
                    # Binary minus
                    expression = sym.Sum((expression, sym.Product((-1, exprs.popleft()))), source=source)
                else:
                    # Unary minus
                    expression = sym.Product((-1, expression), source=source)
            elif op == '*':
                expression = sym.Product((expression, exprs.popleft()), source=source)
            elif op == '/':
                expression = sym.Quotient(numerator=expression, denominator=exprs.popleft(), source=source)
            elif op == '**':
                expression = sym.Power(base=expression, exponent=exprs.popleft(), source=source)
            elif op in ('==', '.eq.'):
                expression = sym.Comparison(expression, '==', exprs.popleft(), source=source)
            elif op in ('/=', '.ne.'):
                expression = sym.Comparison(expression, '!=', exprs.popleft(), source=source)
            elif op in ('>', '.gt.'):
                expression = sym.Comparison(expression, '>', exprs.popleft(), source=source)
            elif op in ('<', '.lt.'):
                expression = sym.Comparison(expression, '<', exprs.popleft(), source=source)
            elif op in ('>=', '.ge.'):
                expression = sym.Comparison(expression, '>=', exprs.popleft(), source=source)
            elif op in ('<=', '.le.'):
                expression = sym.Comparison(expression, '<=', exprs.popleft(), source=source)
            elif op == '.and.':
                expression = sym.LogicalAnd((expression, exprs.popleft()), source=source)
            elif op == '.or.':
                expression = sym.LogicalOr((expression, exprs.popleft()), source=source)
            elif op == '.not.':
                expression = sym.LogicalNot(expression, source=source)
            elif op == '.eqv.':
                e = (expression, exprs.popleft())
                expression = sym.LogicalOr((sym.LogicalAnd(e), sym.LogicalNot(sym.LogicalOr(e))), source=source)
            elif op == '.neqv.':
                e = (expression, exprs.popleft())
                expression = sym.LogicalAnd((sym.LogicalNot(sym.LogicalAnd(e)), sym.LogicalOr(e)), source=source)
            elif op == '//':
                expression = StringConcat((expression, exprs.popleft()), source=source)
            else:
                raise RuntimeError('OFP: Unknown expression operator: %s' % op)

        if o.find('parenthesized_expr') is not None:
            # Force explicitly parenthesised operations
            if isinstance(expression, sym.Sum):
                expression = ParenthesisedAdd(expression.children, source=source)
            if isinstance(expression, sym.Product):
                expression = ParenthesisedMul(expression.children, source=source)
            if isinstance(expression, sym.Power):
                expression = ParenthesisedPow(expression.base, expression.exponent, source=source)

        assert len(exprs) == 0
        return expression

    def visit_operator(self, o, label=None, source=None):
        return o.attrib['operator']

    def create_typedef_declaration(self, t, comps, attr=None, scope=None, source=None):
        """
        Utility method to create individual declarations from a group of AST nodes.
        """
        attrs = {}
        if attr:
            attrs = [a.attrib['attrKeyword'].upper()
                     for a in attr.findall('attribute/component-attr-spec')]
        typename = t.attrib['name']
        t_source = extract_source(t.attrib, self._raw_source)
        kind = t.find('kind/name')
        if kind is not None:
            if kind.attrib['id'].isnumeric():
                kind = sym.Literal(value=kind.attrib['id'])
            else:
                kind = sym.Variable(name=kind.attrib['id'], scope=self.scope)
        # We have an intrinsic Fortran type
        if t.attrib['type'] == 'intrinsic':
            stype = SymbolType(BasicType.from_fortran_type(typename), kind=kind,
                               pointer='POINTER' in attrs,
                               allocatable='ALLOCATABLE' in attrs, source=t_source)
        else:
            # This is a derived type. Let's see if we know it already
            dtype = self.scope.types.lookup(typename, recursive=True)
            if dtype is None:
                dtype = DerivedType(name=typename, typedef=BasicType.DEFERRED)
            stype = SymbolType(dtype, kind=kind, pointer='POINTER' in attrs,
                               allocatable='ALLOCATABLE' in attrs,
                               variables=OrderedDict(), source=t_source)

        # Derive variables for this declaration entry
        variables = []
        for v in comps.findall('component'):
            if len(v.attrib) == 0:
                continue
            if 'DIMENSION' in attrs:
                # Dimensions are provided via `dimension` keyword
                attrib = attr.findall('attribute')[attrs.index('DIMENSION')]
                deferred_shape = attrib.find('deferred-shape-spec-list')
            else:
                deferred_shape = v.find('deferred-shape-spec-list')
            if deferred_shape is not None:
                dim_count = int(deferred_shape.attrib['count'])
                dimensions = [sym.RangeIndex((None, None, None), source=source)
                              for _ in range(dim_count)]
            else:
                dimensions = as_tuple(self.visit(c) for c in v)
            dimensions = as_tuple(d for d in dimensions if d is not None)
            dimensions = dimensions if len(dimensions) > 0 else None
            v_source = extract_source(v.attrib, self._raw_source)
            v_type = stype.clone(shape=dimensions, source=v_source)
            v_name = v.attrib['name']
            if dimensions:
                dimensions = sym.ArraySubscript(dimensions, source=source) if dimensions else None

            variables += [sym.Variable(name=v_name, type=v_type, dimensions=dimensions,
                                       scope=scope, source=source)]

        return ir.Declaration(variables=variables, source=t_source)
