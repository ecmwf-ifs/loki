# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines
from collections import deque
from collections.abc import Iterable
from pathlib import Path
import re
from codetiming import Timer

try:
    import open_fortran_parser

    HAVE_OFP = True
    """Indicate whether OpenFortranParser frontend is available."""
except ImportError:
    HAVE_OFP = False

from loki.frontend.source import extract_source, extract_source_from_range
from loki.frontend.preprocessing import sanitize_registry
from loki.frontend.util import OFP, sanitize_ir
from loki.visitors import GenericVisitor
from loki import ir
import loki.expression.symbols as sym
from loki.expression.operations import (
    ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow, ParenthesisedDiv,
    StringConcat
)
from loki.expression import ExpressionDimensionsMapper, AttachScopesMapper
from loki.tools import (
    as_tuple, disk_cached, flatten, gettempdir, filehash, CaseInsensitiveDict,
)
from loki.pragma_utils import attach_pragmas, process_dimension_pragmas, detach_pragmas, pragmas_attached
from loki.logging import debug, info, warning, error
from loki.types import BasicType, DerivedType, ProcedureType, SymbolAttributes
from loki.config import config


__all__ = ['HAVE_OFP', 'parse_ofp_file', 'parse_ofp_source', 'parse_ofp_ast']


@Timer(logger=debug, text=lambda s: f'[Loki::OFP] Executed parse_ofp_file in {s:.2f}s')
@disk_cached(argname='filename', suffix='ofpast')
def parse_ofp_file(filename):
    """
    Read and parse a source file using the Open Fortran Parser (OFP).

    Note: The parsing is cached on disk in ``<filename>.cache``.
    """
    if not HAVE_OFP:
        error('OpenFortranParser is not available.')

    filepath = Path(filename)
    info(f'[Loki::OFP] Parsing {filepath}')
    return open_fortran_parser.parse(filepath, raise_on_error=True)


@Timer(logger=debug, text=lambda s: f'[Loki::OFP] Executed parse_ofp_source in {s:.2f}s')
def parse_ofp_source(source, filepath=None):
    """
    Read and parse a source string using the Open Fortran Parser (OFP).
    """
    # Use basename of filepath if given
    if filepath is None:
        filepath = Path(filehash(source, prefix='ofp-', suffix='.f90'))
    else:
        filepath = filepath.with_suffix(f'.ofp{filepath.suffix}')

    # Always store intermediate flies in tmp dir
    filepath = gettempdir()/filepath.name

    debug(f'[Loki::OFP] Writing temporary source {filepath}')
    with filepath.open('w') as f:
        f.write(source)

    return parse_ofp_file(filename=filepath)


@Timer(logger=debug, text=lambda s: f'[Loki::OFP] Executed parse_ofp_ast in {s:.2f}s')
def parse_ofp_ast(ast, pp_info=None, raw_source=None, definitions=None, scope=None):
    """
    Generate an internal IR from the raw OFP parser AST.
    """
    # Parse the raw OFP language AST
    _ir = OFP2IR(definitions=definitions, raw_source=raw_source, pp_info=pp_info, scope=scope).visit(ast)

    # Apply postprocessing rules to re-insert information lost during preprocessing
    # and perform some minor sanitation tasks
    _ir = sanitize_ir(_ir, OFP, pp_registry=sanitize_registry[OFP], pp_info=pp_info)

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
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    def __init__(self, raw_source, definitions=None, pp_info=None, scope=None):
        super().__init__()

        self._raw_source = raw_source
        self.definitions = CaseInsensitiveDict((d.name, d) for d in as_tuple(definitions))
        self.pp_info = pp_info
        self.default_scope = scope

    @staticmethod
    def warn_or_fail(msg):
        if config['frontend-strict-mode']:
            error(msg)
            raise NotImplementedError
        warning(msg)

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

    def get_source(self, o, label=None):
        """Helper method that builds the source object for a node"""
        try:
            source = extract_source(o.attrib, self._raw_source, label=label)
        except KeyError:
            source = None
        return source

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        if isinstance(o, Iterable):
            return super().visit(o, **kwargs)

        kwargs['label'] = self.get_label(o)
        kwargs.setdefault('scope', self.default_scope)
        kwargs['source'] = self.get_source(o, kwargs['label'])
        return super().visit(o, **kwargs)

    def visit_tuple(self, o, **kwargs):
        return as_tuple(flatten(self.visit(c, **kwargs) for c in o))

    visit_list = visit_tuple

    def visit_Element(self, o, label=None, source=None, **kwargs):
        """
        Universal default for XML element types
        """
        children = tuple(self.visit(c, **kwargs) for c in o)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_file(self, o, **kwargs):
        body = [self.visit(c, **kwargs) for c in o]
        return ir.Section(body=as_tuple(body))

    def visit_specification(self, o, **kwargs):
        body = tuple(self.visit(c, **kwargs) for c in o)
        body = tuple(c for c in body if c is not None)
        return ir.Section(body=body, label=kwargs['label'], source=kwargs['source'])

    def visit_body(self, o, **kwargs):
        body = tuple(self.visit(c, **kwargs) for c in o)
        body = tuple(c for c in body if c is not None)
        return body

    def visit_loop(self, o, **kwargs):
        body = as_tuple(self.visit(o.find('body'), **kwargs))
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
                condition = self.visit(o.find('header'), **kwargs)
            return ir.WhileLoop(condition=condition, body=body, loop_label=loop_label,
                                name=construct_name, has_end_do=has_end_do,
                                label=label, source=source)

        # We are processing a regular for/do loop with bounds
        vname = o.find('header/index-variable').attrib['name']
        variable = sym.Variable(name=vname, source=source)
        lower = self.visit(o.find('header/index-variable/lower-bound'), **kwargs)
        upper = self.visit(o.find('header/index-variable/upper-bound'), **kwargs)
        step = None
        if o.find('header/index-variable/step') is not None:
            step = self.visit(o.find('header/index-variable/step'), **kwargs)
        bounds = sym.LoopRange((lower, upper, step), source=source)
        return ir.Loop(variable=variable, body=body, bounds=bounds, loop_label=loop_label,
                       label=label, name=construct_name, has_end_do=has_end_do,
                       source=source)

    def visit_if(self, o, **kwargs):
        # process all conditions and bodies
        conditions = [self.visit(h, **kwargs) for h in o.findall('header')]
        bodies = [as_tuple(self.visit(b, **kwargs)) for b in o.findall('body')]
        ncond = len(conditions)
        if len(bodies) > ncond:
            else_body = bodies[-1]
            bodies = bodies[:-1]
        else:
            else_body = None
        assert ncond == len(bodies)
        # shortcut for inline conditionals
        if o.find('if-then-stmt') is None:
            assert ncond == 1 and else_body is None
            return ir.Conditional(condition=conditions[0], body=bodies[0], else_body=(),
                                  inline=True, has_elseif=False, label=kwargs['label'],
                                  source=kwargs['source'])
        # extract labels, names and source
        lend, cend = int(o.attrib['line_end']), int(o.attrib['col_end'])
        names, labels, sources = [], [], []
        for stmt in reversed(o.findall('else-if-stmt')):
            names += [None]
            labels += [self.get_label(stmt)]
            lstart, cstart = int(stmt.attrib['line_begin']), int(stmt.attrib['col_begin'])
            sources += [extract_source_from_range((lstart, lend), (cstart, cend), self._raw_source, label=labels[-1])]
        names += [o.find('if-then-stmt').attrib['id'] or None]
        labels += [self.get_label(o.find('if-then-stmt'))]
        sources += [kwargs['source']]
        # build IR nodes from inside out, using else-if branches as else bodies
        conditions.reverse()
        bodies.reverse()
        node = ir.Conditional(condition=conditions[0], body=bodies[0], else_body=else_body,
                              inline=False, has_elseif=False, label=kwargs['label'],
                              name=names[0], source=sources[0])
        for idx in range(1, ncond):
            node = ir.Conditional(condition=conditions[idx], body=bodies[idx], else_body=(node,),
                                  inline=False, has_elseif=True, label=labels[idx], name=names[idx],
                                  source=sources[idx])
        return node

    def visit_select(self, o, **kwargs):
        expr = self.visit(o.find('header'), **kwargs)
        body = o.find('body')
        case_stmts, case_stmt_index = zip(*[(c, i) for i, c in enumerate(body) if c.tag == 'case'])
        values, bodies = zip(*[self.visit(c, **kwargs) for c in case_stmts])
        if None in values:
            else_index = values.index(None)
            else_body = as_tuple(bodies[else_index])
            values = values[:else_index] + values[else_index+1:]
            bodies = bodies[:else_index] + bodies[else_index+1:]
        else:
            else_body = ()
        # Retain comments before the first case statement
        pre = as_tuple(self.visit(c, **kwargs) for c in body[:case_stmt_index[0]])
        # Retain any comments in-between cases
        bodies = list(bodies)
        for case_idx, stmt_idx in enumerate(case_stmt_index[1:]):
            start_idx = case_stmt_index[case_idx] + 1
            bodies[case_idx-1] += as_tuple(self.visit(c, **kwargs) for c in body[start_idx:stmt_idx])
        bodies = as_tuple(bodies)
        construct_name = o.find('select-case-stmt').attrib['id'] or None
        label = self.get_label(o.find('select-case-stmt'))
        return (
            *pre,
            ir.MultiConditional(expr=expr, values=values, bodies=bodies, else_body=else_body,
                                label=label, name=construct_name, source=kwargs['source'])
        )

    def visit_case(self, o, **kwargs):
        value = self.visit(o.find('header'), **kwargs)
        if isinstance(value, tuple) and len(value) > int(o.find('header/value-ranges').attrib['count']):
            value = sym.RangeIndex(value, source=kwargs['source'])
        body = self.visit(o.find('body'), **kwargs)
        return as_tuple(value) or None, as_tuple(body)

    # TODO: Deal with line-continuation pragmas!
    _re_pragma = re.compile(r'^\s*\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

    def visit_comment(self, o, **kwargs):
        match_pragma = self._re_pragma.search(kwargs['source'].string)
        if match_pragma:
            # Found pragma, generate this instead
            gd = match_pragma.groupdict()
            return ir.Pragma(keyword=gd['keyword'], content=gd['content'], source=kwargs['source'])
        return ir.Comment(text=o.attrib['text'], label=kwargs['label'], source=kwargs['source'])

    def visit_statement(self, o, **kwargs):
        # TODO: Hacky pre-emption for special-case statements
        if o.find('name/nullify-stmt') is not None:
            variable = self.visit(o.find('name'), **kwargs)
            return ir.Nullify(variables=as_tuple(variable), label=kwargs['label'], source=kwargs['source'])
        if o.find('cycle') is not None:
            return self.visit(o.find('cycle'), **kwargs)
        if o.find('where-construct-stmt') is not None or o.find('where-stmt') is not None:
            # Parse WHERE statement(s)...
            # They can appear multiple times within a single statement node and
            # alongside other, non-WHERE-statement nodes. Conveniently, conditions
            # and body are also flat in the node and therefore not marked explicitly.
            # We have to step through them and do our best at picking them out...
            children = [self.visit(c, **kwargs) for c in o]
            children = [c for c in children if c is not None]

            stmts = []
            # Pick out all nodes that belong to this WHERE construct
            # TODO: this breaks for nested where constructs...
            while True:
                # Find position of relevant constructs
                if 'ENDWHERE_CONSTRUCT' in children:
                    construct_start = children.index('WHERE_CONSTRUCT')
                else:
                    construct_start = len(children)
                if 'WHERE_STATEMENT' in children:
                    statement_start = children.index('WHERE_STATEMENT')
                else:
                    statement_start = len(children)

                # Start with where construct if it comes first
                if construct_start < statement_start:
                    if construct_start > 1:
                        # There's stuff before that we retain flat
                        stmts += children[:construct_start-1]

                    # That's the stuff that belongs to this construct
                    iend = children.index('ENDWHERE_CONSTRUCT')
                    w_children = children[construct_start-1:iend]

                    # That's the stuff that follows after
                    children = children[iend+1:]

                    # condition of the WHERE statement
                    conditions = [w_children[0]]
                    bodies = []  # picking out bodies lags behind
                    assert w_children[1] == 'WHERE_CONSTRUCT'
                    w_children = w_children[2:]

                    # any ELSEWHERE statements with a condition?
                    while 'MASKED_ELSEWHERE_CONSTRUCT' in w_children:
                        iw = w_children.index('MASKED_ELSEWHERE_CONSTRUCT')
                        bodies += [as_tuple(w_children[:iw-1])]  # that's the body from the previous case
                        conditions += [w_children[iw-1]]  # that's the condition for the next case
                        w_children = w_children[iw+1:]

                    # any ELSEWHERE statement without a condition?
                    if 'ELSEWHERE_CONSTRUCT' in w_children:
                        iw = w_children.index('ELSEWHERE_CONSTRUCT')
                        bodies += [as_tuple(w_children[:iw])]  # again, body from before
                        default = as_tuple(w_children[iw+1:]) # The default body

                    else:  # No ELSEWHERE, so only body for previous conditional
                        bodies += [as_tuple(w_children)]
                        default = ()

                    # Build the masked statement
                    stmts += [
                        ir.MaskedStatement(
                            conditions=as_tuple(conditions), bodies=as_tuple(bodies), default=default,
                            label=kwargs['label'], source=kwargs['source']
                        )
                    ]

                # Where-statement comes first
                elif statement_start < len(children):
                    # This is always the sequence of [condition, assignment, WHERE_STATEMENT]
                    if statement_start > 2:
                        # There's stuff before that we retain flat
                        stmts += children[:statement_start-2]

                    # Pick out condition and body, update remaining children list
                    conditions = [children[statement_start-2]]
                    bodies = [as_tuple(children[statement_start-1])]
                    assert children[statement_start] == 'WHERE_STATEMENT'
                    children = children[statement_start+1:]

                    # Build the masked statement
                    stmts += [
                        ir.MaskedStatement(
                            conditions=as_tuple(conditions), bodies=as_tuple(bodies), default=(),
                            inline=True, label=kwargs['label'], source=kwargs['source']
                        )
                    ]

                else: # Found neither: terminate loop
                    break

            if children:
                stmts += children
            return as_tuple(stmts)

        if o.find('goto-stmt') is not None:
            target_label = o.find('goto-stmt').attrib['target_label']
            return ir.Intrinsic(text=f'go to {target_label}', label=kwargs['label'], source=kwargs['source'])
        return self.visit_Element(o, **kwargs)

    def visit_where_construct_stmt(self, o, **kwargs):
        # Only used as a marker above
        return 'WHERE_CONSTRUCT'

    def visit_masked_elsewhere_stmt(self, o, **kwargs):
        # Only used as a marker above
        return 'MASKED_ELSEWHERE_CONSTRUCT'

    def visit_elsewhere_stmt(self, o, **kwargs):
        # Only used as a marker above
        return 'ELSEWHERE_CONSTRUCT'

    def visit_end_where_stmt(self, o, **kwargs):
        # Only used as a marker above
        return 'ENDWHERE_CONSTRUCT'

    def visit_where_stmt(self, o, **kwargs):
        # Only used as a marker above
        return 'WHERE_STATEMENT'

    def visit_assignment(self, o, **kwargs):
        lhs = self.visit(o.find('target'), **kwargs)
        rhs = self.visit(o.find('value'), **kwargs)
        return ir.Assignment(lhs=lhs, rhs=rhs, label=kwargs['label'], source=kwargs['source'])

    def visit_pointer_assignment(self, o, **kwargs):
        lhs = self.visit(o.find('target'), **kwargs)
        rhs = self.visit(o.find('value'), **kwargs)
        return ir.Assignment(lhs=lhs, rhs=rhs, ptr=True, label=kwargs['label'], source=kwargs['source'])

    def visit_type(self, o, **kwargs):
        if o.attrib['type'] == 'intrinsic':
            _type = self.visit(o.find('intrinsic-type-spec'), **kwargs)
            if o.attrib['hasKind'] == 'true':
                kind = self.visit(o.find('kind'), **kwargs)
                _type = _type.clone(kind=kind)
            if o.find('length') is not None:
                _type = _type.clone(length=self.visit(o.find('length'), **kwargs))
            return _type

        if o.attrib['type'] == 'derived':
            dtype = self.visit(o.find('derived-type-spec'), **kwargs)

            # Look for a previous definition of this type
            _type = kwargs['scope'].symbol_attrs.lookup(dtype.name)
            if _type is None or _type.dtype is BasicType.DEFERRED:
                _type = SymbolAttributes(dtype)

            decl_spec = o.find('declaration-type-spec')
            if decl_spec is not None and decl_spec.get('udtKeyword') == 'class':
                _type = _type.clone(polymorphic=True)

            # Strip import annotations
            return _type.clone(imported=None, module=None)

        raise ValueError(f'Unknown type {o.attrib("type")}')

    def visit_intrinsic_type_spec(self, o, **kwargs):
        dtype = BasicType.from_str(o.attrib['keyword1'])
        return SymbolAttributes(dtype)

    def visit_derived_type_spec(self, o, **kwargs):
        dtype = DerivedType(o.attrib['typeName'])
        return dtype

    def visit_kind(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_kind_selector(self, o, **kwargs):
        assert o.attrib['token1'] == '*'
        return sym.IntLiteral(int(o.attrib['token2']))

    def visit_attribute_parameter(self, o, **kwargs):
        return self.visit(o.findall('attr-spec'), **kwargs)

    visit_attribute_pointer = visit_attribute_parameter

    def visit_attr_spec(self, o, **kwargs):
        return (o.attrib['attrKeyword'].lower(), True)

    def visit_type_attr_spec(self, o, **kwargs):
        if o.attrib['keyword'].lower() == 'extends':
            return ('extends', o.attrib['id'].lower())
        return (o.attrib['keyword'].lower(), True)

    def visit_intent(self, o, **kwargs):
        return ('intent', o.attrib['type'].lower())

    def visit_access_spec(self, o, **kwargs):
        return (o.attrib['keyword'], True)

    def visit_type_param_value(self, o, **kwargs):
        if o.attrib['hasAsterisk'] == 'true':
            return '*'
        if o.attrib['hasColon'] == 'true':
            return ':'
        return None

    def visit_entity_decl(self, o, **kwargs):
        return sym.Variable(name=o.attrib['id'], source=kwargs['source'])

    def visit_variables(self, o, **kwargs):
        count = int(o.attrib['count'])
        return self.visit(o.findall('variable')[:count], **kwargs)

    def visit_variable(self, o, **kwargs):
        var = self.visit(o.find('entity-decl'), **kwargs)

        if o.attrib['hasInitialValue'] == 'true':
            init = self.visit(o.find('initial-value'), **kwargs)
            var = var.clone(type=var.type.clone(initial=init))

        dimensions = o.find('dimensions')
        if dimensions is not None:
            dimensions = self.visit(dimensions, **kwargs)
            var = var.clone(dimensions=dimensions, type=var.type.clone(shape=dimensions))

        length = o.find('length')
        if length is not None:
            length = self.visit(length, **kwargs)
            var = var.clone(type=var.type.clone(length=length))

        return var

    def visit_procedures(self, o, **kwargs):
        count = int(o.attrib['count'])
        nodes = [self.visit(c, **kwargs) for c in o]
        nodes = [c for c in nodes if c is not None]
        symbols = []
        initial = None
        for c in nodes:
            if not isinstance(c, sym.TypedSymbol):
                assert initial is None
                initial = c
            else:
                if initial is not None:
                    symbols += [c.clone(type=c.type.clone(initial=initial))]
                    initial = None
                else:
                    symbols += [c]
        assert initial is None
        assert len(symbols) == count
        return symbols

    def visit_procedure(self, o, **kwargs):
        var = self.visit(o.find('proc-decl'), **kwargs)
        return var

    def visit_null_init(self, o, **kwargs):
        name = sym.Variable(name='NULL')
        return sym.InlineCall(name, parameters=(), source=kwargs['source'])

    visit_proc_decl = visit_entity_decl
    visit_proc_interface = visit_entity_decl

    def visit_specific_binding(self, o, **kwargs):
        # pylint: disable=no-member  # There is no Variable.member but Variable is a factory
        scope = kwargs['scope']
        attrs = kwargs.pop('attrs')

        # Name of the binding
        var = sym.Variable(name=o.attrib['bindingName'], source=kwargs['source'])

        interface = o.attrib['interfaceName'] or None
        if interface is not None:
            # Interface provided (PROCEDURE(<interface>))
            assert not o.attrib['procedureName']

            # Figure out interface type
            interface_scope = scope.get_symbol_scope(interface)
            if interface_scope is not None:
                interface = sym.Variable(name=interface, scope=interface_scope)
            else:
                interface = sym.Variable(
                    name=interface,
                    type=SymbolAttributes(ProcedureType(interface))
                )

            _type = interface.type

        elif o.attrib['procedureName']:
            # Binding provided (<bindingName> => <procedureName>)
            procedure_scope = scope.get_symbol_scope(o.attrib['procedureName'])
            bind_name = sym.Variable(name=o.attrib['procedureName'], scope=procedure_scope)
            _type = bind_name.type.clone(bind_names=as_tuple(bind_name))

        else:
            # Binding has the same name as procedure
            _type = scope.symbol_attrs.lookup(var.name)
            if _type is None:
                _type = SymbolAttributes(ProcedureType(var.name))

        if attrs:
            _type = _type.clone(**attrs)

        # Update symbol table and rescope symbols
        scope.symbol_attrs[var.name] = _type
        var = var.rescope(scope=scope)
        return ir.ProcedureDeclaration(
            symbols=(var,), interface=interface, source=kwargs['source'], label=kwargs['label']
        )

    def visit_generic_binding(self, o, **kwargs):
        scope = kwargs['scope']
        spec = kwargs.pop('spec')
        attrs = kwargs.pop('attrs')
        names = kwargs.pop('names')

        var = self.visit(spec, **kwargs)
        bind_names = [self.visit(name, **kwargs) for name in names]
        bind_names = AttachScopesMapper()(bind_names, scope=scope)
        _type = SymbolAttributes(ProcedureType(var.name, is_generic=True), bind_names=as_tuple(bind_names))

        if attrs:
            _type = _type.clone(**attrs)
        scope.symbol_attrs[var.name] = _type
        var = var.rescope(scope=scope)
        return ir.ProcedureDeclaration(
            symbols=(var,), generic=True, source=kwargs['source'], label=kwargs['label']
        )

    def visit_final_binding(self, o, **kwargs):
        scope = kwargs['scope']
        spec = kwargs.pop('spec')
        attrs = kwargs.pop('attrs')
        names = kwargs.pop('names')

        assert spec is None
        assert not attrs

        symbols = [self.visit(name, **kwargs) for name in names]
        symbols = AttachScopesMapper()(symbols, scope=scope)

        return ir.ProcedureDeclaration(
            symbols=symbols, final=True, source=kwargs['source'], label=kwargs['label']
        )

    def visit_private_components_stmt(self, o, **kwargs):
        return ir.Intrinsic(o.attrib['privateKeyword'], source=kwargs['source'], label=kwargs['label'])

    def visit_sequence_stmt(self, o, **kwargs):
        return ir.Intrinsic(o.attrib['sequenceKeyword'], source=kwargs['source'], label=kwargs['label'])

    visit_binding_private_stmt = visit_private_components_stmt

    def visit_declaration(self, o, **kwargs):
        label = kwargs['label']
        source = kwargs['source']
        if not o.attrib:
            return None  # Skip empty declarations

        # Dispatch to certain other declarations
        if not 'type' in o.attrib:
            if o.find('access-spec') is not None:
                # access-stmt for module
                from loki.module import Module  # pylint: disable=import-outside-toplevel,cyclic-import
                assert isinstance(kwargs['scope'], Module)
                access_spec = o.find('access-spec').attrib['keyword'].lower()
                assert access_spec in ('public', 'private')
                names = o.findall('name')
                if not names:
                    # default access specification
                    kwargs['scope'].default_access_spec = access_spec
                else:
                    names = [name.attrib['id'].lower() for name in names]
                    if access_spec == 'public':
                        kwargs['scope'].public_access_spec += as_tuple(names)
                    else:
                        kwargs['scope'].private_access_spec += as_tuple(names)
                return None

            if o.find('save-stmt') is not None:
                return ir.Intrinsic(text=source.string.strip(), label=label, source=source)
            if o.find('interface') is not None:
                return self.visit(o.find('interface'), **kwargs)
            if o.find('subroutine') is not None:
                return self.visit(o.find('subroutine'), **kwargs)
            if o.find('function') is not None:
                return self.visit(o.find('function'), **kwargs)
            if o.find('module-nature') is not None:
                return self.visit(o.find('module-nature'), **kwargs)
            if o.find('enum-def-stmt') is not None:
                return self.create_enum(o, **kwargs)
            raise ValueError('Unsupported declaration')
        if o.attrib['type'] in ('implicit', 'intrinsic', 'parameter'):
            return ir.Intrinsic(text=source.string.strip(), label=label, source=source)

        if o.attrib['type'] == 'external':
            # External stmt (not as attribute in a declaration)
            assert o.find('external-stmt') is not None
            assert o.find('type') is None

            variables = self.visit(o.findall('names'), **kwargs)
            for var in variables:
                _type = kwargs['scope'].symbol_attrs.lookup(var.name)
                if _type is None:
                    _type = SymbolAttributes(dtype=ProcedureType(var.name, is_function=False), external=True)
                else:
                    _type = _type.clone(external=True)
                kwargs['scope'].symbol_attrs[var.name] = _type

            variables = tuple(v.clone(scope=kwargs['scope']) for v in variables)
            declaration = ir.ProcedureDeclaration(symbols=variables, external=True, source=source, label=label)
            return declaration

        if o.attrib['type'] == 'data':
            self.warn_or_fail('data declaration not implemented')
            return ir.Intrinsic(text=source.string.strip(), label=label, source=source)

        if o.find('derived-type-stmt') is not None:
            # Derived type definition
            type_stmt = o.find('derived-type-stmt')

            # Derived type attributes
            if type_stmt.attrib['hasTypeAttrSpecList'] == 'true':
                attrs = dict(self.visit(attr, **kwargs) for attr in o.findall('type-attr-spec'))
                abstract = attrs.get('abstract', False)
                extends = attrs.get('extends', None)
                bind_c = attrs.get('bind', False)
                access_spec = o.find('access-spec')
                private = access_spec is not None and access_spec.attrib['keyword'].lower() == 'private'
                public = access_spec is not None and access_spec.attrib['keyword'].lower() == 'public'
            else:
                abstract = False
                extends = None
                bind_c = False
                private = False
                public = False

            # Instantiate the TypeDef without its body
            # Note: This creates the symbol table for the declarations and
            # the typedef object registers itself in the parent scope
            typedef = ir.TypeDef(
                name=type_stmt.attrib['id'], body=(), abstract=abstract, extends=extends, bind_c=bind_c,
                private=private, public=public, label=label, source=source, parent=kwargs['scope'])
            kwargs['scope'] = typedef

            body = []
            if o.find('sequence-stmt') is not None:
                body.append(self.visit(o.find('sequence-stmt'), **kwargs))
            if o.find('private-components-stmt') is not None:
                body.append(self.visit(o.find('private-components-stmt'), **kwargs))

            # Less pretty than before but due to variable components being grouped
            # and procedure components not, we have to step through children and
            # collect type, attributes, etc. along the way.

            contains_stmt = o.find('contains-stmt')
            if contains_stmt is not None:
                contains_idx = list(o).index(contains_stmt)
            else:
                contains_idx = len(list(o))

            # For variable declarations
            t = None
            attr = None

            # For procedure declarations
            iface = None
            proc_attrs = []

            for c in o[:contains_idx]:
                # Variable declarations
                if c.tag == 'type':
                    assert iface is None and not proc_attrs
                    t = c
                elif c.tag == 'attributes':
                    assert iface is None and not proc_attrs
                    attr = c
                elif c.tag == 'components':
                    assert iface is None and not proc_attrs
                    decl = self.create_typedef_variable_declaration(
                        t=t, comps=c, attr=attr, scope=typedef, source=source
                    )
                    body.append(decl)
                    t = None
                    attr = None

                # Procedure declarations
                elif c.tag == 'proc-interface':
                    assert t is None and attr is None
                    iface = c
                elif c.tag == 'proc-component-attr-spec':
                    assert t is None and attr is None
                    proc_attrs += [c]
                elif c.tag == 'procedures':
                    assert t is None and attr is None
                    decl = self.create_typedef_procedure_declaration(
                        iface=iface, comps=c, attrs=proc_attrs, scope=typedef, source=source
                    )
                    body.append(decl)
                    iface = None
                    proc_attrs = []

                elif c.tag in ('comment',):  # Add here other node types. Unfortunately we can't allow all
                                             # because some produce spurious nodes that have been dealt with before
                    node = self.visit(c, **kwargs)
                    if node is not None:
                        body += [node]

            assert t is None and attr is None
            assert iface is None and not proc_attrs

            if contains_stmt is not None:
                # The derived type contains type-bound procedures
                body.append(self.visit(contains_stmt, **kwargs))
                start_idx = contains_idx

                # Once again, this is _hell_ with OFP because binding attributes are not
                # grouped (like they are for components) and therefore we have to step through
                # the flat list of items, picking up the grouped names for generic/final-bindings
                # on the way...
                attrs = {}
                names = []
                generic_spec = None
                for i in o[start_idx:]:
                    if i.tag in ('comment', 'binding-private-stmt'):
                        body.append(self.visit(i, **kwargs))

                    if i.tag == 'binding-attr':
                        if i.attrib['bindingAttr'].upper() == 'PASS':
                            attrs['pass_attr'] = i.attrib['id'] or True
                        elif i.attrib['bindingAttr'].upper() == 'NOPASS':
                            attrs['pass_attr'] = False
                        elif i.attrib['bindingAttr']:
                            attrs[i.attrib['bindingAttr'].lower()] = True

                    if i.tag == 'access-spec':
                        attrs[i.attrib['keyword'].lower()] = True

                    if i.tag == 'names':
                        names += i.findall('name')
                    if i.tag == 'name':
                        generic_spec = i.find('generic_spec')

                    if i.tag == 'specific-binding':
                        assert not names
                        body.append(self.visit(i, attrs=attrs, **kwargs))
                        attrs = {}

                    elif i.tag in ('generic-binding', 'final-binding'):
                        body.append(self.visit(i, attrs=attrs, names=names, spec=generic_spec, **kwargs))
                        names = []
                        attrs = {}
                        generic_spec = None

                assert not attrs
                assert not names
                assert generic_spec is None

            # Infer any additional shape information from `!$loki dimension` pragmas
            body = attach_pragmas(body, ir.VariableDeclaration)
            body = process_dimension_pragmas(body)
            body = detach_pragmas(body, ir.VariableDeclaration)

            # Finally: update the typedef with its body
            typedef._update(body=body)
            typedef.rescope_symbols()
            return typedef

        # First, declaration attributes
        attrs = {}
        if o.find('intent') is not None:
            intent = self.visit(o.find('intent'), **kwargs)
            attrs.update((intent,))

        if o.find('attribute-parameter') is not None:
            parameter = self.visit(o.find('attribute-parameter'), **kwargs)
            attrs.update((parameter,))

        if o.find('attribute-optional') is not None:
            optional = self.visit(o.find('attribute-optional'), **kwargs)
            attrs.update((optional,))

        if o.find('attribute-allocatable') is not None:
            allocatable = self.visit(o.find('attribute-allocatable'), **kwargs)
            attrs.update((allocatable,))

        if o.find('attribute-pointer') is not None:
            pointer = self.visit(o.find('attribute-pointer'), **kwargs)
            attrs.update((pointer,))

        if o.find('attribute-target') is not None:
            target = self.visit(o.find('attribute-target'), **kwargs)
            attrs.update((target,))

        if o.find('access-spec') is not None:
            access_spec = self.visit(o.find('access-spec'), **kwargs)
            attrs.update((access_spec,))

        if o.find('variables') is not None:
            # This is probably a variable declaration
            _type = self.visit(o.find('type'), **kwargs)

            if _type.dtype == BasicType.CHARACTER:
                char_selector = o.find('char-selector')
                if _type.length is None and char_selector is not None:
                    selector_idx = list(o).index(char_selector)

                    if selector_idx > 0:
                        tk1 = char_selector.get('tk1')
                        tk2 = char_selector.get('tk2')

                        length = None
                        kind = None
                        if tk1 in ('', 'len'):
                            # The first child _should_ be the length selector
                            length = self.visit(o[0], **kwargs)

                            if tk2 == 'kind' or selector_idx > 2:
                                # There is another value, presumably the kind specifier, which
                                # should be right before the char-selector
                                kind = self.visit(o[selector_idx-1], **kwargs)
                        elif tk1 == 'kind':
                            # The first child _should_ be the kind selector
                            kind = self.visit(o[0], **kwargs)

                            if tk2 == 'len':
                                # The second child should then be the length selector
                                assert selector_idx > 2
                                length = self.visit(o[1], **kwargs)

                        attrs['length'] = length
                        attrs['kind'] = kind

            # Then, build the common symbol type for all variables
            _type = _type.clone(**attrs)

            # Last, instantiate declared variables
            variables = as_tuple(self.visit(o.find('variables'), **kwargs))

            # check if we have a dimensions keyword
            if o.find('dimensions') is not None:
                dimensions = self.visit(o.find('dimensions'), **kwargs)
                _type = _type.clone(shape=dimensions)
                # Attach dimension attribute to variable declaration for uniform
                # representation of variables in declarations
                variables = as_tuple(v.clone(dimensions=dimensions) for v in variables)

            # Make sure KIND (which can be a name) is in the right scope
            scope = kwargs['scope']
            if _type.kind is not None:
                kind = AttachScopesMapper()(_type.kind, scope=scope)
                _type = _type.clone(kind=kind)

            # EXTERNAL attribute means this is actually a function or subroutine
            # Since every symbol refers to a different function we have to update the
            # type definition for every symbol individually
            external = o.find('attribute-external') is not None
            if external:
                _type = _type.clone(external=True)
                for var in variables:
                    type_kwargs = _type.__dict__.copy()
                    if _type.dtype is not None:
                        return_type = SymbolAttributes(_type.dtype)
                        type_kwargs['dtype'] = ProcedureType(var.name, is_function=True, return_type=return_type)
                    else:
                        type_kwargs['dtype'] = ProcedureType(var.name, is_function=False)
                    scope.symbol_attrs[var.name] = var.type.clone(**type_kwargs)

                variables = tuple(var.rescope(scope=scope) for var in variables)
                return ir.ProcedureDeclaration(symbols=variables, external=True, source=source, label=label)

            # Update symbol table entries and rescope
            scope.symbol_attrs.update({var.name: var.type.clone(**_type.__dict__) for var in variables})
            variables = tuple(var.rescope(scope=scope) for var in variables)
            return ir.VariableDeclaration(symbols=variables, dimensions=_type.shape, source=source, label=label)

        if o.find('procedures') is not None:
            # This is probably a procedure declaration
            scope = kwargs['scope']

            interface = None
            if o.find('type') is not None:
                _type = self.visit(o.find('type'), **kwargs)
            elif o.find('proc-interface') is not None:
                interface = self.visit(o.find('proc-interface'), **kwargs)
                interface = interface.rescope(scope.get_symbol_scope(interface.name))
                _type = interface.type
            else:
                self.warn_or_fail('No type or proc-interface given')
                _type = SymbolAttributes(BasicType.DEFERRED)

            if o.find('proc-attr-spec') is not None:
                # Apparently, the POINTER attribute doesn't show up explicitly anywhere,
                # but a proc-attr-spec node seems to be always present when a declaration
                # carries the POINTER attribute...
                _type = _type.clone(pointer=True)

            _type = _type.clone(**attrs)

            # Build the declared symbols
            symbols = self.visit(o.find('procedures'), **kwargs)

            # Update symbol table entries
            if isinstance(_type.dtype, ProcedureType):
                scope.symbol_attrs.update({var.name: var.type.clone(**_type.__dict__) for var in symbols})
            else:
                # This is (presumably!) an external or dummy function with implicit interface,
                # which is declared as `PROCEDURE(<return_type>) [::] <name>`. Easy, isn't it...?
                # Great, now we have to update each symbol's type one-by-one...
                assert o.find('procedure-declaration-stmt').get('hasProcInterface')
                interface = _type.dtype
                for var in symbols:
                    dtype = ProcedureType(var.name, is_function=True, return_type=_type)
                    scope.symbol_attrs[var.name] = var.type.clone(dtype=dtype)

            # Rescope variables so they know their type
            symbols = tuple(var.rescope(scope=scope) for var in symbols)
            return ir.ProcedureDeclaration(symbols=symbols, interface=interface, source=source, label=label)

        if o.find('import-stmt') is not None:
            # This is an IMPORT statement in a subroutine declaration inside of
            # an interface body
            symbols = self.visit(o.find('names'), **kwargs)
            symbols = AttachScopesMapper()(symbols, scope=kwargs['scope'])
            return ir.Import(
                module = None, symbols=symbols, f_import=True, source=kwargs['source']
            )

        if o.find('prefix-spec') is not None:
            # This is the prefix specification of a subroutine/function. We can't
            # handle this, yet
            return None

        # This should never happen:
        raise ValueError('Unknown Declaration')

    def visit_interface(self, o, **kwargs):
        scope = kwargs['scope']
        abstract = o.get('type') == 'abstract'

        if o.find('header/defined-operator') is not None:
            spec = self.visit(o.find('header/defined-operator'), **kwargs)
        elif o.find('header/name') is not None:
            spec = self.visit(o.find('header/name'), **kwargs)
        else:
            spec = None

        if spec is not None:
            if spec.name not in scope.symbol_attrs:
                scope.symbol_attrs[spec.name] = SymbolAttributes(ProcedureType(name=spec.name, is_generic=True))
            spec = spec.rescope(scope=scope)

        body = []
        grouped_elems = match_tag_sequence(o.find('body/specification/declaration'), [
            ('names', 'procedure-stmt'),
            ('function', ),
            ('subroutine', ),
            ('comment', ),
        ])

        for group in grouped_elems:
            if len(group) == 1:
                # Process indidividual comments/pragmas/functions/subroutines
                body.append(self.visit(group[0], **kwargs))

            elif len(group) == 2:
                # Process procedure declarations
                body.append(self.create_interface_declaration(
                    names=group[0], proc_stmt=group[1], scope=scope, source=kwargs['source']
                ))

        return ir.Interface(abstract=abstract, body=as_tuple(body), spec=spec,
                            label=kwargs['label'], source=kwargs['source'])

    def visit_generic_spec(self, o, **kwargs):
        name = o.attrib['name']
        if o.get('keyword').upper() == 'ASSIGNMENT':
            name = 'ASSIGNMENT(=)'
        return sym.Variable(name=name, source=kwargs['source'])

    def visit_defined_operator(self, o, **kwargs):
        name = f'OPERATOR({o.attrib["definedOp"]})'
        return sym.Variable(name=name, source=kwargs['source'])

    def _create_Subroutine_object(self, o, scope):
        """Helper method to instantiate a Subroutine object"""
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel,cyclic-import
        assert o.tag in ('subroutine', 'function')
        name = o.attrib['name']

        # Check if the Subroutine node has been created before by looking it up in the scope
        routine = None
        if scope is not None and name in scope.symbol_attrs:
            proc_type = scope.symbol_attrs[name]  # Look-up only in current scope!
            if proc_type and proc_type.dtype.procedure != BasicType.DEFERRED:
                routine = proc_type.dtype.procedure
                if not routine._incomplete:
                    # We return the existing object right away, unless it exists from a
                    # previous incomplete parse for which we have to make sure we get a
                    # full parse first
                    return routine

        # Dummy args and procedure attributes
        header_ast = o.find('header')
        if o.tag == 'function':
            is_function = True
            args = [a.attrib['id'].upper() for a in header_ast.findall('names/name')]
        else:
            is_function = False
            args = [a.attrib['name'].upper() for a in header_ast.findall('arguments/argument')]
            if header_ast.find('subroutine-stmt').attrib['hasBindingSpec'] == 'true':
                self.warn_or_fail('binding-spec not implemented')

        if routine is None:
            routine = Subroutine(
                name=name, args=args, prefix=None, bind=None,
                is_function=is_function, parent=scope,
                ast=o, source=self.get_source(o)
            )
        else:
            routine.__init__(  # pylint: disable=unnecessary-dunder-call
                name=name, args=args, docstring=routine.docstring, spec=routine.spec, body=routine.body,
                contains=routine.contains, prefix=None, bind=None, is_function=is_function,
                ast=o, source=self.get_source(o), parent=routine.parent, symbol_attrs=routine.symbol_attrs,
                incomplete=routine._incomplete
            )

        return routine

    def visit_subroutine(self, o, **kwargs):
        # Extract known sections
        body_ast = list(o.find('body'))
        spec_ast = o.find('body/specification')
        spec_ast_idx = body_ast.index(spec_ast)
        docs_ast, body_ast = body_ast[:spec_ast_idx], body_ast[spec_ast_idx+1:]
        members_ast = o.find('members')

        # Instantiate the object
        routine = self._create_Subroutine_object(o, kwargs['scope'])
        kwargs['scope'] = routine

        # Create IRs for the docstring and the declaration spec
        docstring = self.visit(docs_ast, **kwargs)
        docstring = sanitize_ir(docstring, OFP, pp_registry=sanitize_registry[OFP], pp_info=self.pp_info)
        spec = self.visit(spec_ast, **kwargs)
        spec = sanitize_ir(spec, OFP, pp_registry=sanitize_registry[OFP], pp_info=self.pp_info)

        # Parse "member" subroutines and functions recursively
        contains = None
        if members_ast is not None:
            contains = self.visit(members_ast, **kwargs)
            contains.prepend(self.visit(o.find('contains-stmt'), **kwargs))

        # Finally, take care of the body
        if not body_ast:
            body = ir.Section(body=())
        else:
            body = ir.Section(body=self.visit(body_ast, **kwargs))
            body = sanitize_ir(body, OFP, pp_registry=sanitize_registry[OFP], pp_info=self.pp_info)

        # Finally, call the subroutine constructor on the object again to register all
        # bits and pieces in place and rescope all symbols
        # pylint: disable=unnecessary-dunder-call
        routine.__init__(
            name=routine.name, args=routine._dummies,
            docstring=docstring, spec=spec, body=body, contains=contains,
            ast=o, prefix=routine.prefix, bind=routine.bind, is_function=routine.is_function,
            rescope_symbols=True, parent=routine.parent, symbol_attrs=routine.symbol_attrs,
            source=kwargs['source'], incomplete=False
        )

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        routine.spec, routine.body = routine._infer_allocatable_shapes(routine.spec, routine.body)

        # Update array shapes with Loki dimension pragmas
        with pragmas_attached(routine, ir.VariableDeclaration):
            routine.spec = process_dimension_pragmas(routine.spec)

        return routine

    visit_function = visit_subroutine

    def visit_members(self, o, **kwargs):
        body = [self.visit(c, **kwargs) for c in o]
        body = [c for c in body if c is not None]
        return ir.Section(body=as_tuple(body), source=kwargs['source'])

    def _create_Module_object(self, o, scope):
        """Helper method to instantiate a Module object"""
        from loki.module import Module  # pylint: disable=import-outside-toplevel,cyclic-import

        name = o.attrib['name']

        # Check if the Module node has been created before by looking it up in the scope
        if scope is not None and name in scope.symbol_attrs:
            module_type = scope.symbol_attrs[name]  # Look-up only in current scope
            if module_type and module_type.dtype.module != BasicType.DEFERRED:
                return module_type.dtype.module

        return Module(name=name, parent=scope)

    def visit_module(self, o, **kwargs):
        # Extract known sections
        body_ast = list(o.find('body'))
        spec_ast = o.find('body/specification')
        spec_ast_idx = body_ast.index(spec_ast)
        docs_ast, body_ast = body_ast[:spec_ast_idx], body_ast[spec_ast_idx+1:]

        # Instantiate the object
        module = self._create_Module_object(o, kwargs['scope'])
        kwargs['scope'] = module

        # Pre-populate symbol table with procedure types declared in this module
        # to correctly classify inline function calls and type-bound procedures
        contains_ast = o.find('members')
        contains_stmt = o.find('contains-stmt')
        if contains_ast is not None and contains_stmt is not None:
            # Note that we overwrite this variable subsequently with the fully parsed subroutines
            # where the visit-method for the subroutine/function statement will pick out the existing
            # subroutine objects using the weakref pointers stored in the symbol table.
            # I know, it's not pretty but alternatively we could hand down this array as part of
            # kwargs but that feels like carrying around a lot of bulk, too.
            contains = [
                self._create_Subroutine_object(member_ast, kwargs['scope'])
                for member_ast in contains_ast if member_ast.tag in ('subroutine', 'function')
            ]

        # Parse the spec
        spec = self.visit(spec_ast, **kwargs)
        spec = sanitize_ir(spec, OFP, pp_registry=sanitize_registry[OFP], pp_info=self.pp_info)

        # Parse the docstring
        if not docs_ast and not spec.body:
            # If the spec is empty the docstring is conveniently put flat into the module instead...
            docs_ast = o.findall('comment')
        docstring = self.visit(docs_ast, **kwargs)
        docstring = sanitize_ir(docstring, OFP, pp_registry=sanitize_registry[OFP], pp_info=self.pp_info)

        # Parse member subroutines and functions
        if contains_ast is not None and contains_stmt is not None:
            contains_stmt = self.visit(contains_stmt, **kwargs)
            contains = self.visit(contains_ast, **kwargs)
            contains.prepend(contains_stmt)
        else:
            contains = None

        # Finally, call the module constructor on the object again to register all
        # bits and pieces in place and rescope all symbols
        # pylint: disable=unnecessary-dunder-call
        module.__init__(
            name=module.name, docstring=docstring, spec=spec, contains=contains,
            default_access_spec=module.default_access_spec, public_access_spec=module.public_access_spec,
            private_access_spec=module.private_access_spec, ast=o, source=kwargs['source'],
            rescope_symbols=True, parent=module.parent, symbol_attrs=module.symbol_attrs,
            incomplete=False
        )
        return module

    def visit_program(self, o, **kwargs):
        self.warn_or_fail('No support for PROGRAM')

    def visit_association(self, o, **kwargs):
        return sym.Variable(name=o.attrib['associate-name'])

    def visit_associate(self, o, **kwargs):
        # Handle the associates
        associations = [(self.visit(a[0], **kwargs), self.visit(a.find('association'), **kwargs))
                        for a in o.findall('header/keyword-arguments/keyword-argument')]

        # Create a scope for the associate
        parent_scope = kwargs['scope']
        associate = ir.Associate(associations=(), body=(), parent=parent_scope,
                                 label=kwargs['label'], source=kwargs['source'])
        kwargs['scope'] = associate

        # Put associate expressions into the right scope and determine type of new symbols
        rescoped_associations = []
        for expr, name in associations:
            # Put symbols in associated expression into the right scope
            expr = AttachScopesMapper()(expr, scope=parent_scope)

            # Determine type of new names
            if isinstance(expr, (sym.TypedSymbol, sym.MetaSymbol)):
                # Use the type of the associated variable
                _type = expr.type.clone(parent=None)
                if isinstance(expr, sym.Array) and expr.dimensions is not None:
                    shape = ExpressionDimensionsMapper()(expr)
                    if shape == (sym.IntLiteral(1),):
                        # For a scalar expression, we remove the shape
                        shape = None
                    _type = _type.clone(shape=shape)
            else:
                # TODO: Handle data type and shape of complex expressions
                shape = ExpressionDimensionsMapper()(expr)
                if shape == (sym.IntLiteral(1),):
                    # For a scalar expression, we remove the shape
                    shape = None
                _type = SymbolAttributes(BasicType.DEFERRED, shape=shape)
            name = name.clone(scope=associate, type=_type)
            rescoped_associations += [(expr, name)]
        associations = as_tuple(rescoped_associations)

        body = as_tuple(self.visit(o.find('body'), **kwargs))
        associate._update(associations=associations, body=body)
        return associate

    def visit_allocate(self, o, **kwargs):
        variables = as_tuple(self.visit(v, **kwargs) for v in o.findall('expressions/expression/name'))
        kw_args = {arg.attrib['name'].lower(): self.visit(arg, **kwargs)
                   for arg in o.findall('keyword-arguments/keyword-argument')}
        return ir.Allocation(variables=variables, label=kwargs['label'],
                             source=kwargs['source'], data_source=kw_args.get('source'),
                             status_var=kw_args.get('stat'))

    def visit_deallocate(self, o, **kwargs):
        variables = as_tuple(self.visit(v, **kwargs) for v in o.findall('expressions/expression/name'))
        kw_args = {arg.attrib['name'].lower(): self.visit(arg, **kwargs)
                   for arg in o.findall('keyword-arguments/keyword-argument')}
        return ir.Deallocation(variables=variables, label=kwargs['label'],
                               source=kwargs['source'], status_var=kw_args.get('stat'))

    def visit_use(self, o, **kwargs):
        name, module = self.visit(o.find('use-stmt'), **kwargs)
        scope = kwargs['scope']
        if o.find('only') is not None:
            # ONLY list given (import only selected symbols)
            symbols = self.visit(o.find('only'), **kwargs)
            # No rename-list
            rename_list = None
            if module is None:
                # Initialize symbol attributes as DEFERRED
                for s in symbols:
                    _type = SymbolAttributes(BasicType.DEFERRED, imported=True)
                    if isinstance(s, tuple):  # Renamed symbol
                        scope.symbol_attrs[s[1].name] = _type.clone(use_name=s[0])
                    else:
                        scope.symbol_attrs[s.name] = _type
            else:
                # Import symbol attributes from module
                for s in symbols:
                    if isinstance(s, tuple):  # Renamed symbol
                        scope.symbol_attrs[s[1].name] = module.symbol_attrs[s[0]].clone(
                            imported=True, module=module, use_name=s[0]
                        )
                    else:
                        scope.symbol_attrs[s.name] = module.symbol_attrs[s.name].clone(
                            imported=True, module=module, use_name=None
                        )
            symbols = tuple(
                s[1].rescope(scope=scope) if isinstance(s, tuple) else s.rescope(scope=scope) for s in symbols
            )
        else:
            # No ONLY list
            symbols = None
            # Rename list
            rename_list = dict(self.visit(s, **kwargs) for s in o.findall('rename/rename'))
            if module is not None:
                # Import symbol attributes from module, if available
                for k, v in module.symbol_attrs.items():
                    if k in rename_list:
                        local_name = rename_list[k].name
                        scope.symbol_attrs[local_name] = v.clone(
                            imported=True, module=module, use_name=k
                        )
                    else:
                        scope.symbol_attrs[k] = v.clone(
                            imported=True, module=module, use_name=None
                        )
            elif rename_list:
                # Module not available but some information via rename-list
                scope.symbol_attrs.update({
                    v.name: v.type.clone(imported=True, use_name=k) for k, v in rename_list.items()
                })
            rename_list = tuple(rename_list.items()) if rename_list else None
        return ir.Import(module=name, symbols=symbols, rename_list=rename_list,
                         label=kwargs['label'], source=kwargs['source'])

    def visit_only(self, o, **kwargs):
        count = int(o.find('only-list').get('count'))
        nodes = [c for c in o if c.tag in ('name', 'rename', 'defined-operator')]
        return as_tuple([self.visit(c, **kwargs) for c in nodes[:count]])

    def visit_rename(self, o, **kwargs):
        if o.attrib['defOp1'] or o.attrib['defOp2']:
            self.warn_or_fail('OPERATOR in rename-list not yet implemented')
            return ()
        return (o.attrib['id2'], sym.Variable(name=o.attrib['id1'], source=kwargs['source']))

    def visit_use_stmt(self, o, **kwargs):
        name = o.attrib['id']
        if o.attrib['hasModuleNature'] != 'false':
            self.warn_or_fail('module nature in USE statement not implemented')
        return name, self.definitions.get(name)

    def visit_directive(self, o, **kwargs):
        source = kwargs['source']
        if '#include' in o.attrib['text']:
            # Straight pipe-through node for header includes (#include ...)
            match = re.search(r'#include\s[\'"](?P<module>.*)[\'"]', o.attrib['text'])
            module = match.groupdict()['module']
            return ir.Import(module=module, c_import=True, source=source)
        return ir.PreprocessorDirective(text=source.string.strip(), source=source)

    def visit_exit(self, o, **kwargs):
        stmt_tag = f'{o.tag}-stmt'
        stmt = self.visit(o.find(stmt_tag), **kwargs)
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

    def create_intrinsic_from_source(self, o, attrib_name, **kwargs):
        label = kwargs['label']
        source = kwargs['source']
        cstart = source.string.lower().find(o.attrib[attrib_name].lower())
        assert cstart != -1
        return ir.Intrinsic(text=source.string[cstart:].strip(), label=label, source=source)

    def visit_exit_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'exitKeyword', **kwargs)

    def visit_return_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'keyword', **kwargs)

    visit_contains_stmt = visit_return_stmt

    def visit_continue_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'continueKeyword', **kwargs)

    def visit_cycle_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'cycleKeyword', label=kwargs['label'], source=kwargs['source'])

    def visit_format_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'formatKeyword', label=kwargs['label'], source=kwargs['source'])

    def visit_print_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'printKeyword', label=kwargs['label'], source=kwargs['source'])

    def visit_open_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'openKeyword', label=kwargs['label'], source=kwargs['source'])

    def visit_close_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'closeKeyword', label=kwargs['label'], source=kwargs['source'])

    def visit_write_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'writeKeyword', label=kwargs['label'], source=kwargs['source'])

    def visit_read_stmt(self, o, **kwargs):
        return self.create_intrinsic_from_source(o, 'readKeyword', label=kwargs['label'], source=kwargs['source'])

    def visit_call(self, o, **kwargs):
        name, subscripts = self.visit(o.find('name'), **kwargs)
        args = tuple(arg for arg in subscripts if not isinstance(arg, tuple))
        kw_args = tuple(arg for arg in subscripts if isinstance(arg, tuple))
        return ir.CallStatement(name=name, arguments=args, kwarguments=kw_args,
                                label=kwargs['label'], source=kwargs['source'])

    def visit_label(self, o, **kwargs):
        assert kwargs['label'] is not None
        return ir.Comment('__STATEMENT_LABEL__', label=kwargs['label'], source=kwargs['source'])

    # Expression parsing below; maye move to its own parser..?

    def visit_subscripts(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o if c.tag in ('subscript', 'argument'))

    def visit_subscript(self, o, **kwargs):
        source = kwargs['source']
        if o.attrib['type'] in ('empty', 'assumed-shape'):
            return sym.RangeIndex((None, None, None), source=source)
        if o.attrib['type'] == 'simple':
            return self.visit(o[0], **kwargs)
        if o.attrib['type'] == 'upper-bound-assumed-shape':
            return sym.RangeIndex((self.visit(o[0], **kwargs), None, None), source=source)
        if o.attrib['type'] == 'range':
            lower, upper, step = None, None, None
            if o.find('range/lower-bound') is not None:
                lower = self.visit(o.find('range/lower-bound'), **kwargs)
            if o.find('range/upper-bound') is not None:
                upper = self.visit(o.find('range/upper-bound'), **kwargs)
            if o.find('range/step') is not None:
                step = self.visit(o.find('range/step'), **kwargs)
            return sym.RangeIndex((lower, upper, step), source=source)
        raise ValueError('Unknown subscript type')

    def visit_dimensions(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o.findall('dimension'))

    visit_dimension = visit_subscript

    def visit_argument(self, o, **kwargs):
        return o.attrib['name'], self.visit(o[0], **kwargs)

    def visit_names(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o.findall('name'))

    def visit_name(self, o, **kwargs):
        source = kwargs['source']

        if o.find('generic-name-list-part') is not None:
            # From an external-stmt or use-stmt
            return sym.Variable(name=o.attrib['id'], source=source)

        if o.find('generic_spec') is not None:
            return self.visit(o.find('generic_spec'), **kwargs)

        num_part_ref = int(o.find('data-ref').attrib['numPartRef'])
        subscripts = [self.visit(s, **kwargs) for s in o.findall('subscripts')]
        name = None
        for i, part_ref in enumerate(o.findall('part-ref')):
            name, parent = self.visit(part_ref, **kwargs), name
            if parent:
                name = name.clone(name=f'{parent.name}%{name.name}', parent=parent)

            if part_ref.attrib['hasSectionSubscriptList'] == 'true':
                if i < num_part_ref - 1 or o.attrib['type'] == 'variable':
                    if subscripts[0]:  # If there are no subscripts it cannot be an array but must
                                       # be a function call
                        arguments = subscripts.pop(0)
                        kwarguments = tuple(arg for arg in arguments if isinstance(arg, tuple))
                        assert not kwarguments
                        name = name.clone(dimensions=arguments)

        # Check for leftover subscripts
        assert len(subscripts) <= 1

        if not 'type' in o.attrib or o.attrib['type'] == 'variable':
            # name is just used as an identifier or without any subscripts
            assert not subscripts
            return name

        if o.attrib['type'] == 'procedure':
            return name, subscripts[0] if subscripts else ()

        if subscripts and not subscripts[0]:
            # Function call with no arguments (gaaawwddd...)
            return sym.InlineCall(name, parameters=(), source=source)

        # unpack into positional and keyword args
        subscripts = subscripts[0] if subscripts else ()
        kwarguments = tuple(arg for arg in subscripts if isinstance(arg, tuple))
        arguments = tuple(arg for arg in subscripts if not isinstance(arg, tuple))

        cast_names = ('REAL', 'INT')
        if str(name).upper() in cast_names:
            assert arguments
            expr = arguments[0]
            if kwarguments:
                assert len(arguments) == 1
                assert len(kwarguments) == 1 and kwarguments[0][0] == 'kind'
                kind = kwarguments[0][1]
            else:
                kind = arguments[1] if len(arguments) > 1 else None
            return sym.Cast(name, expr, kind=kind, source=source)

        if subscripts:
            # This may potentially be an inline call
            intrinsic_calls = (
                'MIN', 'MAX', 'EXP', 'SQRT', 'ABS', 'LOG', 'MOD', 'SELECTED_INT_KIND',
                'SELECTED_REAL_KIND', 'ALLOCATED', 'PRESENT', 'SIGN', 'EPSILON', 'NULL',
                'SIZE', 'LBOUND', 'UBOUND'
            )
            if str(name).upper() in intrinsic_calls or kwarguments:
                return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments, source=source)

            _type = name._lookup_type(kwargs['scope'])
            if subscripts and _type and isinstance(_type.dtype, ProcedureType):
                return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments, source=source)

        # We end up here if it is ambiguous (i.e., OFP did not know what the symbol is)
        # which is either a function call, an array with subscript or a name without anything further
        assert o.attrib['type'] == 'ambiguous' and not getattr(name, 'dimensions', None)
        assert not kwarguments
        return name.clone(dimensions=arguments or None)

    def visit_part_ref(self, o, **kwargs):
        return sym.Variable(name=o.attrib['id'], source=kwargs['source'])

    def visit_literal(self, o, **kwargs):
        source = kwargs['source']
        boz_literal = o.find('boz-literal-constant')
        if boz_literal is not None:
            return sym.IntrinsicLiteral(boz_literal.attrib['constant'], source=source)

        kw_args = {'source': source}
        value = o.attrib['value']
        _type = o.attrib['type'] if 'type' in o.attrib else None
        if _type is not None:
            tmap = {'bool': BasicType.LOGICAL, 'int': BasicType.INTEGER,
                    'real': BasicType.REAL, 'char': BasicType.CHARACTER}
            _type = tmap[_type] if _type in tmap else BasicType.from_fortran_type(_type)
            kw_args['type'] = _type
        kind_param = o.find('kind-param')
        if kind_param is not None:
            kind = kind_param.attrib['kind']
            if kind.isnumeric():
                kw_args['kind'] = sym.Literal(value=int(kind))
            else:
                kw_args['kind'] = AttachScopesMapper()(sym.Variable(name=kind), scope=kwargs['scope'])
        return sym.Literal(value, **kw_args)

    def visit_array_constructor_values(self, o, **kwargs):
        values = [self.visit(v, **kwargs) for v in o.findall('value')]
        values = [v for v in values if v is not None]  # Filter empy values
        return sym.LiteralList(values=as_tuple(values), source=kwargs['source'])

    def visit_operation(self, o, **kwargs):
        """
        Construct expressions from individual operations, using left-recursion.
        """
        ops = [self.visit(op, **kwargs) for op in o.findall('operator')]
        ops = [str(op).lower() for op in ops if op is not None]  # Filter empty ops
        exprs = [self.visit(c, **kwargs) for c in o.findall('operand')]
        exprs = [e for e in exprs if e is not None]  # Filter empty operands

        # Left-recurse on the list of operations and expressions
        source = kwargs['source']
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
                raise RuntimeError(f'OFP: Unknown expression operator: {op}')

        if o.find('parenthesized_expr') is not None:
            # Force explicitly parenthesised operations
            if isinstance(expression, sym.Sum):
                expression = ParenthesisedAdd(expression.children, source=source)
            if isinstance(expression, sym.Product):
                expression = ParenthesisedMul(expression.children, source=source)
            if isinstance(expression, sym.Power):
                expression = ParenthesisedPow(expression.base, expression.exponent, source=source)
            if isinstance(expression, sym.Quotient):
                expression = ParenthesisedDiv(expression.numerator, expression.denominator, source=source)

        assert len(exprs) == 0
        return expression

    def visit_operator(self, o, **kwargs):
        return o.attrib['operator']

    def create_typedef_procedure_declaration(self, comps, iface=None, attrs=None, scope=None, source=None):
        """
        Utility method to create individual declarations from a group of AST nodes.
        """
        if iface is not None:
            iface = self.visit(iface, scope=scope)
            iface = AttachScopesMapper()(iface, scope=scope)

        if attrs is not None:
            attrs = [a.get('attrSpecKeyword').upper() for a in attrs]
        else:
            attrs = []

        type_attrs = {
            'pointer': 'POINTER' in attrs,
        }

        if iface:
            type_attrs['dtype'] = iface.type.dtype

        symbols = self.visit(comps, scope=scope, source=source)
        scope.symbol_attrs.update({s.name: s.type.clone(**type_attrs) for s in symbols})
        symbols = tuple(s.rescope(scope=scope) for s in symbols)
        return ir.ProcedureDeclaration(symbols=symbols, interface=iface, source=source)


    def create_typedef_variable_declaration(self, t, comps, attr=None, scope=None, source=None):
        """
        Utility method to create individual declarations from a group of AST nodes.
        """
        attrs = []
        if attr:
            attrs = [a.attrib['attrKeyword'].upper()
                     for a in attr.findall('attribute/component-attr-spec')]

        type_attrs = {
            'pointer': 'POINTER' in attrs,
            'allocatable': 'ALLOCATABLE' in attrs,
        }
        stype = self.visit(t, scope=scope).clone(**type_attrs)

        # Derive variables for this declaration entry
        variables = []
        for v in comps.findall('component'):
            if not v.attrib:
                continue
            if 'DIMENSION' in attrs:
                # Dimensions are provided via `dimension` keyword
                attrib = attr.findall('attribute')[attrs.index('DIMENSION')]
                deferred_shape = attrib.find('deferred-shape-spec-list')
            else:
                deferred_shape = v.find('deferred-shape-spec-list')

            initial = None
            dimensions = []

            v_children = list(v)
            if v.get('hasInitialComponentValue') == 'true':
                init_idx = v_children.index(v.find('component-initialization'))
                initial = self.visit(v_children[init_idx-1], scope=scope)
                initial = AttachScopesMapper()(initial, scope=scope)
                v_children = v_children[:init_idx-1] + v_children[init_idx+1:]

            if deferred_shape is not None:
                dim_count = int(deferred_shape.attrib['count'])
                dimensions = [sym.RangeIndex((None, None, None), source=source)
                              for _ in range(dim_count)]
            else:
                dimensions = as_tuple(self.visit(c, scope=scope) for c in v_children)
            dimensions = as_tuple(
                AttachScopesMapper()(d, scope=scope) for d in dimensions if d is not None
            ) or None

            v_source = extract_source(v.attrib, self._raw_source)
            v_type = stype.clone(shape=dimensions, initial=initial)
            v_name = v.attrib['name']

            scope.symbol_attrs[v_name] = v_type
            variables += [sym.Variable(name=v_name, scope=scope, dimensions=dimensions, source=v_source)]

        return ir.VariableDeclaration(symbols=variables, source=source)

    def create_interface_declaration(self, names, proc_stmt, scope=None, source=None):
        """
        Utility method to create individual procedure declarations for an interface
        from a group of AST nodes
        """
        symbols = self.visit(names, scope=scope)
        symbols = AttachScopesMapper()(symbols, scope=scope)
        module_proc = proc_stmt.get('module').lower() == 'module'
        return ir.ProcedureDeclaration(symbols=symbols, module=module_proc, source=source)

    def create_enum(self, o, **kwargs):
        """
        Utility method to create an ``ENUM`` IR node from a declaration node
        """
        scope = kwargs['scope']
        symbols = []
        comments = []
        value = None

        # Step through the items and keep values and enumerator stmts to build symbol list
        for i in o:

            if i.tag == 'enumerator':
                # The actual enumerator stmt (this gives us the symbol name)
                if i.attrib['hasExpr'] == 'true':
                    # This has a value assigned, which we must have seen before
                    assert value is not None
                    _type = SymbolAttributes(BasicType.INTEGER, initial=value)
                    value = None
                else:
                    assert value is None
                    _type = SymbolAttributes(BasicType.INTEGER)
                symbols += [sym.Variable(name=i.attrib['id'], type=_type, scope=scope)]

            else:
                # Something else: let's just recurse on it and save it as a value if
                # it doesn't yield None
                item = self.visit(i, **kwargs)
                if isinstance(item, ir.Comment):
                    comments += [item]
                elif item is not None:
                    assert value is None
                    value = item

        return (ir.Enumeration(symbols=as_tuple(symbols), source=kwargs['source'], label=kwargs['label']), *comments)
