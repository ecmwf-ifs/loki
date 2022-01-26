from collections import deque
from collections.abc import Iterable
from pathlib import Path
import re

try:
    import open_fortran_parser

    HAVE_OFP = True
    """Indicate whether OpenFortranParser frontend is available."""
except ImportError:
    HAVE_OFP = False

from loki.frontend.source import extract_source, extract_source_from_range
from loki.frontend.preprocessing import sanitize_registry
from loki.frontend.util import (
    inline_comments, cluster_comments, inline_labels, OFP, combine_multiline_pragmas
)
from loki.visitors import GenericVisitor
from loki import ir
import loki.expression.symbols as sym
from loki.expression.operations import (
    ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow, ParenthesisedDiv,
    StringConcat
)
from loki.expression import ExpressionDimensionsMapper, AttachScopesMapper
from loki.tools import (
    as_tuple, timeit, disk_cached, flatten, gettempdir, filehash, CaseInsensitiveDict,
)
from loki.pragma_utils import attach_pragmas, process_dimension_pragmas, detach_pragmas
from loki.logging import info, debug, DEBUG, error, warning
from loki.types import BasicType, DerivedType, ProcedureType, SymbolAttributes
from loki.config import config


__all__ = ['HAVE_OFP', 'parse_ofp_file', 'parse_ofp_source', 'parse_ofp_ast']


@timeit(log_level=DEBUG)
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


@timeit(log_level=DEBUG)
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


@timeit(log_level=DEBUG)
def parse_ofp_ast(ast, pp_info=None, raw_source=None, definitions=None, scope=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    # Parse the raw OMNI language AST
    _ir = OFP2IR(definitions=definitions, raw_source=raw_source, scope=scope).visit(ast)

    # Apply postprocessing rules to re-insert information lost during preprocessing
    for r_name, rule in sanitize_registry[OFP].items():
        _info = pp_info[r_name] if pp_info is not None and r_name in pp_info else None
        _ir = rule.postprocess(_ir, _info)

    # Perform some minor sanitation tasks
    _ir = inline_comments(_ir)
    _ir = cluster_comments(_ir)
    _ir = inline_labels(_ir)
    _ir = combine_multiline_pragmas(_ir)

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

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        if isinstance(o, Iterable):
            return super().visit(o, **kwargs)

        kwargs['label'] = self.get_label(o)
        kwargs.setdefault('scope', self.default_scope)

        try:
            kwargs['source'] = extract_source(o.attrib, self._raw_source, label=kwargs['label'])
        except KeyError:
            kwargs['source'] = None
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
    _re_pragma = re.compile(r'\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

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
        if o.find('where-construct-stmt') is not None:
            # Parse a WHERE statement(s)...
            children = [self.visit(c, **kwargs) for c in o]
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
                                             label=kwargs['label'], source=kwargs['source'])]
                children = children[iend+1:]

            # TODO: Deal with alternative conditions (multiple ELSEWHERE)
            return as_tuple(stmts)
        if o.find('goto-stmt') is not None:
            target_label = o.find('goto-stmt').attrib['target_label']
            return ir.Intrinsic(text=f'go to {target_label}', label=kwargs['label'], source=kwargs['source'])
        return self.visit_Element(o, **kwargs)

    def visit_elsewhere_stmt(self, o, **kwargs):
        # Only used as a marker above
        return 'ELSEWHERE_CONSTRUCT'

    def visit_end_where_stmt(self, o, **kwargs):
        # Only used as a marker above
        return 'ENDWHERE_CONSTRUCT'

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
            if o.find('length'):
                _type = _type.clone(length=self.visit(o.find('length'), **kwargs))
            return _type

        if o.attrib['type'] == 'derived':
            dtype = self.visit(o.find('derived-type-spec'), **kwargs)

            # Look for a previous definition of this type
            _type = kwargs['scope'].symbol_attrs.lookup(dtype.name)
            if _type is None or _type.dtype is BasicType.DEFERRED:
                _type = SymbolAttributes(dtype)

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

    def visit_attribute_parameter(self, o, **kwargs):
        return self.visit(o.findall('attr-spec'), **kwargs)

    visit_attribute_pointer = visit_attribute_parameter

    def visit_attr_spec(self, o, **kwargs):
        return (o.attrib['attrKeyword'].lower(), True)

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

    def visit_derived_type_stmt(self, o, **kwargs):
        if o.attrib['keyword'].lower() != 'type':
            self.warn_or_fail(f'Type keyword {o.attrib["keyword"]} not implemented')
        if o.attrib['hasTypeAttrSpecList'] != 'false':
            self.warn_or_fail('type-attr-spec-list not implemented')
        if o.attrib['hasGenericNameList'] != 'false':
            self.warn_or_fail('generic-name-list not implemented')
        return o.attrib['id']

    def visit_declaration(self, o, **kwargs):
        label = kwargs['label']
        source = kwargs['source']
        if not o.attrib:
            return None  # Skip empty declarations
        if not 'type' in o.attrib:
            if o.find('access-spec') is not None or o.find('save-stmt') is not None:
                return ir.Intrinsic(text=source.string.strip(), label=label, source=source)
            if o.find('interface') is not None:
                return self.visit(o.find('interface'), **kwargs)
            if o.find('subroutine') is not None:
                return self.visit(o.find('subroutine'), **kwargs)
            if o.find('function') is not None:
                return self.visit(o.find('function'), **kwargs)
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
            declaration = ir.Declaration(variables=variables, external=True, source=source, label=label)
            return declaration

        if o.attrib['type'] == 'data':
            self.warn_or_fail('data declaration not implemented')
            return ir.Intrinsic(text=source.string.strip(), label=label, source=source)

        if o.find('derived-type-stmt') is not None:
            # Derived type definition
            name = self.visit(o.find('derived-type-stmt'), **kwargs)

            # Instantiate the TypeDef without its body
            # Note: This creates the symbol table for the declarations and
            # the typedef object registers itself in the parent scope
            typedef = ir.TypeDef(name=name, body=(), label=label, source=source, parent=kwargs['scope'])
            kwargs['scope'] = typedef

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
                    body.append(self.visit(group[0], **kwargs))

                elif len(group) == 2:
                    # Process declarations without attributes
                    decl = self.create_typedef_declaration(t=group[0], comps=group[1],
                                                           scope=typedef, source=source)
                    body.append(decl)

                elif len(group) == 3:
                    # Process declarations with attributes
                    decl = self.create_typedef_declaration(t=group[0], attr=group[1], comps=group[2],
                                                           scope=typedef, source=source)
                    body.append(decl)

                else:
                    raise RuntimeError("OFP: Unknown tag grouping in TypeDef declaration processing")

            # Infer any additional shape information from `!$loki dimension` pragmas
            body = attach_pragmas(body, ir.Declaration)
            body = process_dimension_pragmas(body)
            body = detach_pragmas(body, ir.Declaration)

            # Finally: update the typedef with its body
            typedef._update(body=body)
            return typedef

        # First, obtain data type and attributes
        _type = self.visit(o.find('type'), **kwargs)

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

        if _type.dtype == BasicType.CHARACTER:
            if _type.length is None and o.find('char-selector') is not None:
                # For _NO_ good reason, the char-length property seems to be
                # always the first item (fingers crossed) but it is not identified
                # by any sensible unique tag...
                attrs['length'] = self.visit(o[0], **kwargs)

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

        # EXTERNAL attribute means this is actually a function or subroutine
        external = o.find('attribute-external') is not None
        if external:
            _type = _type.clone(external=True)

        # Make sure KIND (which can be a name) is in the right scope
        if _type.kind is not None:
            kind = AttachScopesMapper()(_type.kind, scope=kwargs['scope'])
            _type = _type.clone(kind=kind)

        # Update symbol table entries
        scope = kwargs['scope']
        for var in variables:
            if external:
                type_kwargs = _type.__dict__.copy()
                if _type.dtype is not None:
                    return_type = SymbolAttributes(_type.dtype)
                    type_kwargs['dtype'] = ProcedureType(var.name, is_function=True, return_type=return_type)
                else:
                    type_kwargs['dtype'] = ProcedureType(var.name, is_function=False)
                scope.symbol_attrs[var.name] = var.type.clone(**type_kwargs)
            else:
                scope.symbol_attrs[var.name] = var.type.clone(**_type.__dict__)

        variables = tuple(v.clone(scope=scope) for v in variables)
        return ir.Declaration(variables=variables, dimensions=_type.shape, external=external,
                              source=source, label=label)

    def visit_interface(self, o, **kwargs):
        spec = self.visit(o.find('interface-stmt'), **kwargs)
        body = self.visit(o.find('body'), **kwargs)
        return ir.Interface(spec=spec, body=body, label=kwargs['label'], source=kwargs['source'])

    def visit_interface_stmt(self, o, **kwargs):
        if o.attrib['abstract_token']:
            return o.attrib['abstract_token']
        if o.attrib['hasGenericSpec'] == 'true':
            self.warn_or_fail('interface with generic spec not implemented')
        return None

    def visit_subroutine(self, o, **kwargs):
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel

        # Name and dummy args
        name = o.attrib['name']
        if o.tag == 'function':
            is_function = True
            args = [a.attrib['id'].upper() for a in o.findall('header/names/name')]
            bind = None
        else:
            is_function = False
            args = [a.attrib['name'].upper() for a in o.findall('header/arguments/argument')]
            bind = None
            if o.find('header/subroutine-stmt').attrib['hasBindingSpec'] == 'true':
                self.warn_or_fail('binding-spec not implemented')

        parent_scope = kwargs['scope']
        routine = Subroutine(name=name, args=args, ast=o, bind=bind, is_function=is_function,
                             source=kwargs['source'], parent=parent_scope)
        kwargs['scope'] = routine

        # Spec
        routine.spec = self.visit(o.find('body/specification'), **kwargs)
        return routine

    visit_function = visit_subroutine

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
                    if isinstance(s, tuple):  # Renamed symbol
                        scope.symbol_attrs[s[1].name] = SymbolAttributes(BasicType.DEFERRED, imported=True, use_name=s[0])
                    else:
                        scope.symbol_attrs[s.name] = SymbolAttributes(BasicType.DEFERRED, imported=True)
            else:
                # Import symbol attributes from module
                for s in symbols:
                    if isinstance(s, tuple):  # Renamed symbol
                        scope.symbol_attrs[s[1].name] = module.symbol_attrs[s[0]].clone(imported=True, module=module,
                                                                              use_name=s[0])
                    else:
                        scope.symbol_attrs[s.name] = module.symbol_attrs[s.name].clone(imported=True, module=module)
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
                        scope.symbol_attrs[local_name] = v.clone(imported=True, module=module, use_name=k)
                    else:
                        scope.symbol_attrs[k] = v.clone(imported=True, module=module)
            elif rename_list:
                # Module not available but some information via rename-list
                scope.symbol_attrs.update({v.name: v.type.clone(imported=True, use_name=k) for k, v in rename_list.items()})
            rename_list = tuple(rename_list.items()) if rename_list else None
        return ir.Import(module=name, symbols=symbols, rename_list=rename_list,
                         label=kwargs['label'], source=kwargs['source'])

    def visit_only(self, o, **kwargs):
        return tuple(self.visit(c, **kwargs) for c in o if c.tag in ('name', 'rename'))

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

        if o.find('generic-name-list-part') is not None or o.find('generic_spec') is not None:
            # From an external-stmt or use-stmt
            return sym.Variable(name=o.attrib['id'], source=source)

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
                'SELECTED_REAL_KIND', 'ALLOCATED', 'PRESENT', 'SIGN', 'EPSILON', 'NULL'
            )
            if str(name).upper() in intrinsic_calls or kwarguments:
                return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments, source=source)

            _type = kwargs['scope'].symbol_attrs.lookup(name.name)
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
        return sym.LiteralList(values=values, source=kwargs['source'])

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

    def create_typedef_declaration(self, t, comps, attr=None, scope=None, source=None):
        """
        Utility method to create individual declarations from a group of AST nodes.
        """
        attrs = {}
        if attr:
            attrs = [a.attrib['attrKeyword'].upper()
                     for a in attr.findall('attribute/component-attr-spec')]

        type_attrs = {
            'pointer': 'POINTER' in attrs,
            'allocatable': 'ALLOCATABLE' in attrs,
        }
        stype = self.visit(t).clone(**type_attrs)

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
            if deferred_shape is not None:
                dim_count = int(deferred_shape.attrib['count'])
                dimensions = [sym.RangeIndex((None, None, None), source=source)
                              for _ in range(dim_count)]
            else:
                dimensions = as_tuple(self.visit(c) for c in v)
            dimensions = as_tuple(d for d in dimensions if d is not None)
            dimensions = dimensions if len(dimensions) > 0 else None
            v_source = extract_source(v.attrib, self._raw_source)
            v_type = stype.clone(shape=dimensions)
            v_name = v.attrib['name']

            scope.symbol_attrs[v_name] = v_type
            variables += [sym.Variable(name=v_name, scope=scope, dimensions=dimensions, source=v_source)]

        return ir.Declaration(variables=variables, source=source)
