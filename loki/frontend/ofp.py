from collections import OrderedDict, deque
from collections.abc import Iterable
from pathlib import Path
import re

import open_fortran_parser

from loki.frontend.source import extract_source
from loki.frontend.preprocessing import blacklist
from loki.frontend.util import (
    inline_comments, cluster_comments, inline_pragmas, inline_labels, process_dimension_pragmas
)
from loki.visitors import GenericVisitor
import loki.ir as ir
import loki.expression.symbol_types as sym
from loki.expression.operations import (
    ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow, StringConcat)
from loki.expression import ExpressionDimensionsMapper
from loki.tools import as_tuple, timeit, disk_cached, flatten, gettempdir, filehash
from loki.logging import info, DEBUG
from loki.types import DataType, SymbolType


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
def parse_ofp_ast(ast, pp_info=None, raw_source=None, typedefs=None, scope=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    # Parse the raw OMNI language AST
    _ir = OFP2IR(typedefs=typedefs, raw_source=raw_source, scope=scope).visit(ast)

    # Apply postprocessing rules to re-insert information lost during preprocessing
    for r_name, rule in blacklist.items():
        _info = pp_info[r_name] if pp_info is not None and r_name in pp_info else None
        _ir = rule.postprocess(_ir, _info)

    # Perform some minor sanitation tasks
    _ir = inline_comments(_ir)
    _ir = cluster_comments(_ir)
    _ir = inline_pragmas(_ir)
    _ir = inline_labels(_ir)

    return _ir


class OFP2IR(GenericVisitor):
    # pylint: disable=no-self-use  # Stop warnings about visitor methods that could do without self
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    def __init__(self, raw_source, typedefs=None, scope=None):
        super(OFP2IR, self).__init__()

        self._raw_source = raw_source
        self.typedefs = typedefs
        self.scope = scope

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        if isinstance(instance, Iterable):
            return super(OFP2IR, self).lookup_method(instance)

        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        return super(OFP2IR, self).lookup_method(instance)

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        if isinstance(o, Iterable):
            return super(OFP2IR, self).visit(o, **kwargs)

        try:
            source = extract_source(o.attrib, self._raw_source)
        except KeyError:
            source = None
        return super(OFP2IR, self).visit(o, source=source, **kwargs)

    def visit_tuple(self, o, source=None):
        return as_tuple(flatten(self.visit(c) for c in o))

    visit_list = visit_tuple

    def visit_Element(self, o, source=None):
        """
        Universal default for XML element types
        """
        children = tuple(self.visit(c) for c in o)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    visit_body = visit_Element

    def visit_loop(self, o, source=None):
        if o.find('header/index-variable') is None:
            # We are processing a while loop
            condition = self.visit(o.find('header'))
            body = as_tuple(self.visit(o.find('body')))
            return ir.WhileLoop(condition=condition, body=body, source=source)

        # We are processing a regular for/do loop with bounds
        vname = o.find('header/index-variable').attrib['name']
        variable = sym.Variable(name=vname, scope=self.scope.symbols)
        lower = self.visit(o.find('header/index-variable/lower-bound'))
        upper = self.visit(o.find('header/index-variable/upper-bound'))
        step = None
        if o.find('header/index-variable/step') is not None:
            step = self.visit(o.find('header/index-variable/step'))
        bounds = sym.LoopRange((lower, upper, step))

        body = as_tuple(self.visit(o.find('body')))
        # Store full lines with loop body for easy replacement
        source = extract_source(o.attrib, self._raw_source, full_lines=True)
        return ir.Loop(variable=variable, body=body, bounds=bounds, source=source)

    def visit_if(self, o, source=None):
        conditions = tuple(self.visit(h) for h in o.findall('header'))
        bodies = tuple([self.visit(b)] for b in o.findall('body'))
        ncond = len(conditions)
        else_body = bodies[-1] if len(bodies) > ncond else None
        inline = o.find('if-then-stmt') is None
        return ir.Conditional(conditions=conditions, bodies=bodies[:ncond],
                              else_body=else_body, inline=inline, source=source)

    def visit_select(self, o, source=None):
        expr = self.visit(o.find('header'))
        values = [self.visit(h) for h in o.findall('body/case/header')]
        bodies = [self.visit(b) for b in o.findall('body/case/body')]
        if None in values:
            else_index = values.index(None)
            values.pop(else_index)
            else_body = as_tuple(bodies.pop(else_index))
        else:
            else_body = ()
        return ir.MultiConditional(expr=expr, values=as_tuple(values), bodies=as_tuple(bodies),
                                   else_body=else_body, source=source)

    # TODO: Deal with line-continuation pragmas!
    _re_pragma = re.compile(r'\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

    def visit_comment(self, o, source=None):
        match_pragma = self._re_pragma.search(source.string)
        if match_pragma:
            # Found pragma, generate this instead
            gd = match_pragma.groupdict()
            return ir.Pragma(keyword=gd['keyword'], content=gd['content'], source=source)
        return ir.Comment(text=o.attrib['text'], source=source)

    def visit_statement(self, o, source=None):
        # TODO: Hacky pre-emption for special-case statements
        if o.find('name/nullify-stmt') is not None:
            variable = self.visit(o.find('name'))
            return ir.Nullify(variables=as_tuple(variable), source=source)
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

                stmts += [ir.MaskedStatement(condition=condition, body=body, default=default)]
                children = children[iend+1:]

            # TODO: Deal with alternative conditions (multiple ELSEWHERE)
            return as_tuple(stmts)
        if o.find('goto-stmt') is not None:
            label = o.find('goto-stmt').attrib['target_label']
            return ir.Intrinsic(text='go to %s' % label, source=source)
        return self.visit_Element(o, source=source)

    def visit_elsewhere_stmt(self, o, source=None):
        # Only used as a marker above
        return 'ELSEWHERE_CONSTRUCT'

    def visit_end_where_stmt(self, o, source=None):
        # Only used as a marker above
        return 'ENDWHERE_CONSTRUCT'

    def visit_assignment(self, o, source=None):
        expr = self.visit(o.find('value'))
        target = self.visit(o.find('target'))
        return ir.Statement(target=target, expr=expr, source=source)

    def visit_pointer_assignment(self, o, source=None):
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return ir.Statement(target=target, expr=expr, ptr=True, source=source)

    def visit_specification(self, o, source=None):
        body = tuple(self.visit(c) for c in o)
        body = tuple(c for c in body if c is not None)
        # Wrap spec area into a separate Scope
        return ir.Section(body=body, source=source)

    def visit_declaration(self, o, source=None):
        if len(o.attrib) == 0:
            return None  # Empty element, skip
        if o.find('save-stmt') is not None:
            return ir.Intrinsic(text=source.string, source=source)
        if o.find('implicit-stmt') is not None:
            return ir.Intrinsic(text=source.string, source=source)
        if o.find('access-spec') is not None:
            # PUBLIC or PRIVATE declarations
            return ir.Intrinsic(text=source.string, source=source)
        if o.attrib['type'] == 'variable':
            if o.find('end-type-stmt') is not None:
                # We are dealing with a derived type
                derived_name = o.find('end-type-stmt').attrib['id']

                # Process any associated comments or pragams
                comments = [self.visit(c) for c in o.findall('comment')]
                pragmas = [c for c in comments if isinstance(c, ir.Pragma)]
                comments = [c for c in comments if not isinstance(c, ir.Pragma)]

                # Create the parent type...
                typedef = ir.TypeDef(name=derived_name, declarations=[],
                                     pragmas=pragmas, comments=comments, source=source)
                parent_type = SymbolType(DataType.DERIVED_TYPE, name=derived_name,
                                         variables=OrderedDict(), source=source)

                # ...and derive all of its components
                # This is customized in our dedicated branch atm,
                # and really, really hacky! :(
                types = o.findall('type')
                components = o.findall('components')
                attributes = [None] * len(types)
                elements = list(o)
                declarations = []
                # YUCK!!!
                for i, (t, comps) in enumerate(zip(types, components)):
                    attributes[i] = elements[elements.index(t)+1:elements.index(comps)]

                for t, comps, attr in zip(types, components, attributes):
                    # Process the type of the individual declaration
                    attrs = {}
                    if len(attr) > 0:
                        attrs = [a.attrib['attrKeyword'].upper()
                                 for a in attr[0].findall('attribute/component-attr-spec')]
                    typename = t.attrib['name']
                    t_source = extract_source(t.attrib, self._raw_source)
                    kind = t.find('kind/name')
                    if kind is not None:
                        kind = kind.attrib['id']
                    # We have an intrinsic Fortran type
                    if t.attrib['type'] == 'intrinsic':
                        _type = SymbolType(DataType.from_fortran_type(typename), kind=kind,
                                           pointer='POINTER' in attrs,
                                           allocatable='ALLOCATABLE' in attrs, source=t_source)
                    else:
                        # This is a derived type. Let's see if we know it already
                        _type = self.scope.types.lookup(typename, recursive=True)
                        if _type is not None:
                            _type = _type.clone(kind=kind, pointer='POINTER' in attrs,
                                                allocatable='ALLOCATABLE' in attrs,
                                                source=t_source)
                        else:
                            _type = SymbolType(DataType.DERIVED_TYPE, name=typename,
                                               kind=kind, pointer='POINTER' in attrs,
                                               allocatable='ALLOCATABLE' in attrs,
                                               variables=OrderedDict(), source=t_source)

                    # Derive variables for this declaration entry
                    variables = []
                    for v in comps.findall('component'):
                        if len(v.attrib) == 0:
                            continue
                        if 'DIMENSION' in attrs:
                            # Dimensions are provided via `dimension` keyword
                            attrib = attr[0].findall('attribute')[attrs.index('DIMENSION')]
                            deferred_shape = attrib.find('deferred-shape-spec-list')
                        else:
                            deferred_shape = v.find('deferred-shape-spec-list')
                        if deferred_shape is not None:
                            dim_count = int(deferred_shape.attrib['count'])
                            dimensions = [sym.RangeIndex((None, None, None))
                                          for _ in range(dim_count)]
                        else:
                            dimensions = as_tuple(self.visit(c) for c in v)
                        dimensions = as_tuple(d for d in dimensions if d is not None)
                        dimensions = dimensions if len(dimensions) > 0 else None
                        v_source = extract_source(v.attrib, self._raw_source)
                        v_type = _type.clone(shape=dimensions, source=v_source)
                        v_name = v.attrib['name']
                        if dimensions:
                            dimensions = sym.ArraySubscript(dimensions) if dimensions else None

                        variables += [sym.Variable(name=v_name, type=v_type, dimensions=dimensions,
                                                   scope=typedef.symbols)]

                    parent_type.variables.update([(v.basename, v) for v in variables])  # pylint: disable=no-member
                    declarations += [ir.Declaration(variables=variables, type=_type, source=t_source)]

                typedef._update(declarations=as_tuple(declarations), symbols=typedef.symbols)

                # Infer any additional shape information from `!$loki dimension` pragmas
                process_dimension_pragmas(typedef)

                # Make that derived type known in the types table
                self.scope.types[derived_name] = parent_type

                return typedef

            # We are dealing with a single declaration, so we retrieve
            # all the declaration-level information first.
            typename = o.find('type').attrib['name']
            kind = o.find('type/kind/name')
            if kind is not None:
                kind = kind.attrib['id']
            intent = o.find('intent').attrib['type'] if o.find('intent') else None
            allocatable = o.find('attribute-allocatable') is not None
            pointer = o.find('attribute-pointer') is not None
            parameter = o.find('attribute-parameter') is not None
            optional = o.find('attribute-optional') is not None
            target = o.find('attribute-target') is not None
            dims = o.find('dimensions')
            dimensions = None if dims is None else as_tuple(self.visit(dims))

            if o.find('type').attrib['type'] == 'intrinsic':
                # Create a basic variable type
                # TODO: Character length attribute
                _type = SymbolType(DataType.from_fortran_type(typename), kind=kind,
                                   intent=intent, allocatable=allocatable, pointer=pointer,
                                   optional=optional, parameter=parameter, shape=dimensions,
                                   target=target, source=source)
            else:
                # Create the local variant of the derived type
                _type = self.scope.types.lookup(typename, recursive=True)
                if _type is not None:
                    _type = _type.clone(kind=kind, intent=intent, allocatable=allocatable,
                                        pointer=pointer, optional=optional, shape=dimensions,
                                        parameter=parameter, target=target, source=source)
                if _type is None:
                    if self.typedefs is not None and typename.lower() in self.typedefs:
                        variables = OrderedDict([(v.basename, v) for v
                                                 in self.typedefs[typename.lower()].variables])
                    else:
                        variables = OrderedDict()
                    _type = SymbolType(DataType.DERIVED_TYPE, name=typename,
                                       variables=variables, kind=kind, intent=intent,
                                       allocatable=allocatable, pointer=pointer,
                                       optional=optional, parameter=parameter,
                                       target=target, source=source)

            variables = [self.visit(v, type=_type, dimensions=dimensions)
                         for v in o.findall('variables/variable')]
            variables = [v for v in variables if v is not None]
            return ir.Declaration(variables=variables, type=_type, dimensions=dimensions,
                                  source=source)
        if o.attrib['type'] in ('implicit', 'intrinsic', 'parameter'):
            return ir.Intrinsic(text=source.string, source=source)
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
                declarations += [ir.DataDeclaration(variable=variable, values=vals, source=source)]
            return as_tuple(declarations)

        raise NotImplementedError('Unknown declaration type encountered: %s' % o.attrib['type'])

    def visit_associate(self, o, source=None):
        associations = OrderedDict()
        for a in o.findall('header/keyword-arguments/keyword-argument'):
            var = self.visit(a.find('name'))
            if isinstance(var, sym.Array):
                shape = ExpressionDimensionsMapper()(var)
            else:
                shape = None
            _type = var.type.clone(name=None, parent=None, shape=shape)
            assoc_name = a.find('association').attrib['associate-name']
            associations[var] = sym.Variable(name=assoc_name, type=_type, scope=self.scope.symbols,
                                             source=source)
        body = self.visit(o.find('body'))
        return ir.Scope(body=as_tuple(body), associations=associations)

    def visit_allocate(self, o, source=None):
        variables = as_tuple(self.visit(v) for v in o.findall('expressions/expression/name'))
        return ir.Allocation(variables=variables, source=source)

    def visit_deallocate(self, o, source=None):
        variables = as_tuple(self.visit(v) for v in o.findall('expressions/expression/name'))
        return ir.Deallocation(variables=variables, source=source)

    def visit_use(self, o, source=None):
        symbols = [n.attrib['id'] for n in o.findall('only/name')]
        return ir.Import(module=o.attrib['name'], symbols=symbols, source=source)

    def visit_directive(self, o, source=None):
        if '#include' in o.attrib['text']:
            # Straight pipe-through node for header includes (#include ...)
            match = re.search(r'#include\s[\'"](?P<module>.*)[\'"]', o.attrib['text'])
            module = match.groupdict()['module']
            return ir.Import(module=module, c_import=True, source=source)
        return ir.Intrinsic(text=source.string, source=source)

    def visit_open(self, o, source=None):
        # return ir.Intrinsic(text=source.string, source=source)
        assert o.tag.lower() in source.string.lower()
        return ir.Intrinsic(text=source.string[source.string.lower().find(o.tag.lower()):], source=source)

    visit_close = visit_open
    visit_read = visit_open
    visit_write = visit_open
    visit_format = visit_open
    visit_print = visit_open
    visit_cycle = visit_open
    visit_exit = visit_open
    visit_return = visit_open

    def visit_call(self, o, source=None):
        # Need to re-think this: the 'name' node already creates
        # a 'Variable', which in this case is wrong...
        name = o.find('name').attrib['id']
        args = tuple(self.visit(i) for i in o.findall('name/subscripts/subscript'))
        kwargs = list([self.visit(i) for i in o.findall('name/subscripts/argument')])
        return ir.CallStatement(name=name, arguments=args, kwarguments=kwargs, source=source)

    def visit_argument(self, o, source=None):
        key = o.attrib['name']
        val = self.visit(o.find('name'))
        return key, val

    def visit_label(self, o, source=None):
        source.label = int(o.attrib['lbl'])
        return ir.Comment('__STATEMENT_LABEL__', source=source)

    # Expression parsing below; maye move to its own parser..?

    def visit_name(self, o, source=None):

        def generate_variable(vname, indices, kwargs, parent, source):
            if vname.upper() == 'RESHAPE':
                # return reshape(indices[0], shape=indices[1])
                raise NotImplementedError()
            if vname.upper() in ['MIN', 'MAX', 'EXP', 'SQRT', 'ABS', 'LOG',
                                 'SELECTED_REAL_KIND', 'ALLOCATED', 'PRESENT']:
                return sym.InlineCall(vname, parameters=indices)
            if vname.upper() in ['REAL', 'INT']:
                kind = kwargs.get('kind', indices[1] if len(indices) > 1 else None)
                return sym.Cast(vname, expression=indices[0], kind=kind)
            if indices is not None and len(indices) == 0:
                # HACK: We (most likely) found a call out to a C routine
                return sym.InlineCall(o.attrib['id'], parameters=indices)

            if parent is not None:
                basename = vname
                vname = '%s%%%s' % (parent.name, vname)

            _type = self.scope.symbols.lookup(vname, recursive=True)

            # If the (possibly external) struct definitions exist
            # try to derive the type from it.
            if _type is None and parent is not None and parent.type is not None:
                if parent.type.dtype == DataType.DERIVED_TYPE:
                    _type = parent.type.variables.get(basename)

            if indices:
                indices = sym.ArraySubscript(indices)

            var = sym.Variable(name=vname, dimensions=indices, parent=parent,
                               type=_type, scope=self.scope.symbols, source=source)
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
        if _type is not None:
            _type = _type.clone(shape=dimensions)
        initial = None if o.find('initial-value') is None else self.visit(o.find('initial-value'))
        if dimensions:
            dimensions = sym.ArraySubscript(dimensions)
        return sym.Variable(name=name, scope=self.scope.symbols, dimensions=dimensions,
                            type=_type, initial=initial, source=source)

    def visit_part_ref(self, o, source=None):
        # Return a pure string, as part of a variable name
        return o.attrib['id']

    def visit_literal(self, o, source=None):
        kwargs = {'source': source}
        value = o.attrib['value']
        _type = o.attrib['type'] if 'type' in o.attrib else None
        if _type is not None:
            tmap = {'bool': DataType.LOGICAL, 'int': DataType.INTEGER,
                    'real': DataType.REAL, 'char': DataType.CHARACTER}
            _type = tmap[_type] if _type in tmap else DataType.from_fortran_type(_type)
            kwargs['type'] = _type
        kind_param = o.find('kind-param')
        if kind_param is not None:
            kwargs['kind'] = kind_param.attrib['kind']
        return sym.Literal(value, **kwargs)

    def visit_subscripts(self, o, source=None):
        return tuple(self.visit(c) for c in o
                     if c.tag in ['subscript', 'name'])

    def visit_subscript(self, o, source=None):
        # TODO: Drop this entire routine, but beware the base-case!
        if o.find('range'):
            lower, upper, step = None, None, None
            if o.find('range/lower-bound') is not None:
                lower = self.visit(o.find('range/lower-bound'))
            if o.find('range/upper-bound') is not None:
                upper = self.visit(o.find('range/upper-bound'))
            if o.find('range/step') is not None:
                step = self.visit(o.find('range/step'))
            return sym.RangeIndex((lower, upper, step))
        if 'type' in o.attrib and o.attrib['type'] == "upper-bound-assumed-shape":
            lower = self.visit(o[0])
            return sym.RangeIndex((lower, None, None))
        if o.find('name'):
            return self.visit(o.find('name'))
        if o.find('literal'):
            return self.visit(o.find('literal'))
        if o.find('operation'):
            return self.visit(o.find('operation'))
        if o.find('array-constructor-values'):
            return self.visit(o.find('array-constructor-values'))
        return sym.RangeIndex((None, None, None))

    visit_dimension = visit_subscript

    def visit_array_constructor_values(self, o, source=None):
        values = [self.visit(v) for v in o.findall('value')]
        values = [v for v in values if v is not None]  # Filter empy values
        return sym.LiteralList(values=values)

    def visit_operation(self, o, source=None):
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
                expression = sym.Sum((expression, exprs.popleft()))
            elif op == '-':
                if len(exprs) > 0:
                    # Binary minus
                    expression = sym.Sum((expression, sym.Product((-1, exprs.popleft()))))
                else:
                    # Unary minus
                    expression = sym.Product((-1, expression))
            elif op == '*':
                expression = sym.Product((expression, exprs.popleft()))
            elif op == '/':
                expression = sym.Quotient(numerator=expression, denominator=exprs.popleft())
            elif op == '**':
                expression = sym.Power(base=expression, exponent=exprs.popleft())
            elif op in ('==', '.eq.'):
                expression = sym.Comparison(expression, '==', exprs.popleft())
            elif op in ('/=', '.ne.'):
                expression = sym.Comparison(expression, '!=', exprs.popleft())
            elif op in ('>', '.gt.'):
                expression = sym.Comparison(expression, '>', exprs.popleft())
            elif op in ('<', '.lt.'):
                expression = sym.Comparison(expression, '<', exprs.popleft())
            elif op in ('>=', '.ge.'):
                expression = sym.Comparison(expression, '>=', exprs.popleft())
            elif op in ('<=', '.le.'):
                expression = sym.Comparison(expression, '<=', exprs.popleft())
            elif op == '.and.':
                expression = sym.LogicalAnd((expression, exprs.popleft()))
            elif op == '.or.':
                expression = sym.LogicalOr((expression, exprs.popleft()))
            elif op == '.not.':
                expression = sym.LogicalNot(expression)
            elif op == '.eqv.':
                e = (expression, exprs.popleft())
                expression = sym.LogicalOr((sym.LogicalAnd(e), sym.LogicalNot(sym.LogicalOr(e))))
            elif op == '.neqv.':
                e = (expression, exprs.popleft())
                expression = sym.LogicalAnd((sym.LogicalNot(sym.LogicalAnd(e)), sym.LogicalOr(e)))
            elif op == '//':
                expression = StringConcat((expression, exprs.popleft()))
            else:
                raise RuntimeError('OFP: Unknown expression operator: %s' % op)

        if o.find('parenthesized_expr') is not None:
            # Force explicitly parenthesised operations
            if isinstance(expression, sym.Sum):
                expression = ParenthesisedAdd(expression.children)
            if isinstance(expression, sym.Product):
                expression = ParenthesisedMul(expression.children)
            if isinstance(expression, sym.Power):
                expression = ParenthesisedPow(expression.base, expression.exponent)

        assert len(exprs) == 0
        return expression

    def visit_operator(self, o, source=None):
        return o.attrib['operator']
