import open_fortran_parser
from collections import OrderedDict, deque, Iterable
from pathlib import Path
import re
from pymbolic.primitives import (Sum, Product, Quotient, Power, Comparison, LogicalNot,
                                 LogicalAnd, LogicalOr)

from loki.frontend.source import extract_source
from loki.frontend.preprocessing import blacklist
from loki.frontend.util import inline_comments, cluster_comments, inline_pragmas
from loki.visitors import GenericVisitor
from loki.ir import (Loop, Statement, Conditional, Call, Comment,
                     Pragma, Declaration, Allocation, Deallocation, Nullify,
                     Import, Scope, Intrinsic, TypeDef, MaskedStatement,
                     MultiConditional, WhileLoop, DataDeclaration, Section)
from loki.expression import (Variable, Literal, RangeIndex, InlineCall, LiteralList, Array, Cast)
from loki.expression.operations import ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow
from loki.types import BaseType, DerivedType
from loki.tools import as_tuple, timeit, disk_cached
from loki.logging import info, DEBUG


__all__ = ['parse_ofp_file', 'parse_ofp_ast']


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


class OFP2IR(GenericVisitor):

    def __init__(self, raw_source, shape_map=None, type_map=None,
                 typedefs=None, cache=None):
        super(OFP2IR, self).__init__()

        self._raw_source = raw_source
        self.shape_map = shape_map
        self.type_map = type_map
        self.typedefs = typedefs

        # Store provided symbol cache for variable generation
        self._cache = None  # cache

    def Variable(self, *args, **kwargs):
        """
        Instantiate cached variable symbols from local symbol cache.
        """
        if self._cache is None:
            return Variable(*args, **kwargs)
        else:
            return self._cache.Variable(*args, **kwargs)

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        if isinstance(instance, Iterable):
            return super(OFP2IR, self).lookup_method(instance)

        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        else:
            return super(OFP2IR, self).lookup_method(instance)

    def visit(self, o, **kwargs):
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
        return tuple(self.visit(c) for c in o)

    visit_list = visit_tuple

    def visit_Element(self, o, source=None):
        """
        Universal default for XML element types
        """
        children = tuple(self.visit(c) for c in o.getchildren())
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        else:
            return children if len(children) > 0 else None

    visit_body = visit_Element

    def visit_loop(self, o, source=None):
        if o.find('header/index-variable') is None:
            # We are processing a while loop
            condition = self.visit(o.find('header'))
            body = as_tuple(self.visit(o.find('body')))
            return WhileLoop(condition=condition, body=body, source=source)
        else:
            # We are processing a regular for/do loop with bounds
            vname = o.find('header/index-variable').attrib['name']
            variable = self.Variable(name=vname)
            lower = self.visit(o.find('header/index-variable/lower-bound'))
            upper = self.visit(o.find('header/index-variable/upper-bound'))
            step = None
            if o.find('header/index-variable/step') is not None:
                step = self.visit(o.find('header/index-variable/step'))
            bounds = lower, upper, step

            body = as_tuple(self.visit(o.find('body')))
            # Store full lines with loop body for easy replacement
            source = extract_source(o.attrib, self._raw_source, full_lines=True)
            return Loop(variable=variable, body=body, bounds=bounds, source=source)

    def visit_if(self, o, source=None):
        conditions = tuple(self.visit(h) for h in o.findall('header'))
        bodies = tuple([self.visit(b)] for b in o.findall('body'))
        ncond = len(conditions)
        else_body = bodies[-1] if len(bodies) > ncond else None
        inline = o.find('if-then-stmt') is None
        return Conditional(conditions=conditions, bodies=bodies[:ncond],
                           else_body=else_body, inline=inline, source=source)

    def visit_select(self, o, source=None):
        expr = self.visit(o.find('header'))
        values = tuple(self.visit(h) for h in o.findall('body/case/header'))
        bodies = tuple(self.visit(b) for b in o.findall('body/case/body'))
        return MultiConditional(expr=expr, values=values, bodies=bodies)

    # TODO: Deal with line-continuation pragmas!
    _re_pragma = re.compile('\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

    def visit_comment(self, o, source=None):
        match_pragma = self._re_pragma.search(source.string)
        if match_pragma:
            # Found pragma, generate this instead
            gd = match_pragma.groupdict()
            return Pragma(keyword=gd['keyword'], content=gd['content'], source=source)
        else:
            return Comment(text=o.attrib['text'], source=source)

    def visit_statement(self, o, source=None):
        # TODO: Hacky pre-emption for special-case statements
        if o.find('name/nullify-stmt') is not None:
            variable = self.visit(o.find('name'))
            return Nullify(variable=variable, source=source)
        elif o.find('cycle') is not None:
            return self.visit(o.find('cycle'))
        elif o.find('where-construct-stmt') is not None:
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

                stmts += [MaskedStatement(condition=condition, body=body, default=default)]
                children = children[iend+1:]

            # TODO: Deal with alternative conditions (multiple ELSEWHERE)
            return as_tuple(stmts)
        else:
            return self.visit_Element(o, source=source)

    def visit_elsewhere_stmt(self, o, source=None):
        # Only used as a marker above
        return 'ELSEWHERE_CONSTRUCT'

    def visit_end_where_stmt(self, o, source=None):
        # Only used as a marker above
        return 'ENDWHERE_CONSTRUCT'

    def visit_cycle(self, o, source=None):
        return Intrinsic(text=source.string, source=source)

    def visit_assignment(self, o, source=None):
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return Statement(target=target, expr=expr, source=source)

    def visit_pointer_assignment(self, o, source=None):
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return Statement(target=target, expr=expr, ptr=True, source=source)

    def visit_specification(self, o, source=None):
        body = tuple(self.visit(c) for c in o.getchildren())
        body = tuple(c for c in body if c is not None)
        # Wrap spec area into a separate Scope
        return Section(body=body, source=source)

    def visit_declaration(self, o, source=None):
        if len(o.attrib) == 0:
            return None  # Empty element, skip
        elif o.find('save-stmt') is not None:
            return Intrinsic(text=source.string, source=source)
        elif o.find('implicit-stmt') is not None:
            return Intrinsic(text=source.string, source=source)
        elif o.find('access-spec') is not None:
            # PUBLIC or PRIVATE declarations
            return Intrinsic(text=source.string, source=source)
        elif o.attrib['type'] == 'variable':
            if o.find('end-type-stmt') is not None:
                # We are dealing with a derived type
                derived_name = o.find('end-type-stmt').attrib['id']
                declarations = []

                # Process any associated comments or pragams
                comments = [self.visit(c) for c in o.findall('comment')]
                pragmas = [c for c in comments if isinstance(c, Pragma)]
                comments = [c for c in comments if not isinstance(c, Pragma)]

                # This is customized in our dedicated branch atm,
                # and really, really hacky! :(
                types = o.findall('type')
                components = o.findall('components')
                attributes = [None] * len(types)
                elements = o.getchildren()
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
                    kind = t.find('kind/name').attrib['id'] if t.find('kind') else None
                    type = BaseType(typename, kind=kind, pointer='POINTER' in attrs,
                                    allocatable='ALLOCATABLE' in attrs,
                                    source=t_source)

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
                            dimensions = [RangeIndex(lower=None, upper=None) for _ in range(dim_count)]
                        else:
                            dimensions = as_tuple(self.visit(c) for c in v)
                        dimensions = as_tuple(d for d in dimensions if d is not None)
                        dimensions = dimensions if len(dimensions) > 0 else None
                        v_source = extract_source(v.attrib, self._raw_source)

                        variables += [self.Variable(name=v.attrib['name'], type=type,
                                                    dimensions=dimensions, shape=dimensions,
                                                    source=v_source)]
                    declarations += [Declaration(variables=variables, type=type, source=t_source)]
                return TypeDef(name=derived_name, declarations=declarations,
                               pragmas=pragmas, comments=comments, source=source)
            else:
                # We are dealing with a single declaration, so we retrieve
                # all the declaration-level information first.
                typename = o.find('type').attrib['name']
                kind = o.find('type/kind/name').attrib['id'] if o.find('type/kind') else None
                intent = o.find('intent').attrib['type'] if o.find('intent') else None
                allocatable = o.find('attribute-allocatable') is not None
                pointer = o.find('attribute-pointer') is not None
                parameter = o.find('attribute-parameter') is not None
                optional = o.find('attribute-optional') is not None
                target = o.find('attribute-target') is not None
                if self.typedefs is not None and typename.lower() in self.typedefs:
                    # Create the local variant of the derived type
                    struct_type = self.typedefs[typename.lower()]
                    _type = DerivedType(name=typename, variables=struct_type.variables,
                                        kind=kind, intent=intent,
                                        allocatable=allocatable, pointer=pointer,
                                        optional=optional, parameter=parameter,
                                        target=target, source=source)
                else:
                    # Create a basic variable type
                    _type = BaseType(name=typename, kind=kind, intent=intent,
                                     allocatable=allocatable, pointer=pointer,
                                     optional=optional, parameter=parameter,
                                     target=target, source=source)
                dims = o.find('dimensions')
                dimensions = None if dims is None else as_tuple(self.visit(dims))
                variables = [self.visit(v, type=_type, dimensions=dimensions)
                             for v in o.findall('variables/variable')]
                variables = [v for v in variables if v is not None]
                return Declaration(variables=variables, type=_type,
                                   dimensions=dimensions, source=source)
        elif o.attrib['type'] == 'implicit':
            return Intrinsic(text=source.string, source=source)
        elif o.attrib['type'] == 'intrinsic':
            return Intrinsic(text=source.string, source=source)
        elif o.attrib['type'] == 'data':
            # Data declaration blocks
            declarations = []
            for variables, values in zip(o.findall('variables'), o.findall('values')):
                variable = self.visit(variables)
                # Lists of literal values are again nested, so extract
                # them recursively.
                lit = values.find('literal')  # We explicitly recurse on l
                vals = []
                while lit.find('literal') is not None:
                    vals += [self.visit(lit)]
                    lit = lit.find('literal')
                vals += [self.visit(lit)]
                declarations += [DataDeclaration(variable=variable, values=vals, source=source)]
            return tuple(declarations)
        else:
            raise NotImplementedError('Unknown declaration type encountered: %s' % o.attrib['type'])

    def visit_associate(self, o, source=None):
        associations = OrderedDict()
        for a in o.findall('header/keyword-arguments/keyword-argument'):
            var = self.visit(a.find('name'))
            assoc_name = a.find('association').attrib['associate-name']
            associations[var] = self.Variable(name=assoc_name)
        body = self.visit(o.find('body'))
        return Scope(body=as_tuple(body), associations=associations)

    def visit_allocate(self, o, source=None):
        variables = as_tuple(self.visit(v) for v in o.findall('expressions/expression/name'))
        return Allocation(variables=variables, source=source)

    def visit_deallocate(self, o, source=None):
        variable = self.visit(o.find('expressions/expression/name'))
        return Deallocation(variable=variable, source=source)

    def visit_use(self, o, source=None):
        symbols = [n.attrib['id'] for n in o.findall('only/name')]
        return Import(module=o.attrib['name'], symbols=symbols, source=source)

    def visit_print(self, o, source=None):
        return Intrinsic(text=source.string, source=source)

    def visit_directive(self, o, source=None):
        if '#include' in o.attrib['text']:
            # Straight pipe-through node for header includes (#include ...)
            match = re.search('#include\s[\'"](?P<module>.*)[\'"]', o.attrib['text'])
            module = match.groupdict()['module']
            return Import(module=module, c_import=True, source=source)
        else:
            return Intrinsic(text=source.string, source=source)

    def visit_open(self, o, source=None):
        return Intrinsic(text=source.string, source=source)

    visit_close = visit_open
    visit_read = visit_open
    visit_write = visit_open
    visit_format = visit_open

    def visit_call(self, o, source=None):
        # Need to re-think this: the 'name' node already creates
        # a 'Variable', which in this case is wrong...
        name = o.find('name').attrib['id']
        args = tuple(self.visit(i) for i in o.findall('name/subscripts/subscript'))
        kwargs = list([self.visit(i) for i in o.findall('name/subscripts/argument')])
        return Call(name=name, arguments=args, kwarguments=kwargs, source=source)

    def visit_argument(self, o, source=None):
        key = o.attrib['name']
        val = self.visit(o.find('name'))
        return key, val

    def visit_exit(self, o, source=None):
        return Intrinsic(text=source.string, source=source)

    # Expression parsing below; maye move to its own parser..?

    def visit_name(self, o, source=None):

        def generate_variable(vname, indices, kwargs, parent, source):
            if vname.upper() == 'RESHAPE':
                return reshape(indices[0], shape=indices[1])
            elif vname.upper() in ['MIN', 'MAX', 'EXP', 'SQRT', 'ABS', 'LOG',
                                   'SELECTED_REAL_KIND', 'ALLOCATED', 'PRESENT']:
                return InlineCall(vname, parameters=indices, kw_parameters=kwargs)
            elif vname.upper() in ['REAL', 'INT']:
                return Cast(vname, expression=indices[0], kind=kwargs.get('kind', None))
            elif indices is not None and len(indices) == 0:
                # HACK: We (most likely) found a call out to a C routine
                return InlineCall(o.attrib['id'], parameters=indices)
            else:
                shape = self.shape_map.get(vname, None) if self.shape_map else None
                _type = self.type_map.get(vname, None) if self.type_map else None

                # If the (possibly external) struct definitions exists
                # derive the shape and real type from it.
                if parent is not None and parent.type is not None:
                    if isinstance(parent.type, DerivedType):
                        typevar = [v for v in parent.type.variables
                                   if v.name.lower() == vname.lower()][0]
                        _type = typevar.type
                        if isinstance(typevar, Array):
                            shape = typevar.shape

                var = self.Variable(name=vname, dimensions=indices, parent=parent,
                                    shape=shape, type=_type, source=source)
                return var

        # Creating compound variables is a bit tricky, so let's first
        # process all our children and shove them into a deque
        _children = deque(self.visit(c) for c in o.getchildren())
        _children = deque(c for c in _children if c is not None)

        # Hack: find kwargs for Casts
        kwargs = list([self.visit(i) for i in o.findall('subscripts/argument')])
        kwargs = {k: v for k, v in kwargs}

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
        _type = kwargs.get('type', None)
        if 'id' not in o.attrib and 'name' not in o.attrib:
            return None
        name = o.attrib['id'] if 'id' in o.attrib else o.attrib['name']
        if o.find('dimensions') is not None:
            dimensions = tuple(self.visit(d) for d in o.find('dimensions'))
            dimensions = tuple(d for d in dimensions if d is not None)
        else:
            dimensions = kwargs.get('dimensions', None)
        initial = None if o.find('initial-value') is None else self.visit(o.find('initial-value'))
        return self.Variable(name=name, dimensions=dimensions, shape=dimensions,
                             type=_type, initial=initial, source=source)

    def visit_part_ref(self, o, source=None):
        # Return a pure string, as part of a variable name
        return o.attrib['id']

    def visit_literal(self, o, source=None):
        value = o.attrib['value']
        type = o.attrib['type'] if 'type' in o.attrib else None
        kind_param = o.find('kind-param')
        kind = kind_param.attrib['kind'] if kind_param is not None else None
        # Cast to the right type if given
        value = int(value) if type == 'int' else value
        value = float(value) if type == 'real' else value
        return Literal(value=value, kind=kind, type=type, source=source)

    def visit_subscripts(self, o, source=None):
        return tuple(self.visit(c)for c in o.getchildren()
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
            return RangeIndex(lower=lower, upper=upper, step=step)
        elif 'type' in o.attrib and o.attrib['type'] == "upper-bound-assumed-shape":
            lower = self.visit(o[0])
            return RangeIndex(lower=lower, upper=None)
        elif o.find('name'):
            return self.visit(o.find('name'))
        elif o.find('literal'):
            return self.visit(o.find('literal'))
        elif o.find('operation'):
            return self.visit(o.find('operation'))
        elif o.find('array-constructor-values'):
            return self.visit(o.find('array-constructor-values'))
        else:
            return RangeIndex(lower=None, upper=None, step=None)

    visit_dimension = visit_subscript

    def visit_array_constructor_values(self, o, source=None):
        values = [self.visit(v) for v in o.findall('value')]
        values = [v for v in values if v is not None]  # Filter empy values
        return LiteralList(values=values)

    def visit_operation(self, o, source=None):
        """
        Construct expressions from individual operations, using left-recursion.
        """
        ops = [self.visit(op) for op in o.findall('operator')]
        ops = [str(op).lower() for op in ops if op is not None]  # Filter empty ops
        exprs = [self.visit(c) for c in o.findall('operand')]
        exprs = [e for e in exprs if e is not None]  # Filter empty operands

#        def booleanize(expr):
#            """
#            Super-hacky helper function to force boolean array when needed
#            """
#            if isinstance(expr, Array) and not expr.is_Boolean:
#                return expr.clone(type=BaseType(name='logical'), cache=self._cache)
#            else:
#                return expr

        # Left-recurse on the list of operations and expressions
        exprs = deque(exprs)
        expression = exprs.popleft()
        for op in ops:

            if op == '+':
                expression = Sum((expression, exprs.popleft()))
            elif op == '-':
                if len(exprs) > 0:
                    # Binary minus
                    expression = Sum((expression, Product((-1, exprs.popleft()))))
                else:
                    # Unary minus
                    expression = Product((-1, expression))
            elif op == '*':
                expression = Product((expression, exprs.popleft()))
            elif op == '/':
                expression = Quotient(numerator=expression, denominator=exprs.popleft())
            elif op == '**':
                expression = Power(base=expression, exponent=exprs.popleft())
            elif op == '==' or op == '.eq.':
                expression = Comparison(expression, '==', exprs.popleft())
            elif op == '/=' or op == '.ne.':
                expression = Comparison(expression, '!=', exprs.popleft())
            elif op == '>' or op == '.gt.':
                expression = Comparison(expression, '>', exprs.popleft())
            elif op == '<' or op == '.lt.':
                expression = Comparison(expression, '<', exprs.popleft())
            elif op == '>=' or op == '.ge.':
                expression = Comparison(expression, '>=', exprs.popleft())
            elif op == '<=' or op == '.le.':
                expression = Comparison(expression, '<=', exprs.popleft())
            elif op == '.and.':
                expression = LogicalAnd((expression, exprs.popleft()))
            elif op == '.or.':
                expression = LogicalOr((expression, exprs.popleft()))
            elif op == '.not.':
                expression = LogicalNot(expression)
            elif op == '.eqv.':
                e = (expression, exprs.popleft())
                expression = LogicalOr((LogicalAnd(e), LogicalNot(LogicalOr(e))))
            elif op == '.neqv.':
                e = (expression, exprs.popleft())
                expression = LogicalAnd((LogicalNot(LogicalAnd(e)), LogicalOr(e)))
            else:
                raise RuntimeError('OFP: Unknown expression operator: %s' % op)

        if o.find('parenthesized_expr') is not None:
            # Force explicitly parenthesised operations
            if isinstance(expression, Sum):
                expression = ParenthesisedAdd(expression.children)
            if isinstance(expression, Product):
                expression = ParenthesisedMul(expression.children)
            if isinstance(expression, Power):
                expression = ParenthesisedPow(expression.base, expression.exponent)

        assert len(exprs) == 0
        return expression

    def visit_operator(self, o, source=None):
        return o.attrib['operator']


@timeit(log_level=DEBUG)
def parse_ofp_ast(ast, pp_info=None, raw_source=None, shape_map=None,
                  type_map=None, typedefs=None, cache=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    # Parse the raw OMNI language AST
    ir = OFP2IR(shape_map=shape_map, type_map=type_map, typedefs=typedefs,
                raw_source=raw_source, cache=cache).visit(ast)

    # Apply postprocessing rules to re-insert information lost during preprocessing
    for r_name, rule in blacklist.items():
        info = pp_info[r_name] if pp_info is not None and r_name in pp_info else None
        ir = rule.postprocess(ir, info)

    # Perform soime minor sanitation tasks
    ir = inline_comments(ir)
    ir = cluster_comments(ir)
    ir = inline_pragmas(ir)

    return ir
