from pathlib import Path
import xml.etree.ElementTree as ET

from loki.frontend.source import Source
from loki.frontend.util import inline_comments, cluster_comments, inline_labels
from loki.visitors import GenericVisitor, FindNodes, Transformer
import loki.ir as ir
import loki.expression.symbols as sym
from loki.expression import ExpressionDimensionsMapper, StringConcat, FindTypedSymbols, SubstituteExpressions
from loki.logging import info, debug, DEBUG, warning
from loki.config import config
from loki.tools import (
    as_tuple, timeit, execute, gettempdir, filehash, CaseInsensitiveDict
)
from loki.types import BasicType, DerivedType, ProcedureType, Scope, SymbolAttributes


__all__ = ['parse_omni_source', 'parse_omni_file', 'parse_omni_ast']


@timeit(log_level=DEBUG)
def parse_omni_file(filename, xmods=None):
    """
    Deploy the OMNI compiler's frontend (F_Front) to generate the OMNI AST.

    Note that the intermediate XML files can be dumped to file via by setting
    the environment variable ``LOKI_OMNI_DUMP_XML``.
    """
    dump_xml_files = config['omni-dump-xml']

    filepath = Path(filename)
    info("[Loki::OMNI] Parsing %s" % filepath)

    xml_path = filepath.with_suffix('.xml')
    xmods = xmods or []

    cmd = ['F_Front', '-fleave-comment']
    for m in xmods:
        cmd += ['-M', '%s' % Path(m)]
    cmd += ['%s' % filepath]

    if dump_xml_files:
        # Parse AST from xml file dumped to disk
        cmd += ['-o', '%s' % xml_path]
        execute(cmd)
        return ET.parse(str(xml_path)).getroot()

    result = execute(cmd, silent=False, capture_output=True, text=True)
    return ET.fromstring(result.stdout)


@timeit(log_level=DEBUG)
def parse_omni_source(source, filepath=None, xmods=None):
    """
    Deploy the OMNI compiler's frontend (F_Front) to AST for a source string.
    """
    # Use basename of filepath if given
    if filepath is None:
        filepath = Path(filehash(source, prefix='omni-', suffix='.f90'))
    else:
        filepath = filepath.with_suffix('.omni{}'.format(filepath.suffix))

    # Always store intermediate flies in tmp dir
    filepath = gettempdir()/filepath.name

    debug('[Loki::OMNI] Writing temporary source {}'.format(str(filepath)))
    with filepath.open('w') as f:
        f.write(source)

    return parse_omni_file(filename=filepath, xmods=xmods)


@timeit(log_level=DEBUG)
def parse_omni_ast(ast, definitions=None, type_map=None, symbol_map=None,
                   raw_source=None, scope=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    # Parse the raw OMNI language AST
    _ir = OMNI2IR(type_map=type_map, definitions=definitions, symbol_map=symbol_map,
                  raw_source=raw_source, scope=scope).visit(ast)

    # Perform some minor sanitation tasks
    _ir = inline_comments(_ir)
    _ir = cluster_comments(_ir)
    _ir = inline_labels(_ir)

    return _ir


class OMNI2IR(GenericVisitor):
    # pylint: disable=no-self-use  # Stop warnings about visitor methods that could do without self
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    _omni_types = {
        'Fint': 'INTEGER',
        'Freal': 'REAL',
        'Flogical': 'LOGICAL',
        'Fcharacter': 'CHARACTER',
        'Fcomplex': 'COMPLEX',
        'int': 'INTEGER',
        'real': 'REAL',
    }

    def __init__(self, definitions=None, type_map=None, symbol_map=None,
                 raw_source=None, scope=None):
        super().__init__()

        self.definitions = CaseInsensitiveDict((d.name, d) for d in as_tuple(definitions))
        self.type_map = type_map
        self.symbol_map = symbol_map
        self.raw_source = raw_source
        self.scope = scope

    def _struct_type_variables(self, o, scope, parent=None, source=None):
        """
        Helper routine to build the list of variables for a `FstructType` node
        """
        variables = []
        for s in o.find('symbols'):
            vname = s.find('name').text
            if parent is not None:
                vname = '%s%%%s' % (parent, vname)
            dimensions = None

            t = s.attrib['type']
            if t in self.type_map:
                vtype = self.visit(self.type_map[t])
                dims = self.type_map[t].findall('indexRange')
                if dims:
                    dimensions = as_tuple(self.visit(d) for d in dims)
                    vtype = vtype.clone(shape=dimensions)
            else:
                typename = self._omni_types.get(t, t)
                vtype = SymbolAttributes(BasicType.from_fortran_type(typename))

            if dimensions:
                dimensions = sym.ArraySubscript(dimensions, source=source)
            variables += [
                sym.Variable(name=vname, dimensions=dimensions, type=vtype, scope=scope, source=source)]
        return variables

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        return super().lookup_method(instance)

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        file = o.attrib.get('file', None)
        lineno = o.attrib.get('lineno', None)
        if lineno:
            lineno = int(lineno)
        source = Source(lines=(lineno, lineno), file=file)
        return super().visit(o, source=source, **kwargs)

    def visit_Element(self, o, source=None, **kwargs):
        """
        Universal default for XML element types
        """
        warning('No specific handler for node type %s', o.__class__.name)
        children = tuple(self.visit(c, **kwargs) for c in o)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_FuseDecl(self, o, source=None):
        name = o.attrib['name']
        module = self.definitions.get(name, None)
        scope = self.scope
        if module is not None:
            for k, v in module.symbols.items():
                scope.symbols[k] = v.clone(imported=True, module=module)
        return ir.Import(module=name, c_import=False, source=source)

    def visit_FuseOnlyDecl(self, o, source=None):
        name = o.attrib['name']
        symbols = tuple(self.visit(c) for c in o.findall('renamable'))
        module = self.definitions.get(name, None)
        scope = self.scope
        if module is None:
            scope.symbols.update({s.name: SymbolAttributes(BasicType.DEFERRED, imported=True) for s in symbols})
        else:
            for s in symbols:
                scope.symbols[s.name] = module.symbols[s.name].clone(imported=True, module=module)
        symbols = tuple(s.clone(scope=scope) for s in symbols)
        return ir.Import(module=name, symbols=symbols, c_import=False, source=source)

    def visit_renamable(self, o, source=None):
        return sym.Variable(name=o.attrib['use_name'], source=source)

    def visit_FinterfaceDecl(self, o, source=None):
        # TODO: We can only deal with interface blocks partially
        # in the frontend, as we cannot yet create `Subroutine` objects
        # cleanly here. So for now we skip this, but we should start
        # moving the `Subroutine` constructors into the frontend.
        spec = None
        body = tuple(self.visit(c) for c in o)
        return ir.Interface(spec=spec, body=body, source=source)

    def visit_FfunctionDecl(self, o, source=None):
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel

        # Create a scope
        scope = Scope(parent=self.scope)

        # Name and dummy args
        name = o.find('name').text
        ftype = self.type_map[o.find('name').attrib['type']]
        is_function = ftype.attrib['return_type'] != 'Fvoid'
        args = tuple(a.text for a in ftype.findall('params/name'))

        # Generate spec
        # HACK: temporarily replace the scope property until we pass down scopes properly
        parent_scope, self.scope = self.scope, scope
        spec = self.visit(o.find('declarations'))
        self.scope = parent_scope

        # Filter out the declaration for the subroutine name but keep it for functions (since
        # this declares the return type)
        if not is_function:
            mapper = {d: None for d in FindNodes(ir.Declaration).visit(spec)
                      if d.variables[0].name == name}
            spec = Transformer(mapper, invalidate_source=False).visit(spec)

        return Subroutine(name=name, args=args, spec=spec, ast=o, scope=scope, is_function=is_function,
                          source=source)

    def visit_declarations(self, o, source=None, **kwargs):
        body = tuple(self.visit(c, **kwargs) for c in o)
        body = tuple(c for c in body if c is not None)
        return ir.Section(body=body, source=source)

    def visit_body(self, o, source=None, **kwargs):
        body = tuple(self.visit(c, **kwargs) for c in o)
        body = tuple(c for c in body if c is not None)
        return body

    def visit_varDecl(self, o, source=None):
        # OMNI has only one variable per declaration, find and create that
        name = o.find('name')
        variable = self.visit(name)

        # Create the declared type
        if name.attrib['type'] in self._omni_types:
            # Intrinsic scalar type
            t = self._omni_types[name.attrib['type']]
            _type = SymbolAttributes(BasicType.from_fortran_type(t))
            dimensions = None

        elif name.attrib['type'] in self.type_map:
            # Type with attributes or derived type
            tast = self.type_map[name.attrib['type']]
            _type = self.visit(tast)

            dimensions = as_tuple(self.visit(d) for d in tast.findall('indexRange'))
            if dimensions:
                _type = _type.clone(shape=dimensions)
                variable = variable.clone(dimensions=dimensions)

            if isinstance(_type.dtype, ProcedureType):
                return_type = tast.attrib['return_type']
                if return_type != 'Fvoid':
                    if return_type in self._omni_types:
                        dtype = BasicType.from_fortran_type(self._omni_types[return_type])
                        _type.return_type = dtype
                    elif return_type in self.type_map:
                        # Any attributes (like kind) are on the return type, therefore we
                        # overwrite the _type here
                        _type = self.visit(self.type_map[return_type])
                        _type.return_type = _type.dtype
                    else:
                        raise ValueError
                    _type.dtype = ProcedureType(variable.name, is_function=True)
                else:
                    _type.dtype = ProcedureType(variable.name, is_function=False)

                if tast.attrib.get('is_external') == 'true':
                    _type.external = True
                else:
                    # This is the declaration of the return type inside a function, which is
                    # why we restore the dtype from return_type
                    _type = _type.clone(dtype=_type.return_type or BasicType.DEFERRED, return_type=None)

        else:
            raise ValueError

        scope = self.scope
        if o.find('value') is not None:
            _type.initial = self.visit(o.find('value'))
            # TODO: Apply some rescope-visitor here
            rescope_map = {var: var.clone(scope=scope) for var in FindTypedSymbols().visit(_type.initial)}
            _type.initial = SubstituteExpressions(rescope_map).visit(_type.initial)
        if _type.kind is not None and isinstance(_type.kind, sym.TypedSymbol):
            # TODO: put it in the right scope (Rescope Visitor)
            _type = _type.clone(kind=_type.kind.clone(scope=scope))

        scope.symbols[variable.name] = _type
        variables = (variable.clone(scope=scope),)
        return ir.Declaration(variables=variables, external=_type.external is True, source=source)

    def visit_FstructDecl(self, o, source=None):
        name = o.find('name')

        # Initialize a local scope for typedef objects
        typedef_scope = Scope(parent=self.scope)

        # Built the list of derived type members
        variables = self._struct_type_variables(self.type_map[name.attrib['type']],
                                                scope=typedef_scope)

        # Build individual declarations for each member
        declarations = as_tuple(ir.Declaration(variables=(v, )) for v in variables)
        typedef = ir.TypeDef(name=name.text, body=as_tuple(declarations), scope=typedef_scope)

        # Now make the typedef known in its scope's type table
        self.scope.symbols[name.text] = SymbolAttributes(DerivedType(name=name.text, typedef=typedef))

        return typedef

    def visit_FdataDecl(self, o, source=None):
        variable = self.visit(o.find('varList'))
        values = self.visit(o.find('valueList'))
        return ir.DataDeclaration(variable=variable, values=values, source=source)

    def visit_varList(self, o, source=None):
        children = tuple(self.visit(c) for c in o)
        children = tuple(c for c in children if c is not None)
        return children

    visit_valueList = visit_varList

    def visit_FbasicType(self, o, source=None):
        ref = o.attrib.get('ref', None)
        if ref in self._omni_types:
            dtype = BasicType.from_fortran_type(self._omni_types[ref])
            kind = self.visit(o.find('kind')) if o.find('kind') is not None else None
            length = self.visit(o.find('len')) if o.find('len') is not None else None
            _type = SymbolAttributes(dtype, kind=kind, length=length)
        elif ref in self.type_map:
            _type = self.visit(self.type_map[ref])
        else:
            raise ValueError

        shape = o.findall('indexRange')
        if shape:
            _type.shape = tuple(self.visit(s) for s in shape)

        # OMNI types are build recursively from references (Matroshka-style)
        _type.intent = o.attrib.get('intent', None)
        _type.allocatable = o.attrib.get('is_allocatable', 'false') == 'true'
        _type.pointer = o.attrib.get('is_pointer', 'false') == 'true'
        _type.optional = o.attrib.get('is_optional', 'false') == 'true'
        _type.parameter = o.attrib.get('is_parameter', 'false') == 'true'
        _type.target = o.attrib.get('is_target', 'false') == 'true'
        _type.contiguous = o.attrib.get('is_contiguous', 'false') == 'true'
        return _type

    def visit_FfunctionType(self, o, source=None):
        if o.attrib['return_type'] == 'Fvoid':
            return_type = None
        elif o.attrib['return_type'] in self._omni_types:
            return_type = BasicType.from_fortran_type(self._omni_types[o.attrib['return_type']])
        elif o.attrib['return_type'] in self.type_map:
            return_type = self.visit(self.type_map[o.attrib['return_type']])
        else:
            raise ValueError

        # OMNI doesn't give us the function name at this point
        dtype = ProcedureType('UNKNOWN', is_function=return_type is not None)
        return SymbolAttributes(dtype, return_type=return_type)

    def visit_FstructType(self, o, source=None):
        # We have encountered a derived type as part of the declaration in the spec
        # of a routine.
        name = o.attrib['type']
        if self.symbol_map is not None and name in self.symbol_map:
            name = self.symbol_map[name].find('name').text

        # Check if we know that type already
        dtype = self.scope.symbols.lookup(name, recursive=True)
        if dtype is None or dtype.dtype == BasicType.DEFERRED:
            dtype = DerivedType(name=name, typedef=BasicType.DEFERRED)
        else:
            dtype = dtype.dtype

        return SymbolAttributes(dtype)

    def visit_value(self, o, source=None):
        return self.visit(o[0])

    visit_kind = visit_value
    visit_len = visit_value

    def visit_associateStatement(self, o, source=None):
        associations = tuple(self.visit(c) for c in o.findall('symbols/id'))

        # Create a scope for the associate
        parent_scope = self.scope
        scope = parent_scope  # TODO: actually create own scope

        # TODO: Apply some rescope-visitor here
        rescoped_associations = []
        for expr, name in associations:
            rescope_map = {var: var.clone(scope=parent_scope) for var in FindTypedSymbols().visit(expr)}
            expr = SubstituteExpressions(rescope_map).visit(expr)
            name = name.clone(scope=scope)
            rescoped_associations += [(expr, name)]
        associations = as_tuple(rescoped_associations)

        # Update symbol table for associates
        for expr, name in associations:
            if isinstance(expr, sym.TypedSymbol):
                # Use the type of the associated variable
                _type = parent_scope.symbols.lookup(expr.name)
                if isinstance(expr, sym.Array) and expr.dimensions is not None:
                    shape = ExpressionDimensionsMapper()(expr)
                    _type = _type.clone(shape=shape)
            else:
                # TODO: Handle data type and shape of complex expressions
                shape = ExpressionDimensionsMapper()(expr)
                _type = SymbolAttributes(BasicType.DEFERRED, shape=shape)
            scope.symbols[name.name] = _type

        body = self.visit(o.find('body'))
        return ir.Associate(body=body, associations=associations, source=source)

    def visit_id(self, o, source=None):
        expr = self.visit(o.find('value'))
        name = self.visit(o.find('name'))
        return expr, name

    def visit_exprStatement(self, o, source=None):
        return self.visit(o[0])

    def visit_FcommentLine(self, o, source=None):
        return ir.Comment(text=o.text, source=source)

    def visit_FpragmaStatement(self, o, source=None):
        keyword = o.text.split(' ')[0]
        content = ' '.join(o.text.split(' ')[1:])
        return ir.Pragma(keyword=keyword, content=content, source=source)

    def visit_FassignStatement(self, o, source=None):
        lhs = self.visit(o[0])
        rhs = self.visit(o[1])
        return ir.Assignment(lhs=lhs, rhs=rhs, source=source)

    def visit_FallocateStatement(self, o, source=None):
        variables = tuple(self.visit(c) for c in o.findall('alloc'))
        if o.find('allocOpt') is not None:
            data_source = self.visit(o.find('allocOpt'))
            return ir.Allocation(variables=variables, data_source=data_source, source=source)
        return ir.Allocation(variables=variables, source=source)

    def visit_allocOpt(self, o, source=None):
        return self.visit(o[0])

    def visit_FdeallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        variables = as_tuple(self.visit(a[0]) for a in allocs)
        return ir.Deallocation(variables=variables, source=source)

    def visit_FnullifyStatement(self, o, source=None):
        variables = tuple(self.visit(c) for c in o.findall('alloc'))
        return ir.Nullify(variables=variables, source=source)

    def visit_alloc(self, o, source=None):
        variable = self.visit(o[0])
        if o.find('arrayIndex') is not None:
            dimensions = tuple(self.visit(c) for c in o.findall('arrayIndex'))
            variable = variable.clone(dimensions=dimensions)
        return variable

    def visit_FwhereStatement(self, o, source=None):
        condition = self.visit(o.find('condition'))
        body = self.visit(o.find('then/body'))
        if o.find('else') is not None:
            default = self.visit(o.find('else/body'))
        else:
            default = ()
        return ir.MaskedStatement(condition, body, default, source=source)

    def visit_FpointerAssignStatement(self, o, source=None):
        target = self.visit(o[0])
        expr = self.visit(o[1])
        return ir.Assignment(lhs=target, rhs=expr, ptr=True, source=source)

    def visit_FdoWhileStatement(self, o, source=None):
        assert o.find('condition') is not None
        assert o.find('body') is not None
        condition = self.visit(o.find('condition'))
        body = self.visit(o.find('body'))
        return ir.WhileLoop(condition=condition, body=body, source=source)

    def visit_FdoStatement(self, o, source=None):
        assert o.find('body') is not None
        body = self.visit(o.find('body'))
        if o.find('Var') is None:
            # We are in an unbound do loop
            return ir.WhileLoop(condition=None, body=body, source=source)
        variable = self.visit(o.find('Var'))
        lower = self.visit(o.find('indexRange/lowerBound'))
        upper = self.visit(o.find('indexRange/upperBound'))
        step = self.visit(o.find('indexRange/step'))
        # Drop OMNI's `:1` step counting for ranges in the name of consistency
        step = None if step == '1' else step
        bounds = sym.LoopRange((lower, upper, step), source=source)
        return ir.Loop(variable=variable, body=body, bounds=bounds, source=source)

    def visit_FdoLoop(self, o, source=None):
        raise NotImplementedError

    def visit_FifStatement(self, o, source=None):
        condition = self.visit(o.find('condition'))
        body = self.visit(o.find('then/body'))
        if o.find('else'):
            else_body = self.visit(o.find('else/body'))
        else:
            else_body = ()
        return ir.Conditional(condition=condition, body=body, else_body=else_body, source=source)

    def visit_condition(self, o, source=None):
        return self.visit(o[0])

    def visit_FselectCaseStatement(self, o, source=None):
        expr = self.visit(o.find('value'))
        cases = [self.visit(case) for case in o.findall('FcaseLabel')]
        values, bodies = zip(*cases)
        if None in values:
            else_index = values.index(None)
            else_body = as_tuple(bodies[else_index])
            values = values[:else_index] + values[else_index+1:]
            bodies = bodies[:else_index] + bodies[else_index+1:]
        else:
            else_body = ()
        return ir.MultiConditional(expr=expr, values=values, bodies=bodies, else_body=else_body, source=source)

    def visit_FcaseLabel(self, o, source=None):
        values = [self.visit(value) for value in list(o) if value.tag in ('value', 'indexRange')]
        if not values:
            values = None
        elif len(values) == 1:
            values = values.pop()
        body = self.visit(o.find('body'))
        return as_tuple(values) or None, as_tuple(body)

    def visit_FmemberRef(self, o, **kwargs):
        parent = self.visit(o.find('varRef'))
        name = '{}%{}'.format(parent.name, o.attrib['member'])
        variable = sym.Variable(name=name, parent=parent)
        return variable

    def visit_name(self, o, **kwargs):
        return sym.Variable(name=o.text, source=kwargs.get('source'))

    visit_Var = visit_name

    def visit_FarrayRef(self, o, source=None):
        var = self.visit(o.find('varRef'))
        dimensions = as_tuple(self.visit(i) for i in o[1:])
        var = var.clone(dimensions=dimensions)
        return var

    def visit_varRef(self, o, source=None):
        return self.visit(o[0])

    def visit_arrayIndex(self, o, source=None):
        return self.visit(o[0])

    def visit_indexRange(self, o, source=None):
        lbound = o.find('lowerBound')
        lower = self.visit(lbound) if lbound is not None else None
        ubound = o.find('upperBound')
        upper = self.visit(ubound) if ubound is not None else None
        st = o.find('step')
        step = self.visit(st) if st is not None else None
        # Drop OMNI's `:1` step counting for ranges in the name of consistency
        step = None if step == '1' else step
        return sym.RangeIndex((lower, upper, step), source=source)

    def visit_lowerBound(self, o, source=None):
        return self.visit(o[0])

    visit_upperBound = visit_lowerBound
    visit_step = visit_lowerBound

    def visit_FrealConstant(self, o, source=None):
        if 'kind' in o.attrib and not 'd' in o.text.lower():
            _type = self.visit(self.type_map[o.attrib.get('type')])
            return sym.Literal(value=o.text, type=BasicType.REAL, kind=_type.kind, source=source)
        return sym.Literal(value=o.text, type=BasicType.REAL, source=source)

    def visit_FlogicalConstant(self, o, source=None):
        return sym.Literal(value=o.text, type=BasicType.LOGICAL, source=source)

    def visit_FcharacterConstant(self, o, source=None):
        return sym.Literal(value='"%s"' % o.text, type=BasicType.CHARACTER, source=source)

    def visit_FintConstant(self, o, source=None):
        if 'kind' in o.attrib:
            _type = self.visit(self.type_map[o.attrib.get('type')])
            return sym.Literal(value=int(o.text), type=BasicType.INTEGER, kind=_type.kind, source=source)
        return sym.Literal(value=int(o.text), type=BasicType.INTEGER, source=source)

    def visit_FcomplexConstant(self, o, source=None):
        value = '({})'.format(', '.join('{}'.format(self.visit(v)) for v in list(o)))
        return sym.IntrinsicLiteral(value=value, source=source)

    def visit_FarrayConstructor(self, o, source=None):
        values = as_tuple(self.visit(v) for v in o)
        return sym.LiteralList(values=values)

    def visit_functionCall(self, o, source=None):
        if o.find('name') is not None:
            name = self.visit(o.find('name'))
        elif o.find('FmemberRef') is not None:
            name = self.visit(o.find('FmemberRef'))
        else:
            raise ValueError

        args = o.find('arguments')
        if args is not None:
            args = as_tuple(self.visit(a) for a in args)
            # Separate keyword argument from positional arguments
            kwargs = as_tuple(arg for arg in args if isinstance(arg, tuple))
            args = as_tuple(arg for arg in args if not isinstance(arg, tuple))
        else:
            args, kwargs = (), ()

        if o.attrib.get('type', 'Fvoid') == 'Fvoid':
            # Subroutine call
            return ir.CallStatement(name=name, arguments=args, kwarguments=kwargs, source=source)

        if name.name.lower() in ('real', 'int'):
            assert args
            expr = args[0]
            if kwargs:
                assert len(args) == 1
                assert len(kwargs) == 1 and kwargs[0][0] == 'kind'
                kind = kwargs[0][1]
            else:
                kind = args[1] if len(args) > 1 else None
            return sym.Cast(name, expr, kind=kind, source=source)

        return sym.InlineCall(name, parameters=args, kw_parameters=kwargs, source=source)

    def visit_FcycleStatement(self, o, source=None):
        # TODO: do-construct-name is not preserved
        return ir.Intrinsic(text='cycle', source=source)

    def visit_continueStatement(self, o, source=None):
        return ir.Intrinsic(text='continue', source=source)

    def visit_FexitStatement(self, o, source=None):
        # TODO: do-construct-name is not preserved
        return ir.Intrinsic(text='exit', source=source)

    def visit_FopenStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        return ir.Intrinsic(text='open(%s)' % nargs, source=source)

    def visit_FcloseStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        return ir.Intrinsic(text='close(%s)' % nargs, source=source)

    def visit_FreadStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        values = [self.visit(v) for v in o.find('valueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        args = ', '.join('%s' % v for v in values)
        return ir.Intrinsic(text='read(%s) %s' % (nargs, args), source=source)

    def visit_FwriteStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        values = [self.visit(v) for v in o.find('valueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        args = ', '.join('%s' % v for v in values)
        return ir.Intrinsic(text='write(%s) %s' % (nargs, args), source=source)

    def visit_FprintStatement(self, o, source):
        values = [self.visit(v) for v in o.find('valueList')]
        args = ', '.join('%s' % v for v in values)
        fmt = o.attrib['format']
        return ir.Intrinsic(text='print %s, %s' % (fmt, args), source=source)

    def visit_FformatDecl(self, o, source):
        fmt = 'FORMAT%s' % o.attrib['format']
        return ir.Intrinsic(text=fmt, source=source)

    def visit_namedValue(self, o, source):
        name = o.attrib['name']
        if 'value' in o.attrib:
            return name, o.attrib['value']
        return name, self.visit(list(o)[0])

    def visit_plusExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Sum(exprs, source=source)

    def visit_minusExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Sum((exprs[0], sym.Product((-1, exprs[1]))), source=source)

    def visit_mulExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Product(exprs, source=source)

    def visit_divExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Quotient(*exprs, source=source)

    def visit_FpowerExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Power(base=exprs[0], exponent=exprs[1], source=source)

    def visit_unaryMinusExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 1
        return sym.Product((-1, exprs[0]), source=source)

    def visit_logOrExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        return sym.LogicalOr(exprs, source=source)

    def visit_logAndExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        return sym.LogicalAnd(exprs, source=source)

    def visit_logNotExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 1
        return sym.LogicalNot(exprs[0], source=source)

    def visit_logLTExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '<', exprs[1], source=source)

    def visit_logLEExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '<=', exprs[1], source=source)

    def visit_logGTExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '>', exprs[1], source=source)

    def visit_logGEExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '>=', exprs[1], source=source)

    def visit_logEQExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '==', exprs[1], source=source)

    def visit_logNEQExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '!=', exprs[1], source=source)

    def visit_logEQVExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.LogicalOr((sym.LogicalAnd(exprs), sym.LogicalNot(sym.LogicalOr(exprs))), source=source)

    def visit_logNEQVExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return sym.LogicalAnd((sym.LogicalNot(sym.LogicalAnd(exprs)), sym.LogicalOr(exprs)), source=source)

    def visit_FconcatExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return StringConcat(exprs, source=source)

    def visit_gotoStatement(self, o, source=None):
        label = int(o.attrib['label_name'])
        return ir.Intrinsic(text='go to %d' % label, source=source)

    def visit_statementLabel(self, o, source=None):
        return ir.Comment('__STATEMENT_LABEL__', label=o.attrib['label_name'], source=source)

    def visit_FreturnStatement(self, o, source=None):
        return ir.Intrinsic(text='return', source=source)
