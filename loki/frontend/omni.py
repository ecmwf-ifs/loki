from pathlib import Path
from shutil import which
import xml.etree.ElementTree as ET

from loki.frontend.source import Source
from loki.frontend.util import inline_comments, cluster_comments, inline_labels
from loki.visitors import GenericVisitor, FindNodes, Transformer
from loki import ir
from loki.expression import (
    symbols as sym, operations as op,
    ExpressionDimensionsMapper, StringConcat, AttachScopesMapper
)
from loki.logging import info, debug, DEBUG, warning, error
from loki.config import config
from loki.tools import (
    as_tuple, timeit, execute, gettempdir, filehash, CaseInsensitiveDict
)
from loki.types import BasicType, DerivedType, ProcedureType, SymbolAttributes


__all__ = ['HAVE_OMNI', 'parse_omni_source', 'parse_omni_file', 'parse_omni_ast']


HAVE_OMNI = which('F_Front') is not None
"""Indicate whether OMNI frontend is available."""


@timeit(log_level=DEBUG)
def parse_omni_file(filename, xmods=None):
    """
    Deploy the OMNI compiler's frontend (F_Front) to generate the OMNI AST.

    Note that the intermediate XML files can be dumped to file via by setting
    the environment variable ``LOKI_OMNI_DUMP_XML``.
    """
    if not HAVE_OMNI:
        error('OMNI is not available. Is "F_Front" in the search path?')

    dump_xml_files = config['omni-dump-xml']

    filepath = Path(filename)
    info(f'[Loki::OMNI] Parsing {filepath}')

    xml_path = filepath.with_suffix('.xml')
    xmods = xmods or []

    cmd = ['F_Front', '-fleave-comment']
    for m in xmods:
        cmd += ['-M', f'{Path(m)}']
    cmd += [f'{filepath}']

    if dump_xml_files:
        # Parse AST from xml file dumped to disk
        cmd += ['-o', f'{xml_path}']
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
        filepath = filepath.with_suffix(f'.omni{filepath.suffix}')

    # Always store intermediate flies in tmp dir
    filepath = gettempdir()/filepath.name

    debug(f'[Loki::OMNI] Writing temporary source {filepath}')
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
        self.default_scope = scope

    @staticmethod
    def warn_or_fail(msg):
        if config['frontend-strict-mode']:
            error(msg)
            raise NotImplementedError
        warning(msg)

    def _struct_type_variables(self, o, scope, parent=None, **kwargs):
        """
        Helper routine to build the list of variables for a `FstructType` node
        """
        variables = []
        for s in o.find('symbols'):
            vname = s.find('name').text
            if parent is not None:
                vname = f'{parent}%%{vname}'
            dimensions = None

            t = s.attrib['type']
            if t in self.type_map:
                vtype = self.visit(self.type_map[t], **kwargs)
                dims = self.type_map[t].findall('indexRange')
                if dims:
                    dimensions = as_tuple(self.visit(d, **kwargs) for d in dims)
                    vtype = vtype.clone(shape=dimensions)
            else:
                typename = self._omni_types.get(t, t)
                vtype = SymbolAttributes(BasicType.from_fortran_type(typename))

            variables += [
                sym.Variable(name=vname, dimensions=dimensions, type=vtype, scope=scope, source=kwargs['source'])]
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
        kwargs['source'] = Source(lines=(lineno, lineno), file=file)
        kwargs.setdefault('scope', self.default_scope)
        return super().visit(o, **kwargs)

    def visit_Element(self, o, **kwargs):
        """
        Universal default for XML element types
        """
        warning('No specific handler for node type %s', o.__class__.name)
        children = tuple(self.visit(c, **kwargs) for c in o)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_FuseDecl(self, o, **kwargs):
        name = o.attrib['name']
        module = self.definitions.get(name, None)
        scope = kwargs['scope']
        if module is not None:
            for k, v in module.symbols.items():
                scope.symbols[k] = v.clone(imported=True, module=module)
        return ir.Import(module=name, c_import=False, source=kwargs['source'])

    def visit_FuseOnlyDecl(self, o, **kwargs):
        name = o.attrib['name']
        symbols = tuple(self.visit(c, **kwargs) for c in o.findall('renamable'))
        module = self.definitions.get(name, None)
        scope = kwargs['scope']
        if module is None:
            scope.symbols.update({s.name: SymbolAttributes(BasicType.DEFERRED, imported=True) for s in symbols})
        else:
            for s in symbols:
                scope.symbols[s.name] = module.symbols[s.name].clone(imported=True, module=module)
        symbols = tuple(s.clone(scope=scope) for s in symbols)
        return ir.Import(module=name, symbols=symbols, c_import=False, source=kwargs['source'])

    def visit_renamable(self, o, **kwargs):
        return sym.Variable(name=o.attrib['use_name'], source=kwargs['source'])

    def visit_FinterfaceDecl(self, o, **kwargs):
        # TODO: We can only deal with interface blocks partially
        # in the frontend, as we cannot yet create `Subroutine` objects
        # cleanly here. So for now we skip this, but we should start
        # moving the `Subroutine` constructors into the frontend.
        spec = None
        body = tuple(self.visit(c, **kwargs) for c in o)
        return ir.Interface(spec=spec, body=body, source=kwargs['source'])

    def visit_FfunctionDecl(self, o, **kwargs):
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel

        # Name and dummy args
        name = o.find('name').text
        ftype = self.type_map[o.find('name').attrib['type']]
        is_function = ftype.attrib['return_type'] != 'Fvoid'
        args = tuple(a.text for a in ftype.findall('params/name'))

        parent_scope = kwargs['scope']
        routine = Subroutine(name=name, args=args, ast=o, is_function=is_function,
                             source=kwargs['source'], parent=parent_scope)
        kwargs['scope'] = routine
        routine.spec = self.visit(o.find('declarations'), **kwargs)

        # Filter out the declaration for the subroutine name but keep it for functions (since
        # this declares the return type)
        if not is_function:
            mapper = {d: None for d in FindNodes(ir.Declaration).visit(routine.spec)
                      if d.variables[0].name == name}
            routine.spec = Transformer(mapper, invalidate_source=False).visit(routine.spec)

        return routine

    def visit_declarations(self, o, **kwargs):
        body = tuple(self.visit(c, **kwargs) for c in o)
        body = tuple(c for c in body if c is not None)
        return ir.Section(body=body, source=kwargs['source'])

    def visit_body(self, o, **kwargs):
        body = tuple(self.visit(c, **kwargs) for c in o)
        body = tuple(c for c in body if c is not None)
        return body

    def visit_varDecl(self, o, **kwargs):
        # OMNI has only one variable per declaration, find and create that
        name = o.find('name')
        variable = self.visit(name, **kwargs)

        # Create the declared type
        if name.attrib['type'] in self._omni_types:
            # Intrinsic scalar type
            t = self._omni_types[name.attrib['type']]
            _type = SymbolAttributes(BasicType.from_fortran_type(t))
            dimensions = None

        elif name.attrib['type'] in self.type_map:
            # Type with attributes or derived type
            tast = self.type_map[name.attrib['type']]
            _type = self.visit(tast, **kwargs)

            dimensions = as_tuple(self.visit(d, **kwargs) for d in tast.findall('indexRange'))
            if dimensions:
                _type = _type.clone(shape=dimensions)
                variable = variable.clone(dimensions=dimensions)

            if isinstance(_type.dtype, ProcedureType):
                return_type = tast.attrib['return_type']
                if return_type != 'Fvoid':
                    if return_type in self._omni_types:
                        return_type = SymbolAttributes(BasicType.from_fortran_type(self._omni_types[return_type]))
                    elif return_type in self.type_map:
                        # Any attributes (like kind) are on the return type, therefore we
                        # overwrite the _type here
                        _type = self.visit(self.type_map[return_type], **kwargs)
                        return_type = SymbolAttributes(_type.dtype)
                    else:
                        raise ValueError
                    _type.dtype = ProcedureType(variable.name, is_function=True, return_type=return_type)
                else:
                    _type.dtype = ProcedureType(variable.name, is_function=False)

                if tast.attrib.get('is_external') == 'true':
                    _type.external = True
                else:
                    # This is the declaration of the return type inside a function, which is
                    # why we restore the dtype from return_type
                    _type = _type.clone(dtype=getattr(_type.dtype.return_type, 'dtype', BasicType.DEFERRED))

        else:
            raise ValueError

        scope = kwargs['scope']
        if o.find('value') is not None:
            _type = _type.clone(initial=AttachScopesMapper()(self.visit(o.find('value'), **kwargs), scope=scope))
        if _type.kind is not None:
            _type = _type.clone(kind=AttachScopesMapper()(_type.kind, scope=scope))

        scope.symbols[variable.name] = _type
        variables = (variable.clone(scope=scope),)
        return ir.Declaration(variables=variables, external=_type.external is True, source=kwargs['source'])

    def visit_FstructDecl(self, o, **kwargs):
        name = o.find('name')

        # Instantiate the TypeDef without its body
        # Note: This creates the symbol table for the declarations and
        # the typedef object registers itself in the parent scope
        typedef = ir.TypeDef(name=name.text, body=(), parent=kwargs['scope'])
        kwargs['scope'] = typedef

        # Build the list of derived type members
        struct_type = self.type_map[name.attrib['type']]
        variables = self._struct_type_variables(struct_type, **kwargs)

        if 'extends' in struct_type.attrib:
            self.warn_or_fail('extends attribute for derived types not implemented')

        # Build individual declarations for each member
        declarations = as_tuple(ir.Declaration(variables=(v, )) for v in variables)

        # Finally: update the typedef with its body
        typedef._update(body=declarations)
        return typedef

    def visit_FdataDecl(self, o, **kwargs):
        variable = self.visit(o.find('varList'), **kwargs)
        values = self.visit(o.find('valueList'), **kwargs)
        return ir.DataDeclaration(variable=variable, values=values, source=kwargs['source'])

    def visit_varList(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o)
        children = tuple(c for c in children if c is not None)
        return children

    visit_valueList = visit_varList

    def visit_FbasicType(self, o, **kwargs):
        ref = o.attrib.get('ref', None)
        if ref in self._omni_types:
            dtype = BasicType.from_fortran_type(self._omni_types[ref])
            kind = self.visit(o.find('kind'), **kwargs) if o.find('kind') is not None else None
            length = self.visit(o.find('len'), **kwargs) if o.find('len') is not None else None
            _type = SymbolAttributes(dtype, kind=kind, length=length)
        elif ref in self.type_map:
            _type = self.visit(self.type_map[ref], **kwargs)
        else:
            raise ValueError

        shape = o.findall('indexRange')
        if shape:
            _type.shape = tuple(self.visit(s, **kwargs) for s in shape)

        # OMNI types are build recursively from references (Matroshka-style)
        _type.intent = o.attrib.get('intent', None)
        _type.allocatable = o.attrib.get('is_allocatable', 'false') == 'true'
        _type.pointer = o.attrib.get('is_pointer', 'false') == 'true'
        _type.optional = o.attrib.get('is_optional', 'false') == 'true'
        _type.parameter = o.attrib.get('is_parameter', 'false') == 'true'
        _type.target = o.attrib.get('is_target', 'false') == 'true'
        _type.contiguous = o.attrib.get('is_contiguous', 'false') == 'true'
        return _type

    def visit_FfunctionType(self, o, **kwargs):
        if o.attrib['return_type'] == 'Fvoid':
            return_type = None
        elif o.attrib['return_type'] in self._omni_types:
            return_type = SymbolAttributes(BasicType.from_fortran_type(self._omni_types[o.attrib['return_type']]))
        elif o.attrib['return_type'] in self.type_map:
            return_type = self.visit(self.type_map[o.attrib['return_type']], **kwargs)
        else:
            raise ValueError

        # OMNI doesn't give us the function name at this point
        dtype = ProcedureType('UNKNOWN', is_function=return_type is not None, return_type=return_type)
        return SymbolAttributes(dtype)

    def visit_FstructType(self, o, **kwargs):
        # We have encountered a derived type as part of the declaration in the spec
        # of a routine.
        name = o.attrib['type']
        if self.symbol_map is not None and name in self.symbol_map:
            name = self.symbol_map[name].find('name').text

        # Check if we know that type already
        dtype = kwargs['scope'].symbols.lookup(name, recursive=True)
        if dtype is None or dtype.dtype == BasicType.DEFERRED:
            dtype = DerivedType(name=name, typedef=BasicType.DEFERRED)
        else:
            dtype = dtype.dtype

        return SymbolAttributes(dtype)

    def visit_value(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    visit_kind = visit_value
    visit_len = visit_value

    def visit_associateStatement(self, o, **kwargs):
        associations = tuple(self.visit(c, **kwargs) for c in o.findall('symbols/id'))

        # Create a scope for the associate
        parent_scope = kwargs['scope']
        associate = ir.Associate(associations=(), body=(), parent=parent_scope, source=kwargs['source'])
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

        body = self.visit(o.find('body'), **kwargs)
        associate._update(associations=associations, body=body)
        return associate

    def visit_id(self, o, **kwargs):
        expr = self.visit(o.find('value'), **kwargs)
        name = self.visit(o.find('name'), **kwargs)
        return expr, name

    def visit_exprStatement(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_FcommentLine(self, o, **kwargs):
        return ir.Comment(text=o.text, source=kwargs['source'])

    def visit_FpragmaStatement(self, o, **kwargs):
        keyword = o.text.split(' ')[0]
        content = ' '.join(o.text.split(' ')[1:])
        return ir.Pragma(keyword=keyword, content=content, source=kwargs['source'])

    def visit_FassignStatement(self, o, **kwargs):
        lhs = self.visit(o[0], **kwargs)
        rhs = self.visit(o[1], **kwargs)
        return ir.Assignment(lhs=lhs, rhs=rhs, source=kwargs['source'])

    def visit_FallocateStatement(self, o, **kwargs):
        variables = tuple(self.visit(c, **kwargs) for c in o.findall('alloc'))
        if o.find('allocOpt') is not None:
            data_source = self.visit(o.find('allocOpt'), **kwargs)
            return ir.Allocation(variables=variables, data_source=data_source, source=kwargs['source'])
        return ir.Allocation(variables=variables, source=kwargs['source'])

    def visit_allocOpt(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_FdeallocateStatement(self, o, **kwargs):
        allocs = o.findall('alloc')
        variables = as_tuple(self.visit(a[0], **kwargs) for a in allocs)
        return ir.Deallocation(variables=variables, source=kwargs['source'])

    def visit_FnullifyStatement(self, o, **kwargs):
        variables = tuple(self.visit(c, **kwargs) for c in o.findall('alloc'))
        return ir.Nullify(variables=variables, source=kwargs['source'])

    def visit_alloc(self, o, **kwargs):
        variable = self.visit(o[0], **kwargs)
        if o.find('arrayIndex') is not None:
            dimensions = tuple(self.visit(c, **kwargs) for c in o.findall('arrayIndex'))
            variable = variable.clone(dimensions=dimensions)
        return variable

    def visit_FwhereStatement(self, o, **kwargs):
        condition = self.visit(o.find('condition'), **kwargs)
        body = self.visit(o.find('then/body'), **kwargs)
        if o.find('else') is not None:
            default = self.visit(o.find('else/body'), **kwargs)
        else:
            default = ()
        return ir.MaskedStatement(condition, body, default, source=kwargs['source'])

    def visit_FpointerAssignStatement(self, o, **kwargs):
        target = self.visit(o[0], **kwargs)
        expr = self.visit(o[1], **kwargs)
        return ir.Assignment(lhs=target, rhs=expr, ptr=True, source=kwargs['source'])

    def visit_FdoWhileStatement(self, o, **kwargs):
        assert o.find('condition') is not None
        assert o.find('body') is not None
        condition = self.visit(o.find('condition'), **kwargs)
        body = self.visit(o.find('body'), **kwargs)
        return ir.WhileLoop(condition=condition, body=body, source=kwargs['source'])

    def visit_FdoStatement(self, o, **kwargs):
        assert o.find('body') is not None
        body = self.visit(o.find('body'), **kwargs)
        if o.find('Var') is None:
            # We are in an unbound do loop
            return ir.WhileLoop(condition=None, body=body, source=kwargs['source'])
        variable = self.visit(o.find('Var'), **kwargs)
        lower = self.visit(o.find('indexRange/lowerBound'), **kwargs)
        upper = self.visit(o.find('indexRange/upperBound'), **kwargs)
        step = self.visit(o.find('indexRange/step'), **kwargs)
        # Drop OMNI's `:1` step counting for ranges in the name of consistency
        step = None if step == '1' else step
        bounds = sym.LoopRange((lower, upper, step), source=kwargs['source'])
        return ir.Loop(variable=variable, body=body, bounds=bounds, source=kwargs['source'])

    def visit_FdoLoop(self, o, **kwargs):
        self.warn_or_fail('implicit do loops not implemented')

    def visit_FifStatement(self, o, **kwargs):
        condition = self.visit(o.find('condition'), **kwargs)
        body = self.visit(o.find('then/body'), **kwargs)
        if o.find('else'):
            else_body = self.visit(o.find('else/body'), **kwargs)
        else:
            else_body = ()
        return ir.Conditional(condition=condition, body=body, else_body=else_body, source=kwargs['source'])

    def visit_condition(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_FselectCaseStatement(self, o, **kwargs):
        expr = self.visit(o.find('value'), **kwargs)
        cases = [self.visit(case, **kwargs) for case in o.findall('FcaseLabel')]
        values, bodies = zip(*cases)
        if None in values:
            else_index = values.index(None)
            else_body = as_tuple(bodies[else_index])
            values = values[:else_index] + values[else_index+1:]
            bodies = bodies[:else_index] + bodies[else_index+1:]
        else:
            else_body = ()
        return ir.MultiConditional(expr=expr, values=values, bodies=bodies, else_body=else_body,
                                   source=kwargs['source'])

    def visit_FcaseLabel(self, o, **kwargs):
        values = [self.visit(value, **kwargs) for value in list(o) if value.tag in ('value', 'indexRange')]
        if not values:
            values = None
        elif len(values) == 1:
            values = values.pop()
        body = self.visit(o.find('body'), **kwargs)
        return as_tuple(values) or None, as_tuple(body)

    def visit_FmemberRef(self, o, **kwargs):
        parent = self.visit(o.find('varRef'), **kwargs)
        name = f'{parent.name}%{o.attrib["member"]}'
        variable = sym.Variable(name=name, parent=parent)
        return variable

    def visit_name(self, o, **kwargs):
        return sym.Variable(name=o.text, source=kwargs.get('source'))

    visit_Var = visit_name

    def visit_FarrayRef(self, o, **kwargs):
        var = self.visit(o.find('varRef'), **kwargs)
        dimensions = as_tuple(self.visit(i, **kwargs) for i in o[1:])
        var = var.clone(dimensions=dimensions)
        return var

    def visit_varRef(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_arrayIndex(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_indexRange(self, o, **kwargs):
        lbound = o.find('lowerBound')
        lower = self.visit(lbound, **kwargs) if lbound is not None else None
        ubound = o.find('upperBound')
        upper = self.visit(ubound, **kwargs) if ubound is not None else None
        st = o.find('step')
        step = self.visit(st, **kwargs) if st is not None else None
        # Drop OMNI's `:1` step counting for ranges in the name of consistency
        step = None if step == '1' else step
        return sym.RangeIndex((lower, upper, step), source=kwargs['source'])

    def visit_lowerBound(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    visit_upperBound = visit_lowerBound
    visit_step = visit_lowerBound

    def visit_FrealConstant(self, o, **kwargs):
        if 'kind' in o.attrib and not 'd' in o.text.lower():
            _type = self.visit(self.type_map[o.attrib.get('type')], **kwargs)
            return sym.Literal(value=o.text, type=BasicType.REAL, kind=_type.kind, source=kwargs['source'])
        return sym.Literal(value=o.text, type=BasicType.REAL, source=kwargs['source'])

    def visit_FlogicalConstant(self, o, **kwargs):
        return sym.Literal(value=o.text, type=BasicType.LOGICAL, source=kwargs['source'])

    def visit_FcharacterConstant(self, o, **kwargs):
        return sym.Literal(value=f'"{o.text}"', type=BasicType.CHARACTER, source=kwargs['source'])

    def visit_FintConstant(self, o, **kwargs):
        if 'kind' in o.attrib:
            _type = self.visit(self.type_map[o.attrib.get('type')], **kwargs)
            return sym.Literal(value=int(o.text), type=BasicType.INTEGER, kind=_type.kind, source=kwargs['source'])
        return sym.Literal(value=int(o.text), type=BasicType.INTEGER, source=kwargs['source'])

    def visit_FcomplexConstant(self, o, **kwargs):
        value = ', '.join(f'{self.visit(v, **kwargs)}' for v in list(o))
        return sym.IntrinsicLiteral(value=f'({value})', source=kwargs['source'])

    def visit_FarrayConstructor(self, o, **kwargs):
        values = as_tuple(self.visit(v, **kwargs) for v in o)
        return sym.LiteralList(values=values)

    def visit_functionCall(self, o, **kwargs):
        if o.find('name') is not None:
            name = self.visit(o.find('name'), **kwargs)
        elif o.find('FmemberRef') is not None:
            name = self.visit(o.find('FmemberRef'), **kwargs)
        else:
            raise ValueError

        args = o.find('arguments')
        if args is not None:
            args = as_tuple(self.visit(a, **kwargs) for a in args)
            # Separate keyword argument from positional arguments
            kw_args = as_tuple(arg for arg in args if isinstance(arg, tuple))
            args = as_tuple(arg for arg in args if not isinstance(arg, tuple))
        else:
            args, kw_args = (), ()

        if o.attrib.get('type', 'Fvoid') == 'Fvoid':
            # Subroutine call
            return ir.CallStatement(name=name, arguments=args, kwarguments=kw_args, source=kwargs['source'])

        if name.name.lower() in ('real', 'int'):
            assert args
            expr = args[0]
            if kw_args:
                assert len(args) == 1
                assert len(kw_args) == 1 and kw_args[0][0] == 'kind'
                kind = kw_args[0][1]
            else:
                kind = args[1] if len(args) > 1 else None
            return sym.Cast(name, expr, kind=kind, source=kwargs['source'])

        return sym.InlineCall(name, parameters=args, kw_parameters=kw_args, source=kwargs['source'])

    def visit_FcycleStatement(self, o, **kwargs):
        # TODO: do-construct-name is not preserved
        return ir.Intrinsic(text='cycle', source=kwargs['source'])

    def visit_continueStatement(self, o, **kwargs):
        return ir.Intrinsic(text='continue', source=kwargs['source'])

    def visit_FexitStatement(self, o, **kwargs):
        # TODO: do-construct-name is not preserved
        return ir.Intrinsic(text='exit', source=kwargs['source'])

    def visit_FopenStatement(self, o, **kwargs):
        nvalues = [self.visit(nv, **kwargs) for nv in o.find('namedValueList')]
        nargs = ', '.join(f'{k}={v}' for k, v in nvalues)
        return ir.Intrinsic(text=f'open({nargs})', source=kwargs['source'])

    def visit_FcloseStatement(self, o, **kwargs):
        nvalues = [self.visit(nv, **kwargs) for nv in o.find('namedValueList')]
        nargs = ', '.join(f'{k}={v}' for k, v in nvalues)
        return ir.Intrinsic(text=f'close({nargs})', source=kwargs['source'])

    def visit_FreadStatement(self, o, **kwargs):
        nvalues = [self.visit(nv, **kwargs) for nv in o.find('namedValueList')]
        values = [self.visit(v, **kwargs) for v in o.find('valueList')]
        nargs = ', '.join(f'{k}={v}' for k, v in nvalues)
        args = ', '.join(f'{v}' for v in values)
        return ir.Intrinsic(text=f'read({nargs}) {args}', source=kwargs['source'])

    def visit_FwriteStatement(self, o, **kwargs):
        nvalues = [self.visit(nv, **kwargs) for nv in o.find('namedValueList')]
        values = [self.visit(v, **kwargs) for v in o.find('valueList')]
        nargs = ', '.join(f'{k}={v}' for k, v in nvalues)
        args = ', '.join(f'{v}' for v in values)
        return ir.Intrinsic(text=f'write({nargs}) {args}', source=kwargs['source'])

    def visit_FprintStatement(self, o, **kwargs):
        values = [self.visit(v, **kwargs) for v in o.find('valueList')]
        args = ', '.join(f'{v}' for v in values)
        fmt = o.attrib['format']
        return ir.Intrinsic(text=f'print {fmt}, {args}', source=kwargs['source'])

    def visit_FformatDecl(self, o, **kwargs):
        fmt = f'FORMAT{o.attrib["format"]}'
        return ir.Intrinsic(text=fmt, source=kwargs['source'])

    def visit_namedValue(self, o, **kwargs):
        name = o.attrib['name']
        if 'value' in o.attrib:
            return name, o.attrib['value']
        return name, self.visit(list(o)[0], **kwargs)

    @staticmethod
    def parenthesize_if_needed(expr, enclosing_cls):
        # Other than FP/OFP, OMNI does not retain any information about parenthesis in the
        # original source. While the parse tree is semantically correct,
        # it may cause problems with some agressively optimising compilers.
        # We inject manual parenthesis here for nested expressions to make sure
        # we capture as much of the evaluation order of the original source as possible.
        # Note: this will result in an abundance of trivial/unnecessary parenthesis!
        if enclosing_cls in (sym.Product, sym.Quotient):
            if isinstance(expr, sym.Product):
                return op.ParenthesisedMul(expr.children, source=expr.source)
            if isinstance(expr, sym.Quotient):
                return op.ParenthesisedDiv(expr.numerator, expr.denominator, source=expr.source)
            if isinstance(expr, sym.Sum):
                return op.ParenthesisedAdd(expr.children, source=expr.source)
            if isinstance(expr, sym.Power):
                return op.ParenthesisedPow(expr.base, expr.exponent, source=expr.source)
        return expr

    def visit_plusExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Sum(exprs, source=kwargs['source'])

    def visit_minusExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Sum((exprs[0], sym.Product((-1, exprs[1]))), source=kwargs['source'])

    def visit_mulExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        exprs = tuple(self.parenthesize_if_needed(c, sym.Product) for c in exprs)
        return sym.Product(exprs, source=kwargs['source'])

    def visit_divExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        exprs = tuple(self.parenthesize_if_needed(c, sym.Quotient) for c in exprs)
        return sym.Quotient(*exprs, source=kwargs['source'])

    def visit_FpowerExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Power(base=exprs[0], exponent=exprs[1], source=kwargs['source'])

    def visit_unaryMinusExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 1
        return sym.Product((-1, exprs[0]), source=kwargs['source'])

    def visit_logOrExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        return sym.LogicalOr(exprs, source=kwargs['source'])

    def visit_logAndExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        return sym.LogicalAnd(exprs, source=kwargs['source'])

    def visit_logNotExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 1
        return sym.LogicalNot(exprs[0], source=kwargs['source'])

    def visit_logLTExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '<', exprs[1], source=kwargs['source'])

    def visit_logLEExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '<=', exprs[1], source=kwargs['source'])

    def visit_logGTExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '>', exprs[1], source=kwargs['source'])

    def visit_logGEExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '>=', exprs[1], source=kwargs['source'])

    def visit_logEQExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '==', exprs[1], source=kwargs['source'])

    def visit_logNEQExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '!=', exprs[1], source=kwargs['source'])

    def visit_logEQVExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.LogicalOr((sym.LogicalAnd(exprs), sym.LogicalNot(sym.LogicalOr(exprs))), source=kwargs['source'])

    def visit_logNEQVExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.LogicalAnd((sym.LogicalNot(sym.LogicalAnd(exprs)), sym.LogicalOr(exprs)), source=kwargs['source'])

    def visit_FconcatExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return StringConcat(exprs, source=kwargs['source'])

    def visit_gotoStatement(self, o, **kwargs):
        label = int(o.attrib['label_name'])
        return ir.Intrinsic(text=f'go to {label: d}', source=kwargs['source'])

    def visit_statementLabel(self, o, **kwargs):
        return ir.Comment('__STATEMENT_LABEL__', label=o.attrib['label_name'], source=kwargs['source'])

    def visit_FreturnStatement(self, o, **kwargs):
        return ir.Intrinsic(text='return', source=kwargs['source'])
