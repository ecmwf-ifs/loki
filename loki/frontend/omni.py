from subprocess import check_output, CalledProcessError
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pymbolic.primitives import (Sum, Product, Quotient, Power, Comparison, LogicalNot,
                                 LogicalAnd, LogicalOr)


from loki.frontend.source import Source
from loki.frontend.util import inline_comments, cluster_comments, inline_pragmas, inline_labels
from loki.visitors import GenericVisitor
from loki.expression import (Variable, Literal, LiteralList, InlineCall, RangeIndex, Cast, Array,
                             ExpressionDimensionsMapper, StringConcat)
from loki.ir import (Scope, Statement, Conditional, CallStatement, Loop, Allocation, Deallocation,
                     Import, Declaration, TypeDef, Intrinsic, Pragma, Comment)
from loki.logging import info, error, DEBUG
from loki.tools import as_tuple, timeit
from loki.types import DataType, SymbolType


__all__ = ['preprocess_omni', 'parse_omni_file', 'parse_omni_ast']


def preprocess_omni(filename, outname, includes=None):
    """
    Call C-preprocessor to sanitize input for OMNI frontend.
    """
    filepath = Path(filename)
    outpath = Path(outname)
    includes = [Path(incl) for incl in includes or []]

    # TODO Make CPP driveable via flags/config
    cmd = ['gfortran', '-E', '-cpp']
    for incl in includes:
        cmd += ['-I', '%s' % Path(incl)]
    cmd += ['-o', '%s' % outpath]
    cmd += ['%s' % filepath]

    try:
        check_output(cmd)
    except CalledProcessError as e:
        error('[OMNI] Preprocessing failed: %s' % ' '.join(cmd))
        raise e


@timeit(log_level=DEBUG)
def parse_omni_file(filename, xmods=None):
    """
    Deploy the OMNI compiler's frontend (F_Front) to generate the OMNI AST.
    """
    filepath = Path(filename)
    info("[Frontend.OMNI] Parsing %s" % filepath.name)

    xml_path = filepath.with_suffix('.xml')
    xmods = xmods or []

    cmd = ['F_Front', '-fleave-comment']
    for m in xmods:
        cmd += ['-M', '%s' % Path(m)]
    cmd += ['-o', '%s' % xml_path]
    cmd += ['%s' % filepath]

    try:
        check_output(cmd)
    except CalledProcessError as e:
        error('[%s] Parsing failed: %s' % ('omni', ' '.join(cmd)))
        raise e

    return ET.parse(str(xml_path)).getroot()


class OMNI2IR(GenericVisitor):

    _omni_types = {
        'Fint': 'INTEGER',
        'Freal': 'REAL',
        'Flogical': 'LOGICAL',
        'Fcharacter': 'CHARACTER',
        'int': 'INTEGER',
        'real': 'REAL',
    }

    def __init__(self, typedefs=None, type_map=None, symbol_map=None,
                 raw_source=None, scope=None):
        super(OMNI2IR, self).__init__()

        self.typedefs = typedefs
        self.type_map = type_map
        self.symbol_map = symbol_map
        self.raw_source = raw_source
        self.scope = scope

    def _struct_type_variables(self, o, scope, parent=None):
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
                vtype = SymbolType(DataType.from_fortran_type(typename))

            variables += [Variable(name=vname, dimensions=dimensions, type=vtype, scope=scope)]
        return variables

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        else:
            return super(OMNI2IR, self).lookup_method(instance)

    def visit(self, o, **kwargs):
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        file = o.attrib.get('file', None)
        lineno = o.attrib.get('lineno', None)
        source = Source(lines=(lineno, lineno), file=file)
        return super(OMNI2IR, self).visit(o, source=source, **kwargs)

    def visit_Element(self, o, source=None, **kwargs):
        """
        Universal default for XML element types
        """
        children = tuple(self.visit(c, **kwargs) for c in o)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        else:
            return children if len(children) > 0 else None

    visit_body = visit_Element

    def visit_FuseOnlyDecl(self, o, source=None):
        symbols = as_tuple(r.attrib['use_name'] for r in o.findall('renamable'))
        return Import(module=o.attrib['name'], symbols=symbols, c_import=False)

    def visit_FinterfaceDecl(self, o, source=None):
        header = Path(o.attrib['file']).name
        return Import(module=header, c_import=True)

    def visit_varDecl(self, o, source=None):
        name = o.find('name')

        # Hack: For some £$%^ reason, OMNI inserts the function name as a varDecl...
        if hasattr(self.scope, 'name') and self.scope.name == name.text:
            return None

        if name.attrib['type'] in self.type_map:
            tast = self.type_map[name.attrib['type']]
            _type = self.visit(tast)

            # If the type definition comes back as deferred, carry out the definition here
            # (this is due to not knowing to which variable instance the type definition
            # belongs further down the tree, which makes scoping hard...)
            if _type.dtype == DataType.DEFERRED and _type.name in self.type_map:
                tname = self.symbol_map[_type.name].find('name').text
                variables = self._struct_type_variables(self.type_map[_type.name],
                                                        scope=self.scope.symbols, parent=name.text)
                variables = OrderedDict([(v.basename, v) for v in variables])

                # Use the previous _type to keep other attributes (like allocatable, pointer, ...)
                _type = _type.clone(dtype=DataType.DERIVED_TYPE, name=tname, variables=variables)

            # If the type node has ranges, create dimensions
            dimensions = as_tuple(self.visit(d) for d in tast.findall('indexRange'))
            dimensions = None if len(dimensions) == 0 else dimensions
        else:
            t = self._omni_types[name.attrib['type']]
            _type = SymbolType(DataType.from_fortran_type(t))
            dimensions = None

        value = self.visit(o.find('value')) if o.find('value') is not None else None
        if _type is not None:
            _type.shape = dimensions
        variable = Variable(name=name.text, dimensions=dimensions, type=_type,
                            initial=value, scope=self.scope.symbols, source=source)
        return Declaration(variables=as_tuple(variable), type=_type, source=source)

    def visit_FstructDecl(self, o, source=None):
        name = o.find('name')
        typedef = TypeDef(name=name.text, declarations=[])

        # Create the derived type...
        _type = SymbolType(DataType.DERIVED_TYPE, name=name.text, variables=OrderedDict())

        # ...and built the list of its members
        variables = self._struct_type_variables(self.type_map[name.attrib['type']],
                                                typedef.symbols)

        # Remember that derived type
        _type.variables.update([(v.basename, v) for v in variables])
        self.scope.types[name.text] = _type

        # Build individual declarations for each member
        declarations = as_tuple(Declaration(variables=(v, ), type=v.type)
                                for v in _type.variables.values())
        typedef._update(declarations=as_tuple(declarations), symbols=typedef.symbols)
        return typedef

    def visit_FbasicType(self, o, source=None):
        ref = o.attrib.get('ref', None)
        if ref in self.type_map:
            _type = self.visit(self.type_map[ref])
        else:
            typename = self._omni_types[ref]
            kind = self.visit(o.find('kind')) if o.find('kind') is not None else None
            length = self.visit(o.find('len')) if o.find('len') is not None else None
            _type = SymbolType(DataType.from_fortran_type(typename), kind=kind, length=length)

        # OMNI types are build recursively from references (Matroshka-style)
        _type.intent = o.attrib.get('intent', None)
        _type.allocatable = o.attrib.get('is_allocatable', 'false') == 'true'
        _type.pointer = o.attrib.get('is_pointer', 'false') == 'true'
        _type.optional = o.attrib.get('is_optional', 'false') == 'true'
        _type.parameter = o.attrib.get('is_parameter', 'false') == 'true'
        _type.target = o.attrib.get('is_target', 'false') == 'true'
        _type.contiguous = o.attrib.get('is_contiguous', 'false') == 'true'
        return _type

    def visit_FstructType(self, o, source=None):
        # We have encountered a derived type as part of the declaration in the spec
        # of a routine.
        name = o.attrib['type']
        if self.symbol_map is not None and name in self.symbol_map:
            name = self.symbol_map[name].find('name').text

        # Check if we know that type already
        parent_type = self.scope.types.lookup(name, recursive=True)
        if parent_type is not None:
            return parent_type.clone()

        # Check if the type was defined externally
        if self.typedefs is not None and name in self.typedefs:
            variables = OrderedDict([(v.name, v) for v in self.typedefs[name].variables])
            return SymbolType(DataType.DERIVED_TYPE, name=name, variables=variables)

        # Otherwise: We have an externally defined type for which we were not given
        # a typedef. We defer the definition
        return SymbolType(DataType.DEFERRED, name=o.attrib['type'])

    def visit_associateStatement(self, o, source=None):
        associations = OrderedDict()
        for i in o.findall('symbols/id'):
            var = self.visit(i.find('value'))
            if isinstance(var, Array):
                shape = ExpressionDimensionsMapper()(var)
            else:
                shape = None
            vname = i.find('name').text
            vtype = var.type.clone(name=None, parent=None, shape=shape)
            associations[var] = Variable(name=vname, type=vtype, scope=self.scope.symbols,
                                         source=source)
        body = self.visit(o.find('body'))
        return Scope(body=as_tuple(body), associations=associations)

    def visit_FcommentLine(self, o, source=None):
        return Comment(text=o.text, source=source)

    def visit_FpragmaStatement(self, o, source=None):
        keyword = o.text.split(' ')[0]
        content = ' '.join(o.text.split(' ')[1:])
        return Pragma(keyword=keyword, content=content, source=source)

    def visit_FassignStatement(self, o, source=None):
        target = self.visit(o[0])
        expr = self.visit(o[1])
        return Statement(target=target, expr=expr, source=source)

    def visit_FpointerAssignStatement(self, o, source=None):
        target = self.visit(o[0])
        expr = self.visit(o[1])
        return Statement(target=target, expr=expr, ptr=True, source=source)

    def visit_FdoStatement(self, o, source=None):
        assert (o.find('Var') is not None)
        assert (o.find('body') is not None)
        variable = self.visit(o.find('Var'))
        body = self.visit(o.find('body'))
        # TODO: What do we do with loop bounds? Tuple or Range?
        lower = self.visit(o.find('indexRange/lowerBound'))
        upper = self.visit(o.find('indexRange/upperBound'))
        step = self.visit(o.find('indexRange/step'))
        bounds = lower, upper, step
        return Loop(variable=variable, body=body, bounds=bounds)

    def visit_FifStatement(self, o, source=None):
        conditions = [self.visit(c) for c in o.findall('condition')]
        bodies = as_tuple([self.visit(o.find('then/body'))])
        else_body = self.visit(o.find('else/body')) if o.find('else') is not None else None
        return Conditional(conditions=as_tuple(conditions),
                           bodies=(bodies, ), else_body=else_body)

    def visit_FmemberRef(self, o, **kwargs):
        vname = o.attrib['member']
        t = o.attrib['type']
        parent = self.visit(o.find('varRef'))

        source = kwargs.get('source', None)
        shape = kwargs.get('shape', None)
        dimensions = kwargs.get('dimensions', None)

        vtype = None
        if parent is not None:
            basename = vname
            vname = '%s%%%s' % (parent.name, vname)

        vtype = self.scope.symbols.lookup(vname, recursive=True)

        # If we have a parent with a type, use that
        if vtype is None and parent is not None and parent.type.dtype == DataType.DERIVED_TYPE:
            vtype = parent.type.variables.get(basename, vtype)
        if vtype is None and t in self.type_map:
            vtype = self.visit(self.type_map[t])
        if vtype is None:
            typename = self._omni_types.get(t, t)
            vtype = SymbolType(DataType.from_fortran_type(typename))

        if shape is not None and vtype is not None and vtype.shape != shape:
            # We need to create a clone of that type as other instances of that
            # derived type might have a different allocation size
            vtype = vtype.clone(shape=shape)

        return Variable(name=vname, type=vtype, parent=parent, scope=self.scope.symbols,
                        dimensions=dimensions, source=source)

    def visit_Var(self, o, **kwargs):
        vname = o.text
        t = o.attrib['type']

        source = kwargs.get('source', None)
        dimensions = kwargs.get('dimensions', None)
        shape = kwargs.get('shape', None)

        vtype = self.scope.symbols.lookup(vname, recursive=True)

        if (vtype is None or vtype.dtype == DataType.DEFERRED) and t in self.type_map:
            vtype = self.visit(self.type_map[t])

        if (vtype is None or vtype.dtype == DataType.DEFERRED) and t in self._omni_types:
            typename = self._omni_types.get(t, t)
            vtype = SymbolType(DataType.from_fortran_type(typename))

        if shape is not None and vtype is not None and vtype.shape != shape:
            # We need to create a clone of that type as other instances of that
            # derived type might have a different allocation size
            vtype = vtype.clone(shape=shape)

        return Variable(name=vname, type=vtype, scope=self.scope.symbols,
                        dimensions=dimensions, source=source)

    def visit_FarrayRef(self, o, source=None):
        # Hack: Get variable components here and derive the dimensions
        # explicitly before constructing our symbolic variable.
        dimensions = as_tuple(self.visit(i) for i in o[1:])
        return self.visit(o.find('varRef'), dimensions=dimensions)

    def visit_arrayIndex(self, o, source=None):
        return self.visit(o[0])

    def visit_indexRange(self, o, source=None):
        lbound = o.find('lowerBound')
        lower = self.visit(lbound) if lbound is not None else None
        ubound = o.find('upperBound')
        upper = self.visit(ubound) if ubound is not None else None
        st = o.find('step')
        step = self.visit(st) if st is not None else None
        return RangeIndex(lower=lower, upper=upper, step=step)

    def visit_FrealConstant(self, o, source=None):
        return Literal(value=o.text, kind=o.attrib.get('kind', None))

    def visit_FlogicalConstant(self, o, source=None):
        return Literal(value=o.text, type=DataType.LOGICAL)

    def visit_FcharacterConstant(self, o, source=None):
        return Literal(value='"%s"' % o.text)

    def visit_FintConstant(self, o, source=None):
        return Literal(value=int(o.text))

    def visit_FarrayConstructor(self, o, source=None):
        values = as_tuple(self.visit(v) for v in o)
        return LiteralList(values=values)

    def visit_functionCall(self, o, source=None):
        if o.find('name') is not None:
            name = o.find('name').text
        elif o[0].tag == 'FmemberRef':
            # TODO: Super-hacky for now!
            # We need to deal with member function (type-bound procedures)
            # and integrate FfunctionType into our own IR hierachy.
            var = self.visit(o[0][0])
            name = '%s%%%s' % (var.name, o[0].attrib['member'])
        else:
            raise RuntimeError('Could not determine name of function call')
        args = o.find('arguments') or tuple()
        args = as_tuple(self.visit(a) for a in args)
        # Separate keyrword argument from positional arguments
        # kwargs = as_tuple(arg for arg in args if isinstance(arg, tuple))
        kwargs = {a[0]: a[1] for a in args if isinstance(a, tuple)}
        args = as_tuple(arg for arg in args if not isinstance(arg, tuple))
        # Slightly hacky: inlining is decided based on return type
        # TODO: Unify the two call types?
        if o.attrib.get('type', 'Fvoid') != 'Fvoid':
            if o.find('name') is not None and o.find('name').text in ['real', 'int']:
                args = o.find('arguments')
                expr = self.visit(args[0])
                if len(args) > 1:
                    kind = self.visit(args[1])
                    if isinstance(kind, tuple):
                        kind = kind[1]  # Yuckk!
                else:
                    kind = None
                return Cast(o.find('name').text, expression=expr, kind=kind)
            else:
                return InlineCall(name, parameters=args, kw_parameters=kwargs)
        else:
            return CallStatement(name=name, arguments=args, kwarguments=kwargs)

    def visit_FallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        variables = []
        data_source = None
        if o.find('allocOpt') is not None:
            data_source = self.visit(o.find('allocOpt'))
        for a in allocs:
            dimensions = as_tuple(self.visit(i) for i in a[1:])
            dimensions = None if len(dimensions) == 0 else dimensions
            variables += [self.visit(a[0], shape=dimensions, dimensions=dimensions)]
#            vname, vtype, parent = self.visit(a[0], shape=dimensions)
#            if parent is not None:
#                vname = '%s%%%s' % (parent.name, vname)
#            variables += [Variable(name=vname, dimensions=dimensions, scope=self.scope,
#                                   type=vtype, parent=parent, source=source)]
        return Allocation(variables=as_tuple(variables), data_source=data_source)

    def visit_FdeallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        deallocations = []
        for a in allocs:
            v = self.visit(a[0])
            deallocations += [Deallocation(variable=v)]
        return deallocations[0] if len(deallocations) == 1 else as_tuple(deallocations)

    def visit_FcycleStatement(self, o, source=None):
        return Intrinsic(text='cycle', source=source)

    def visit_FopenStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        return Intrinsic(text='open(%s)' % nargs, source=source)

    def visit_FcloseStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        return Intrinsic(text='close(%s)' % nargs, source=source)

    def visit_FreadStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        values = [self.visit(v) for v in o.find('valueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        args = ', '.join('%s' % v for v in values)
        return Intrinsic(text='read(%s) %s' % (nargs, args), source=source)

    def visit_FwriteStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        values = [self.visit(v) for v in o.find('valueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        args = ', '.join('%s' % v for v in values)
        return Intrinsic(text='write(%s) %s' % (nargs, args), source=source)

    def visit_FprintStatement(self, o, source):
        values = [self.visit(v) for v in o.find('valueList')]
        args = ', '.join('%s' % v for v in values)
        fmt = o.attrib['format']
        return Intrinsic(text='print %s, %s' % (fmt, args), source=source)

    def visit_FformatDecl(self, o, source):
        # Hackery galore; this is wrong on soooo many levels! :(
        lineno = int(o.attrib['lineno'])
        line = self.raw_source.splitlines(keepends=False)[lineno-1]
        return Intrinsic(text=line, source=source)

    def visit_namedValue(self, o, source):
        name = o.attrib['name']
        if 'value' in o.attrib:
            return name, o.attrib['value']
        else:
            return name, self.visit(list(o)[0])

    def visit_plusExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Sum(exprs)

    def visit_minusExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Sum((exprs[0], Product((-1, exprs[1]))))

    def visit_mulExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Product(exprs)

    def visit_divExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Quotient(*exprs)

    def visit_FpowerExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Power(base=exprs[0], exponent=exprs[1])

    def visit_unaryMinusExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 1
        return Product((-1, exprs[0]))

    def visit_logOrExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        return LogicalOr(exprs)

    def visit_logAndExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        return LogicalAnd(exprs)

    def visit_logNotExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 1
        return LogicalNot(exprs[0])

    def visit_logLTExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Comparison(exprs[0], '<', exprs[1])

    def visit_logLEExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Comparison(exprs[0], '<=', exprs[1])

    def visit_logGTExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Comparison(exprs[0], '>', exprs[1])

    def visit_logGEExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Comparison(exprs[0], '>=', exprs[1])

    def visit_logEQExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Comparison(exprs[0], '==', exprs[1])

    def visit_logNEQExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return Comparison(exprs[0], '!=', exprs[1])

    def visit_logEQVExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return LogicalOr((LogicalAnd(exprs), LogicalNot(LogicalOr(exprs))))

    def visit_logNEQVExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return LogicalAnd((LogicalNot(LogicalAnd(exprs)), LogicalOr(exprs)))

    def visit_FconcatExpr(self, o, source=None):
        exprs = tuple(self.visit(c) for c in o)
        assert len(exprs) == 2
        return StringConcat(exprs)

    def visit_gotoStatement(self, o, source=None):
        label = int(o.attrib['label_name'])
        return Intrinsic(text='go to %d' % label, source=source)

    def visit_statementLabel(self, o, source=None):
        source.label = int(o.attrib['label_name'])
        return Comment('__STATEMENT_LABEL__', source=source)

    def visit_FreturnStatement(self, o, source=None):
        return Intrinsic(text='return', source=source)


@timeit(log_level=DEBUG)
def parse_omni_ast(ast, typedefs=None, type_map=None, symbol_map=None,
                   raw_source=None, scope=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    # Parse the raw OMNI language AST
    ir = OMNI2IR(type_map=type_map, typedefs=typedefs, symbol_map=symbol_map,
                 raw_source=raw_source, scope=scope).visit(ast)

    # Perform soime minor sanitation tasks
    ir = inline_comments(ir)
    ir = cluster_comments(ir)
    ir = inline_pragmas(ir)
    ir = inline_labels(ir)

    return ir
