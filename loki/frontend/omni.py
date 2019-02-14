from subprocess import check_call, CalledProcessError
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import OrderedDict
from functools import reduce
import operator
from sympy import Add, Mul, Pow, Equality, Unequality


from loki.frontend.source import Source
from loki.frontend.util import inline_comments, cluster_comments, inline_pragmas
from loki.visitors import GenericVisitor
from loki.expression import Variable, Literal, LiteralList, InlineCall, RangeIndex, Cast, SymbolCache
from loki.ir import (Scope, Statement, Conditional, Call, Loop, Allocation, Deallocation,
                     Import, Declaration, TypeDef, Intrinsic, Pragma, Comment)
from loki.types import BaseType, DerivedType, DataType
from loki.logging import info, error, DEBUG
from loki.tools import as_tuple, timeit


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
        check_call(cmd)
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
        check_call(cmd)
    except CalledProcessError as e:
        error('[%s] Parsing failed: %s' % ('omni', ' '.join(cmd)))
        raise e

    return ET.parse(str(xml_path)).getroot()


class OMNI2IR(GenericVisitor):

    def __init__(self, typedefs=None, type_map=None, symbol_map=None, shape_map=None,
                 raw_source=None, cache=None):
        super(OMNI2IR, self).__init__()

        self.typedefs = typedefs
        self.type_map = type_map
        self.symbol_map = symbol_map
        self.shape_map = shape_map
        self.raw_source = raw_source

        # Use provided symbol cache for variable generation
        self._cache = cache

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
        if name.attrib['type'] in self.type_map:
            tast = self.type_map[name.attrib['type']]
            _type = self.visit(tast)

            # Hacky..: Override derived type meta-info with provided ``typedefs``.
            # This is needed to get the Loki-specific (pragma-driven) dimensions on
            # derived-type components, that are otherwise deferred.
            if _type is not None and self.typedefs is not None:
                typedef= self.typedefs.get(_type.name.lower(), None)
                if typedef is not None:
                    _type = DerivedType(name=typedef.name, variables=typedef.variables,
                                        intent=_type.intent, allocatable=_type.allocatable,
                                        pointer=_type.pointer, optional=_type.optional,
                                        parameter=_type.parameter, target=_type.target,
                                        contiguous=_type.contiguous)

            # If the type node has ranges, create dimensions
            dimensions = as_tuple(self.visit(d) for d in tast.findall('indexRange'))
            dimensions = None if len(dimensions) == 0 else dimensions
        else:
            t = name.attrib['type']
            _type = BaseType(name=BaseType._omni_types.get(t, t))
            dimensions = None

        value = self.visit(o.find('value')) if o.find('value') is not None else None
        variable = self.Variable(name=name.text, dimensions=dimensions, type=_type,
                                 shape=dimensions, initial=value)
        return Declaration(variables=as_tuple(variable), type=_type, source=source)

    def visit_FstructDecl(self, o, source=None):
        # TODO: THIS IS A MASSIVE HACK!
        # Since we create derived type definitions from pre-processed OMNI-AST,
        # we need to prevent accidental variable aliasing between the definitions
        # in the type and the routine. For this, we basically stash the cache here
        #and re-instate it once we have all type variables.
        previous_cache = self._cache
        self._cache = SymbolCache()

        name = o.find('name')
        derived = self.visit(self.type_map[name.attrib['type']])
        decls = as_tuple(Declaration(variables=(v, ), type=v.type)
                         for v in derived.variables)

        self._cache = previous_cache
        return TypeDef(name=name.text, declarations=decls)

    def visit_FbasicType(self, o, source=None):
        ref = o.attrib.get('ref', None)
        if ref in self.type_map:
            _type = self.visit(self.type_map[ref])
        else:
            name = BaseType._omni_types.get(ref, ref)
            kind = self.visit(o.find('kind')) if o.find('kind') is not None else None
            _type = BaseType(name, kind=kind)

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
        name = o.attrib['type']
        if self.symbol_map is not None and name in self.symbol_map:
            name = self.symbol_map[name].find('name').text
        variables = []

        # TODO: THIS IS A MASSIVE HACK!
        # Since we create derived type definitions from pre-processed OMNI-AST,
        # we need to prevent accidental variable aliasing between the definitions
        # in the type and the routine. For this, we basically stash the cache here
        #and re-instate it once we have all type variables.
        previous_cache = self._cache
        self._cache = SymbolCache()

        for s in o.find('symbols'):
            vname = s.find('name').text
            t = s.attrib['type']
            dimensions = None
            if t in self.type_map:
                vtype = self.visit(self.type_map[t])
                if len(self.type_map[t]) > 0:
                    dimensions = as_tuple(self.visit(d) for d in self.type_map[t])
            else:
                vtype = BaseType(name=BaseType._omni_types.get(t, t))
            variables += [self.Variable(name=vname, dimensions=dimensions,
                                        shape=dimensions, type=vtype)]

        self._cache = previous_cache
        return DerivedType(name=name, variables=as_tuple(variables))

    def visit_associateStatement(self, o, source=None):
        associations = OrderedDict()
        for i in o.findall('symbols/id'):
            var = self.visit(i.find('value'))
            associations[var] = self.Variable(name=i.find('name').text)
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
        return Statement(target=target, expr=expr)

    def visit_FpointerAssignStatement(self, o, source=None):
        target = self.visit(o[0])
        expr = self.visit(o[1])
        return Statement(target=target, expr=expr, ptr=True)

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

    def visit_FmemberRef(self, o, lookahead=False, source=None):
        vname = o.attrib['member']
        t = o.attrib['type']
        if t in self.type_map:
            vtype = self.visit(self.type_map[t])
        else:
            vtype = BaseType(name=BaseType._omni_types.get(t, t))
        parent = self.visit(o.find('varRef'))

        if lookahead:
            # Hack: Return components of Variable symbol to allow deferred creation
            return vname, vtype, parent

        shape = None
        if parent is not None:
            # If we have a parent, get the shape info from it
            assert isinstance(parent.type, DerivedType)
            typevar = [v for v in parent.type.variables
                       if v.name.lower() == vname.lower()][0]
            shape = None
            if typevar.is_Array:
                shape = typevar.shape or typevar.dimensions

        return self.Variable(name=vname, type=vtype, parent=parent, shape=shape)

    def visit_Var(self, o, lookahead=False, source=None):
        vname = o.text
        t = o.attrib['type']
        if t in self.type_map:
            vtype = self.visit(self.type_map[t])

            # Inject derived-type definition override :(
            if vtype is not None and self.typedefs is not None:
                typedef = self.typedefs.get(vtype.name.lower(), None)
                if typedef is not None:
                    vtype = DerivedType(name=typedef.name, variables=typedef.variables,
                                        intent=vtype.intent, allocatable=vtype.allocatable,
                                        pointer=vtype.pointer, optional=vtype.optional,
                                        parameter=vtype.parameter, target=vtype.target,
                                        contiguous=vtype.contiguous)
        else:
            vtype = BaseType(name=BaseType._omni_types.get(t, t))

        if lookahead:
            return vname, vtype, None

        shape = None
        if self.shape_map is not None:
            shape = self.shape_map.get(vname, None)
        return self.Variable(name=vname, type=vtype, shape=shape)

    def visit_FarrayRef(self, o, source=None):
        # Hack: Get variable components here and derive the dimensions
        # explicitly before constructing our symbolic variable.
        vname, vtype, parent = self.visit(o.find('varRef'), lookahead=True)
        dimensions = as_tuple(self.visit(i) for i in o[1:])
        shape = self.shape_map.get(vname, None)

        if parent is not None:
            # If we have a parent, get the shape info from it
            assert isinstance(parent.type, DerivedType)
            typevar = [v for v in parent.type.variables
                       if v.name.lower() == vname.lower()][0]
            shape = typevar.shape or typevar.dimensions

        return self.Variable(name=vname, dimensions=dimensions,
                             shape=shape, type=vtype, parent=parent)

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
        return Literal(value=float(o.text), kind=o.attrib.get('kind', None))

    def visit_FlogicalConstant(self, o, source=None):
        return Literal(value=o.text, type=DataType.BOOL)

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
        kwargs = as_tuple(arg for arg in args if isinstance(arg, tuple))
        args = as_tuple(arg for arg in args if not isinstance(arg, tuple))
        # Slightly hacky: inlining is decided based on return type
        # TODO: Unify the two call types?
        if o.attrib.get('type', 'Fvoid') != 'Fvoid':
            if o.find('name') is not None and o.find('name').text in ['real']:
                args = o.find('arguments')
                expr = self.visit(args[0])
                kind = self.visit(args[1])
                if isinstance(kind, tuple):
                    kind = kind[1]  # Yuckk!
                dtype = BaseType(name=o.find('name').text, kind=kind)
                return Cast(dtype.name, expression=expr, kind=dtype.kind)
            else:
                return InlineCall(name=name, arguments=args, kwarguments=kwargs)
        else:
            return Call(name=name, arguments=args, kwarguments=kwargs)
        return o.text

    def visit_FallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        variables = []
        data_source = None
        if o.find('allocOpt') is not None:
            data_source = self.visit(o.find('allocOpt'))
        for a in allocs:
            vname, vtype, parent = self.visit(a[0], lookahead=True)
            dimensions = as_tuple(self.visit(i) for i in a[1:])
            dimensions = None if len(dimensions) == 0 else dimensions
            variables += [self.Variable(name=vname, dimensions=dimensions,
                                        type=vtype, parent=parent)]
        return Allocation(variables=as_tuple(variables), data_source=data_source)

    def visit_FdeallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        deallocations = []
        for a in allocs:
            v = self.visit(a[0])
            deallocations += [Deallocation(variable=v)]
        return deallocations[0] if len(deallocations) == 1 else as_tuple(deallocations)

    def visit_FcycleStatement(self, o, source=None):
        return Intrinsic(text='cycle')

    def visit_FopenStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        return Intrinsic(text='open(%s)' % nargs)

    def visit_FcloseStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        return Intrinsic(text='close(%s)' % nargs)

    def visit_FreadStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        values = [self.visit(v) for v in o.find('valueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        args = ', '.join('%s' % v for v in values)
        return Intrinsic(text='read(%s) %s' % (nargs, args))

    def visit_FwriteStatement(self, o, source):
        nvalues = [self.visit(nv) for nv in o.find('namedValueList')]
        values = [self.visit(v) for v in o.find('valueList')]
        nargs = ', '.join('%s=%s' % (k, v) for k, v in nvalues)
        args = ', '.join('%s' % v for v in values)
        return Intrinsic(text='write(%s) %s' % (nargs, args))

    def visit_FprintStatement(self, o, source):
        values = [self.visit(v) for v in o.find('valueList')]
        args = ', '.join('%s' % v for v in values)
        fmt = o.attrib['format']
        return Intrinsic(text='print %s, %s' % (fmt, args))

    def visit_FformatDecl(self, o, source):
        # Hackery galore; this is wrong on soooo many levels! :(
        lineno = int(o.attrib['lineno'])
        line = self.raw_source.splitlines(keepends=False)[lineno-1]
        return Intrinsic(text=line)

    def visit_namedValue(self, o, source):
        name = o.attrib['name']
        if 'value' in o.attrib:
            return name, o.attrib['value']
        else:
            return name, self.visit(o.getchildren()[0])

    def visit_plusExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 2
        return Add(exprs[0], exprs[1], evaluate=False)

    def visit_minusExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 2
        return Add(exprs[0], Mul(-1, exprs[1], evaluate=False), evaluate=False)

    def visit_mulExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 2
        return Mul(exprs[0], exprs[1], evaluate=False)

    def visit_divExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 2
        return Mul(exprs[0], Pow(exprs[1], -1, evaluate=False), evaluate=False)

    def visit_FpowerExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 2
        return Pow(exprs[0], exprs[1], evaluate=False)

    def visit_unaryMinusExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 1
        return Mul(-1, exprs[0], evaluate=False)

    def visit_logOrExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return reduce(operator.or_, exprs)

    def visit_logAndExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return reduce(operator.and_, exprs)

    def visit_logNotExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 1
        return operator.invert(exprs[0])

    def visit_logLTExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return reduce(operator.lt, exprs)

    def visit_logLEExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return reduce(operator.le, exprs)

    def visit_logGTExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return reduce(operator.gt, exprs)

    def visit_logGEExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return reduce(operator.ge, exprs)

    def visit_logEQExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 2
        return Equality(exprs[0], exprs[1], evaluate=False)

    def visit_logNEQExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        assert len(exprs) == 2
        return Unequality(exprs[0], exprs[1], evaluate=False)


@timeit(log_level=DEBUG)
def parse_omni_ast(ast, typedefs=None, type_map=None, symbol_map=None, shape_map=None,
                   raw_source=None, cache=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    # Parse the raw OMNI language AST
    ir = OMNI2IR(type_map=type_map, typedefs=typedefs, symbol_map=symbol_map,
                 shape_map=shape_map, raw_source=raw_source, cache=cache).visit(ast)

    # Perform soime minor sanitation tasks
    ir = inline_comments(ir)
    ir = cluster_comments(ir)
    ir = inline_pragmas(ir)

    return ir
