from subprocess import check_call, CalledProcessError
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import OrderedDict

from loki.frontend.source import Source
from loki.visitors import GenericVisitor
from loki.expression import Variable, Literal, LiteralList, Operation, InlineCall, RangeIndex
from loki.ir import (Scope, Statement, Conditional, Call, Loop, Allocation, Deallocation,
                     Import, Declaration, TypeDef, Intrinsic, Pragma, Comment)
from loki.types import BaseType, DerivedType
from loki.logging import info, error, DEBUG
from loki.tools import as_tuple, timeit


__all__ = ['preprocess_omni', 'parse_omni', 'convert_omni2ir']


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
def parse_omni(filename, xmods=None):
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

    def __init__(self, type_map=None, symbol_map=None, raw_source=None):
        super(OMNI2IR, self).__init__()

        self.type_map = type_map
        self.symbol_map = symbol_map
        self.raw_source = raw_source

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        else:
            return super(OMNI2IR, self).lookup_method(instance)

    def visit(self, o):
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        file = o.attrib.get('file', None)
        lineno = o.attrib.get('lineno', None)
        source = Source(lines=(lineno, None), file=file)
        return super(OMNI2IR, self).visit(o, source=source)

    def visit_Element(self, o, source=None):
        """
        Universal default for XML element types
        """
        children = tuple(self.visit(c) for c in o)
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
            type = self.visit(tast)
            dimensions = as_tuple(self.visit(d) for d in tast.findall('indexRange'))
            # Flatten trivial dimension to variables (eg. `1:v` - > `v`)
            dimensions = as_tuple(d.upper if d == d.upper else d for d in dimensions)
            dimensions = None if len(dimensions) == 0 else dimensions
        else:
            t = name.attrib['type']
            type = BaseType(name=BaseType._omni_types.get(t, t))
            dimensions = None

        value = self.visit(o.find('value')) if o.find('value') is not None else None
        variable = Variable(name=name.text, dimensions=dimensions, type=type, initial=value)
        return Declaration(variables=as_tuple(variable), type=type, source=source)

    def visit_FstructDecl(self, o, source=None):
        name = o.find('name')
        derived = self.visit(self.type_map[name.attrib['type']])
        decls = as_tuple(Declaration(variables=(v, ), type=v.type)
                         for v in derived._variables)
        return TypeDef(name=name.text, declarations=decls)

    def visit_FbasicType(self, o, source=None):
        ref = o.attrib.get('ref', None)
        if ref in self.type_map:
            t = self.visit(self.type_map[ref])
            name = t.name
            kind = t.kind
        else:
            name = BaseType._omni_types.get(ref, ref)
            kind = self.visit(o.find('kind')) if o.find('kind') is not None else None
        intent = o.attrib.get('intent', None)
        allocatable = o.attrib.get('is_allocatable', 'false') == 'true'
        pointer = o.attrib.get('is_pointer', 'false') == 'true'
        optional = o.attrib.get('is_optional', 'false') == 'true'
        parameter = o.attrib.get('is_parameter', 'false') == 'true'
        target = o.attrib.get('is_target', 'false') == 'true'
        contiguous = o.attrib.get('is_contiguous', 'false') == 'true'
        return BaseType(name=name, kind=kind, intent=intent, allocatable=allocatable,
                        pointer=pointer, optional=optional, parameter=parameter,
                        target=target, contiguous=contiguous)

    def visit_FstructType(self, o, source=None):
        name = o.attrib['type']
        if self.symbol_map is not None and name in self.symbol_map:
            name = self.symbol_map[name].find('name').text
        variables = []
        for s in o.find('symbols'):
            vname = s.find('name').text
            t = s.attrib['type']
            if t in self.type_map:
                vtype = self.visit(self.type_map[t])
                dimensions = [self.visit(d) for d in self.type_map[t]]
            else:
                vtype = BaseType(name=BaseType._omni_types.get(t, t))
                dimensions = None
            variables += [Variable(name=vname, dimensions=dimensions, type=vtype)]
        return DerivedType(name=name, variables=as_tuple(variables))

    def visit_associateStatement(self, o, source=None):
        associations = OrderedDict()
        for i in o.findall('symbols/id'):
            var = self.visit(i.find('value'))
            associations[var] = Variable(name=i.find('name').text)
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
        bounds = self.visit(o.find('indexRange'))
        return Loop(variable=variable, body=body, bounds=bounds)

    def visit_FifStatement(self, o, source=None):
        conditions = as_tuple(self.visit(c) for c in o.findall('condition'))
        bodies = as_tuple([self.visit(o.find('then/body'))])
        else_body = self.visit(o.find('else/body')) if o.find('else') is not None else None
        return Conditional(conditions=conditions, bodies=(bodies, ), else_body=else_body)

    def visit_FmemberRef(self, o, source=None):
        t = o.attrib['type']
        if t in self.type_map:
            vtype = self.visit(self.type_map[t])
        else:
            vtype = BaseType(name=BaseType._omni_types.get(t, t))
        var = Variable(name=o.attrib['member'], type=vtype)
        var.ref = self.visit(o.find('varRef'))
        return var

    def visit_Var(self, o, source=None):
        return Variable(name=o.text)

    def visit_FarrayRef(self, o, source=None):
        v = self.visit(o[0])
        v.dimensions = as_tuple(self.visit(i) for i in o[1:])
        return v

    def visit_arrayIndex(self, o, source=None):
        return self.visit(o[0])

    def visit_indexRange(self, o, source=None):
        if 'is_assumed_shape' in o.attrib and o.attrib['is_assumed_shape'] == 'true':
            return RangeIndex(lower=None, upper=None, step=None)
        else:
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
        return Literal(value=o.text)

    def visit_FcharacterConstant(self, o, source=None):
        return Literal(value='"%s"' % o.text)

    def visit_FintConstant(self, o, source=None):
        return Literal(value=o.text)

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
            return InlineCall(name=name, arguments=args, kwarguments=kwargs)
        else:
            return Call(name=name, arguments=args, kwarguments=kwargs)
        return o.text

    def visit_FallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        allocations = []
        data_source = None
        if o.find('allocOpt') is not None:
            data_source = self.visit(o.find('allocOpt'))
        for a in allocs:
            v = self.visit(a[0])
            v.dimensions = as_tuple(self.visit(i) for i in a[1:])
            allocations += [Allocation(variable=v, data_source=data_source)]
        return allocations[0] if len(allocations) == 1 else as_tuple(allocations)

    def visit_FdeallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        deallocations = []
        for a in allocs:
            v = self.visit(a[0])
            v.dimensions = as_tuple(self.visit(i) for i in a[1:])
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
        return Operation(ops=['+'], operands=exprs)

    def visit_minusExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['-'], operands=exprs)

    def visit_mulExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['*'], operands=exprs)

    def visit_divExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['/'], operands=exprs)

    def visit_FpowerExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['**'], operands=exprs)

    def visit_unaryMinusExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['-'], operands=exprs)

    def visit_logOrExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['.or.'], operands=exprs)

    def visit_logAndExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['.and.'], operands=exprs)

    def visit_logNotExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['.not.'], operands=exprs)

    def visit_logLTExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['<'], operands=exprs)

    def visit_logLEExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['<='], operands=exprs)

    def visit_logGTExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['>'], operands=exprs)

    def visit_logGEExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['>='], operands=exprs)

    def visit_logEQExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['=='], operands=exprs)

    def visit_logNEQExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['/='], operands=exprs)


def convert_omni2ir(omni_ast, type_map=None, symbol_map=None, raw_source=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    return OMNI2IR(type_map=type_map, symbol_map=symbol_map,
                   raw_source=raw_source).visit(omni_ast)
