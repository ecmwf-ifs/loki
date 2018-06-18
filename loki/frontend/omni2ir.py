from subprocess import check_call, CalledProcessError
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import OrderedDict

from loki.frontend import OMNI
from loki.frontend.source import extract_source
from loki.visitors import GenericVisitor
from loki.expression import Variable, Literal, Index, Operation, InlineCall, RangeIndex
from loki.ir import Scope, Statement, Conditional, Call, Loop, Allocation, Deallocation
from loki.logging import info, error, DEBUG
from loki.tools import as_tuple, timeit, disk_cached


__all__ = ['preprocess_omni', 'parse_omni', 'convert_omni_to_ir']


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
        error('[%s] Preprocessing failed: %s' % (OMNI, ' '.join(cmd)))
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

    cmd = ['F_Front']
    for m in xmods:
        cmd += ['-M', '%s' % Path(m)]
    cmd += ['-o', '%s' % xml_path]
    cmd += ['%s' % filepath]

    try:
        check_call(cmd)
    except CalledProcessError as e:
        error('[%s] Parsing failed: %s' % (OMNI, ' '.join(cmd)))
        raise e

    return ET.parse(str(xml_path)).getroot()


class OMNI2IR(GenericVisitor):

    def __init__(self, raw_source):
        super(OMNI2IR, self).__init__()

        self._raw_source = raw_source

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
        try:
            source = extract_source(o.attrib, self._raw_source)
        except KeyError:
            source = None
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

    def visit_associateStatement(self, o, source=None):
        associations = OrderedDict()
        for i in o.findall('symbols/id'):
            var = self.visit(i.find('value'))
            associations[var] = Variable(name=i.find('name').text)
        body = self.visit(o.find('body'))
        return Scope(body=as_tuple(body), associations=associations)

    def visit_FassignStatement(self, o, source=None):
        target = self.visit(o[0])
        expr = self.visit(o[1])
        return Statement(target=target, expr=expr)

    def visit_FdoStatement(self, o, source=None):
        assert (o.find('Var') is not None)
        assert (o.find('body') is not None)
        variable = self.visit(o.find('Var'))
        body = self.visit(o.find('body'))
        bounds = self.visit(o.find('indexRange'))
        return Loop(variable=variable, body=body, bounds=bounds)

    def visit_FifStatement(self, o, source=None):
        conditions = as_tuple(self.visit(c) for c in o.findall('condition'))
        bodies = [self.visit(o.find('then/body'))]
        else_body = self.visit(o.find('else/body')) if o.find('else') is not None else None
        return Conditional(conditions=conditions, bodies=bodies, else_body=else_body)

    def visit_FmemberRef(self, o, source=None):
        var = self.visit(o.find('varRef'))
        var.subvar = Variable(name=o.attrib['member'])
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
            return Index(name=':')
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

    def visit_FintConstant(self, o, source=None):
        return Literal(value=o.text)

    def visit_functionCall(self, o, source=None):
        name = o.find('name').text
        args = o.find('arguments') or tuple()
        args = as_tuple(self.visit(a) for a in args)
        # TODO: Slightly hacky logic to determine inline calls
        # from generic subroutine calls. Should we follow suit
        # and unify both types?
        inline = o.attrib.get('is_intrinsic', 'false') == 'true'
        inline = inline or o.attrib.get('type', 'Fvoid') != 'Fvoid'
        if inline:
            return InlineCall(name=name, arguments=args)
        else:
            return Call(name=name, arguments=args)
        return o.text

    def visit_FallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        allocations = []
        for a in allocs:
            v = self.visit(a[0])
            v.dimensions = as_tuple(self.visit(i) for i in a[1:])
            allocations += [Allocation(variable=v)]
        return allocations[0] if len(allocations) == 1 else as_tuple(allocations)

    def visit_FdeallocateStatement(self, o, source=None):
        allocs = o.findall('alloc')
        deallocations = []
        for a in allocs:
            v = self.visit(a[0])
            v.dimensions = as_tuple(self.visit(i) for i in a[1:])
            deallocations += [Deallocation(variable=v)]
        return deallocations[0] if len(deallocations) == 1 else as_tuple(deallocations)

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

    def visit_divExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['/'], operands=exprs)

    def visit_logOrExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['.or.'], operands=exprs)

    def visit_logAndExpr(self, o, source=None):
        exprs = [self.visit(c) for c in o]
        return Operation(ops=['.and.'], operands=exprs)

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

def convert_omni_to_ir(omni_ast, raw_source):
    """
    Generate an internal IR from the raw OMNI parser AST.
    output.
    """

    # Parse the OFP AST into a raw IR
    ir = OMNI2IR(raw_source).visit(omni_ast)

    return ir
