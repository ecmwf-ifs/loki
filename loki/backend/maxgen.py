from sympy import evaluate
from sympy.codegen.ast import real, float32

from loki.expression import indexify
from loki.ir import Import
from loki.tools import chunks
from loki.types import DataType
from loki.visitors import Visitor, FindNodes


class MaxjCodegen(Visitor):
    """
    Tree visitor to generate Maxeler maxj kernel code from IR.
    """

    def __init__(self, depth=0, linewidth=90, chunking=6):
        super(MaxjCodegen, self).__init__()
        self.linewidth = linewidth
        self.chunking = chunking
        self._depth = depth

    @property
    def indent(self):
        return '  ' * self._depth

    def segment(self, arguments, chunking=None):
        chunking = chunking or self.chunking
        delim = ',\n%s  ' % self.indent
        args = list(chunks(list(arguments), chunking))
        return delim.join(', '.join(c) for c in args)

    def visit_Node(self, o):
        return self.indent + '// <%s>' % o.__class__.__name__

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    visit_list = visit_tuple

    def visit_Module(self, o):
        # Assuming this will be put in header files...
        spec = self.visit(o.spec)
        routines = self.visit(o.routines)
        return spec + '\n\n' + routines

    def visit_Subroutine(self, o):
        # Re-generate variable declarations
        o._externalize()

        package = 'package %s;\n\n' % o.name

        # Some boilerplate imports...
        imports = 'import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;\n'
        imports += 'import com.maxeler.maxcompiler.v2.kernelcompiler.KernelParameters;\n'
        imports += 'import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;\n'
        imports += 'import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;\n'
        imports += 'import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;\n'
        imports += self.visit(FindNodes(Import).visit(o.spec))

        # Standard Kernel definitions
        header = 'class %sKernel extends Kernel {\n\n' % o.name
        self._depth += 1
        header += '%s%sKernel(KernelParameters parameters) {\n' % (self.indent, o.name)
        self._depth += 1
        header += self.indent + 'super(parameters);\n'

        # Generate declarations and body
        spec = self.visit(o.spec)
        body = self.visit(o.body)

        # Insert outflow statements for output variables
        out_vars = [v for v in o.arguments if v.type.intent.lower() in ('out', 'inout')]
        out_types = [self.visit(v.type) for v in out_vars]
        out_funcs = ['output' if v.type.pointer or v.type.allocatable else 'scalarOutput'
                     for v in out_vars]
        outflow = ['%sio.%s("%s", %s, %s);' % (self.indent, f, v.name, v.name, t)
                   for v, t, f in zip(out_vars, out_types, out_funcs)]
        outflow = self.segment(outflow)

        self._depth -= 1
        footer = '\n%s}\n}' % self.indent
        self._depth -= 1

        return package + imports + '\n' + header + spec + '\n' + body + '\n' + outflow + footer

    def visit_Section(self, o):
        return self.visit(o.body) + '\n'

    def visit_Import(self, o):
        raise NotImplementedError('Imports not supported')

    def visit_Declaration(self, o):
        # Ignore parameters
        if o.type.parameter:
            return ''

        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''

        # Determine the underlying data type and initialization value
        base_type = self.visit(o.type)
        if o.type.intent is not None and o.type.intent.lower() in ('in', 'inout'):
            vfunc = ['input' if v.type.pointer or v.type.allocatable else 'scalarInput'
                     for v in o.variables]
            vinit = ['io.%s("%s", %s)' % (f, v.name, base_type)
                     for v, f in zip(o.variables, vfunc)]
        else:
            vinit = [('%s.newInstance(this)' % base_type)] * len(o.variables)

        variables = ['%sDFEVar %s = %s;' % (self.indent, v.name, i)
                     for v, i in zip(o.variables, vinit)]
        return self.segment(variables) + comment

    def visit_BaseType(self, o):
        return o.dtype.maxjtype

    def visit_DerivedType(self, o):
        return 'DFEStructType %s' % o.name

    def visit_TypeDef(self, o):
        self._depth += 1
        decls = self.visit(o.declarations)
        self._depth -= 1
        return 'DFEStructType %s {\n%s\n} ;' % (o.name, decls)

    def visit_Comment(self, o):
        text = o._source.string if o.text is None else o.text
        return self.indent + text.replace('!', '//')

    def visit_CommentBlock(self, o):
        comments = [self.visit(c) for c in o.comments]
        return '\n'.join(comments)

    def visit_Loop(self, o):
        raise NotImplementedError('Loops not yet added')

    def visit_Statement(self, o):
        # Suppress evaluation of expressions to avoid accuracy errors
        # due to symbolic expression re-writing.
        with evaluate(False):
            target = indexify(o.target)
            expr = indexify(o.expr)

        type_aliases = {}
        if o.target.type and o.target.type.dtype == DataType.FLOAT32:
            type_aliases[real] = float32

        stmt = '%s = %s;' % (target, expr)
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        return self.indent + stmt + comment

    def visit_Conditional(self, o):
        raise NotImplementedError('Conditionals not yet added')

    def visit_Intrinsic(self, o):
        return o.text


class MaxjManagerCodegen(object):

    def __init__(self, depth=0, linewidth=90, chunking=6):
        self.linewidth = linewidth
        self.chunking = chunking
        self._depth = depth

    @property
    def indent(self):
        return '  ' * self._depth

    def gen(self, o):
        # Standard boilerplate header
        imports = 'package %s;\n\n' % o.name
        imports += 'import com.maxeler.maxcompiler.v2.build.EngineParameters;\n'
        imports += 'import com.maxeler.maxcompiler.v2.kernelcompiler.Kernel;\n'
        imports += 'import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;\n'
        imports += 'import com.maxeler.platform.max5.manager.MAX5CManager;\n'

        header = 'public class %sManager extends MAX5CManager {\n\n' % o.name
        self._depth += 1
        header += self.indent + 'public %sManager(EngineParameters params) {\n' % o.name
        self._depth += 1

        body = [self.indent + 'super(params);\n']
        body += ['Kernel kernel = new %sKernel(makeKernelParameters("%sKernel"));\n'
                 % (o.name, o.name)]
        body += ['KernelBlock kernelBlock = addKernel(kernel);\n']

        # Insert in/out streams
        in_vars = [v for v in o.arguments
                   if v.type.intent.lower() in ('in', 'inout') and (v.type.pointer or v.type.allocatable)]
        body += ['kernelBlock.getInput("%s") <== addStreamFromCPU("%s");\n' % (v.name, v.name)
                 for v in in_vars]
        out_vars = [v for v in o.arguments
                    if v.type.intent.lower() in ('out', 'inout') and (v.type.pointer or v.type.allocatable)]
        body += ['addStreamToCPU("%s") <== kernelBlock.getOutput("%s");\n' % (v.name, v.name)
                 for v in out_vars]

        body = self.indent.join(body)

        self._depth -= 1
        main_header = self.indent + '}\n\n'
        main_header += self.indent + 'public static void main(String[] args) {\n'
        self._depth += 1

        main_body = [self.indent + 'EngineParameters params = new EngineParameters(args);\n']
        main_body += ['MAX5CManager manager = new %sManager(params);\n' % o.name]
        main_body += ['manager.build();\n']
        main_body = self.indent.join(main_body)

        self._depth -= 1
        footer = self.indent + '}\n'
        self._depth -= 1
        footer += self.indent + '}'

        return imports + header + body + main_header + main_body + footer


def maxjgen(ir):
    """
    Generate Maxeler maxj kernel code from one or many IR objects/trees.
    """
    return MaxjCodegen().visit(ir)


def maxjmanagergen(ir):
    """
    Generate Maxeler maxj manager for the given IR objects/trees.
    """
    return MaxjManagerCodegen().gen(ir)
