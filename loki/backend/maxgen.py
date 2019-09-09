from functools import reduce
from sympy import evaluate
from sympy.codegen.ast import real, float32

from loki.backend import CCodegen, csymgen
from loki.expression import indexify, Variable
from loki.ir import Call, Import, Declaration
from loki.tools import chunks, flatten
from loki.types import BaseType, DataType
from loki.visitors import Visitor, FindNodes, Transformer


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

    def type_and_stream(self, v, is_input=True):
        """
        Builds the string representation of the nested parameterized type for vectors and scalars
        and the matching initialization function or output stream.
        """
        base_type = self.visit(v.type)
        L = len(v.dimensions) if v.is_Array else 0

        # Build nested parameterized type
        types = ['DFEVar'] + ['DFEVector<%s>'] * L
        types = [reduce(lambda p, n: n % p, types[:i]) for i in range(1, L+2)]

        # Deduce matching type constructor
        init_templates = ['new DFEVectorType<%s>(%s, %s)'] * L
        inits = [base_type]
        for i in range(L):
            inits += [init_templates[i] % (types[i], inits[i], v.dimensions[-(i+1)])]

        if is_input:
            # Determine matching initialization routine
            if v.type.intent is not None and v.type.intent.lower() in ('in', 'inout'):
                stream = 'io.%s("%s_in", %s)' % ('input' if v.is_Array else 'scalarInput',
                                                 v.name, inits[-1])
            else:
                stream = '%s.newInstance(this)' % inits[-1]

        else:
            # Matching outflow statement
            if v.type.intent is not None and v.type.intent.lower() in ('out', 'inout'):
                stream = 'io.{0}utput("{1}_out", {1}, {2})'.format('o' if v.is_Array else 'scalarO',
                                                                   v.name, inits[-1])
            else:
                stream = None

        return types[-1], stream

    def visit_Node(self, o):
        return self.indent + '// <%s>' % o.__class__.__name__

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    visit_list = visit_tuple

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

        # Generate declarations for local variables
        local_vars = [v for v in o.variables if v not in o.arguments]
        spec = ['%s %s;\n' % (v.type.dtype.jtype, v) for v in local_vars]
        spec = self.indent.join(spec)

        # Remove any declarations for variables that are not arguments
        decl_map = {}
        for d in FindNodes(Declaration).visit(o.spec):
            if any([v in local_vars for v in d.variables]):
                decl_map[d] = None
        o.spec = Transformer(decl_map).visit(o.spec)

        # Generate remaining declarations
        spec += self.visit(o.spec)

        # Remove pointer type from scalar arguments
        decl_map = {}
        for d in FindNodes(Declaration).visit(o.spec):
            if d.type.pointer:
                new_type = d.type
                new_type.pointer = False
                decl_map[d] = d.clone(type=new_type)
        o.spec = Transformer(decl_map).visit(o.spec)

        # Generate body
        body = self.visit(o.body)

        # Insert outflow statements for output variables
        outflow = [self.type_and_stream(v, is_input=False)
                   for v in o.arguments if v.type.intent.lower() in ('out', 'inout')]
        outflow = '\n'.join(['%s%s;' % (self.indent, a[1]) for a in outflow])

        self._depth -= 1
        footer = '\n%s}\n}' % self.indent
        self._depth -= 1

        return package + imports + '\n' + header + spec + '\n' + body + '\n' + outflow + footer

    def visit_Section(self, o):
        return self.visit(o.body) + '\n'

    def visit_Declaration(self, o):
        # Ignore parameters
        if o.type.parameter:
            return ''

        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''

        # Determine the underlying data type and initialization value
        vtype, vinit = zip(*[self.type_and_stream(v, is_input=True) for v in o.variables])
        variables = ['%s%s %s = %s;' % (self.indent, t, v.name, i)
                     for v, t, i in zip(o.variables, vtype, vinit)]
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

    def visit_Statement(self, o):
        # Suppress evaluation of expressions to avoid accuracy errors
        # due to symbolic expression re-writing.
        with evaluate(False):
            target = indexify(o.target)
            expr = indexify(o.expr)

        type_aliases = {}
        if o.target.type and o.target.type.dtype == DataType.FLOAT32:
            type_aliases[real] = float32

        if o.target.is_Array:
#            stmt = '%s <== %s;\n' % (target, expr)
            stmt = '%s <== %s;\n' % (target, csymgen(expr, type_aliases=type_aliases))
        else:
            stmt = csymgen(expr, assign_to=target, type_aliases=type_aliases)
        comment = '  %s' % self.visit(o.comment) if o.comment is not None else ''
        return self.indent + stmt + comment

    def visit_Intrinsic(self, o):
        return o.text

    def visit_Loop(self, o):
        self._depth += 1
        body = self.visit(o.body)
        self._depth -= 1
        header = self.indent + 'for ({0} = {1}; {0} <= {2}; {0} += {3}) '
        header = header.format(o.variable.name, o.bounds[0], o.bounds[1],
                               o.bounds[2] or 1)
        return header + '{\n' + body + '\n' + self.indent + '}\n'


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
        # imports += 'import com.maxeler.maxcompiler.v2.managers.engine_interfaces.CPUTypes;\n'
        # imports += 'import com.maxeler.maxcompiler.v2.managers.engine_interfaces.EngineInterface;\n'
        imports += 'import com.maxeler.platform.max5.manager.MAX5CManager;\n'
        imports += '\n'

        # Class definitions
        header = 'public class %sManager extends MAX5CManager {\n\n' % o.name
        self._depth += 1
        header += self.indent + 'public static final String kernelName = "%sKernel";\n\n' % o.name
        header += self.indent + 'public %sManager(EngineParameters params) {\n' % o.name
        self._depth += 1

        # Making the kernel known
        body = [self.indent + 'super(params);\n']
        body += ['Kernel kernel = new %sKernel(makeKernelParameters(kernelName));\n' % o.name]
        body += ['KernelBlock kernelBlock = addKernel(kernel);\n']

        # Insert in/out streams
        in_vars = [v for v in o.arguments if v.is_Array and v.type.intent.lower() in ('in', 'inout')]
        out_vars = [v for v in o.arguments if v.is_Array and v.type.intent.lower() in ('out', 'inout')]
        body += ['\n']
        body += ['kernelBlock.getInput("{0}_in") <== addStreamFromCPU("{0}_in");\n'.format(v.name)
                 for v in in_vars]
        body += ['addStreamToCPU("{0}_out") <== kernelBlock.getOutput("{0}_out");\n'.format(v.name)
                 for v in out_vars]

        # Specify default values for interface parameters
        # body += ['\n']
        # body += ['EngineInterface ei = new EngineInterface("kernel");\n']
        # body += ['ei.setTicks(kernelName, 1000);\n']  # TODO: Put a useful value here!

        # Specify sizes of streams
        # stream_template = 'ei.setStream("{0}", {1}, {2} * {1}.sizeInBytes());\n'
        # in_sizes = [', '.join([str(d) for d in v.dimensions]) for v in in_vars]
        # out_sizes = [', '.join([str(d) for d in v.dimensions]) for v in out_vars]
        # body += ['\n']
        # body += [stream_template.format(v.name + '_in', v.type.dtype.maxjManagertype, s)
        #          for v, s in zip(in_vars, in_sizes)]
        # body += [stream_template.format(v.name + '_out', v.type.dtype.maxjManagertype, s)
        #          for v, s in zip(out_vars, out_sizes)]

        # body += ['\n']
        # body += ['createSLiCinterface(ei);\n']
        body = self.indent.join(body)

        # Writing the main for maxJavaRun
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


class MaxjCCodegen(CCodegen):

    def visit_Call(self, o):
        #astr = [csymgen(a) for a in o.arguments]
        #astr = ['*%s' % arg.name if not arg.is_Array and arg.type.pointer else arg.name
        #        for arg in o.arguments]
        astr = [arg.name for arg in o.arguments]
        return '%s%s(%s);' % (self.indent, o.name, ', '.join(astr))


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


def maxjcgen(ir):
    """
    Generate a C routine that wraps the call to the Maxeler kernel.
    """
    return MaxjCCodegen().visit(ir)
