from pathlib import Path
import dace

from loki.visitors import Visitor, Transformer, FindNodes
from loki.transform import Transformation, FortranCTransformation
from loki.types import DataType
from loki.expression import symbol_types as sym, FindVariables, SubstituteExpressions
from loki.backend import cgen, pygen
from loki import ir, Subroutine, SourceFile


class SDFGVisitor(Visitor):

    # pylint: disable=no-self-use

    def __init__(self, name):
        super().__init__()

        self.sdfg = dace.SDFG(name)
        self.var_map = {}

    @staticmethod
    def get_dtype(_type):
        type_map = {
            DataType.LOGICAL: lambda kind: dace.bool,
            DataType.INTEGER: lambda kind: dace.int32,
            DataType.REAL: lambda kind: dace.float32 if kind in ('real32',) else dace.float64,
            }
        return type_map[_type.dtype](_type.kind)

    def get_var(self, var):
        if not var in self.var_map:
            if isinstance(var, sym.Array):
                shape = [self.get_var(d) for d in var.shape]
                if var.type.intent is not None:
                    self.var_map[var] = self.sdfg.add_array(var.name, shape=shape,
                                                            dtype=self.get_dtype(var.type))
                else:
                    self.var_map[var] = self.sdfg.add_transient(var.name, shape=shape,
                                                                dtype=self.get_dtype(var.type))
            else:
                self.var_map[var] = dace.symbol(var.name)
        return self.var_map[var]

    def visit_Subroutine(self, o, **kwargs):
        for var in o.variables:
            self.get_var(var)
        state = self.sdfg.add_state()
        self.visit(o.body, state=state, **kwargs)

    def visit_Loop(self, o, **kwargs):
        state = kwargs.pop('state')
        var_name = str(o.variable)
        loop_vars = kwargs.pop('loop_vars', set()) | {var_name}
        map_entry, map_exit = state.add_map(repr(o), {var_name: str(o.bounds)})
        tasklets = self.visit(o.body, state=state, loop_vars=loop_vars, **kwargs)
        state.add_edge(map_entry, None, tasklets[0], None, dace.Memlet())
        for t1, t2 in zip(tasklets[:-1], tasklets[1:]):
            state.add_edge(t1, None, t2, None, dace.Memlet())
        state.add_edge(tasklets[-1], None, map_exit, None, dace.Memlet())

    def visit_Statement(self, o, **kwargs):
        state = kwargs.pop('state')
        loop_vars = kwargs.pop('loop_vars', set())
        inputs = set(var.name for var in FindVariables().visit(o.expr)) - loop_vars
        outputs = set(o.target.name)
        code = '{} = {}'.format(o.target.name, pygen(o.expr))
        return state.add_tasklet(repr(o), inputs, outputs, code)


class FortranSDFGTransformation(Transformation):

    def transform_subroutine(self, routine, **kwargs):
        path = Path(kwargs.get('path'))

        # Generate Python kernel
        kernel = self.generate_kernel(routine, **kwargs)
        self.py_path = (path/kernel.name.lower()).with_suffix('.py')
        SourceFile.to_file(source=pygen(kernel, with_dace=True), path=self.py_path)

    @staticmethod
    def generate_kernel(routine, **kwargs):
        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = ir.Section(body=())
        body = ir.Section(body=Transformer({}).visit(routine.body))
        kernel = Subroutine(name='{}_py'.format(routine.name), spec=spec, body=body)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        # Force all variables to lower-caps, as Python/DaCe is case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(kernel.body)
                if isinstance(v, (sym.Scalar, sym.Array)) and not v.name.islower()}
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        # Resolve implicit struct mappings through "associates"
        assoc_map = {}
        vmap = {}
        for assoc in FindNodes(ir.Scope).visit(kernel.body):
            invert_assoc = {v.name: k for k, v in assoc.associations.items()}
            for v in FindVariables(unique=False).visit(kernel.body):
                if v.name in invert_assoc:
                    vmap[v] = invert_assoc[v.name]
            assoc_map[assoc] = assoc.body
        kernel.body = Transformer(assoc_map).visit(kernel.body)
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        FortranCTransformation._resolve_vector_notation(kernel, **kwargs)
        FortranCTransformation._resolve_omni_size_indexing(kernel, **kwargs)
        FortranCTransformation._invert_array_indices(kernel, **kwargs)
        FortranCTransformation._shift_to_zero_indexing(kernel, **kwargs)
        # self._replace_intrinsics(kernel, **kwargs)

        return kernel
