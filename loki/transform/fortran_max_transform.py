from pathlib import Path

from loki.transform.transformation import BasicTransformation
from loki.backend import maxjgen, maxjmanagergen
from loki.expression import FindVariables, InlineCall, SubstituteExpressions
from loki.ir import Declaration, Import, Section, Statement
from loki.module import Module
from loki.sourcefile import SourceFile
from loki.subroutine import Subroutine
from loki.tools import as_tuple, flatten
from loki.visitors import Transformer, FindNodes


__all__ = ['FortranMaxTransformation']


class FortranMaxTransformation(BasicTransformation):
    """
    Fortran-to-Maxeler transformation that translates the given routine
    into .maxj and generates a matching manager and host code with
    corresponding ISO-C wrappers.
    """

    def __init__(self):
        pass

    def _pipeline(self, source, **kwargs):
        path = kwargs.get('path')

        if isinstance(source, Module):
            # TODO
            raise NotImplementedError('Module translation not yet done')

        elif isinstance(source, Subroutine):
            # Generate maxj kernel that is to be run on the FPGA
            maxj_kernel = self.generate_maxj_kernel(source)
            self.maxj_kernel_path = (path / maxj_kernel.name).with_suffix('.maxj')
            SourceFile.to_file(source=maxjgen(maxj_kernel), path=self.maxj_kernel_path)

            # Generate matching kernel manager
            self.maxj_manager_path = Path('%sManager.maxj' % (path / maxj_kernel.name))
            SourceFile.to_file(source=maxjmanagergen(source), path=self.maxj_manager_path)

        else:
            raise RuntimeError('Can only translate Module or Subroutine nodes')

    def generate_maxj_kernel(self, routine, **kwargs):
        # Change imports to C header includes
        imports = []
        getter_calls = []
#        header_map = {m.name.lower(): m for m in as_tuple(self.header_modules)}
#        for imp in FindNodes(Import).visit(routine.spec):
#            if imp.module.lower() in header_map:
#                # Create a C-header import
#                imports += [Import(module='%s_c.h' % imp.module, c_import=True)]
#
#                # For imported modulevariables, create a declaration and call the getter
#                module = header_map[imp.module]
#                mod_vars = flatten(d.variables for d in FindNodes(Declaration).visit(module.spec))
#                mod_vars = {v.name.lower(): v for v in mod_vars}
#
#                for s in imp.symbols:
#                    if s.lower() in mod_vars:
#                        var = mod_vars[s.lower()]
#
#                        decl = Declaration(variables=[var], type=var.type)
#                        getter = '%s__get__%s' % (module.name.lower(), var.name.lower())
#                        vget = Statement(target=var, expr=InlineCall(name=getter, arguments=()))
#                        getter_calls += [decl, vget]

        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = Section(body=imports)
        body = Transformer({}).visit(routine.body)
        body = as_tuple(getter_calls) + as_tuple(body)

        # Force all variables to lower-case, as Java is case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(body)
                if (v.is_Scalar or v.is_Array) and not v.name.islower()}
        body = SubstituteExpressions(vmap).visit(body)

        kernel = Subroutine(name=routine.name, spec=spec, body=body, cache=routine._cache)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        # Force pointer on reference-passed arguments
        for arg in kernel.arguments:
            if not (arg.type.intent.lower() == 'in' and arg.is_Scalar):
                arg.type.pointer = True
        # Propagate that reference pointer to all variables
        arg_map = {a.name: a for a in kernel.arguments}
        for v in FindVariables(unique=False).visit(kernel.body):
            if v.name in arg_map:
                if v.type:
                    v.type.pointer = arg_map[v.name].type.pointer
                else:
                    v._type = arg_map[v.name].type

        return kernel
