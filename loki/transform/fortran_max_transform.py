from collections import OrderedDict
from pathlib import Path

from loki.transform.transformation import BasicTransformation
from loki.backend import maxjgen, maxjcgen, maxjmanagergen
from loki.expression import (Array, FindVariables, InlineCall, Literal, RangeIndex,
                             SubstituteExpressions, Variable)
from loki.ir import Call, Import, Interface, Intrinsic, Loop, Section, Statement
from loki.module import Module
from loki.sourcefile import SourceFile
from loki.subroutine import Subroutine
from loki.tools import as_tuple
from loki.types import BaseType, DerivedType
from loki.visitors import Transformer, FindNodes


__all__ = ['FortranMaxTransformation']


class FortranMaxTransformation(BasicTransformation):
    """
    Fortran-to-Maxeler transformation that translates the given routine
    into .maxj and generates a matching manager and host code with
    corresponding ISO-C wrappers.
    """

    def __init__(self):
        super().__init__()

    def _pipeline(self, source, **kwargs):
        path = kwargs.get('path')

        # Maps from original type name to ISO-C and C-struct types
        c_structs = OrderedDict()

        if isinstance(source, Module):
            # TODO
            raise NotImplementedError('Module translation not yet done')

        elif isinstance(source, Subroutine):
            # Generate maxj kernel that is to be run on the FPGA
            maxj_kernel = self.generate_maxj_kernel(source)
            target_dir = path / maxj_kernel.name
            target_dir.mkdir(exist_ok=True)
            self.maxj_kernel_path = (target_dir / maxj_kernel.name).with_suffix('.maxj')
            SourceFile.to_file(source=maxjgen(maxj_kernel), path=self.maxj_kernel_path)

            # Generate matching kernel manager
            self.maxj_manager_path = Path('%sManager.maxj' % (target_dir / maxj_kernel.name))
            SourceFile.to_file(source=maxjmanagergen(source), path=self.maxj_manager_path)

            # Generate C host code
            c_kernel = self.generate_c_kernel(source)
            self.c_path = (target_dir / c_kernel.name).with_suffix('.c')
            SourceFile.to_file(source=maxjcgen(c_kernel), path=self.c_path)

            # Generate Fortran wrapper routine
            wrapper = self.generate_iso_c_wrapper_routine(source, c_structs)
            self.wrapperpath = (target_dir / wrapper.name.lower()).with_suffix('.f90')
            self.write_to_file(wrapper, filename=self.wrapperpath, module_wrap=True)

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

        self._resolve_vector_notation(kernel, **kwargs)

        # TODO: This is NOT how you do it! This is silly!
        # (But it might work for the very first kernel...)
#        vmap = {}
#        for stmt in FindNodes(Statement).visit(kernel.body):
#            for v in FindNodes(Variable).visit(stmt):
#                if not isinstance(v, Array):
#                    continue
#                vmap[v] = v.clone(dimensions=(1,))
#        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        self._resolve_omni_size_indexing(kernel, **kwargs)

        return kernel

    def generate_c_kernel(self, routine, **kwargs):
        """
        Re-generate the C kernel and insert wrapper-specific peculiarities,
        such as the explicit getter calls for imported module-level variables.
        """

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

        # Add import of kernel header
        imports += [Import(module='%s.h' % routine.name, c_import=True)]

        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = Section(body=imports)
        body = Transformer({}).visit(routine.body)
        body = as_tuple(getter_calls) + as_tuple(body)

        # Force all variables to lower-caps, as C/C++ is case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(body)
                if (v.is_Scalar or v.is_Array) and not v.name.islower()}
        body = SubstituteExpressions(vmap).visit(body)

        kernel = Subroutine(name='%s_c' % routine.name, spec=spec, body=body, cache=routine._cache)
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

        # Resolve implicit struct mappings through "associates"
#        assoc_map = {}
#        vmap = {}
#        for assoc in FindNodes(Scope).visit(kernel.body):
#            invert_assoc = {v: k for k, v in assoc.associations.items()}
#            for v in FindVariables(unique=False).visit(kernel.body):
#                if v in invert_assoc:
#                    vmap[v] = v.clone(parent=invert_assoc[v].parent)
#            assoc_map[assoc] = assoc.body
#        kernel.body = Transformer(assoc_map).visit(kernel.body)
#        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        self._resolve_vector_notation(kernel, **kwargs)
        self._resolve_omni_size_indexing(kernel, **kwargs)

        # TODO: Resolve reductions (eg. SUM(myvar(:)))
#        self._invert_array_indices(kernel, **kwargs)
#        self._shift_to_zero_indexing(kernel, **kwargs)
#        self._replace_intrinsics(kernel, **kwargs)

        return kernel

    def generate_iso_c_interface(self, routine, c_structs):
        """
        Generate the ISO-C subroutine interface
        """
        intf_name = '%s_iso_c' % routine.name
        isoc_import = Import(module='iso_c_binding',
                             symbols=('c_int', 'c_double', 'c_float'))
        intf_spec = Section(body=as_tuple(isoc_import))
        intf_spec.body += as_tuple(Intrinsic(text='implicit none'))
        intf_spec.body += as_tuple(c_structs.values())
        intf_routine = Subroutine(name=intf_name, spec=intf_spec, args=(),
                                  body=None, bind='%s_c' % routine.name)

        # Generate variables and types for argument declarations
        for arg in routine.arguments:
            if isinstance(arg.type, DerivedType):
                ctype = DerivedType(name=c_structs[arg.type.name.lower()].name, variables=None)
            else:
                ctype = arg.type.dtype.isoctype
                # Only scalar, intent(in) arguments are pass by value
                ctype.value = arg.is_Scalar and arg.type.intent.lower() == 'in'
                # Pass by reference for array types
            dimensions = arg.dimensions if arg.is_Array else None
            shape = arg.shape if arg.is_Array else None
            var = intf_routine.Variable(name=arg.name, dimensions=dimensions,
                                        shape=shape, type=ctype)
            intf_routine.variables += [var]
            intf_routine.arguments += [var]

        return Interface(body=(intf_routine, ))

    def generate_iso_c_wrapper_routine(self, routine, c_structs):
        interface = self.generate_iso_c_interface(routine, c_structs)

        # Generate the wrapper function
        wrapper_spec = Transformer().visit(routine.spec)
        wrapper_spec.prepend(Import(module='iso_c_binding',
                                    symbols=('c_int', 'c_double', 'c_float')))
        wrapper_spec.append(c_structs.values())
        wrapper_spec.append(interface)

        # Create the wrapper function with casts and interface invocation
        local_arg_map = OrderedDict()
        casts_in = []
        casts_out = []
        for arg in routine.arguments:
            if isinstance(arg.type, DerivedType):
                ctype = DerivedType(name=c_structs[arg.type.name.lower()].name, variables=None)
                cvar = Variable(name='%s_c' % arg.name, type=ctype)
                cast_in = InlineCall(name='transfer', arguments=as_tuple(arg),
                                     kwarguments=as_tuple([('mold', cvar)]))
                casts_in += [Statement(target=cvar, expr=cast_in)]

                cast_out = InlineCall(name='transfer', arguments=as_tuple(cvar),
                                      kwarguments=as_tuple([('mold', arg)]))
                casts_out += [Statement(target=arg, expr=cast_out)]
                local_arg_map[arg.name] = cvar

        arguments = [local_arg_map[a] if a in local_arg_map else a for a in routine.argnames]
        wrapper_body = casts_in
        wrapper_body += [Call(name=interface.body[0].name, arguments=arguments)]
        wrapper_body += casts_out
        wrapper = Subroutine(name='%s_fc' % routine.name, spec=wrapper_spec, body=wrapper_body)

        # Copy internal argument and declaration definitions
        wrapper.variables = routine.arguments + [v for _, v in local_arg_map.items()]
        wrapper.arguments = routine.arguments
        return wrapper

    def _resolve_vector_notation(self, kernel, **kwargs):
        """
        Resolve implicit vector notation by inserting explicit loops
        """
        loop_map = {}
        index_vars = set()
        vmap = {}
        for stmt in FindNodes(Statement).visit(kernel.body):
            # Loop over all variables and replace them with loop indices
            vdims = []
            shape_index_map = {}
            index_range_map = {}
            for v in FindVariables(unique=False).visit(stmt):
                if not isinstance(v, Array):
                    continue

                for dim, s in zip(v.dimensions, as_tuple(v.shape)):
                    if isinstance(dim, RangeIndex):
                        # Create new index variable
                        vtype = BaseType(name='integer', kind='4')
                        ivar = Variable(name='i_%s' % s.upper, type=vtype)
                        shape_index_map[s] = ivar
                        index_range_map[ivar] = s

                        if ivar not in vdims:
                            vdims.append(ivar)

                # Add index variable to range replacement
                new_dims = as_tuple(shape_index_map.get(s, d)
                                    for d, s in zip(v.dimensions, v.shape))
                vmap[v] = v.clone(dimensions=new_dims)

            index_vars.update(list(vdims))

            # Recursively build new loop nest over all implicit dims
            if len(vdims) > 0:
                loop = None
                body = stmt
                for ivar in vdims:
                    irange = index_range_map[ivar]
                    bounds = (irange.lower or Literal(1), irange.upper, irange.step)
                    loop = Loop(variable=ivar, body=body, bounds=bounds)
                    body = loop

                loop_map[stmt] = loop

        if len(loop_map) > 0:
            kernel.body = Transformer(loop_map).visit(kernel.body)
        kernel.variables += list(set(index_vars))

        # Apply variable substitution
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

    def _resolve_omni_size_indexing(self, kernel, **kwargs):
        """
        Replace the ``(1:size)`` indexing in array sizes that OMNI introduces
        """
        vmap = {}
        for v in kernel.variables:
            if isinstance(v, Array):
                new_dims = as_tuple(d.upper if isinstance(d, RangeIndex) \
                                    and d.lower == 1 and d.step is None else d
                                    for d in v.dimensions)
                vmap[v] = v.clone(dimensions=new_dims)
        kernel.arguments = [vmap.get(v, v) for v in kernel.arguments]
        kernel.variables = [vmap.get(v, v) for v in kernel.variables]
