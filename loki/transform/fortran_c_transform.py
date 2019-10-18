from collections import OrderedDict
from sympy import evaluate

from loki.transform.transformation import BasicTransformation
from loki.sourcefile import SourceFile
from loki.backend import cgen
from loki.ir import (Section, Import, Intrinsic, Interface, Call, Declaration,
                     TypeDef, Statement, Scope, Loop)
from loki.subroutine import Subroutine
from loki.module import Module
from loki.types import BaseType, DerivedType
from loki.expression import (Variable, FindVariables, InlineCall, RangeIndex,
                             Literal, Array, SubstituteExpressions, ExpressionFinder)
from loki.visitors import Transformer, FindNodes
from loki.tools import as_tuple, flatten


__all__ = ['FortranCTransformation']


class FortranCTransformation(BasicTransformation):
    """
    Fortran-to-C transformation that translates the given routine
    into C and generates the corresponding ISO-C wrappers.
    """

    def __init__(self, header_modules=None):
        # Fortran modules that can be imported as C headers
        self.header_modules = header_modules or None

    def _pipeline(self, source, **kwargs):
        path = kwargs.get('path')

        # Maps from original type name to ISO-C and C-struct types
        c_structs = OrderedDict()

        if isinstance(source, Module):
            for name, td in source.typedefs.items():
                c_structs[name.lower()] = self.c_struct_typedef(td)

            # Generate Fortran wrapper module
            wrapper = self.generate_iso_c_wrapper_module(source, c_structs)
            self.wrapperpath = (path/wrapper.name.lower()).with_suffix('.f90')
            self.write_to_file(wrapper, filename=self.wrapperpath, module_wrap=False)

            # Generate C header file from module
            c_header = self.generate_c_header(source)
            self.c_path = (path/c_header.name.lower()).with_suffix('.h')
            SourceFile.to_file(source=cgen(c_header), path=self.c_path)

        elif isinstance(source, Subroutine):
            for arg in source.arguments:
                if isinstance(arg.type, DerivedType):
                    c_structs[arg.type.name.lower()] = self.c_struct_typedef(arg.type)

            # Generate Fortran wrapper module
            wrapper = self.generate_iso_c_wrapper_routine(source, c_structs)
            self.wrapperpath = (path/wrapper.name.lower()).with_suffix('.f90')
            self.write_to_file(wrapper, filename=self.wrapperpath, module_wrap=True)

            # Generate C source file from Loki IR
            c_kernel = self.generate_c_kernel(source)
            self.c_path = (path/c_kernel.name.lower()).with_suffix('.c')
            SourceFile.to_file(source=cgen(c_kernel), path=self.c_path)

        else:
            raise RuntimeError('Can only translate Module or Subroutine nodes')

    def c_struct_typedef(self, derived):
        """
        Create the :class:`TypeDef` for the C-wrapped struct definition.
        """
        decls = []
        for v in derived.variables:
            ctype = v.type.dtype.isoctype
            vnew = v.clone(name=v.name.lower())
            decls += [Declaration(variables=(vnew, ), type=ctype)]
            typename = '%s_c' % derived.name
        return TypeDef(name=typename, bind_c=True, declarations=decls)

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

    def generate_iso_c_wrapper_module(self, module, c_structs):
        """
        Generate the ISO-C wrapper module for a raw Fortran module.
        """
        # Generate bind(c) intrinsics for module variables
        original_import = Import(module=module.name)
        isoc_import = Import(module='iso_c_binding',
                             symbols=('c_int', 'c_double', 'c_float'))
        implicit_none = Intrinsic(text='implicit none')
        spec = [original_import, isoc_import, implicit_none]

        # Add module-based derived type/struct definitions
        spec += list(c_structs.values())

        # Create getter methods for module-level variables (I know... :( )
        wrappers = []
        for decl in FindNodes(Declaration).visit(module.spec):
            for v in decl.variables:
                if v.type.dtype is None or v.type.pointer or v.type.allocatable:
                    continue
                isoctype = v.type.dtype.isoctype
                gettername = '%s__get__%s' % (module.name.lower(), v.name.lower())
                getterspec = Section(body=[Import(module=module.name, symbols=[v.name])])
                if isoctype.kind in ['c_int', 'c_float', 'c_double']:
                    getterspec.append(Import(module='iso_c_binding', symbols=[isoctype.kind]))
                getterbody = [Statement(target=Variable(name=gettername), expr=v)]
                getter = Subroutine(name=gettername, bind=gettername, spec=getterspec,
                                    body=getterbody, is_function=True)
                getter.variables = as_tuple(Variable(name=gettername, type=isoctype))
                wrappers += [getter]

        return Module(name='%s_fc' % module.name, spec=spec, routines=wrappers)

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

    def generate_c_header(self, module, **kwargs):
        """
        Re-generate the C header as a module with all pertinent nodes,
        but not Fortran-specific intrinsics (eg. implicit none or save).
        """
        # Generate stubs for getter functions
        spec = []
        for decl in FindNodes(Declaration).visit(module.spec):
            assert len(decl.variables) == 1
            v = decl.variables[0]
            # Bail if not a basic type
            if v.type.dtype is None:
                continue
            tmpl_function = '%s %s__get__%s();' % (
                v.type.dtype.ctype, module.name.lower(), v.name.lower())
            spec += [Intrinsic(text=tmpl_function)]

        # Re-create type definitions with range indices (``:``) replaced by pointers
        for td in FindNodes(TypeDef).visit(module.spec):
            vmap = {}
            for decl in td.declarations:
                for v in decl.variables:
                    # Note that we force lower-case on all struct variables
                    if v.is_Array:
                        new_dims = as_tuple(d for d in v.dimensions if not isinstance(d, RangeIndex))
                        vmap[v] = v.clone(name=v.name.lower(), dimensions=new_dims)
                    else:
                        vmap[v] = v.clone(name=v.name.lower())
            spec += [SubstituteExpressions(vmap).visit(td)]

        # Re-generate header module without subroutines
        return Module(name='%s_c' % module.name, spec=spec)

    def generate_c_kernel(self, routine, **kwargs):
        """
        Re-generate the C kernel and insert wrapper-specific peculiarities,
        such as the explicit getter calls for imported module-level variables.
        """

        # Change imports to C header includes
        imports = []
        getter_calls = []
        header_map = {m.name.lower(): m for m in as_tuple(self.header_modules)}
        for imp in FindNodes(Import).visit(routine.spec):
            if imp.module.lower() in header_map:
                # Create a C-header import
                imports += [Import(module='%s_c.h' % imp.module, c_import=True)]

                # For imported modulevariables, create a declaration and call the getter
                module = header_map[imp.module]
                mod_vars = flatten(d.variables for d in FindNodes(Declaration).visit(module.spec))
                mod_vars = {v.name.lower(): v for v in mod_vars}

                for s in imp.symbols:
                    if s.lower() in mod_vars:
                        var = mod_vars[s.lower()]

                        decl = Declaration(variables=[var], type=var.type)
                        getter = '%s__get__%s' % (module.name.lower(), var.name.lower())
                        vget = Statement(target=var, expr=InlineCall(name=getter, arguments=()))
                        getter_calls += [decl, vget]

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
        arg_map = {}
        for arg in kernel.arguments:
            if not(arg.type.intent.lower() == 'in' and arg.is_Scalar):
                dtype = arg.type
                dtype.pointer = True
                arg_map[arg.name.lower()] = arg.clone(type=dtype)
        kernel.arguments = [arg_map.get(arg.name.lower(), arg) for arg in kernel.arguments]
        kernel.variables = [arg_map.get(arg.name.lower(), arg) for arg in kernel.variables]

        vmap = {}
        for v in FindVariables(unique=True).visit(kernel.body):
            if v.name.lower() in arg_map:
                dtype = v.type or arg_map[v.name.lower()].type
                dtype.pointer = True
                vmap[v] = v.clone(type=dtype)
        SubstituteExpressions(vmap).visit(kernel.body)

        # Resolve implicit struct mappings through "associates"
        assoc_map = {}
        vmap = {}
        for assoc in FindNodes(Scope).visit(kernel.body):
            invert_assoc = {v: k for k, v in assoc.associations.items()}
            for v in FindVariables(unique=False).visit(kernel.body):
                if v in invert_assoc:
                    vmap[v] = v.clone(parent=invert_assoc[v].parent)
            assoc_map[assoc] = assoc.body
        kernel.body = Transformer(assoc_map).visit(kernel.body)
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        self._resolve_vector_notation(kernel, **kwargs)
        self._resolve_omni_size_indexing(kernel, **kwargs)

        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        self._invert_array_indices(kernel, **kwargs)
        self._shift_to_zero_indexing(kernel, **kwargs)
        self._replace_intrinsics(kernel, **kwargs)

        return kernel

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

    def _invert_array_indices(self, kernel, **kwargs):
        """
        Invert data/loop accesses from column to row-major

        TODO: Take care of the indexing shift between C and Fortran.
        Basically, we are relying on the CGen to shift the iteration
        indices and dearly hope that nobody uses the index's value.
        """
        # Invert array indices in the kernel body
        vmap = {}
        for v in FindVariables(unique=True).visit(kernel.body):
            if isinstance(v, Array):
                vmap[v] = v.clone(dimensions=as_tuple(reversed(v.dimensions)))
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        # Invert variable and argument dimensions for the automatic cast generation
        for v in kernel.variables:
            if isinstance(v, Array):
                vmap[v] = v.clone(dimensions=as_tuple(reversed(v.dimensions)))
        kernel.arguments = [vmap.get(v, v) for v in kernel.arguments]
        kernel.variables = [vmap.get(v, v) for v in kernel.variables]

    def _shift_to_zero_indexing(self, kernel, **kwargs):
        """
        Shift each array indices to adjust to C indexing conventions
        """
        vmap = {}
        for v in FindVariables(unique=True).visit(kernel.body):
            if isinstance(v, Array):
                with evaluate(False):
                    new_dims = as_tuple(d - 1 for d in v.dimensions)
                    vmap[v] = v.clone(dimensions=new_dims)
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

    def _replace_intrinsics(self, kernel, **kwargs):
        """
        Replace known numerical intrinsic functions.
        """
        def q_inlinecall(expr):
            return isinstance(expr, InlineCall)

        def retrieve_inlinecall(expr):
            return expr.find(q_inlinecall)

        _intrinsic_map = {
            'epsilon': 'DBL_EPSILON',
            'min': 'fmin', 'max': 'fmax',
            'abs': 'fabs', 'sign': 'copysign',
        }

        callmap = {}
        for c in ExpressionFinder(retrieve=retrieve_inlinecall).visit(kernel.body):
            cname = c.name.text.lower()
            if cname in _intrinsic_map:
                with evaluate(False):
                    if cname == 'epsilon':
                        callmap[c] = kernel.Variable(name=_intrinsic_map[cname])
                    else:
                        callmap[c] = InlineCall(name=_intrinsic_map[cname],
                                                arguments=c.arguments,
                                                kwarguments=c.kwarguments)

        # Capture nesting by applying map to itself before applying to the kernel
        with evaluate(False):
            callmap = {k: v.xreplace(callmap) for k, v in callmap.items()}
            callmap = {k: v.xreplace(callmap) for k, v in callmap.items()}
        kernel.body = SubstituteExpressions(callmap).visit(kernel.body)
