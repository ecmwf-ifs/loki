from collections import OrderedDict

from loki.transform.transformation import BasicTransformation
from loki.sourcefile import SourceFile
from loki.backend import fgen, cgen
from loki.ir import (Section, Import, Intrinsic, Interface, Call, Declaration,
                     TypeDef, Statement, Scope, Loop)
from loki.subroutine import Subroutine
from loki.module import Module
from loki.types import BaseType, DerivedType, DataType
from loki.expression import (Variable, FindVariables, InlineCall, RangeIndex,
                             Literal, ExpressionVisitor, Operation)
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
            decls += [Declaration(variables=(v, ), type=ctype)]
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
                ctype.value = (arg.dimensions is None or len(arg.dimensions) == 0) \
                              and arg.type.intent.lower() == 'in'
                # Pass by reference for array types
            var = Variable(name=arg.name, dimensions=arg.dimensions,
                           shape=arg.shape, type=ctype)
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
            assert len(decl.variables) == 1;
            v = decl.variables[0]
            # Bail if not a basic type
            if v.type.dtype is None:
                continue
            tmpl_function = '%s %s__get__%s();' % (
                v.type.dtype.ctype, module.name.lower(), v.name.lower())
            spec += [Intrinsic(text=tmpl_function)]

        # Re-create spec with getters and typedefs to wipe Fortran-specifics
        spec += FindNodes(TypeDef).visit(module.spec)

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

        kernel = Subroutine(name='%s_c' % routine.name, spec=spec, body=body)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        # Force pointer on reference-passed arguments
        for arg in kernel.arguments:
            if not (arg.type.intent.lower() == 'in' and \
                    (arg.dimensions is None or len(arg.dimensions) == 0)):
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
        assoc_map = {}
        for assoc in FindNodes(Scope).visit(kernel.body):
            invert_assoc = {v: k for k, v in assoc.associations.items()}
            for v in FindVariables(unique=False).visit(kernel.body):
                if v in invert_assoc:
                    v.ref = invert_assoc[v].ref
            assoc_map[assoc] = assoc.body
        kernel.body = Transformer(assoc_map).visit(kernel.body)

        # TODO: Resolve reductions (eg. SUM(myvar(:)))

        # Resolve implicit vector notation by inserting explicit loops
        loop_map = {}
        index_vars = set()
        for stmt in FindNodes(Statement).visit(kernel.body):
            # Loop over all variables and replace them with loop indices
            vdims = set()
            for v in FindVariables(unique=False).visit(stmt):
                for dim, shape in zip(v.dimensions, as_tuple(v.shape)):
                    if isinstance(dim, RangeIndex):
                        vtype = BaseType(name='integer', kind='4')
                        ivar = Variable(name='i_%s' % shape, type=vtype)
                        vdims.add(ivar)
                        v.dimensions = as_tuple(d if d is not dim else ivar
                                                for d in v.dimensions)
            index_vars.update(list(vdims))

            # Recursively build new loop nest over all implicit dims
            if len(vdims) > 0:
                loop = None
                body = stmt
                for ivar in vdims:
                    # TODO: Handle more complex ranges
                    bounds = RangeIndex(lower=Literal(value='1'), upper=shape)
                    loop = Loop(variable=ivar, body=body, bounds=bounds)
                    body = loop

                loop_map[stmt] = loop

        if len(loop_map) > 0:
            kernel.body = Transformer(loop_map).visit(kernel.body)
        kernel.variables += list(index_vars)

        # Invert data/loop accesses from column to row-major
        # TODO: Take care of the indexing shift between C and Fortran.
        # Basically, we are relying on the CGen to shuft the iteration
        # indices and dearly hope that nobody uses the index's value.
        for v in FindVariables(unique=False).visit(kernel.body):
            v.dimensions = as_tuple(reversed(v.dimensions))

        for v in kernel.variables:
            v.dimensions = as_tuple(reversed(v.dimensions))

        # Shift each array indices to adjust to C indexing conventions
        def minus_one(dim):
            # TODO: Symbolics should make this neater
            return Operation(ops=('-',), operands=(dim, Literal(value='1')))

        for v in FindVariables(unique=False).visit(kernel.body):
            v.dimensions = as_tuple(minus_one(d) for d in v.dimensions)

        # Replace known numerical intrinsic functions
        class IntrinsicVisitor(ExpressionVisitor):
            _intrinsic_map = {
                'epsilon': 'DBL_EPSILON',
                'min': 'fmin', 'max': 'fmax',
                'abs': 'fabs', 'sign': 'copysign',
            }

            def visit_InlineCall(self, o):
                if o.name.lower() in self._intrinsic_map:
                    o.name = self._intrinsic_map[o.name.lower()]

                for c in o.children:
                    self.visit(c)

        intrinsic = IntrinsicVisitor()
        for stmt in FindNodes(Statement).visit(kernel.body):
            intrinsic.visit(stmt.expr)

        return kernel
