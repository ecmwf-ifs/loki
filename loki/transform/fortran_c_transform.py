from pathlib import Path
from collections import OrderedDict

from loki.transform.transformation import Transformation
from loki.transform.transform_array_indexing import (
    shift_to_zero_indexing, invert_array_indices,
    resolve_vector_notation, normalize_range_indexing
)
from loki.transform.transform_utilities import (
    convert_to_lower_case, replace_intrinsics, resolve_associates
)
from loki.sourcefile import SourceFile
from loki.backend import cgen, fgen
from loki.ir import (
    Section, Import, Intrinsic, Interface, CallStatement, Declaration,
    TypeDef, Assignment
)
from loki.subroutine import Subroutine
from loki.module import Module
from loki.expression import (
    Variable, FindVariables, InlineCall, RangeIndex, Scalar, Array,
    SubstituteExpressions, ProcedureSymbol
)
from loki.visitors import Transformer, FindNodes
from loki.tools import as_tuple, flatten, CaseInsensitiveDict
from loki.types import BasicType, SymbolType, DerivedType, TypeTable, Scope


__all__ = ['FortranCTransformation']


class FortranCTransformation(Transformation):
    """
    Fortran-to-C transformation that translates the given routine
    into C and generates the corresponding ISO-C wrappers.
    """
    # pylint: disable=unused-argument

    def __init__(self, header_modules=None):
        # Fortran modules that can be imported as C headers
        self.header_modules = header_modules or None

        # Maps from original type name to ISO-C and C-struct types
        self.c_structs = OrderedDict()

    def transform_module(self, module, **kwargs):
        path = Path(kwargs.get('path'))

        for name, td in module.typedefs.items():
            self.c_structs[name.lower()] = self.c_struct_typedef(td)

        # Generate Fortran wrapper module
        wrapper = self.generate_iso_c_wrapper_module(module, self.c_structs)
        self.wrapperpath = (path/wrapper.name.lower()).with_suffix('.F90')
        SourceFile.to_file(source=fgen(wrapper), path=self.wrapperpath)

        # Generate C header file from module
        c_header = self.generate_c_header(module)
        self.c_path = (path/c_header.name.lower()).with_suffix('.h')
        SourceFile.to_file(source=cgen(c_header), path=self.c_path)

    def transform_subroutine(self, routine, **kwargs):
        path = Path(kwargs.get('path'))

        for arg in routine.arguments:
            if isinstance(arg.type.dtype, DerivedType):
                self.c_structs[arg.type.dtype.name.lower()] = self.c_struct_typedef(arg.type)

        # Generate Fortran wrapper module
        wrapper = self.generate_iso_c_wrapper_routine(routine, self.c_structs)
        self.wrapperpath = (path/wrapper.name.lower()).with_suffix('.F90')
        module = Module(name='%s_MOD' % wrapper.name.upper(), routines=[wrapper])
        SourceFile.to_file(source=fgen(module), path=self.wrapperpath)

        # Generate C source file from Loki IR
        c_kernel = self.generate_c_kernel(routine)
        self.c_path = (path/c_kernel.name.lower()).with_suffix('.c')
        SourceFile.to_file(source=cgen(c_kernel), path=self.c_path)

    @classmethod
    def c_struct_typedef(cls, derived):
        """
        Create the :class:`TypeDef` for the C-wrapped struct definition.
        """
        typename = '%s_c' % (derived.name if isinstance(derived, TypeDef) else derived.dtype.name)
        scope = Scope()
        if isinstance(derived, TypeDef):
            variables = derived.variables
        else:
            variables = derived.dtype.typedef.variables
        declarations = []
        for v in variables:
            ctype = v.type.clone(kind=cls.iso_c_intrinsic_kind(v.type))
            vnew = v.clone(name=v.basename.lower(), scope=scope, type=ctype)
            declarations += (Declaration(variables=(vnew,)),)
        return TypeDef(name=typename.lower(), bind_c=True, body=declarations, scope=scope)

    @staticmethod
    def iso_c_intrinsic_kind(_type):
        if _type.dtype == BasicType.INTEGER:
            return 'c_int'
        if _type.dtype == BasicType.REAL:
            kind = str(_type.kind)
            if kind.lower() in ('real32', 'c_float'):
                return 'c_float'
            if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
                return 'c_double'
        return None

    @staticmethod
    def c_intrinsic_kind(_type):
        if _type.dtype == BasicType.LOGICAL:
            return 'int'
        if _type.dtype == BasicType.INTEGER:
            return 'int'
        if _type.dtype == BasicType.REAL:
            kind = str(_type.kind)
            if kind.lower() in ('real32', 'c_float'):
                return 'float'
            if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
                return 'double'
        return None

    @classmethod
    def generate_iso_c_wrapper_routine(cls, routine, c_structs, bind_name=None):
        wrapper_scope = Scope()

        if bind_name is None:
            bind_name = '%s_c' % routine.name
        interface = cls.generate_iso_c_interface(routine, bind_name, c_structs, scope=wrapper_scope)

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
            if isinstance(arg.type.dtype, DerivedType):
                ctype = SymbolType(DerivedType(name=c_structs[arg.type.dtype.name.lower()].name))
                cvar = Variable(name='%s_c' % arg.name, type=ctype, scope=wrapper_scope)
                cast_in = InlineCall(ProcedureSymbol('transfer', scope=wrapper_scope),
                                     parameters=(arg,), kw_parameters={'mold': cvar})
                casts_in += [Assignment(lhs=cvar, rhs=cast_in)]

                cast_out = InlineCall(ProcedureSymbol('transfer', scope=wrapper_scope),
                                      parameters=(cvar,), kw_parameters={'mold': arg})
                casts_out += [Assignment(lhs=arg, rhs=cast_out)]
                local_arg_map[arg.name] = cvar

        arguments = [local_arg_map[a] if a in local_arg_map else a for a in routine.argnames]
        wrapper_body = casts_in
        wrapper_body += [CallStatement(name=interface.body[0].name, arguments=arguments)]
        wrapper_body += casts_out
        wrapper_body = Section(body=wrapper_body)
        wrapper = Subroutine(name='%s_fc' % routine.name, spec=wrapper_spec, body=wrapper_body,
                             scope=wrapper_scope)

        # Copy internal argument and declaration definitions
        wrapper.variables = routine.arguments + tuple(v for _, v in local_arg_map.items())
        wrapper.arguments = routine.arguments
        return wrapper

    @classmethod
    def generate_iso_c_wrapper_module(cls, module, c_structs):
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

        module_scope = Scope()

        # Create getter methods for module-level variables (I know... :( )
        wrappers = []
        for decl in FindNodes(Declaration).visit(module.spec):
            for v in decl.variables:
                if isinstance(v.type.dtype, DerivedType) or v.type.pointer or v.type.allocatable:
                    continue
                gettername = '%s__get__%s' % (module.name.lower(), v.name.lower())
                getter_scope = Scope(parent=module_scope)

                getterspec = Section(body=[Import(module=module.name, symbols=[v.name])])
                isoctype = SymbolType(v.type.dtype, kind=cls.iso_c_intrinsic_kind(v.type))
                if isoctype.kind in ['c_int', 'c_float', 'c_double']:
                    getterspec.append(Import(module='iso_c_binding', symbols=[isoctype.kind]))
                getterbody = Section(body=[
                    Assignment(lhs=Variable(name=gettername, scope=getter_scope), rhs=v)])

                getter = Subroutine(name=gettername, scope=getter_scope, spec=getterspec,
                                    body=getterbody, bind=gettername, is_function=True)
                getter.variables = as_tuple(Variable(name=gettername, type=isoctype,
                                                     scope=getter.scope))
                wrappers += [getter]

        modname = '{}_fc'.format(module.name)
        return Module(name=modname, spec=spec, routines=wrappers, scope=module_scope)

    @classmethod
    def generate_iso_c_interface(cls, routine, bind_name, c_structs, scope):
        """
        Generate the ISO-C subroutine interface
        """
        intf_name = '%s_iso_c' % routine.name
        isoc_import = Import(module='iso_c_binding',
                             symbols=('c_int', 'c_double', 'c_float'))
        intf_spec = Section(body=as_tuple(isoc_import))
        intf_spec.append(im for im in FindNodes(Import).visit(routine.spec) if not im.c_import)
        intf_spec.append(Intrinsic(text='implicit none'))
        intf_spec.append(c_structs.values())
        intf_routine = Subroutine(name=intf_name, spec=intf_spec, body=None,
                                  args=(), scope=scope, bind=bind_name)

        # Generate variables and types for argument declarations
        for arg in routine.arguments:
            if isinstance(arg.type.dtype, DerivedType):
                struct_name = c_structs[arg.type.dtype.name.lower()].name
                ctype = SymbolType(DerivedType(name=struct_name), shape=arg.type.shape,
                                   name=struct_name)
            else:
                # Only scalar, intent(in) arguments are pass by value
                # Pass by reference for array types
                value = isinstance(arg, Scalar) and arg.type.intent.lower() == 'in'
                ctype = SymbolType(arg.type.dtype, value=value,
                                   kind=cls.iso_c_intrinsic_kind(arg.type))
            dimensions = arg.dimensions if isinstance(arg, Array) else None
            var = Variable(name=arg.name, dimensions=dimensions, type=ctype,
                           scope=intf_routine.scope)
            intf_routine.variables += (var,)
            intf_routine.arguments += (var,)

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
            if isinstance(v.type.dtype, DerivedType):
                continue
            tmpl_function = '%s %s__get__%s();' % (
                self.c_intrinsic_kind(v.type), module.name.lower(), v.name.lower())
            spec += [Intrinsic(text=tmpl_function)]

        # Re-create type definitions with range indices (``:``) replaced by pointers
        for td in FindNodes(TypeDef).visit(module.spec):
            declarations = []
            for decl in td.declarations:
                variables = []
                for v in decl.variables:
                    # Note that we force lower-case on all struct variables
                    if isinstance(v, Array):
                        new_shape = as_tuple(d for d in v.shape if not isinstance(d, RangeIndex))
                        new_type = v.type.clone(shape=new_shape)
                        variables += [v.clone(name=v.name.lower(), type=new_type)]
                    else:
                        variables += [v.clone(name=v.name.lower())]
                declarations += [Declaration(variables=variables, dimensions=decl.dimensions,
                                             comment=decl.comment, pragma=decl.pragma)]
            td = td.clone(body=declarations)
            spec += [td]

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
        header_map = CaseInsensitiveDict({m.name: m for m in as_tuple(self.header_modules)})
        for imp in FindNodes(Import).visit(routine.spec):
            if imp.module in header_map:
                # Create a C-header import
                imports += [Import(module='%s_c.h' % imp.module.lower(), c_import=True)]

                # For imported modulevariables, create a declaration and call the getter
                module = header_map[imp.module]
                mod_vars = flatten(d.variables for d in FindNodes(Declaration).visit(module.spec))
                mod_vars = {v.name.lower(): v for v in mod_vars}

                for s in imp.symbols:
                    if str(s).lower() in mod_vars:
                        var = mod_vars[str(s).lower()]

                        decl = Declaration(variables=(var,))
                        getter = '%s__get__%s' % (module.name.lower(), var.name.lower())
                        vget = Assignment(lhs=var, rhs=InlineCall(ProcedureSymbol(getter, scope=var.scope)))
                        getter_calls += [decl, vget]

        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = Section(body=imports)
        body = Transformer({}).visit(routine.body)
        body = Section(body=as_tuple(getter_calls) + as_tuple(body))

        kernel = Subroutine(name='%s_c' % routine.name, spec=spec, body=body)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        # Force all variables to lower-caps, as C/C++ is case-sensitive
        convert_to_lower_case(kernel)

        # Force pointer on reference-passed arguments
        arg_map = {}
        for arg in kernel.arguments:
            if not(arg.type.intent.lower() == 'in' and isinstance(arg, Scalar)):
                dtype = arg.type
                # Down-case type names for derived types
                if isinstance(dtype.dtype, DerivedType):
                    dtype= dtype.clone(dtype=DerivedType(name=dtype.dtype.name.lower(),
                                                         typedef=dtype.typedef))
                dtype.pointer = True
                arg_map[arg.name.lower()] = arg.clone(type=dtype)
        kernel.arguments = [arg_map.get(arg.name.lower(), arg) for arg in kernel.arguments]
        kernel.variables = [arg_map.get(arg.name.lower(), arg) for arg in kernel.variables]

        vmap = {}
        for v in FindVariables(unique=False).visit(kernel.body):
            if v.name.lower() in arg_map:
                dtype = v.type or arg_map[v.name.lower()].type
                dtype.pointer = True
                vmap[v] = v.clone(type=dtype)
        SubstituteExpressions(vmap).visit(kernel.body)

        # Resolve implicit struct mappings through "associates"
        resolve_associates(kernel)

        # Clean up Fortran vector notation
        resolve_vector_notation(kernel)
        normalize_range_indexing(kernel)

        # Convert array indexing to C conventions
        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        invert_array_indices(kernel)
        shift_to_zero_indexing(kernel)

        symbol_map = {'epsilon': 'DBL_EPSILON'}
        function_map = {'min': 'fmin', 'max': 'fmax', 'abs': 'fabs', 'sign': 'copysign'}
        replace_intrinsics(kernel, symbol_map=symbol_map, function_map=function_map)

        return kernel
