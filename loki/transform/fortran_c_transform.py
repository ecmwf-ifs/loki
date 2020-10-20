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
from loki.transform.transform_inline import (
    inline_constant_parameters
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

        # Maps from original type name to ISO-C and C-struct types
        self.c_structs = OrderedDict()

    def transform_module(self, module, **kwargs):
        path = Path(kwargs.get('path'))
        role = kwargs.get('role', 'kernel')

        for name, td in module.typedefs.items():
            self.c_structs[name.lower()] = self.c_struct_typedef(td)

        if role == 'header':
            # Generate Fortran wrapper module
            wrapper = self.generate_iso_c_wrapper_module(module)
            self.wrapperpath = (path/wrapper.name.lower()).with_suffix('.F90')
            SourceFile.to_file(source=fgen(wrapper), path=self.wrapperpath)

            # Generate C header file from module
            c_header = self.generate_c_header(module)
            self.c_path = (path/c_header.name.lower()).with_suffix('.h')
            SourceFile.to_file(source=cgen(c_header), path=self.c_path)

    def transform_subroutine(self, routine, **kwargs):
        path = Path(kwargs.get('path'))
        role = kwargs.get('role', 'kernel')

        for arg in routine.arguments:
            if isinstance(arg.type.dtype, DerivedType):
                self.c_structs[arg.type.dtype.name.lower()] = self.c_struct_typedef(arg.type)

        if role == 'kernel':
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
    def generate_iso_c_wrapper_module(cls, module):
        """
        Generate the ISO-C wrapper module for a raw Fortran module.

        Note, we only create getter functions for module variables here,
        since certain type definitions cannot be used in ISO-C interfaces
        due to pointer variables, etc.
        """
        # Generate bind(c) intrinsics for module variables
        original_import = Import(module=module.name)
        isoc_import = Import(module='iso_c_binding',
                             symbols=('c_int', 'c_double', 'c_float'))
        implicit_none = Intrinsic(text='implicit none')
        spec = [original_import, isoc_import, implicit_none]

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

        kernel = routine
        kernel.name = '%s_c' % kernel.name.lower()

        # Clean up Fortran vector notation
        resolve_vector_notation(kernel)
        normalize_range_indexing(kernel)

        # Convert array indexing to C conventions
        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        invert_array_indices(kernel)
        shift_to_zero_indexing(kernel)

        # Inline all known parameters, since they can be used in declarations,
        # and thus need to be known before we can fetch them via getters.
        inline_constant_parameters(kernel, external_only=True)

        # Create calls to getter routines for module variables
        getter_calls = []
        for im in FindNodes(Import).visit(kernel.spec):
            for s in im.symbols:
                if isinstance(s, Scalar) and s.type.dtype is not BasicType.DEFERRED:
                    # Skip parameters, as they will be inlined
                    if s.type.parameter:
                        continue
                    decl = Declaration(variables=(s,))
                    getter = '%s__get__%s' % (im.module.lower(), s.name.lower())
                    vget = Assignment(lhs=s, rhs=InlineCall(ProcedureSymbol(getter, scope=s.scope)))
                    getter_calls += [decl, vget]
        kernel.body.prepend(getter_calls)

        # Change imports to C header includes
        import_map = {}
        for im in FindNodes(Import).visit(kernel.spec):
            if not im.c_import and im.symbols:
                # Create a C-header import for any converted modules
                import_map[im] = im.clone(module='%s_c.h' % im.module.lower(), c_import=True)
            else:
                # Remove other imports, as they might include untreated Fortran code
                import_map[im] = None
        kernel.spec = Transformer(import_map).visit(kernel.spec)

        # Remove intrinsics from spec (eg. implicit none)
        intrinsic_map = {i: None for i in FindNodes(Intrinsic).visit(kernel.spec)
                         if 'implicit' in i.text.lower()}
        kernel.spec = Transformer(intrinsic_map).visit(kernel.spec)

        # Resolve implicit struct mappings through "associates"
        resolve_associates(kernel)

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

        symbol_map = {'epsilon': 'DBL_EPSILON'}
        function_map = {'min': 'fmin', 'max': 'fmax', 'abs': 'fabs',
                        'EXP': 'exp', 'SQRT': 'sqrt', 'sign': 'copysign'}
        replace_intrinsics(kernel, symbol_map=symbol_map, function_map=function_map)

        return kernel
