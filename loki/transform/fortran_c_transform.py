# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from collections import OrderedDict

from loki.transform.transformation import Transformation
from loki.transform.transform_array_indexing import (
    shift_to_zero_indexing, invert_array_indices,
    resolve_vector_notation, normalize_array_shape_and_access,
    flatten_arrays
)
from loki.transform.transform_associates import resolve_associates
from loki.transform.transform_utilities import (
    convert_to_lower_case, replace_intrinsics, sanitise_imports
)
from loki.transform.transform_inline import (
    inline_constant_parameters, inline_elemental_functions
)
from loki.sourcefile import Sourcefile
from loki.backend import cgen, fgen
from loki.ir import (
    Section, Import, Intrinsic, Interface, CallStatement, VariableDeclaration,
    TypeDef, Assignment
)
from loki.subroutine import Subroutine
from loki.module import Module
from loki.expression import (
    Variable, InlineCall, RangeIndex, Scalar, Array,
    ProcedureSymbol, SubstituteExpressions, Dereference
)
from loki.visitors import Transformer, FindNodes
from loki.tools import as_tuple, flatten
from loki.types import BasicType, DerivedType, SymbolAttributes

__all__ = ['FortranCTransformation']


class FortranCTransformation(Transformation):
    """
    Fortran-to-C transformation that translates the given routine
    into C and generates the corresponding ISO-C wrappers.
    """
    # pylint: disable=unused-argument

    # Set of standard module names that have no C equivalent
    __fortran_intrinsic_modules = ['ISO_FORTRAN_ENV', 'ISO_C_BINDING']

    def __init__(self, header_modules=None, inline_elementals=True):
        self.inline_elementals = inline_elementals

        # Maps from original type name to ISO-C and C-struct types
        self.c_structs = OrderedDict()

    def transform_file(self, sourcefile, **kwargs):
        for module in sourcefile.modules:
            self.transform_module(module, **kwargs)

        for routine in sourcefile.subroutines:
            self.transform_subroutine(routine, **kwargs)

    def transform_module(self, module, **kwargs):
        path = Path(kwargs.get('path'))
        role = kwargs.get('role', 'kernel')

        for name, td in module.typedef_map.items():
            self.c_structs[name.lower()] = self.c_struct_typedef(td)

        if role == 'header':
            # Generate Fortran wrapper module
            wrapper = self.generate_iso_c_wrapper_module(module)
            self.wrapperpath = (path/wrapper.name.lower()).with_suffix('.F90')
            Sourcefile.to_file(source=fgen(wrapper), path=self.wrapperpath)

            # Generate C header file from module
            c_header = self.generate_c_header(module)
            self.c_path = (path/c_header.name.lower()).with_suffix('.h')
            Sourcefile.to_file(source=cgen(c_header), path=self.c_path)

        for routine in module.subroutines:
            self.transform_subroutine(routine, **kwargs)

    def transform_subroutine(self, routine, **kwargs):
        path = Path(kwargs.get('path'))
        role = kwargs.get('role', 'kernel')

        for arg in routine.arguments:
            if isinstance(arg.type.dtype, DerivedType):
                self.c_structs[arg.type.dtype.name.lower()] = self.c_struct_typedef(arg.type)

        if role == 'kernel':
            # Generate Fortran wrapper module
            wrapper = self.generate_iso_c_wrapper_routine(routine, self.c_structs)
            contains = Section(body=(Intrinsic('CONTAINS'), wrapper))
            self.wrapperpath = (path/wrapper.name.lower()).with_suffix('.F90')
            module = Module(name=f'{wrapper.name.upper()}_MOD', contains=contains)
            Sourcefile.to_file(source=fgen(module), path=self.wrapperpath)

            # Generate C source file from Loki IR
            c_kernel = self.generate_c_kernel(routine)
            self.c_path = (path/c_kernel.name.lower()).with_suffix('.c')
            Sourcefile.to_file(source=cgen(c_kernel), path=self.c_path)

    @classmethod
    def c_struct_typedef(cls, derived):
        """
        Create the :class:`TypeDef` for the C-wrapped struct definition.
        """
        typename = f'{derived.name if isinstance(derived, TypeDef) else derived.dtype.name}_c'
        typedef = TypeDef(name=typename.lower(), body=(), bind_c=True)  # pylint: disable=unexpected-keyword-arg
        if isinstance(derived, TypeDef):
            variables = derived.variables
        else:
            variables = derived.dtype.typedef.variables
        declarations = []
        for v in variables:
            ctype = v.type.clone(kind=cls.iso_c_intrinsic_kind(v.type, typedef))
            vnew = v.clone(name=v.basename.lower(), scope=typedef, type=ctype)
            declarations += (VariableDeclaration(symbols=(vnew,)),)
        typedef._update(body=as_tuple(declarations))
        return typedef

    @staticmethod
    def iso_c_intrinsic_import(scope):
        symbols = as_tuple(Variable(name=name, scope=scope) for name in ['c_int', 'c_double', 'c_float'])
        isoc_import = Import(module='iso_c_binding', symbols=symbols)
        return isoc_import

    @staticmethod
    def iso_c_intrinsic_kind(_type, scope):
        if _type.dtype == BasicType.INTEGER:
            return Variable(name='c_int', scope=scope)
        if _type.dtype == BasicType.REAL:
            kind = str(_type.kind)
            if kind.lower() in ('real32', 'c_float'):
                return Variable(name='c_float', scope=scope)
            if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
                return Variable(name='c_double', scope=scope)
        return None

    @staticmethod
    def c_intrinsic_kind(_type, scope):
        if _type.dtype == BasicType.LOGICAL:
            return Variable(name='int', scope=scope)
        if _type.dtype == BasicType.INTEGER:
            return Variable(name='int', scope=scope)
        if _type.dtype == BasicType.REAL:
            kind = str(_type.kind)
            if kind.lower() in ('real32', 'c_float'):
                return Variable(name='float', scope=scope)
            if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
                return Variable(name='double', scope=scope)
        return None

    @classmethod
    def generate_iso_c_wrapper_routine(cls, routine, c_structs, bind_name=None):
        wrapper = Subroutine(name=f'{routine.name}_fc')

        if bind_name is None:
            bind_name = f'{routine.name.lower()}_c'
        interface = cls.generate_iso_c_interface(routine, bind_name, c_structs, scope=wrapper)

        # Generate the wrapper function
        wrapper_spec = Transformer().visit(routine.spec)
        wrapper_spec.prepend(cls.iso_c_intrinsic_import(wrapper))
        wrapper_spec.append(struct.clone(parent=wrapper) for struct in c_structs.values())
        wrapper_spec.append(interface)
        wrapper.spec = wrapper_spec

        # Create the wrapper function with casts and interface invocation
        local_arg_map = OrderedDict()
        casts_in = []
        casts_out = []
        for arg in routine.arguments:
            if isinstance(arg.type.dtype, DerivedType):
                ctype = SymbolAttributes(DerivedType(name=c_structs[arg.type.dtype.name.lower()].name))
                cvar = Variable(name=f'{arg.name}_c', type=ctype, scope=wrapper)
                cast_in = InlineCall(ProcedureSymbol('transfer', scope=wrapper),
                                     parameters=(arg,), kw_parameters={'mold': cvar})
                casts_in += [Assignment(lhs=cvar, rhs=cast_in)]

                cast_out = InlineCall(ProcedureSymbol('transfer', scope=wrapper),
                                      parameters=(cvar,), kw_parameters={'mold': arg})
                casts_out += [Assignment(lhs=arg, rhs=cast_out)]
                local_arg_map[arg.name] = cvar

        arguments = tuple(local_arg_map[a] if a in local_arg_map else Variable(name=a)
                          for a in routine.argnames)
        wrapper_body = casts_in
        wrapper_body += [
            CallStatement(name=Variable(name=interface.body[0].name), arguments=arguments)  # pylint: disable=unsubscriptable-object
        ]
        wrapper_body += casts_out
        wrapper.body = Section(body=as_tuple(wrapper_body))

        # Copy internal argument and declaration definitions
        wrapper.variables = tuple(arg.clone(scope=wrapper) for arg in routine.arguments) + tuple(local_arg_map.values())
        wrapper.arguments = tuple(arg.clone(scope=wrapper) for arg in routine.arguments)

        # Remove any unused imports
        sanitise_imports(wrapper)
        return wrapper

    @classmethod
    def generate_iso_c_wrapper_module(cls, module):
        """
        Generate the ISO-C wrapper module for a raw Fortran module.

        Note, we only create getter functions for module variables here,
        since certain type definitions cannot be used in ISO-C interfaces
        due to pointer variables, etc.
        """
        modname = f'{module.name}_fc'
        wrapper_module = Module(name=modname)

        # Generate bind(c) intrinsics for module variables
        original_import = Import(module=module.name)
        isoc_import = cls.iso_c_intrinsic_import(module)
        implicit_none = Intrinsic(text='implicit none')
        spec = [original_import, isoc_import, implicit_none]

        # Create getter methods for module-level variables (I know... :( )
        wrappers = []
        for decl in FindNodes(VariableDeclaration).visit(module.spec):
            for v in decl.symbols:
                if isinstance(v.type.dtype, DerivedType) or v.type.pointer or v.type.allocatable:
                    continue
                gettername = f'{module.name.lower()}__get__{v.name.lower()}'
                getter = Subroutine(name=gettername, bind=gettername, is_function=True, parent=wrapper_module)

                getter.spec = Section(body=(Import(module=module.name, symbols=(v.clone(scope=getter), )), ))
                isoctype = SymbolAttributes(v.type.dtype, kind=cls.iso_c_intrinsic_kind(v.type, getter))
                if isoctype.kind in ['c_int', 'c_float', 'c_double']:
                    getter.spec.append(Import(module='iso_c_binding', symbols=(isoctype.kind, )))
                getter.body = Section(body=(Assignment(lhs=Variable(name=gettername, scope=getter), rhs=v),))
                getter.variables = as_tuple(Variable(name=gettername, type=isoctype, scope=getter))
                wrappers += [getter]
        wrapper_module.contains = Section(body=(Intrinsic('CONTAINS'), *wrappers))

        # Create function interface definitions for module functions
        intfs = []
        for fct in module.subroutines:
            if fct.is_function:
                intf_fct = fct.clone(bind=f'{fct.name.lower()}')
                intf_fct.body = Section(body=())

                intf_args = []
                for arg in intf_fct.arguments:
                    # Only scalar, intent(in) arguments are pass by value
                    # Pass by reference for array types
                    value = isinstance(arg, Scalar) and arg.type.intent and arg.type.intent.lower() == 'in'
                    kind = cls.iso_c_intrinsic_kind(arg.type, intf_fct)
                    ctype = SymbolAttributes(arg.type.dtype, value=value, kind=kind)
                    dimensions = arg.dimensions if isinstance(arg, Array) else None
                    var = Variable(name=arg.name, dimensions=dimensions, type=ctype, scope=intf_fct)
                    intf_args += (var,)
                intf_fct.arguments = intf_args
                sanitise_imports(intf_fct)
                intfs.append(intf_fct)
        spec.append(Interface(body=(as_tuple(intfs),)))

        # Remove any unused imports
        sanitise_imports(wrapper_module)
        return wrapper_module

    @classmethod
    def generate_iso_c_interface(cls, routine, bind_name, c_structs, scope):
        """
        Generate the ISO-C subroutine interface
        """
        intf_name = f'{routine.name}_iso_c'
        intf_routine = Subroutine(name=intf_name, body=None, args=(), parent=scope, bind=bind_name)
        intf_spec = Section(body=as_tuple(cls.iso_c_intrinsic_import(intf_routine)))
        for im in FindNodes(Import).visit(routine.spec):
            if not im.c_import:
                im_symbols = tuple(s.clone(scope=intf_routine) for s in im.symbols)
                intf_spec.append(im.clone(symbols=im_symbols))
        intf_spec.append(Intrinsic(text='implicit none'))
        intf_spec.append(c_structs.values())
        intf_routine.spec = intf_spec

        # Generate variables and types for argument declarations
        for arg in routine.arguments:
            if isinstance(arg.type.dtype, DerivedType):
                struct_name = c_structs[arg.type.dtype.name.lower()].name
                ctype = SymbolAttributes(DerivedType(name=struct_name), shape=arg.type.shape)
            else:
                # Only scalar, intent(in) arguments are pass by value
                # Pass by reference for array types
                value = isinstance(arg, Scalar) and arg.type.intent.lower() == 'in'
                kind = cls.iso_c_intrinsic_kind(arg.type, intf_routine)
                ctype = SymbolAttributes(arg.type.dtype, value=value, kind=kind)
            dimensions = arg.dimensions if isinstance(arg, Array) else None
            var = Variable(name=arg.name, dimensions=dimensions, type=ctype, scope=intf_routine)
            intf_routine.variables += (var,)
            intf_routine.arguments += (var,)

        sanitise_imports(intf_routine)

        return Interface(body=(intf_routine, ))

    def generate_c_header(self, module, **kwargs):
        """
        Re-generate the C header as a module with all pertinent nodes,
        but not Fortran-specific intrinsics (eg. implicit none or save).
        """
        header_module = Module(name=f'{module.name}_c')

        # Generate stubs for getter functions
        spec = []
        for decl in FindNodes(VariableDeclaration).visit(module.spec):
            assert len(decl.symbols) == 1
            v = decl.symbols[0]
            # Bail if not a basic type
            if isinstance(v.type.dtype, DerivedType):
                continue
            ctype = self.c_intrinsic_kind(v.type, module)
            tmpl_function = f'{ctype} {module.name.lower()}__get__{v.name.lower()}();'
            spec += [Intrinsic(text=tmpl_function)]

        # Re-create type definitions with range indices (``:``) replaced by pointers
        for td in FindNodes(TypeDef).visit(module.spec):
            header_td = TypeDef(name=td.name.lower(), body=(), parent=header_module)  # pylint: disable=unexpected-keyword-arg
            declarations = []
            for decl in td.declarations:
                variables = []
                for v in decl.symbols:
                    # Note that we force lower-case on all struct variables
                    if isinstance(v, Array):
                        new_shape = as_tuple(d for d in v.shape if not isinstance(d, RangeIndex))
                        new_type = v.type.clone(shape=new_shape)
                        variables += [v.clone(name=v.name.lower(), type=new_type, scope=header_td)]
                    else:
                        variables += [v.clone(name=v.name.lower(), scope=header_td)]
                declarations += [VariableDeclaration(symbols=as_tuple(variables), dimensions=decl.dimensions,
                                                     comment=decl.comment, pragma=decl.pragma)]
            header_td._update(body=as_tuple(declarations))
            spec += [header_td]

        # Generate a header declaration for module routines
        for fct in module.subroutines:
            if fct.is_function:
                fct_type = 'void'
                if fct.name in fct.variables:
                    fct_type = self.c_intrinsic_kind(fct.variable_map[fct.name.lower()].type, header_module)

                args = [f'{self.c_intrinsic_kind(a.type, header_module)} {a.name.lower()}'
                        for a in fct.arguments]
                fct_decl = f'{fct_type} {fct.name.lower()}({", ".join(args)});'
                spec.append(Intrinsic(text=fct_decl))

        header_module.spec = spec
        header_module.rescope_symbols()
        return header_module

    def generate_c_kernel(self, routine, **kwargs):
        """
        Re-generate the C kernel and insert wrapper-specific peculiarities,
        such as the explicit getter calls for imported module-level variables.
        """

        kernel = routine
        kernel.name = f'{kernel.name.lower()}_c'

        # Clean up Fortran vector notation
        resolve_vector_notation(kernel)
        normalize_array_shape_and_access(kernel)

        # Convert array indexing to C conventions
        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        invert_array_indices(kernel)
        shift_to_zero_indexing(kernel)
        flatten_arrays(kernel, order='C', start_index=0)

        # Inline all known parameters, since they can be used in declarations,
        # and thus need to be known before we can fetch them via getters.
        inline_constant_parameters(kernel, external_only=True)

        if self.inline_elementals:
            # Inline known elemental function via expression substitution
            inline_elemental_functions(kernel)

        # Create declarations for module variables
        module_variables = {
            im.module.lower(): [
                s.clone(scope=kernel, type=s.type.clone(imported=None, module=None)) for s in im.symbols
                if isinstance(s, Scalar) and s.type.dtype is not BasicType.DEFERRED and not s.type.parameter
            ]
            for im in kernel.imports
        }
        kernel.variables += as_tuple(flatten(list(module_variables.values())))

        # Create calls to getter routines for module variables
        getter_calls = []
        for module, variables in module_variables.items():
            for var in variables:
                getter = f'{module}__get__{var.name.lower()}'
                vget = Assignment(lhs=var, rhs=InlineCall(ProcedureSymbol(getter, scope=var.scope)))
                getter_calls += [vget]
        kernel.body.prepend(getter_calls)

        # Change imports to C header includes
        import_map = {}
        for im in kernel.imports:
            if str(im.module).upper() in self.__fortran_intrinsic_modules:
                # Remove imports of Fortran intrinsic modules
                import_map[im] = None

            elif not im.c_import and im.symbols:
                # Create a C-header import for any converted modules
                import_map[im] = im.clone(module=f'{im.module.lower()}_c.h', c_import=True, symbols=())

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
        var_map = {}
        for arg in kernel.arguments:
            if not(arg.type.intent.lower() == 'in' and isinstance(arg, Scalar)):
                _type = arg.type.clone(pointer=True)
                if isinstance(arg.type.dtype, DerivedType):
                    # Lower case type names for derived types
                    typedef = _type.dtype.typedef.clone(name=_type.dtype.typedef.name.lower())
                    _type = _type.clone(dtype=typedef.dtype)
                var_map[arg] = Dereference(arg)
                kernel.symbol_attrs[arg.name] = _type
        if var_map:
            routine.body = SubstituteExpressions(var_map).visit(routine.body)
        symbol_map = {'epsilon': 'DBL_EPSILON'}
        function_map = {'min': 'fmin', 'max': 'fmax', 'abs': 'fabs',
                        'exp': 'exp', 'sqrt': 'sqrt', 'sign': 'copysign'}
        replace_intrinsics(kernel, symbol_map=symbol_map, function_map=function_map)

        # Remove redundant imports
        sanitise_imports(kernel)

        return kernel
