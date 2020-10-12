from pathlib import Path
from collections import OrderedDict
from itertools import count

from loki.transform.transformation import Transformation
from loki.sourcefile import SourceFile
from loki.backend import cgen, fgen
from loki.ir import (Section, Import, Intrinsic, Interface, CallStatement, Declaration,
                     TypeDef, Statement, Scope, Loop)
from loki.subroutine import Subroutine
from loki.module import Module
from loki.expression import (Variable, FindVariables, InlineCall, RangeIndex, Scalar,
                             IntLiteral, Literal, Array, SubstituteExpressions, FindInlineCalls,
                             SubstituteExpressionsMapper, LoopRange, ArraySubscript,
                             retrieve_expressions)
from loki.visitors import Transformer, FindNodes
from loki.tools import as_tuple, flatten
from loki.types import DataType, SymbolType, DerivedType, TypeTable


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
        symbols = TypeTable()
        if isinstance(derived, TypeDef):
            variables = derived.variables
        else:
            variables = derived.dtype.typedef.variables
        declarations = []
        for v in variables:
            ctype = v.type.clone(kind=cls.iso_c_intrinsic_kind(v.type))
            vnew = v.clone(name=v.basename.lower(), scope=symbols, type=ctype)
            declarations += (Declaration(variables=(vnew,)),)
        return TypeDef(name=typename.lower(), bind_c=True, body=declarations, symbols=symbols)

    @staticmethod
    def iso_c_intrinsic_kind(_type):
        if _type.dtype == DataType.INTEGER:
            return 'c_int'
        if _type.dtype == DataType.REAL:
            kind = str(_type.kind)
            if kind.lower() in ('real32', 'c_float'):
                return 'c_float'
            if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
                return 'c_double'
        return None

    @staticmethod
    def c_intrinsic_kind(_type):
        if _type.dtype == DataType.LOGICAL:
            return 'int'
        if _type.dtype == DataType.INTEGER:
            return 'int'
        if _type.dtype == DataType.REAL:
            kind = str(_type.kind)
            if kind.lower() in ('real32', 'c_float'):
                return 'float'
            if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
                return 'double'
        return None

    @classmethod
    def generate_iso_c_wrapper_routine(cls, routine, c_structs, bind_name=None):
        # Create initial object to have a scope
        wrapper = Subroutine(name='%s_fc' % routine.name)

        if bind_name is None:
            bind_name = '%s_c' % routine.name
        interface = cls.generate_iso_c_interface(routine, bind_name, c_structs, wrapper)

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
                cvar = Variable(name='%s_c' % arg.name, type=ctype, scope=wrapper.symbols)
                cast_in = InlineCall('transfer', parameters=(arg,), kw_parameters={'mold': cvar})
                casts_in += [Statement(target=cvar, expr=cast_in)]

                cast_out = InlineCall('transfer', parameters=(cvar,), kw_parameters={'mold': arg})
                casts_out += [Statement(target=arg, expr=cast_out)]
                local_arg_map[arg.name] = cvar

        arguments = [local_arg_map[a] if a in local_arg_map else a for a in routine.argnames]
        wrapper_body = casts_in
        wrapper_body += [CallStatement(name=interface.body[0].name, arguments=arguments)]
        wrapper_body += casts_out
        wrapper_body = Section(body=wrapper_body)
        wrapper.__init__(name='%s_fc' % routine.name, spec=wrapper_spec, body=wrapper_body,
                         symbols=wrapper.symbols, types=wrapper.types)

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

        obj = Module(name='%s_fc' % module.name)

        # Create getter methods for module-level variables (I know... :( )
        wrappers = []
        for decl in FindNodes(Declaration).visit(module.spec):
            for v in decl.variables:
                if isinstance(v.type.dtype, DerivedType) or v.type.pointer or v.type.allocatable:
                    continue
                gettername = '%s__get__%s' % (module.name.lower(), v.name.lower())
                getter = Subroutine(name=gettername, parent=obj)

                getterspec = Section(body=[Import(module=module.name, symbols=[v.name])])
                isoctype = SymbolType(v.type.dtype, kind=cls.iso_c_intrinsic_kind(v.type))
                if isoctype.kind in ['c_int', 'c_float', 'c_double']:
                    getterspec.append(Import(module='iso_c_binding', symbols=[isoctype.kind]))
                getterbody = Section(body=[
                    Statement(target=Variable(name=gettername, scope=getter.symbols), expr=v)])

                getter.__init__(name=gettername, bind=gettername, spec=getterspec,
                                body=getterbody, is_function=True, parent=obj,
                                symbols=getter.symbols, types=getter.types)
                getter.variables = as_tuple(Variable(name=gettername, type=isoctype,
                                                     scope=getter.symbols))
                wrappers += [getter]

        obj.__init__(name=obj.name, spec=spec, routines=wrappers, types=obj.types,
                     symbols=obj.symbols)
        return obj

    @classmethod
    def generate_iso_c_interface(cls, routine, bind_name, c_structs, parent):
        """
        Generate the ISO-C subroutine interface
        """
        intf_name = '%s_iso_c' % routine.name
        isoc_import = Import(module='iso_c_binding',
                             symbols=('c_int', 'c_double', 'c_float'))
        intf_spec = Section(body=as_tuple(isoc_import))
        intf_spec.body += as_tuple(Intrinsic(text='implicit none'))
        intf_spec.body += as_tuple(c_structs.values())
        intf_routine = Subroutine(name=intf_name, spec=intf_spec, args=(), parent=parent,
                                  body=None, bind=bind_name)

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
                           scope=intf_routine.symbols)
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
                    if str(s).lower() in mod_vars:
                        var = mod_vars[str(s).lower()]

                        decl = Declaration(variables=(var,))
                        getter = '%s__get__%s' % (module.name.lower(), var.name.lower())
                        vget = Statement(target=var, expr=InlineCall(function=getter))
                        getter_calls += [decl, vget]

        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = Section(body=imports)
        body = Transformer({}).visit(routine.body)
        body = Section(body=as_tuple(getter_calls) + as_tuple(body))

        # Force all variables to lower-caps, as C/C++ is case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(body)
                if isinstance(v, (Scalar, Array)) and not v.name.islower()}
        body = SubstituteExpressions(vmap).visit(body)

        kernel = Subroutine(name='%s_c' % routine.name, spec=spec, body=body)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        # Force pointer on reference-passed arguments
        arg_map = {}
        for arg in kernel.arguments:
            if not(arg.type.intent.lower() == 'in' and isinstance(arg, Scalar)):
                dtype = arg.type
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
        assoc_map = {}
        vmap = {}
        for assoc in FindNodes(Scope).visit(kernel.body):
            invert_assoc = {v.name: k for k, v in assoc.associations.items()}
            for v in FindVariables(unique=False).visit(kernel.body):
                if v.name in invert_assoc:
                    vmap[v] = invert_assoc[v.name]
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

    @staticmethod
    def _resolve_vector_notation(kernel, **kwargs):
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

                ivar_basename = 'i_%s' % stmt.target.basename
                for i, dim, s in zip(count(), v.dimensions.index_tuple, as_tuple(v.shape)):
                    if isinstance(dim, RangeIndex):
                        # Create new index variable
                        vtype = SymbolType(DataType.INTEGER)
                        ivar = Variable(name='%s_%s' % (ivar_basename, i), type=vtype,
                                        scope=kernel.symbols)
                        shape_index_map[s] = ivar
                        index_range_map[ivar] = s

                        if ivar not in vdims:
                            vdims.append(ivar)

                # Add index variable to range replacement
                new_dims = as_tuple(shape_index_map.get(s, d)
                                    for d, s in zip(v.dimensions.index_tuple, as_tuple(v.shape)))
                vmap[v] = v.clone(dimensions=ArraySubscript(new_dims))

            index_vars.update(list(vdims))

            # Recursively build new loop nest over all implicit dims
            if len(vdims) > 0:
                loop = None
                body = stmt
                for ivar in vdims:
                    irange = index_range_map[ivar]
                    if isinstance(irange, RangeIndex):
                        bounds = LoopRange(irange.children)
                    else:
                        bounds = LoopRange((Literal(1), irange, Literal(1)))
                    loop = Loop(variable=ivar, body=as_tuple(body), bounds=bounds)
                    body = loop

                loop_map[stmt] = loop

        if len(loop_map) > 0:
            kernel.body = Transformer(loop_map).visit(kernel.body)
        kernel.variables += tuple(set(index_vars))

        # Apply variable substitution
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

    @staticmethod
    def _resolve_omni_size_indexing(kernel, **kwargs):
        """
        Replace the ``(1:size)`` indexing in array sizes that OMNI introduces
        """
        def is_omni_index(dim):
            return (isinstance(dim, RangeIndex) and dim.lower in (1, IntLiteral(1)) and
                    dim.step is None)
        vmap = {}
        for v in kernel.variables:
            if isinstance(v, Array):
                new_dims = [d.upper if is_omni_index(d) else d for d in v.dimensions.index_tuple]
                new_shape = [d.upper if is_omni_index(d) else d for d in v.shape]
                new_type = v.type.clone(shape=as_tuple(new_shape))
                vmap[v] = v.clone(dimensions=ArraySubscript(as_tuple(new_dims)), type=new_type)
        kernel.variables = [vmap.get(v, v) for v in kernel.variables]

    @staticmethod
    def _invert_array_indices(kernel, **kwargs):
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
                rdim = as_tuple(reversed(v.dimensions.index_tuple))
                vmap[v] = v.clone(dimensions=ArraySubscript(rdim))
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

        # Invert variable and argument dimensions for the automatic cast generation
        for v in kernel.variables:
            if isinstance(v, Array):
                rdim = as_tuple(reversed(v.dimensions.index_tuple))
                if v.shape:
                    rshape = as_tuple(reversed(v.shape))
                    vmap[v] = v.clone(dimensions=ArraySubscript(rdim),
                                      type=v.type.clone(shape=rshape))
                else:
                    vmap[v] = v.clone(dimensions=ArraySubscript(rdim))
        # kernel.arguments = [vmap.get(v, v) for v in kernel.arguments]
        kernel.variables = [vmap.get(v, v) for v in kernel.variables]

    @staticmethod
    def _shift_to_zero_indexing(kernel, **kwargs):
        """
        Shift each array indices to adjust to C indexing conventions
        """
        vmap = {}
        for v in FindVariables(unique=False).visit(kernel.body):
            if isinstance(v, Array):
                new_dims = as_tuple(d - 1 for d in v.dimensions.index_tuple)
                vmap[v] = v.clone(dimensions=ArraySubscript(new_dims))
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

    @staticmethod
    def _replace_intrinsics(kernel, **kwargs):
        """
        Replace known numerical intrinsic functions.
        """
        _intrinsic_map = {
            'epsilon': 'DBL_EPSILON',
            'min': 'fmin', 'max': 'fmax',
            'abs': 'fabs', 'sign': 'copysign',
        }

        callmap = {}
        for c in FindInlineCalls(unique=False).visit(kernel.body):
            cname = c.name.lower()
            if cname in _intrinsic_map:
                if cname == 'epsilon':
                    callmap[c] = Variable(name=_intrinsic_map[cname], scope=kernel.symbols)
                else:
                    callmap[c] = InlineCall(_intrinsic_map[cname], parameters=c.parameters,
                                            kw_parameters=c.kw_parameters)

        # Capture nesting by applying map to itself before applying to the kernel
        for _ in range(2):
            mapper = SubstituteExpressionsMapper(callmap)
            callmap = {k: mapper(v) for k, v in callmap.items()}

        kernel.body = SubstituteExpressions(callmap).visit(kernel.body)
