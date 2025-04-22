# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from collections import OrderedDict

from loki.backend import cgen, fgen, cudagen, cppgen
from loki.batch import Transformation, ProcedureItem, ModuleItem
from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, SubstituteExpressions, Transformer
)
from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine
from loki.types import BasicType, DerivedType, SymbolAttributes
from loki.tools import as_tuple

from loki.transformations.utilities import sanitise_imports


__all__ = [
    'c_intrinsic_kind', 'iso_c_intrinsic_import',
    'iso_c_intrinsic_kind', 'c_struct_typedef',
    'generate_iso_c_interface', 'generate_iso_c_wrapper_routine',
    'generate_iso_c_wrapper_module', 'generate_c_header',
    'FortranISOCWrapperTransformation'
]


class FortranISOCWrapperTransformation(Transformation):
    """
    Wrapper transformation that generates ISO-C Fortran wrappers and C
    headers for translated kernels or additional header modules.

    In addition to :any:`Subroutine` objects with the role
    ``'kernel'``, this transformation will process :any:`Module`
    objects with the role ``'header'``. This will generate ISO-C
    wrappers for derived types and the corresponding C-compatible
    structs in C header files.

    Parameters
    ----------
    use_c_ptr : bool, optional
        Use ``c_ptr`` for array declarations and ``c_loc(...)`` to
        pass the corresponding argument. Default is ``False``.
    language : string
        Actual C-style language to generate; must be on of ``'c'``,
        ``'cpp'`` or ``'cuda'`` for C, C++ and CUDA respectively.
    """

    item_filter = (ProcedureItem, ModuleItem)

    _supported_languages = ['c', 'cpp', 'cuda']

    def __init__(self, use_c_ptr=False, language='c'):
        self.use_c_ptr = use_c_ptr
        self.language = language.lower()

        if self.language == 'c':
            self.codegen = cgen
        elif self.language == 'cpp':
            self.codegen = cppgen
        elif self.language == 'cuda':
            self.codegen = cudagen
        else:
            raise ValueError(f'language "{self.language}" is not supported!'
                             f' (supported languages: "{self._supported_languages}")')

    def transform_module(self, module, **kwargs):
        if 'path' in kwargs:
            path = kwargs.get('path')
        else:
            build_args = kwargs.get('build_args')
            path = Path(build_args.get('output_dir'))

        role = kwargs.get('role', 'kernel')

        if role == 'header':
            # Generate Fortran wrapper module
            wrapper = generate_iso_c_wrapper_module(
                module, use_c_ptr=self.use_c_ptr, language=self.language
            )
            wrapperpath = (path/wrapper.name.lower()).with_suffix('.F90')
            Sourcefile.to_file(source=fgen(wrapper), path=wrapperpath)

            # Generate C header file from module
            c_header = generate_c_header(module)
            c_path = (path/c_header.name.lower()).with_suffix('.h')
            Sourcefile.to_file(source=self.codegen(c_header), path=c_path)


    def transform_subroutine(self, routine, **kwargs):
        if 'path' in kwargs:
            path = kwargs.get('path')
        else:
            build_args = kwargs.get('build_args')
            path = Path(build_args.get('output_dir'))

        role = kwargs.get('role', 'kernel')

        if role == 'kernel': #  and routine.name.lower() == 'implsch':
            c_structs = {}
            for arg in routine.arguments:
                if isinstance(arg.type.dtype, DerivedType):
                    c_structs[arg.type.dtype.name.lower()] = c_struct_typedef(arg.type, use_c_ptr=self.use_c_ptr)

            # Generate Fortran wrapper module
            bind_name = None if self.language in ['c', 'cpp'] else f'{routine.name.lower()}_c_launch'
            wrapper = generate_iso_c_wrapper_routine(
                routine, c_structs, bind_name=bind_name,
                use_c_ptr=self.use_c_ptr, language=self.language
            )
            contains = ir.Section(body=(ir.Intrinsic('CONTAINS'), wrapper))
            wrapperpath = (path/wrapper.name.lower()).with_suffix('.F90')
            module = Module(name=f'{wrapper.name.upper()}_MOD', contains=contains)
            module.spec = ir.Section(body=(ir.Import(module='iso_c_binding'),))
            Sourcefile.to_file(source=fgen(module), path=wrapperpath)


def c_intrinsic_kind(_type, scope):
    """
    Determine the intrinsic C-type for a given symbol table entry.

    Parameters
    ----------
    _type : :any:`SymbolAttr`
        The symbols type attribute to determine type and kind
    scope : :any:`Scope`
        The containing scope in which to clone the type symbol
    """
    if _type.dtype == BasicType.LOGICAL:
        return sym.Variable(name='int', scope=scope)
    if _type.dtype == BasicType.INTEGER:
        return sym.Variable(name='int', scope=scope)
    if _type.dtype == BasicType.REAL:
        kind = str(_type.kind)
        if kind.lower() in ('real32', 'c_float'):
            return sym.Variable(name='float', scope=scope)
        if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
            return sym.Variable(name='double', scope=scope)
    return None


def iso_c_intrinsic_import(scope, use_c_ptr=False):
    """
    Create :any:`Import` object for the intrinsic C base types.

    Parameters
    ----------
    scope : :any:`Scope`
        The scope in which to create the import node and type symbols.
    use_c_ptr : bool, optional
        Use ``c_ptr`` for array declarations and ``c_loc(...)`` to
        pass the corresponding argument. Default is ``False``.
    """
    import_symbols = ['c_int', 'c_double', 'c_float']
    if use_c_ptr:
        import_symbols += ['c_ptr', 'c_loc']
    symbols = as_tuple(sym.Variable(name=name, scope=scope) for name in import_symbols)
    isoc_import = ir.Import(module='iso_c_binding', symbols=symbols)
    return isoc_import


def iso_c_intrinsic_kind(_type, scope, is_array=False, use_c_ptr=False):
    """
    Determine the intrinsic ISO-C type for a given symbol table entry.

    Parameters
    ----------
    _type : :any:`SymbolAttr`
        The symbols type attribute to determine type and kind
    is_array : bool
        Flag indicating if the passed type belongs to an array symbol.
    use_c_ptr : bool, optional
        Use ``c_ptr`` for array declarations and ``c_loc(...)`` to
        pass the corresponding argument. Default is ``False``.
    """
    if _type.dtype == BasicType.INTEGER:
        return sym.Variable(name='c_int', scope=scope)

    if _type.dtype == BasicType.REAL:
        kind = str(_type.kind)
        if kind.lower() in ('real32', 'c_float'):
            return sym.Variable(name='c_float', scope=scope)
        if kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double', 'c_ptr'):
            if use_c_ptr and is_array:
                return sym.Variable(name='c_ptr', scope=scope)
            return sym.Variable(name='c_double', scope=scope)

    return None


def c_struct_typedef(derived, use_c_ptr=False):
    """
    Create the :class:`TypeDef` for the C-wrapped struct definition.

    Parameters
    ----------
    use_c_ptr : bool, optional
        Use ``c_ptr`` for array declarations and ``c_loc(...)`` to
        pass the corresponding argument. Default is ``False``.
    """
    typename = f'{derived.name if isinstance(derived, ir.TypeDef) else derived.dtype.name}_c'
    typedef = ir.TypeDef(name=typename.lower(), body=(), bind_c=True)  # pylint: disable=unexpected-keyword-arg
    if isinstance(derived, ir.TypeDef):
        variables = derived.variables
    else:
        variables = derived.dtype.typedef.variables
    declarations = []
    for v in variables:
        ctype = v.type.clone(kind=iso_c_intrinsic_kind(v.type, typedef, use_c_ptr=use_c_ptr))
        vnew = v.clone(name=v.basename.lower(), scope=typedef, type=ctype)
        declarations += (ir.VariableDeclaration(symbols=(vnew,)),)
    typedef._update(body=as_tuple(declarations))
    return typedef


def generate_iso_c_interface(routine, bind_name, c_structs, scope, use_c_ptr=False, language='c'):
    """
    Generate the ISO-C subroutine :any:`Interface` object for a given :any:`Subroutine`.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine for which to generate the interface
    bind_name : str
        Name of the C-function to which this interface corresponds.
    c_structs : dict of str to str
        Map from Fortran derived type name to  C-struct type name
    scope : :any:`Scope`
        Parent scope in which to create the :any:`Interface`
    use_c_ptr : bool, optional
        Use ``c_ptr`` for array declarations and ``c_loc(...)`` to
        pass the corresponding argument. Default is ``False``.
    language : string
        C-style language to generate; if this is ``'c'``, we resolve
        non-C imports.
    """
    intf_name = f'{routine.name}_iso_c'
    intf_routine = Subroutine(name=intf_name, body=None, args=(), parent=scope, bind=bind_name)
    intf_spec = ir.Section(
        body=as_tuple(iso_c_intrinsic_import(intf_routine, use_c_ptr=use_c_ptr))
    )
    if language == 'c':
        for im in FindNodes(ir.Import).visit(routine.spec):
            if not im.c_import:
                im_symbols = tuple(s.clone(scope=intf_routine) for s in im.symbols)
                intf_spec.append(im.clone(symbols=im_symbols))
    intf_spec.append(ir.Intrinsic(text='implicit none'))
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
            value = isinstance(arg, sym.Scalar) and arg.type.intent.lower() == 'in' and not arg.type.optional
            kind = iso_c_intrinsic_kind(arg.type, intf_routine, is_array=isinstance(arg, sym.Array))
            if use_c_ptr:
                if isinstance(arg, sym.Array):
                    ctype = SymbolAttributes(DerivedType(name="c_ptr"), value=True, kind=None)
                else:
                    ctype = SymbolAttributes(arg.type.dtype, value=value, kind=kind)
            else:
                ctype = SymbolAttributes(arg.type.dtype, value=value, kind=kind)
        if use_c_ptr:
            dimensions = None
        else:
            dimensions = arg.dimensions if isinstance(arg, sym.Array) else None
        var = sym.Variable(name=arg.name, dimensions=dimensions, type=ctype, scope=intf_routine)
        intf_routine.variables += (var,)
        intf_routine.arguments += (var,)

    sanitise_imports(intf_routine)

    return ir.Interface(body=(intf_routine, ))


def generate_iso_c_wrapper_routine(routine, c_structs, bind_name=None, use_c_ptr=False, language='c'):
    """
    Generate Fortran ISO-C wrapper :any:`Subroutine` that corresponds
    to a transpiled C method.

    The new wrapper subroutine will have the suffix ``'_fc'`` appended
    to the name original subroutine name and bind to a C function with
    the suffix ``'_c'``.

    This method will call :meth:`generate_iso_c_interface` to generate
    the ISO-C compatible interface for the C function and generate a
    wrapper :any:`Subroutine` that converts the native Fortran arguments
    to a call to the C function with ISO-C compatible arguments.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine for which to generate the interface
    c_structs : dict of str to str
        Map from Fortran derived type name to  C-struct type name
    bind_name : str
        Name of the C-function to which this interface corresponds.
    use_c_ptr : bool, optional
        Use ``c_ptr`` for array declarations and ``c_loc(...)`` to
        pass the corresponding argument. Default is ``False``.
    language : string
        C-style language to generate; if this is ``'c'``, we resolve
        non-C imports.
    """
    wrapper = Subroutine(name=f'{routine.name}_fc')

    if bind_name is None:
        bind_name = f'{routine.name.lower()}_c'
    interface = generate_iso_c_interface(
        routine, bind_name, c_structs, scope=wrapper, use_c_ptr=use_c_ptr, language=language
    )

    # Generate the wrapper function
    wrapper_spec = Transformer().visit(routine.spec)
    wrapper_spec.prepend(iso_c_intrinsic_import(wrapper, use_c_ptr=use_c_ptr))
    wrapper_spec.append(struct.clone(parent=wrapper) for struct in c_structs.values())
    wrapper_spec.append(interface)
    wrapper.spec = wrapper_spec

    pragmas = FindNodes(ir.Pragma).visit(wrapper.ir)
    pragma_map = {pragma: None for pragma in pragmas if 'routine' in pragma.content.lower()}
    wrapper.spec = Transformer(pragma_map).visit(wrapper.spec)

    # Create the wrapper function with casts and interface invocation
    local_arg_map = OrderedDict()
    casts_in = []
    casts_out = []
    for arg in routine.arguments:
        if isinstance(arg.type.dtype, DerivedType):
            ctype = SymbolAttributes(DerivedType(name=c_structs[arg.type.dtype.name.lower()].name))
            cvar = sym.Variable(name=f'{arg.name}_c', type=ctype, scope=wrapper)
            cast_in = sym.InlineCall(sym.ProcedureSymbol('transfer', scope=wrapper),
                                     parameters=(arg,), kw_parameters={'mold': cvar})
            casts_in += [ir.Assignment(lhs=cvar, rhs=cast_in)]

            cast_out = sym.InlineCall(sym.ProcedureSymbol('transfer', scope=wrapper),
                                      parameters=(cvar,), kw_parameters={'mold': arg})
            casts_out += [ir.Assignment(lhs=arg, rhs=cast_out)]
            local_arg_map[arg.name] = cvar

    arguments = tuple(local_arg_map[a] if a in local_arg_map else sym.Variable(name=a)
                      for a in routine.argnames)
    use_device_addr = []
    if use_c_ptr:
        arg_map = {}
        for arg in routine.arguments:
            if isinstance(arg, sym.Array):
                new_dims = tuple(sym.RangeIndex((None, None)) for _ in arg.dimensions)
                arg_map[arg] = arg.clone(dimensions=new_dims, type=arg.type.clone(target=True))
        routine.spec = SubstituteExpressions(arg_map).visit(routine.spec)

        call_arguments = []
        for arg in routine.arguments:
            if isinstance(arg, sym.Array):
                new_arg = arg.clone(dimensions=None)
                c_loc = sym.InlineCall(
                    function=sym.ProcedureSymbol(name="c_loc", scope=routine),
                    parameters=(new_arg,))
                call_arguments.append(c_loc)
                use_device_addr.append(arg.name)
            elif isinstance(arg.type.dtype, DerivedType):
                cvar = sym.Variable(name=f'{arg.name}_c', type=ctype, scope=wrapper)
                call_arguments.append(cvar)
            else:
                call_arguments.append(arg)
    else:
        call_arguments = arguments

    wrapper_body = casts_in
    if language in ['cuda', 'hip']:
        wrapper_body += [
            ir.Pragma(keyword='acc', content=f'host_data use_device({", ".join(use_device_addr)})')
        ]
    wrapper_body += [
        ir.CallStatement(name=sym.Variable(name=interface.body[0].name), arguments=call_arguments)
    ]
    if language in ['cuda', 'hip']:
        wrapper_body += [ir.Pragma(keyword='acc', content='end host_data')]
    wrapper_body += casts_out
    wrapper.body = ir.Section(body=as_tuple(wrapper_body))

    # Copy internal argument and declaration definitions
    wrapper.variables = tuple(arg.clone(scope=wrapper) for arg in routine.arguments) + tuple(local_arg_map.values())
    wrapper.arguments = tuple(arg.clone(scope=wrapper) for arg in routine.arguments)

    # Remove any unused imports
    sanitise_imports(wrapper)
    return wrapper


def generate_iso_c_wrapper_module(module, use_c_ptr=False, language='c'):
    """
    Generate the ISO-C wrapper module for a raw Fortran module.

    The new wrapper module will have the suffix ``'_fc'`` appended to
    the name and contain ISO-C function interfaces for contained
    :any:`Subroutine` objects. This method will call
    :meth:`generate_iso_c_routine` to generate the ISO-C compatible
    procedure interfaces.

    Note
    ----
    If the module contains global variables, we generate templated
    getter functions here, as global Fortran variables are not
    accessible via ISO-C interfaces.

    Parameters
    ----------
    module : :any:`Module`
        The module for which to generate the interface module
    use_c_ptr : bool, optional
        Use ``c_ptr`` for array declarations and ``c_loc(...)`` to
        pass the corresponding argument. Default is ``False``.
    language : string
        C-style language to generate; if this is ``'c'``, we resolve
        non-C imports.
    """
    modname = f'{module.name}_fc'
    wrapper_module = Module(name=modname)

    # Create getter methods for module-level variables (I know... :( )
    if language == 'c':
        wrappers = []
        for decl in FindNodes(ir.VariableDeclaration).visit(module.spec):
            for v in decl.symbols:
                if isinstance(v.type.dtype, DerivedType) or v.type.pointer or v.type.allocatable:
                    continue
                gettername = f'{module.name.lower()}__get__{v.name.lower()}'
                getter = Subroutine(name=gettername, bind=gettername, is_function=True, parent=wrapper_module)

                getter.spec = ir.Section(
                    body=(ir.Import(module=module.name, symbols=(v.clone(scope=getter), )), )
                )
                isoctype = SymbolAttributes(
                    v.type.dtype, kind=iso_c_intrinsic_kind(v.type, getter, use_c_ptr=use_c_ptr)
                )
                if isoctype.kind in ['c_int', 'c_float', 'c_double']:
                    getter.spec.append(ir.Import(module='iso_c_binding', symbols=(isoctype.kind, )))
                getter.body = ir.Section(
                    body=(ir.Assignment(lhs=sym.Variable(name=gettername, scope=getter), rhs=v),)
                )
                getter.variables = as_tuple(sym.Variable(name=gettername, type=isoctype, scope=getter))
                wrappers += [getter]
        wrapper_module.contains = ir.Section(body=(ir.Intrinsic('CONTAINS'), *wrappers))

    # Remove any unused imports
    sanitise_imports(wrapper_module)
    return wrapper_module


def generate_c_header(module):
    """
    Re-generate the C header as a module with all pertinent nodes,
    but not Fortran-specific intrinsics (eg. implicit none or save).

    The new header module will have the suffix ``'_c'`` appended to
    the original module name.

    Parameters
    ----------
    module : :any:`Module`
        The module for which to generate the C header
    """
    header_module = Module(name=f'{module.name}_c')

    # Generate stubs for getter functions
    spec = []
    for decl in FindNodes(ir.VariableDeclaration).visit(module.spec):
        assert len(decl.symbols) == 1
        v = decl.symbols[0]
        # Bail if not a basic type
        if isinstance(v.type.dtype, DerivedType):
            continue
        ctype = c_intrinsic_kind(v.type, scope=module)
        tmpl_function = f'{ctype} {module.name.lower()}__get__{v.name.lower()}();'
        spec += [ir.Intrinsic(text=tmpl_function)]

    # Re-create type definitions with range indices (``:``) replaced by pointers
    for td in FindNodes(ir.TypeDef).visit(module.spec):
        header_td = ir.TypeDef(name=td.name.lower(), body=(), parent=header_module)  # pylint: disable=unexpected-keyword-arg
        declarations = []
        for decl in td.declarations:
            variables = []
            for v in decl.symbols:
                # Note that we force lower-case on all struct variables
                if isinstance(v, sym.Array):
                    new_shape = as_tuple(d for d in v.shape if not isinstance(d, sym.RangeIndex))
                    new_type = v.type.clone(shape=new_shape)
                    variables += [v.clone(name=v.name.lower(), type=new_type, scope=header_td)]
                else:
                    variables += [v.clone(name=v.name.lower(), scope=header_td)]
            declarations += [ir.VariableDeclaration(
                symbols=as_tuple(variables), dimensions=decl.dimensions,
                comment=decl.comment, pragma=decl.pragma
            )]
        header_td._update(body=as_tuple(declarations))
        spec += [header_td]

    header_module.spec = spec
    header_module.rescope_symbols()
    return header_module
