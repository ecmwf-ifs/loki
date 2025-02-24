# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

from loki.backend import cgen, cudagen, cppgen
from loki.batch import Transformation
from loki.expression import (
    symbols as sym, Variable, InlineCall, Scalar, Array,
    ProcedureSymbol, Dereference, Reference, ExpressionRetriever,
    SubstituteExpressionsMapper
)
from loki.ir import (
    Import, Intrinsic, Interface, CallStatement, Assignment,
    Transformer, FindNodes, Comment, SubstituteExpressions,
    FindInlineCalls
)
from loki.logging import debug
from loki.sourcefile import Sourcefile
from loki.tools import as_tuple, flatten
from loki.types import BasicType, DerivedType

from loki.transformations.array_indexing import (
    shift_to_zero_indexing, invert_array_indices,
    resolve_vector_notation, normalize_array_shape_and_access,
    flatten_arrays
)
from loki.transformations.inline import (
    inline_constant_parameters, inline_elemental_functions
)
from loki.transformations.sanitise import do_resolve_associates
from loki.transformations.utilities import (
    convert_to_lower_case, replace_intrinsics, sanitise_imports
)


__all__ = ['FortranCTransformation']


class DeReferenceTrafo(Transformer):
    """
    Transformation to apply/insert Dereference = `*` and
    Reference/*address-of* = `&` operators.

    Parameters
    ----------
    vars2dereference : list
        Variables to be dereferenced. Ususally the arguments except
        for scalars with `intent=in`.
    """
    # pylint: disable=unused-argument

    def __init__(self, vars2dereference):
        super().__init__()
        self.retriever = ExpressionRetriever(self.is_dereference)
        self.vars2dereference = vars2dereference

    @staticmethod
    def is_dereference(symbol):
        return isinstance(symbol, (DerivedType, Array, Scalar)) and not (
            isinstance(symbol, Array) and symbol.dimensions is not None
            and not all(dim == sym.RangeIndex((None, None)) for dim in symbol.dimensions)
        )

    def visit_Expression(self, o, **kwargs):
        symbol_map = {
            symbol: Dereference(symbol.clone()) for symbol in self.retriever.retrieve(o)
            if symbol.name.lower() in self.vars2dereference
        }
        return SubstituteExpressionsMapper(symbol_map)(o)

    def visit_CallStatement(self, o, **kwargs):
        new_args = ()
        if o.routine is BasicType.DEFERRED:
            debug(f'DeReferenceTrafo: Skipping call to {o.name!s} due to missing procedure enrichment')
            return o
        call_arg_map = dict((v,k) for k,v in o.arg_map.items())
        for arg in o.arguments:
            if not self.is_dereference(arg) and (isinstance(call_arg_map[arg], Array)\
                    or call_arg_map[arg].type.intent.lower() != 'in'):
                new_args += (Reference(arg.clone()),)
            else:
                if isinstance(arg, Scalar) and call_arg_map[arg].type.intent.lower() != 'in':
                    new_args += (Reference(arg.clone()),)
                else:
                    new_args += (arg,)
        o._update(arguments=new_args)
        return o


class FortranCTransformation(Transformation):
    """
    Fortran-to-C transformation that translates the given routine into C.

    Parameters
    ----------
    inline_elementals : bool, optional
        Inline known elemental function via expression substitution. Default is ``True``.
    language : str
        C-style language to generate; should be one of ``['c', 'cpp', 'cuda']``.
    """
    # pylint: disable=unused-argument

    # Set of standard module names that have no C equivalent
    __fortran_intrinsic_modules = ['ISO_FORTRAN_ENV', 'ISO_C_BINDING']

    def __init__(self, inline_elementals=True, language='c'):
        self.inline_elementals = inline_elementals
        self.language = language.lower()
        self._supported_languages = ['c', 'cpp', 'cuda']

        if self.language == 'c':
            self.codegen = cgen
        elif self.language == 'cpp':
            self.codegen = cppgen
        elif self.language == 'cuda':
            self.codegen = cudagen
        else:
            raise ValueError(f'language "{self.language}" is not supported!'
                             f' (supported languages: "{self._supported_languages}")')

    def file_suffix(self):
        if self.language == 'cpp':
            return '.cpp'
        return '.c'

    def transform_subroutine(self, routine, **kwargs):
        if 'path' in kwargs:
            path = kwargs.get('path')
        else:
            build_args = kwargs.get('build_args')
            path = Path(build_args.get('output_dir'))

        role = kwargs.get('role', 'kernel')
        item = kwargs.get('item', None)
        depths = kwargs.get('depths', None)
        targets = kwargs.get('targets', None)
        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = as_tuple(sub_sgraph.successors(item)) if sub_sgraph is not None else ()

        depth = 0
        if depths is None:
            if role == 'driver':
                depth = 0
            elif role == 'kernel':
                depth = 1
        else:
            depth = depths[item]

        if role == 'driver':
            self.interface_to_import(routine, targets)
            return

        # for calls and inline calls: convert kwarguments to arguments
        self.convert_kwargs_to_args(routine, targets)

        if role == 'kernel':
            # Generate C source file from Loki IR
            c_kernel = self.generate_c_kernel(routine, targets=targets)

            for successor in successors:
                c_kernel.spec.prepend(Import(module=f'{successor.ir.name.lower()}_c.h', c_import=True))

            if depth == 1:
                if self.language != 'c':
                    c_kernel_launch = c_kernel.clone(name=f"{c_kernel.name}_launch", prefix="extern_c")
                    self.generate_c_kernel_launch(c_kernel_launch, c_kernel)
                    c_path = (path/c_kernel_launch.name.lower()).with_suffix('.h')
                    Sourcefile.to_file(source=self.codegen(c_kernel_launch, extern=True), path=c_path)

            assignments = FindNodes(Assignment).visit(c_kernel.body)
            assignments2remove = ['griddim', 'blockdim']
            assignment_map = {assignment: None for assignment in assignments
                    if assignment.lhs.name.lower() in assignments2remove}
            c_kernel.body = Transformer(assignment_map).visit(c_kernel.body)

            if depth > 1:
                c_kernel.spec.prepend(Import(module=f'{c_kernel.name.lower()}.h', c_import=True))
            c_path = (path/c_kernel.name.lower()).with_suffix(self.file_suffix())
            Sourcefile.to_file(source=self.codegen(c_kernel, extern=self.language=='cpp'), path=c_path)
            header_path = (path/c_kernel.name.lower()).with_suffix('.h')
            Sourcefile.to_file(source=self.codegen(c_kernel, header=True), path=header_path)

    def convert_kwargs_to_args(self, routine, targets):
        # calls (to subroutines)
        for call in as_tuple(FindNodes(CallStatement).visit(routine.body)):
            if str(call.name).lower() in as_tuple(targets):
                call.convert_kwargs_to_args()
        # inline calls (to functions)
        inline_call_map = {}
        for inline_call in as_tuple(FindInlineCalls().visit(routine.body)):
            if str(inline_call.name).lower() in as_tuple(targets) and inline_call.routine is not BasicType.DEFERRED:
                inline_call_map[inline_call] = inline_call.clone_with_kwargs_as_args()
        if inline_call_map:
            routine.body = SubstituteExpressions(inline_call_map).visit(routine.body)

    def interface_to_import(self, routine, targets):
        """
        Convert interface to import.
        """
        for call in FindNodes(CallStatement).visit(routine.body):
            if str(call.name).lower() in as_tuple(targets):
                call.convert_kwargs_to_args()
        intfs = FindNodes(Interface).visit(routine.spec)
        removal_map = {}
        for i in intfs:
            for s in i.symbols:
                if targets and s in targets:
                    # Create a new module import with explicitly qualified symbol
                    new_symbol = s.clone(name=f'{s.name}_FC', scope=routine)
                    modname = f'{new_symbol.name}_MOD'
                    new_import = Import(module=modname, c_import=False, symbols=(new_symbol,))
                    routine.spec.prepend(new_import)
                    # Mark current import for removal
                    removal_map[i] = None
        # Apply any scheduled interface removals to spec
        if removal_map:
            routine.spec = Transformer(removal_map).visit(routine.spec)

    @staticmethod
    def apply_de_reference(routine):
        """
        Utility method to apply/insert Dereference = `*` and
        Reference/*address-of* = `&` operators.
        """
        to_be_dereferenced = []
        for arg in routine.arguments:
            if not(arg.type.intent.lower() == 'in' and isinstance(arg, Scalar)) or arg.type.optional:
                to_be_dereferenced.append(arg.name.lower())

        routine.body = DeReferenceTrafo(to_be_dereferenced).visit(routine.body)

    def generate_c_kernel(self, routine, targets, **kwargs):
        """
        Re-generate the C kernel and insert wrapper-specific peculiarities,
        such as the explicit getter calls for imported module-level variables.
        """

        # CAUTION! Work with a copy of the original routine to not break the
        #  dependency graph of the Scheduler through the rename
        kernel = routine.clone()
        kernel.name = f'{kernel.name.lower()}_c'

        # Clean up Fortran vector notation
        resolve_vector_notation(kernel)
        normalize_array_shape_and_access(kernel)

        # Convert array indexing to C conventions
        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        invert_array_indices(kernel)
        shift_to_zero_indexing(kernel, ignore=() if self.language == 'c' else ('jl', 'ibl'))
        flatten_arrays(kernel, order='C', start_index=0)

        # Inline all known parameters, since they can be used in declarations,
        # and thus need to be known before we can fetch them via getters.
        inline_constant_parameters(kernel, external_only=True)

        if self.inline_elementals:
            # Inline known elemental function via expression substitution
            inline_elemental_functions(kernel)

        # Create declarations for module variables
        if self.language == 'c':
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
        do_resolve_associates(kernel)

        # Force all variables to lower-caps, as C/C++ is case-sensitive
        convert_to_lower_case(kernel)

        # Force pointer on reference-passed arguments (and lower case type names for derived types)
        for arg in kernel.arguments:

            if not(arg.type.intent.lower() == 'in' and isinstance(arg, Scalar)):
                _type = arg.type.clone(pointer=True)
                if isinstance(arg.type.dtype, DerivedType):
                    # Lower case type names for derived types
                    typedef = _type.dtype.typedef.clone(name=_type.dtype.typedef.name.lower())
                    _type = _type.clone(dtype=typedef.dtype)
                kernel.symbol_attrs[arg.name] = _type

        # apply dereference and reference where necessary
        self.apply_de_reference(kernel)

        # adapt call and inline call names -> '<call name>_c'
        self.convert_call_names(kernel, targets)

        symbol_map = {'epsilon': 'DBL_EPSILON'}
        function_map = {'min': 'fmin', 'max': 'fmax', 'abs': 'fabs',
                        'exp': 'exp', 'sqrt': 'sqrt', 'sign': 'copysign'}
        replace_intrinsics(kernel, symbol_map=symbol_map, function_map=function_map)

        # Remove redundant imports
        sanitise_imports(kernel)

        return kernel

    def convert_call_names(self, routine, targets):
        # calls (to subroutines)
        calls = FindNodes(CallStatement).visit(routine.body)
        for call in calls:
            if call.name not in as_tuple(targets):
                continue
            call._update(name=Variable(name=f'{call.name}_c'.lower()))
        # inline calls (to functions)
        callmap = {}
        for call in FindInlineCalls(unique=False).visit(routine.body):
            if call.routine is not BasicType.DEFERRED and (targets is None or call.name in as_tuple(targets)):
                callmap[call.function] = call.function.clone(name=f'{call.name}_c')
        routine.body = SubstituteExpressions(callmap).visit(routine.body)

    def generate_c_kernel_launch(self, kernel_launch, kernel, **kwargs):
        import_map = {}
        for im in FindNodes(Import).visit(kernel_launch.spec):
            import_map[im] = None
        kernel_launch.spec = Transformer(import_map).visit(kernel_launch.spec)

        kernel_call = kernel.clone()
        call_arguments = []
        for arg in kernel_call.arguments:
            call_arguments.append(arg)

        griddim = None
        blockdim = None
        if 'griddim' in kernel_launch.variable_map:
            griddim = kernel_launch.variable_map['griddim']
        if 'blockdim' in kernel_launch.variable_map:
            blockdim = kernel_launch.variable_map['blockdim']
        assignments = FindNodes(Assignment).visit(kernel_launch.body)
        griddim_assignment = None
        blockdim_assignment = None
        for assignment in assignments:
            if assignment.lhs == griddim:
                griddim_assignment = assignment.clone()
            if assignment.lhs == blockdim:
                blockdim_assignment = assignment.clone()
        kernel_launch.body = (Comment(text="! here should be the launcher ...."),
                griddim_assignment, blockdim_assignment, CallStatement(name=Variable(name=kernel.name),
                    arguments=call_arguments, chevron=(sym.Variable(name="griddim"),
                        sym.Variable(name="blockdim"))))
