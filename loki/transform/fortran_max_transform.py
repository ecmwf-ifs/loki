# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import OrderedDict, defaultdict
from pathlib import Path
from hashlib import sha256

from loki.backend import maxjgen, fgen, cgen
from loki.batch import Transformation
from loki.expression import (
    FindVariables, SubstituteExpressions, ExpressionCallbackMapper,
    SubstituteExpressionsMapper, ExpressionRetriever, symbols as sym
)
from loki.ir import (
    nodes as ir, Transformer, FindNodes, is_loki_pragma, pragmas_attached
)
from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine
from loki.tools import as_tuple, flatten

from loki.transform.fortran_c_transform import FortranCTransformation
from loki.transform.transform_array_indexing import (
    shift_to_zero_indexing, invert_array_indices,
    resolve_vector_notation, normalize_range_indexing
)
from loki.transform.transform_utilities import replace_intrinsics
from loki.types import SymbolAttributes, BasicType, DerivedType


__all__ = ['FortranMaxTransformation']


class FortranMaxTransformation(Transformation):
    """
    Fortran-to-Maxeler transformation that translates the given routine
    into .maxj and generates a matching manager and host code with
    corresponding ISO-C wrappers.
    """

    def __init__(self):
        super().__init__()

        # Maps from original type name to ISO-C and C-struct types
        self.c_structs = OrderedDict()

    def transform_subroutine(self, routine, **kwargs):
        self.maxj_src = Path(kwargs.get('path')) / routine.name
        self.maxj_src.mkdir(exist_ok=True)

        f2c_transformation = FortranCTransformation(use_c_ptr=False)
        for arg in routine.arguments:
            if isinstance(arg.type.dtype, DerivedType):
                self.c_structs[arg.type.name.lower()] = f2c_transformation.c_struct_typedef(arg.type)

        # Create a copy of the kernel and apply some common transformations
        routine = routine.clone()
        routine = self._convert_variables_to_lowercase(routine)
        routine = self._remove_implicit_statements(routine)

        # Generate the DFE kernel
        maxj_kernel = self._generate_dfe_kernel(routine)
        self.maxj_kernel_path = (self.maxj_src / maxj_kernel.name).with_suffix('.maxj')
        maxj_module = self._generate_dfe_kernel_module(maxj_kernel)
        Sourcefile.to_file(source=maxjgen(maxj_module), path=self.maxj_kernel_path)

        # Generate the SLiC host interface
        host_interface = self._generate_slic_interface(routine, maxj_kernel)

        # Generate Fortran wrapper module
        wrapper = f2c_transformation.generate_iso_c_wrapper_routine(
            host_interface, self.c_structs, bind_name=host_interface.name)
        self.wrapperpath = (self.maxj_src / wrapper.name.lower()).with_suffix('.f90')
        contains = ir.Section(body=(ir.Intrinsic('CONTAINS'), wrapper))
        module = Module(name=f'{wrapper.name.upper()}_MOD', contains=contains)
        Sourcefile.to_file(source=fgen(module), path=self.wrapperpath)

        # Generate C host code
        host_interface.spec.prepend(ir.Import(f'{routine.name}.h', c_import=True))
        host_interface = self._convert_arguments_to_pointer(host_interface)
        self.c_path = (self.maxj_src / host_interface.name).with_suffix('.c')
        Sourcefile.to_file(source=cgen(host_interface), path=self.c_path)

        # Generate kernel manager
        maxj_manager_intf = self._generate_dfe_manager_intf(maxj_kernel)
        self.maxj_manager_intf_path = (self.maxj_src / maxj_manager_intf.name).with_suffix('.maxj')
        Sourcefile.to_file(source=maxjgen(maxj_manager_intf), path=self.maxj_manager_intf_path)

        maxj_manager = self._generate_dfe_manager(maxj_kernel)
        self.maxj_manager_path = (self.maxj_src / maxj_manager.name).with_suffix('.maxj')
        Sourcefile.to_file(source=maxjgen(maxj_manager), path=self.maxj_manager_path)

    @staticmethod
    def _convert_variables_to_lowercase(routine):
        """
        Convert all variables to lower-case, as Java and C are case-sensitive.
        """
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(routine.ir)
                if not v.name.islower()}
        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
        routine.body = SubstituteExpressions(vmap).visit(routine.body)
        return routine

    @staticmethod
    def _remove_implicit_statements(routine):
        """
        Remove all IMPLICIT statements.
        """
        stmt_map = {stmt: None for stmt in FindNodes(ir.Intrinsic).visit(routine.spec)
                    if stmt.text.lstrip().lower().startswith('implicit')}
        routine.spec = Transformer(stmt_map).visit(routine.spec)
        return routine

    @staticmethod
    def _split_and_order_arguments(routine):
        """
        Copy all arguments and split 'inout' arguments into a new 'in'-argument and
        the original 'inout' argument that has the 'in'-argument as initial value assigned.
        """
        declarations = FindNodes(ir.VariableDeclaration).visit(routine.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.symbols)

        arguments = []
        out_map = {}
        var_map = {}
        for arg in routine.arguments:
            if arg.type.intent.lower() == 'inout':
                # Create matching instream argument
                in_type = arg.type.clone(intent='in', dfevar=True)
                arg_in = arg.clone(name=f'{arg.name}_in', type=in_type)
                # Modify existing argument
                arg_out = arg.clone(type=arg.type.clone(dfevar=True, initial=arg_in))
                var_map[arg] = arg_out
                arguments += [arg_in, arg_out]
                # Enlist declaration of modified argument for removal
                decl = decl_map[arg]
                if len(decl.symbols) > 1:
                    out_map[decl] = decl.clone(symbols=tuple(v for v in decl.symbols if v != arg))
                else:
                    out_map[decl] = None
            elif arg.type.intent is not None:
                arguments += [arg.clone(type=arg.type.clone(dfevar=True))]
            else:
                arguments += [arg]

        # In the SLiC interface, scalars are kept in-order apparently, followed by
        # instreams (alphabetically) and then outstreams (alphabetically)
        scalar_arguments = [arg for arg in arguments
                            if arg.type.intent.lower() == 'in' and isinstance(arg, sym.Scalar)]
        in_arguments = [arg for arg in arguments
                        if arg.type.intent.lower() == 'in' and isinstance(arg, sym.Array)]
        out_arguments = [arg for arg in arguments if arg.type.intent.lower() in ('inout', 'out')]
        in_arguments.sort(key=lambda a: a.name)
        out_arguments.sort(key=lambda a: a.name)

        # Remove declarations of inout-arguments that now depend on the new in arguments and
        # thus have to be declared at the end
        routine.spec = Transformer(out_map).visit(routine.spec)

        # Update the argument list of the routine
        # This also re-creates the deleted declarations
        routine.arguments = scalar_arguments + in_arguments + out_arguments

        return routine

    @staticmethod
    def _convert_arguments_to_pointer(routine):
        """
        Force pointer on reference-passed arguments (i.e., all except input scalars).
        """
        routine.arguments = [arg if arg.type.intent.lower() == 'in' and isinstance(arg, sym.Scalar)
                             else arg.clone(type=arg.type.clone(pointer=True)) for arg in routine.arguments]
        return routine

    def _generate_dfe_kernel(self, routine, **kwargs):  # pylint: disable=unused-argument
        # Create a copy for the MaxJ kernel
        max_kernel = routine.clone(name=f'{routine.name}Kernel')

        # Transform arguments list
        max_kernel = self._split_and_order_arguments(max_kernel)

        # Remove parameter declarations for data types
        max_kernel.variables = [v for v in max_kernel.variables
                                if not (isinstance(v.initial, sym.InlineCall) and
                                        'select_real_kind' in v.initial.name)]

        # Some vector notation sanitation
        resolve_vector_notation(max_kernel)
        normalize_range_indexing(max_kernel)

        # Remove dataflow loops
        loop_map = {}
        var_map = {}
        dataflow_indices = []
        with pragmas_attached(max_kernel, ir.Loop):
            for loop in FindNodes(ir.Loop).visit(max_kernel.body):
                if is_loki_pragma(loop.pragma, starts_with='dataflow'):
                    loop_map[loop] = loop.body
                    # We have to add 1 since FORTRAN counts from 1
                    call_fct = sym.ProcedureSymbol('control.count.simpleCounter', scope=max_kernel)
                    vinit = sym.Sum((sym.InlineCall(call_fct, parameters=(sym.Literal(32),)), sym.Literal(1)))
                    # TODO: Add support for wrap point
                    #                      parameters=(Literal(32), loop.bounds[1]))
                    var_map[loop.variable] = loop.variable.clone(
                        type=loop.variable.type.clone(dfevar=True, initial=vinit))
                    dataflow_indices += [str(loop.variable)]
        max_kernel.spec = SubstituteExpressions(var_map).visit(max_kernel.spec)
        max_kernel.body = Transformer(loop_map).visit(max_kernel.body)

        # Replace conditionals by conditional statements
        # TODO: This does not handle nested conditionals!
        cond_map = {}
        cond_type = SymbolAttributes(BasicType.LOGICAL)
        for cnt, cond in enumerate(FindNodes(ir.Conditional).visit(max_kernel.body)):
            body = []

            # Extract conditions as separate variables
            cond_var = sym.Variable(name=f'cond_{cnt}', type=cond_type.clone(), scope=max_kernel)
            body += [ir.Assignment(lhs=cond_var, rhs=cond.condition)]
            max_kernel.variables += as_tuple(cond_var)

            # Hacky heuristic: We use body and else body to hangle us along the order of statements
            # TODO: Do this in a better way!
            else_stmts = FindNodes(ir.Assignment).visit(cond.else_body)
            for stmt in FindNodes(ir.Assignment).visit(cond.body):
                # Try to find a matching statement in else_stmts
                for i, s in enumerate(else_stmts):
                    if s.lhs == stmt.lhs:
                        else_index = i
                        break
                else:
                    else_index = -1

                # Create conditional assignment
                if else_index == -1:
                    # no matching else-stmt: insert only body-stmt
                    cond_stmt = ir.ConditionalAssignment(lhs=stmt.lhs, condition=cond_var,
                                                         rhs=stmt.rhs, else_rhs=stmt.lhs)
                    body += [cond_stmt]
                else:
                    # insert any else-stmts before the matching stmt
                    for else_stmt in else_stmts[:else_index]:
                        cond_stmt = ir.ConditionalAssignment(lhs=else_stmt.lhs, condition=cond_var,
                                                             rhs=else_stmt.lhs, else_rhs=else_stmt.rhs)
                        body += [cond_stmt]

                    # conditional assignment with body-stmt rhs and else-stmt rhs
                    cond_stmt = ir.ConditionalAssignment(lhs=stmt.lhs, condition=cond_var,
                                                         rhs=stmt.rhs, else_rhs=else_stmts[else_index].rhs)
                    body += [cond_stmt]

                    # remove processed else_stmts
                    else_stmts = else_stmts[else_index+1:]

            cond_map[cond] = body
        max_kernel.body = Transformer(cond_map).visit(max_kernel.body)

        # Mark DFEVar variables:
        # Add the attribute `dfevar` to the type of all variables that depend on a
        # `dfevar` variable (which are initially all 'in', 'inout'-arguments and dataflow loop
        # variables)
        def is_dfevar(expr, *args, **kwargs):  # pylint: disable=unused-argument
            return {isinstance(expr, (sym.Scalar, sym.Array)) and expr.type.dfevar is True}
        dfevar_mapper = ExpressionCallbackMapper(callback=is_dfevar, combine=lambda v: {any(v)})
        node_fields = {ir.Assignment: ('rhs',),
                       ir.ConditionalAssignment: ('condition', 'rhs', 'else_rhs')}

        for stmt in FindNodes(tuple(node_fields.keys())).visit(max_kernel.body):
            is_dfe = any(dfevar_mapper(getattr(stmt, attr)).pop()
                         for attr in node_fields[stmt.__class__])
            if not stmt.lhs.type.dfevar and is_dfe:
                max_kernel.symbol_attrs[stmt.lhs.name] = stmt.lhs.type.clone(dfevar=is_dfe)

        # Replace array access by stream inflow
        if dataflow_indices:
            scalar_retriever = ExpressionRetriever(lambda e: isinstance(e, sym.Scalar))
            vmap = {}
            arr_args = {arg.name: arg for arg in max_kernel.arguments if isinstance(arg, sym.Array)}
            for v in FindVariables(unique=False).visit(max_kernel.ir):
                # All array subscripts must be transformed to streams/stream offsets
                if isinstance(v, sym.Array) and v.dimensions is not None:
                    dfe_dims = {d: d.name in dataflow_indices for d in scalar_retriever.retrieve(v)}
                    if not any(dfe_dims.values()) or v in vmap:
                        continue

                    # TODO: we can only handle 1D arrays for now
                    dim = dataflow_indices[0]
                    if len(v.dimensions) > 1:
                        raise NotImplementedError('Can not handle >1 dataflow dimensions!')
                    index = v.dimensions[0]

                    # Make sure the stream for that array exists
                    assert v.name in arr_args
                    # if v.name not in in_streams:
                    #     in_streams[v.name] = v.clone(dimensions=None, type=v_type, initial=None)

                    # Stream should have no shape but the stream property instead
                    if arr_args[v.name] not in vmap:
                        new_type = arr_args[v.name].type.clone(shape=None, dfestream=True)
                        stream = arr_args[v.name].clone(dimensions=None, type=new_type)
                        vmap[arr_args[v.name]] = stream
                    else:
                        stream = vmap[arr_args[v.name]]

                    # Array subscript corresponds to current stream position: Replace array
                    # subscript by stream argument
                    if str(index) == dim:
                        vmap[v] = stream
                    elif v not in vmap:  # We have to create/use an offset stream
                        # Hacky: Replace dataflow index (loop variable) by zero
                        retriever = ExpressionRetriever(lambda e, dim=dim: isinstance(e, sym.Scalar) and str(e) == dim)
                        dmap = {d: sym.Literal(0) for d in retriever.retrieve(v)}
                        offset = SubstituteExpressionsMapper(dmap)(index)
                        # Create the offset-variable
                        fct_symbol = sym.ProcedureSymbol('stream.offset', scope=max_kernel)
                        initial = sym.InlineCall(fct_symbol, parameters=(stream, offset))
                        var_hash = sha256(str(v).encode('utf-8')).hexdigest()[:10]
                        name = f'{v.name}_{var_hash}'
                        vmap[v] = v.clone(name=name, dimensions=None,
                                          type=stream.type.clone(intent=None, initial=initial))
            max_kernel.spec = SubstituteExpressions(vmap).visit(max_kernel.spec)
            max_kernel.body = SubstituteExpressions(vmap).visit(max_kernel.body)

            # Update list of variables to include new offset streams
            variables = [v for v in max_kernel.variables if v not in vmap or v in vmap.values()]
            new_vars = [v for v in vmap.values() if v not in variables]
            max_kernel.variables = as_tuple(variables + new_vars)

        # Find out which streams are actually unneeded and remove them
        # (e.g., because they got obsolete with the removal of the dataflow loop or due to 'inout'
        # being split up into 'in' and 'out' but the argument being actually purely 'in' or 'out')
        used_var_names = [v.name for v in FindVariables().visit(max_kernel.body)]
        for v in max_kernel.variables:
            # We have to add initialization variables by hand because the mapper does not recurse
            # to them
            if v.initial is not None:
                used_var_names += [i.name for i in FindVariables().visit(v.initial)]
        obsolete_args = [arg for arg in max_kernel.arguments if arg.name not in used_var_names]
        max_kernel.variables = [v for v in max_kernel.variables if v not in obsolete_args]

        # Add casts to dataflow constants for literal assignments
        smap = {}
        for stmt in FindNodes(ir.Assignment).visit(max_kernel.body):
            if stmt.lhs.type.dfevar:
                if isinstance(stmt.rhs, (sym.FloatLiteral, sym.IntLiteral)):
                    _type = sym.InlineCall(sym.ProcedureSymbol(f'{stmt.lhs.name}.getType',
                                                               scope=max_kernel))
                    rhs = sym.InlineCall(sym.ProcedureSymbol('constant.var', scope=max_kernel),
                                         parameters=(_type, stmt.rhs))
                    smap[stmt] = ir.Assignment(lhs=stmt.lhs, rhs=rhs)
        max_kernel.body = Transformer(smap).visit(max_kernel.body)

        def base_type(var_type):
            # TODO: put this somewhere else
            if var_type.dtype == BasicType.LOGICAL:
                return sym.InlineCall(sym.ProcedureSymbol('dfeBool', scope=max_kernel))
            if var_type.dtype == BasicType.INTEGER:
                # TODO: Distinguish between signed and unsigned
                return sym.InlineCall(sym.ProcedureSymbol('dfeUInt', scope=max_kernel),
                                      parameters=(sym.IntLiteral(32),))
            if var_type.dtype == BasicType.REAL:
                if var_type.kind in ['real32']:
                    parameters = (sym.IntLiteral(8), sym.IntLiteral(24))
                else:
                    parameters = (sym.IntLiteral(11), sym.IntLiteral(53))
                return sym.InlineCall(sym.ProcedureSymbol('dfeFloat', scope=max_kernel),
                                      parameters=parameters)
            raise ValueError()

        def decl_type(var_type):
            # TODO: put this somewhere else
            if not var_type.shape:
                return 'DFEVar'
            return f'DFEVector<{decl_type(var_type.clone(shape=var_type.shape[:-1]))}>'

        def init_type(var_type):
            # TODO: put this somewhere else
            if not var_type.shape:
                return base_type(var_type)
            sub_type = var_type.clone(shape=var_type.shape[:-1])
            name = f'new DFEVectorType<{decl_type(sub_type)}>'
            parameters = (init_type(sub_type), var_type.shape[-1])
            return sym.InlineCall(sym.ProcedureSymbol(name, scope=max_kernel), parameters=parameters)

        # Initialization of dfevars
        var_map = {}
        for var in max_kernel.variables:
            if var.type.dfevar:
                if var.type.intent and var.type.intent.lower() == 'in':
                    if isinstance(var, sym.Array) or var.type.dfestream:
                        name = 'io.input'
                    else:
                        name = 'io.scalarInput'
                    parameters = (sym.StringLiteral(f'"{var.name}"'), init_type(var.type))
                    initial = sym.InlineCall(sym.ProcedureSymbol(name, scope=max_kernel),
                                             parameters=parameters)
                    var_map[var] = var.clone(type=var.type.clone(initial=initial))
                elif var.initial is None:
                    name = f'{init_type(var.type)}.newInstance'
                    if isinstance(var, sym.Array):
                        parameters = (sym.IntrinsicLiteral('this'),)
                    else:
                        parameters = (sym.IntrinsicLiteral('this'), sym.IntLiteral(0))
                    initial = sym.InlineCall(sym.ProcedureSymbol(name, scope=max_kernel),
                                             parameters=parameters)
                    var_map[var] = var.clone(type=var.type.clone(initial=initial))
        max_kernel.spec = SubstituteExpressions(var_map).visit(max_kernel.spec)

        # Insert outflow statements for output variables
        for var in max_kernel.arguments:
            if var.type.intent.lower() in ('inout', 'out'):
                if isinstance(var, sym.Array) or var.type.dfestream:
                    name = sym.Variable(name='io.output')
                else:
                    name = sym.Variable(name='io.scalarOutput')
                parameters = (sym.StringLiteral(f'"{var.name}"'),
                              var.clone(dimensions=None), init_type(var.type))
                stmt = ir.CallStatement(name, arguments=parameters)
                max_kernel.body.append(stmt)

        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        shift_to_zero_indexing(max_kernel)
        invert_array_indices(max_kernel)
        replace_intrinsics(max_kernel, function_map={'mod': 'KernelMath.modulo'})

        return max_kernel

    @staticmethod
    def _generate_dfe_kernel_module(kernel, **kwargs):  # pylint: disable=unused-argument
        """
        Create the Maxj kernel module that wraps the DFE kernel routine.
        """
        # Some boilerplate imports
        standard_imports = ['Kernel', 'KernelParameters', 'stdlib.KernelMath', 'types.base.DFEVar',
                            'types.composite.DFEVector', 'types.composite.DFEVectorType']
        standard_imports_basepath = 'com.maxeler.maxcompiler.v2.kernelcompiler.'
        spec = [ir.Import(standard_imports_basepath + imprt) for imprt in standard_imports]

        max_module = Module(name=kernel.name, spec=ir.Section(body=as_tuple(spec)))
        max_kernel = kernel.clone(parent=max_module)

        # Remove all arguments (as they are streamed in now) and insert parameter argument
        arg_type = SymbolAttributes(DerivedType('KernelParameters'), intent='in')
        arg = sym.Variable(name='params', type=arg_type, scope=max_kernel)
        max_kernel.arguments = as_tuple(arg)
        max_kernel.spec.prepend(ir.CallStatement(sym.Variable(name='super'), arguments=(arg,)))

        # Add kernel to wrapper module
        max_module.contains = ir.Section(body=as_tuple(max_kernel))
        return max_module

    @staticmethod
    def _generate_dfe_manager_intf(kernel, **kwargs):  # pylint: disable=unused-argument
        """
        Create the Maxj manager interface that configures the DFE kernel.
        """
        name = kernel.name.replace('Kernel', '')

        # Create the manager class
        # TODO: Use a TypeDef once we have typebound procedures etc.
        spec = [ir.Import('com.maxeler.maxcompiler.v2.kernelcompiler.Kernel')]
        spec += [ir.Import('com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock')]
        spec += [ir.Import('com.maxeler.maxcompiler.v2.managers.custom.api.ManagerPCIe')]
        spec += [ir.Import('com.maxeler.maxcompiler.v2.managers.custom.api.ManagerKernel')]
        spec += [ir.Intrinsic(f'static final String kernelName = "{kernel.name}";')]
        manager = Module(name=f'{name}Manager', spec=ir.Section(body=as_tuple(spec)))

        # Create the setup
        setup = Subroutine(name='default void setup', parent=manager, spec=ir.Section(body=()))

        body = [ir.Intrinsic(f'Kernel kernel = new {kernel.name}(makeKernelParameters(kernelName));')]

        # Insert in/out streams
        streams = defaultdict(list)
        for arg in kernel.arguments:
            if arg.type.intent and (arg.type.dfestream or isinstance(arg, sym.Array)):
                streams[arg.type.intent.lower()] += [arg]

        if streams:
            body += [ir.Intrinsic('KernelBlock kernelBlock = addKernel(kernel);')]
        else:
            body += [ir.Intrinsic('addKernel(kernel);')]

        for stream in streams['in']:
            body += [ir.Intrinsic(
                f'kernelBlock.getInput("{stream.name}") <== addStreamFromCPU("{stream.name}");')]
        for stream in streams['inout'] + streams['out']:
            body += [ir.Intrinsic(
                f'addStreamToCPU("{stream.name}") <== kernelBlock.getOutput("{stream.name}");')]
        setup.body = ir.Section(body=as_tuple(body))

        # Insert functions into manager class
        manager.contains = ir.Section(body=as_tuple(setup))
        return manager

    @staticmethod
    def _generate_dfe_manager(kernel, **kwargs):  # pylint: disable=unused-argument
        """
        Create the Maxj manager for the MAX5C platform.
        """
        name = kernel.name.replace('Kernel', '')

        # Create the manager class
        # TODO: Use a TypeDef once we have typebound procedures etc.
        standard_imports = ['maxcompiler.v2.build.EngineParameters',
                            'platform.max5.manager.MAX5CManager']
        standard_imports_basepath = 'com.maxeler.'
        spec = [ir.Import(standard_imports_basepath + imprt) for imprt in standard_imports]
        manager = Module(name=f'{name}ManagerMAX5C', spec=ir.Section(body=as_tuple(spec)))

        # Create the constructor
        constructor = Subroutine(name=manager.name, parent=manager, spec=ir.Section(body=()))
        params_type = SymbolAttributes(DerivedType('EngineParameters'), intent='in')
        params = sym.Variable(name='params', type=params_type, scope=constructor)
        body = [ir.CallStatement(sym.Variable(name='super'), arguments=(params,)),
                ir.CallStatement(sym.Variable(name='setup'), arguments=())]
        constructor.arguments = as_tuple(params)
        constructor.body = ir.Section(body=as_tuple(body))

        # Create the main function for maxJavaRun
        main = Subroutine(name='public static void main', parent=manager,
                          spec=ir.Section(body=()))
        args_type = SymbolAttributes(DerivedType('String[]'), intent='in')
        args = sym.Variable(name='args', type=args_type, scope=main)
        main.arguments = as_tuple(args)

        params_type = SymbolAttributes(
            DerivedType('EngineParameters'), initial=sym.InlineCall(
                sym.ProcedureSymbol('new EngineParameters', scope=main), parameters=(args,)))
        params = sym.Variable(name='params', type=params_type, scope=main)
        mgr_type = SymbolAttributes(
            DerivedType('MAX5CManager'), initial=sym.InlineCall(
                sym.ProcedureSymbol(f'new {manager.name}', scope=main), parameters=(params,)))
        mgr = sym.Variable(name='manager', type=mgr_type, scope=main)
        main.variables += as_tuple([params, mgr])
        body = (ir.CallStatement(sym.Variable(name='manager.build'), arguments=()), )
        main.body = ir.Section(body=body)

        # Insert functions into manager class
        manager.contains = ir.Section(body=(constructor, main))
        return manager

    @staticmethod
    def _generate_slic_interface(routine, kernel, **kwargs):  # pylint: disable=unused-argument
        """
        Create the SLiC interface that calls the DFE kernel.
        """
        # Create a copy of the routine that has only the routine's spec
        slic_routine = routine.clone(name=f'{routine.name}_c', body=None)

        # Add an argument for ticks
        size_t_type = SymbolAttributes(BasicType.INTEGER, intent='in')  # TODO: make this size_t
        ticks_argument = sym.Variable(name='ticks', type=size_t_type, scope=kernel)
        arguments = (ticks_argument,) + slic_routine.arguments

        # The DFE wants to know array sizes, so we replace array arguments by pairs (arg, arg_size)
        def generate_arg_tuple(arg):
            if isinstance(arg, sym.Array):
                return (arg, sym.Variable(name=f'{arg.name}_size', type=size_t_type, scope=slic_routine))
            return arg
        arguments = flatten([generate_arg_tuple(arg) for arg in arguments])

        # Remove initial values from arguments
        arguments = tuple(arg.clone(type=arg.type.clone(initial=None)) for arg in arguments)

        # Update the routine's arguments and remove any other variables
        slic_routine.variables = ()  # [v for v in slic_routine.variables if v.type.parameter]
        slic_routine.arguments = arguments

        # Create the call to the DFE kernel
        variable_map = slic_routine.variable_map

        # TODO: actually create a CallStatement. Not possible currently because we don't represent
        #       all the cases around pointer dereferencing etc that are necessary here
        call_arguments = [ticks_argument.name]  # pylint: disable=no-member
        for arg in kernel.arguments:
            if arg.name.endswith('_in') and arg.name not in variable_map:
                if isinstance(arg, sym.Array) or arg.type.dfestream:
                    call_arguments += [arg.name[:-3]]
                else:
                    call_arguments += [f'*{arg.name[:-3]}']
            else:
                call_arguments += [arg.name]
            if isinstance(arg, sym.Array) or arg.type.dfestream:
                call_arguments += [f'{call_arguments[-1]}_size']
        call = ir.Intrinsic(f'{routine.name}({", ".join(call_arguments)});')

        # Assign the body of the SLiC interface routine
        slic_routine.body = ir.Section(body=as_tuple(call))
        return slic_routine
