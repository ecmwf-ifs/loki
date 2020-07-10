from collections import OrderedDict
from pathlib import Path
from hashlib import sha256

from loki.transform import Transformation, FortranCTransformation
from loki.backend import maxjgen, fgen, cgen
from loki.expression import (FindVariables, FindInlineCalls, SubstituteExpressions,
                             ExpressionCallbackMapper, SubstituteExpressionsMapper,
                             retrieve_expressions)
import loki.ir as ir
from loki.expression import symbol_types as sym
from loki.module import Module
from loki.subroutine import Subroutine
from loki.sourcefile import SourceFile
from loki.tools import as_tuple, flatten
from loki.types import SymbolType, DataType
from loki.visitors import Transformer, FindNodes


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

        for arg in routine.arguments:
            if arg.type.dtype == DataType.DERIVED_TYPE:
                self.c_structs[arg.type.name.lower()] = self.c_struct_typedef(arg.type)

        # Create a copy of the kernel and apply some common transformations
        routine = routine.clone()
        routine = self._convert_variables_to_lowercase(routine)
        routine = self._remove_implicit_statements(routine)

        # Generate the DFE kernel
        maxj_kernel = self._generate_dfe_kernel(routine)
        self.maxj_kernel_path = (self.maxj_src / maxj_kernel.name).with_suffix('.maxj')
        maxj_module = self._generate_dfe_kernel_module(maxj_kernel)
        SourceFile.to_file(source=maxjgen(maxj_module), path=self.maxj_kernel_path)

        # Generate the SLiC host interface
        host_interface = self._generate_slic_interface(routine, maxj_kernel)

        # Generate Fortran wrapper module
        wrapper = FortranCTransformation.generate_iso_c_wrapper_routine(
            host_interface, self.c_structs, bind_name=host_interface.name)
        self.wrapperpath = (self.maxj_src / wrapper.name.lower()).with_suffix('.f90')
        module = Module(name='%s_MOD' % wrapper.name.upper(), routines=[wrapper])
        SourceFile.to_file(source=fgen(module), path=self.wrapperpath)

        # Generate C host code
        host_interface = self._convert_arguments_to_pointer(host_interface)
        self.c_path = (self.maxj_src / host_interface.name).with_suffix('.c')
        SourceFile.to_file(source=cgen(host_interface), path=self.c_path)

        # Generate kernel manager
        maxj_manager = self._generate_dfe_manager(maxj_kernel)
        self.maxj_manager_path = (self.maxj_src / maxj_manager.name).with_suffix('.maxj')
        SourceFile.to_file(source=maxjgen(maxj_manager), path=self.maxj_manager_path)

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
        declarations = FindNodes(ir.Declaration).visit(routine.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.variables)

        arguments = []
        out_map = {}
        for arg in routine.arguments:
            if arg.type.intent.lower() == 'inout':
                # Create matching instream argument
                in_type = arg.type.clone(intent='in', dfevar=True)
                arg_in = arg.clone(name='{}_in'.format(arg.name), type=in_type)
                # Modify existing argument
                arg.type.initial = arg_in
                arg.type.dfevar = True
                arguments += [arg_in, arg]
                # Enlist declaration of modified argument for removal
                decl = decl_map[arg]
                if len(decl.variables) > 1:
                    out_map[decl] = decl.clone(variables=[v for v in decl.variables if v != arg])
                else:
                    out_map[decl] = None
            elif arg.type.intent is not None:
                arg.type.dfevar = True
                arguments += [arg]
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
        for arg in routine.arguments:
            if not (arg.type.intent.lower() == 'in' and isinstance(arg, sym.Scalar)):
                arg.type.pointer = True
        return routine

    def _generate_dfe_kernel(self, routine, **kwargs):
        # Create a copy for the MaxJ kernel
        max_kernel = routine.clone(name='{}Kernel'.format(routine.name))

        # Transform arguments list
        max_kernel = self._split_and_order_arguments(max_kernel)

        # Remove parameter declarations
        max_kernel.variables = [v for v in max_kernel.variables if not v.type.parameter]

        # Some vector notation sanitation
        FortranCTransformation._resolve_vector_notation(max_kernel, **kwargs)
        FortranCTransformation._resolve_omni_size_indexing(max_kernel, **kwargs)

        # Remove dataflow loops
        loop_map = {}
        dataflow_indices = []
        for loop in FindNodes(ir.Loop).visit(max_kernel.body):
            if (loop.pragma is not None and loop.pragma.keyword == 'loki' and
                    'dataflow' in loop.pragma.content):
                loop_map[loop] = loop.body
                # We have to add 1 since FORTRAN counts from 1
                call_name = 'control.count.simpleCounter'
                vinit = sym.Sum((sym.InlineCall(call_name, parameters=(sym.Literal(32),)), sym.Literal(1)))
                # TODO: Add support for wrap point
                #                      parameters=(Literal(32), loop.bounds[1]))
                loop.variable.type.initial = vinit
                loop.variable.type.dfevar = True
                dataflow_indices += [str(loop.variable)]
        max_kernel.body = Transformer(loop_map).visit(max_kernel.body)

        # Replace conditionals by conditional statements
        # TODO: This does not handle nested conditionals!
        cond_map = {}
        cond_type = SymbolType(DataType.LOGICAL)
        for cnt, cond in enumerate(FindNodes(ir.Conditional).visit(max_kernel.body)):
            body = []

            # Extract conditions as separate variables
            cond_vars = []
            for i, condition in enumerate(cond.conditions):
                cond_vars += [sym.Variable(name='cond_{cnt}_{i}'.format(cnt=cnt, i=i),
                                           type=cond_type.clone(), scope=max_kernel.symbols)]
                body += [ir.Statement(target=cond_vars[-1], expr=condition)]
            max_kernel.variables += as_tuple(cond_vars)

            # Build list of dicts with all the statements from all bodies of the conditional
            stmts = []
            for cond_body in cond.bodies:
                body_stmts = OrderedDict()
                for stmt in FindNodes(ir.Statement).visit(cond_body):
                    body_stmts[stmt.target] = body_stmts.get(stmt.target, []) + [stmt]
                stmts += [body_stmts]

            else_stmts = OrderedDict()
            for stmt in FindNodes(ir.Statement).visit(cond.else_body):
                else_stmts[stmt.target] = else_stmts.get(stmt.target, []) + [stmt]

            # Collect all the statements grouped by their target
            targets = set([t for slist in (stmts + [else_stmts]) for t in slist.keys()])
            target_stmts = {t: [slist.get(t, []) for slist in stmts] for t in targets}

            # Hacky heuristic: We use the first body to hangle us along the order of statements
            # TODO: Do this in a better way!
            for stmt in FindNodes(ir.Statement).visit(cond.bodies[0]):
                t = stmt.target
                cond_stmt = else_stmts[t].pop(0).expr if else_stmts.get(t, []) else t
                for var, slist in zip(reversed(cond_vars), reversed(target_stmts.get(t, []))):
                    cond_stmt = ir.ConditionalStatement(target=t, condition=var,
                                                        expr=slist.pop(0).expr if slist else t,
                                                        else_expr=cond_stmt)
                body += [cond_stmt]

            # Add all remaining statements of all targets at the end
            for t in targets:
                while else_stmts.get(t, []) or any(target_stmts.get(t, [])):
                    cond_stmt = else_stmts[t].pop(0).expr if else_stmts.get(t, []) else t
                    for var, slist in zip(reversed(cond_vars), reversed(target_stmts.get(t, []))):
                        cond_stmt = ir.ConditionalStatement(target=t, condition=var,
                                                            expr=slist.pop(0).expr if slist else t,
                                                            else_expr=cond_stmt)
                    body += [cond_stmt]

            cond_map[cond] = body
        max_kernel.body = Transformer(cond_map).visit(max_kernel.body)

        # Mark DFEVar variables:
        # Add the attribute `dfevar` to the type of all variables that depend on a
        # `dfevar` variable (which are initially all 'in', 'inout'-arguments and dataflow loop
        # variables)
        def is_dfevar(expr, *args, **kwargs):
            return {isinstance(expr, (sym.Scalar, sym.Array)) and expr.type.dfevar is True}
        dfevar_mapper = ExpressionCallbackMapper(callback=is_dfevar, combine=lambda v: {any(v)})
        node_fields = {ir.Statement: ('expr',),
                       ir.ConditionalStatement: ('condition', 'expr', 'else_expr')}

        for stmt in FindNodes(tuple(node_fields.keys())).visit(max_kernel.body):
            is_dfe = any(dfevar_mapper(getattr(stmt, attr)).pop()
                         for attr in node_fields[stmt.__class__])
            stmt.target.type.dfevar = stmt.target.type.dfevar or is_dfe

        # Replace array access by stream inflow
        if len(dataflow_indices) > 0:
            vmap = {}
            arr_args = {arg.name: arg for arg in max_kernel.arguments if isinstance(arg, sym.Array)}
            for v in FindVariables(unique=False).visit(max_kernel.ir):
                # All array subscripts must be transformed to streams/stream offsets
                if isinstance(v, sym.Array) and v.dimensions is not None:
                    dfe_dims = {d: d.name in dataflow_indices
                                for d in retrieve_expressions(v, lambda e: isinstance(e, sym.Scalar))}
                    if not any(dfe_dims.values()) or v in vmap:
                        continue

                    # TODO: we can only handle 1D arrays for now
                    dim = dataflow_indices[0]
                    if len(v.dimensions.index_tuple) > 1:
                        raise NotImplementedError('Can not handle >1 dataflow dimensions!')
                    index = v.dimensions.index_tuple[0]

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
                        dmap = {d: sym.Literal(0) for d in retrieve_expressions(
                            v, lambda e: isinstance(e, sym.Scalar) and str(e) == dim)}
                        offset = SubstituteExpressionsMapper(dmap)(index)
                        # Create the offset-variable
                        initial = sym.InlineCall('stream.offset', parameters=(stream, offset))
                        var_hash = sha256(str(v).encode('utf-8')).hexdigest()[:10]
                        name = '{}_{}'.format(v.name, var_hash)
                        vmap[v] = v.clone(name=name, dimensions=None,
                                          type=stream.type.clone(initial=initial, intent=None))
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
            if v.type.initial is not None:
                used_var_names += [i.name for i in retrieve_expressions(
                    v.type.initial, lambda e: isinstance(e, (sym.Scalar, sym.Array)))]
        obsolete_args = [arg for arg in max_kernel.arguments if arg.name not in used_var_names]
        max_kernel.variables = [v for v in max_kernel.variables if v not in obsolete_args]

        # Add casts to dataflow constants for literal assignments
        smap = {}
        for stmt in FindNodes(ir.Statement).visit(max_kernel.body):
            if stmt.target.type.dfevar:
                if isinstance(stmt.expr, (sym.FloatLiteral, sym.IntLiteral)):
                    _type = sym.InlineCall('%s.getType' % stmt.target.name)
                    expr = sym.InlineCall('constant.var', parameters=(_type, stmt.expr))
                    smap[stmt] = ir.Statement(target=stmt.target, expr=expr)
        max_kernel.body = Transformer(smap).visit(max_kernel.body)

        def base_type(var_type):
            # TODO: put this somewhere else
            if var_type.dtype == DataType.LOGICAL:
                return sym.InlineCall('dfeBool')
            if var_type.dtype == DataType.INTEGER:
                # TODO: Distinguish between signed and unsigned
                return sym.InlineCall('dfeUInt', parameters=(sym.IntLiteral(32),))
            if var_type.dtype == DataType.REAL:
                if var_type.kind in ['real32']:
                    parameters = (sym.IntLiteral(8), sym.IntLiteral(24))
                else:
                    parameters = (sym.IntLiteral(11), sym.IntLiteral(53))
                return sym.InlineCall('dfeFloat', parameters=parameters)
            raise ValueError()

        # Initialization of dfevars
        for var in max_kernel.variables:
            if var.type.dfevar:
                if var.type.intent and var.type.intent.lower() == 'in':
                    if isinstance(var, sym.Array) or var.type.dfestream:
                        name = 'io.input'
                    else:
                        name = 'io.scalarInput'
                    parameters = (sym.StringLiteral('"{}"'.format(var.name)), base_type(var.type))
                    var.type.initial = sym.InlineCall(name, parameters=parameters)
                elif var.type.initial is None:
                    name = '{}.newInstance'.format(base_type(var.type))
                    if isinstance(var, sym.Array):
                        parameters = (sym.IntrinsicLiteral('this'),)
                    else:
                        parameters = (sym.IntrinsicLiteral('this'), sym.IntLiteral(0))
                    var.type.initial = sym.InlineCall(name, parameters=parameters)

        # Insert outflow statements for output variables
        for var in max_kernel.arguments:
            if var.type.intent.lower() in ('inout', 'out'):
                if isinstance(var, sym.Array) or var.type.dfestream:
                    name = 'io.output'
                else:
                    name = 'io.scalarOutput'
                parameters = (sym.StringLiteral('"{}"'.format(var.name)), var, base_type(var.type))
                stmt = ir.CallStatement(name, arguments=parameters)
                max_kernel.body.append(stmt)

        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        FortranCTransformation._invert_array_indices(max_kernel, **kwargs)
        FortranCTransformation._shift_to_zero_indexing(max_kernel, **kwargs)
        FortranCTransformation._replace_intrinsics(max_kernel, **kwargs)

        return max_kernel

    def _generate_dfe_kernel_module(self, kernel, **kwargs):
        max_module = Module(name=kernel.name)
        max_kernel = kernel.clone(parent=max_module)

        # Remove all arguments (as they are streamed in now) and insert parameter argument
        arg_type = SymbolType(DataType.DEFERRED, name='KernelParameters', intent='in')
        arg = sym.Variable(name='params', type=arg_type, scope=max_kernel.symbols)
        max_kernel.arguments = as_tuple(arg)
        max_kernel.spec.prepend([ir.Intrinsic('super(params);')])

        # Add kernel to wrapper module
        max_module.routines = as_tuple(max_kernel)
        return max_module

    def _generate_dfe_manager(self, kernel, **kwargs):
        """
        Create the Maxj manager that configures the DFE kernel.
        """
        name = kernel.name.replace('Kernel', '')

        # Create the manager class
        spec = (ir.Intrinsic('public static final String kernelName = "{}";'.format(kernel.name)),)
        manager = Module(name='{}Manager'.format(name), spec=ir.Section(body=spec))

        # Create the constructor
        constructor = Subroutine(name=manager.name, parent=manager, spec=ir.Section(body=()))
        arg_type = SymbolType(DataType.DEFERRED, name='EngineParameters', intent='in')
        arg = sym.Variable(name='params', type=arg_type, scope=constructor.symbols)
        constructor.arguments = as_tuple(arg)

        body = [
            ir.Intrinsic('super(params);'),
            ir.Intrinsic('Kernel kernel = new {}(makeKernelParameters(kernelName));'.format(kernel.name)),
            ir.Intrinsic('KernelBlock kernelBlock = addKernel(kernel);'),
        ]

        # Insert in/out streams
        in_streams = [arg for arg in kernel.arguments
                      if arg.type.dfestream and arg.type.intent.lower() == 'in']
        out_streams = [arg for arg in kernel.arguments
                       if arg.type.dfestream and arg.type.intent.lower() in ('inout', 'out')]
        for stream in in_streams:
            body += [ir.Intrinsic(
                'kernelBlock.getInput("{name}") <== addStreamFromCPU("{name}");'.format(
                    name=stream.name))]
        for stream in out_streams:
            body += [ir.Intrinsic(
                'addStreamToCPU("{name}") <== kernelBlock.getOutput("{name}");'.format(
                    name=stream.name))]
        constructor.body = ir.Section(body=body)

        # Create the main function for maxJavaRun
        main = Subroutine(name='public static void main', parent=manager, spec=ir.Section(body=()))
        arg_type = SymbolType(DataType.DEFERRED, name='String[]', intent='in')
        main.arguments += (sym.Variable(name='args', type=arg_type, scope=main.symbols),)

        body = [
            ir.Intrinsic('EngineParameters params = new EngineParameters(args);'),
            ir.Intrinsic('MAX5CManager manager = new {}(params);'.format(manager.name)),
            ir.Intrinsic('manager.build();')
        ]
        main.body = ir.Section(body=body)

        # Insert functions into manager class
        manager.routines = (constructor, main)
        return manager

    def _generate_slic_interface(self, routine, kernel, **kwargs):
        """
        Create the SLiC interface that calls the DFE kernel.
        """
        # Create a copy of the routine that has only the routine's spec
        slic_routine = routine.clone(name='{}_c'.format(routine.name), body=None)

        # Add include file
        slic_routine.spec.prepend(ir.Import('{}.h'.format(routine.name), c_import=True))

        # Add an argument for ticks
        size_t_type = SymbolType(DataType.INTEGER, intent='in')  # TODO: make this size_t
        ticks_argument = sym.Variable(name='ticks', type=size_t_type, scope=kernel.symbols)
        arguments = (ticks_argument,) + slic_routine.arguments

        # The DFE wants to know array sizes, so we replace array arguments by pairs (arg, arg_size)
        def generate_arg_tuple(arg):
            if isinstance(arg, sym.Array):
                return (arg, sym.Variable(name='{}_size'.format(arg.name), type=size_t_type,
                                          scope=slic_routine.symbols))
            return arg
        arguments = flatten([generate_arg_tuple(arg) for arg in arguments])

        # Remove initial values from arguments
        for arg in arguments:
            arg.initial = None

        # Update the routine's arguments and remove any other variables
        slic_routine.arguments = arguments
        slic_routine.variables = [v for v in slic_routine.variables
                                  if v in arguments or v.type.parameter]

        # Create the call to the DFE kernel
        variable_map = slic_routine.variable_map

        # TODO: actually create a CallStatement
        # def generate_call_arg_tuple(arg):
        #     if arg.name.endswith('_in') and arg.name not in variable_map:
        #         # For the split-up arguments we reuse the existing argument
        #         arg_name = arg.name[:-3]
        #     else:
        #         arg_name = arg.name
        #     if isinstance(arg, sym.Array) or arg.type.dfestream:
        #         return (variable_map[arg_name].clone(dimensions=None),
        #                 variable_map['{}_size'.format(arg_name)])
        #     return variable_map[arg_name].clone()

        # call_arguments = flatten([generate_call_arg_tuple(arg) for arg in kernel.arguments])
        # call_arguments = [ticks_argument] + call_arguments
        # call = ir.CallStatement(name=routine.name, arguments=call_arguments)

        call_arguments = [ticks_argument.name]
        for arg in kernel.arguments:
            if arg.name.endswith('_in') and arg.name not in variable_map:
                if isinstance(arg, sym.Array) or arg.type.dfestream:
                    call_arguments += [arg.name[:-3]]
                else:
                    call_arguments += ['*{}'.format(arg.name[:-3])]
            else:
                call_arguments += [arg.name]
            if isinstance(arg, sym.Array) or arg.type.dfestream:
                call_arguments += ['{}_size'.format(call_arguments[-1])]
        call = ir.Intrinsic('{}({});'.format(routine.name, ', '.join(call_arguments)))

        # Assign the body of the SLiC interface routine
        slic_routine.body = ir.Section(body=as_tuple(call))
        return slic_routine

    def _replace_intrinsics(self, kernel, **kwargs):
        """
        Replace known numerical intrinsic functions.
        """
        _intrinsic_map = {'mod': 'KernelMath.modulo'}

        callmap = {}
        for c in FindInlineCalls(unique=False).visit(kernel.body):
            cname = c.name.lower()
            if cname in _intrinsic_map:
                callmap[c] = sym.InlineCall(_intrinsic_map[cname], parameters=c.parameters,
                                            kw_parameters=c.kw_parameters)

        kernel.body = SubstituteExpressions(callmap).visit(kernel.body)
