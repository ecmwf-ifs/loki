from collections import OrderedDict
from functools import reduce
from pathlib import Path
import operator

from loki.transform.transformation import BasicTransformation
from loki.backend import maxjgen, maxjcgen, maxjmanagergen
from loki.expression import (Array, FindVariables, InlineCall, Literal, RangeIndex,
                             SubstituteExpressions, Variable, Scalar, ExpressionCallbackMapper,
                             retrieve_variables)
from loki.ir import (Call, Import, Interface, Intrinsic, Loop, Section, Statement,
                     Conditional, ConditionalStatement)
from loki.module import Module
from loki.sourcefile import SourceFile
from loki.subroutine import Subroutine
from loki.tools import as_tuple, flatten
from loki.types import SymbolType, DataType
from loki.visitors import Transformer, FindNodes


__all__ = ['FortranMaxTransformation']


class FortranMaxTransformation(BasicTransformation):
    """
    Fortran-to-Maxeler transformation that translates the given routine
    into .maxj and generates a matching manager and host code with
    corresponding ISO-C wrappers.
    """

    def __init__(self):
        super().__init__()

    def _pipeline(self, source, **kwargs):
        path = kwargs.get('path')

        # Maps from original type name to ISO-C and C-struct types
        c_structs = OrderedDict()

        if isinstance(source, Module):
            # TODO
            raise NotImplementedError('Module translation not yet done')

        elif isinstance(source, Subroutine):
            self.maxj_src = path / source.name
            self.maxj_src.mkdir(exist_ok=True)

            # Create a copy of the kernel and apply some common transformations
            kernel, argument_map = self.transform_kernel(source.clone())  # kernel)

            # Create the host interface
            host_interface = self.generate_host_interface(kernel, argument_map)

            # Generate Fortran wrapper routine
            wrapper = self.generate_iso_c_wrapper_routine(host_interface, c_structs)
            self.wrapperpath = (self.maxj_src / wrapper.name.lower()).with_suffix('.f90')
            self.write_to_file(wrapper, filename=self.wrapperpath, module_wrap=True)

            # Generate C host code
            c_interface = self.generate_c_interface_routine(host_interface)
            self.c_path = (self.maxj_src / host_interface.name).with_suffix('.c')
            SourceFile.to_file(source=maxjcgen(c_interface), path=self.c_path)

            # Strip the Fortran-specific boilerplate
#            spec = Section(body=[])
#            max_kernel = Subroutine(name=source.name, spec=spec, body=kernel.body)
#            max_kernel.variables = [v.clone(scope=max_kernel.symbols) for v in kernel.variables]
#            max_kernel.arguments = [a.clone(scope=max_kernel.symbols) for a in kernel.arguments]
            kernel.spec = Section(body=[])

            # Generate kernel manager
            self.maxj_manager_path = Path('%sManager.maxj' % (self.maxj_src / kernel.name))
            SourceFile.to_file(source=maxjmanagergen(kernel), path=self.maxj_manager_path)

            # Finally, generate MaxJ kernel that is to be run on the FPGA
            maxj_kernel = self.generate_maxj_kernel(kernel)
            self.maxj_kernel_path = (self.maxj_src / maxj_kernel.name).with_suffix('.maxj')
            SourceFile.to_file(source=maxjgen(maxj_kernel), path=self.maxj_kernel_path)

        else:
            raise RuntimeError('Can only translate Module or Subroutine nodes')

    def transform_kernel(self, routine):
        """
        Copies all arguments and splits up 'inout' arguments into a new 'in'-argument and
        the original 'inout' argument with the new 'in'-argument as initial value assigned.
        """
        arguments = []
        argument_map = {}
        for arg in routine.arguments:
            if arg.type.intent.lower() == 'inout':
                in_type = arg.type.clone(intent='in')
                arg_in = arg.clone(name='%s_in' % arg.name.lower(), scope=routine.symbols,
                                   type=in_type)
                new_arg = arg.clone(name=arg.name.lower(), scope=routine.symbols, initial=arg_in)
                arguments += [arg_in, new_arg]
                argument_map.update({arg_in.name: arg.name, new_arg.name: arg.name})
            else:
                new_arg = arg.clone(name=arg.name.lower(), scope=routine.symbols)
                arguments += [new_arg]
                argument_map.update({new_arg.name: arg.name})

        variables = [v for v in routine.variables if v not in routine.arguments]

        # In the SLiC interface, scalars are kept in-order apparently, followed by
        # instreams (alphabetically) and then outstreams (alphabetically)
        scalar_arguments = [arg for arg in arguments
                            if arg.type.intent.lower() == 'in' and isinstance(arg, Scalar)]
        in_arguments = [arg for arg in arguments
                        if arg.type.intent.lower() == 'in' and isinstance(arg, Array)]
        out_arguments = [arg for arg in arguments if arg.type.intent.lower() in ('inout', 'out')]
        in_arguments.sort(key=lambda a: a.name)
        out_arguments.sort(key=lambda a: a.name)
        routine.arguments = scalar_arguments + in_arguments + out_arguments
        routine.variables = routine.arguments + variables

        # Force all variables to lower-case, as Java and C are case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(routine.body)
                if (isinstance(v, Scalar) or isinstance(v, Array)) and not v.name.islower()}
        routine.body = SubstituteExpressions(vmap).visit(routine.body)

        return routine, argument_map

    def generate_maxj_kernel(self, kernel, **kwargs):
        # Create a copy for the MaxJ kernel
        max_kernel = kernel.clone()

        # Assign matching 'in' argument as initial value to 'inout' arguments
        arguments = []
        max_kernel.variables = [v.clone(scope=max_kernel.symbols) for v in kernel.variables]
        for arg in kernel.arguments:
            arg_type = arg.type.clone(dfevar=arg.type.intent in ['in', 'out', 'inout'],
                                      pointer=None)
            if arg.type.intent.lower() == 'inout':
                arg_in = next(a for a in kernel.arguments if a.name == '%s_in' % arg.name)
                arguments += [arg.clone(name=arg.name, initial=arg_in, type=arg_type,
                                        scope=max_kernel.symbols)]
            else:
                arguments += [arg.clone(type=arg_type, scope=max_kernel.symbols)]
        max_kernel.arguments = arguments

        # TODO: hacky, do this properly!
        max_kernel.variables = [v for v in max_kernel.variables if v.name.lower() not in ['jprb']]

        self._resolve_vector_notation(max_kernel, **kwargs)
        self._resolve_omni_size_indexing(max_kernel, **kwargs)

        # Remove dataflow loops
        loop_map = {}
        dataflow_indices = []
        for loop in FindNodes(Loop).visit(max_kernel.body):
            if (loop.pragma is not None and loop.pragma.keyword == 'loki' and \
                    'dataflow' in loop.pragma.content):
                loop_map[loop] = loop.body
                var_init = InlineCall('control.count.simpleCounter', parameters=(Literal(32),))
                # TODO: Add support for wrap point
                #                      parameters=(Literal(32), loop.bounds[1]))
                var_index = max_kernel.variables.index(loop.variable)
#                max_kernel.variables[var_index].initial = var_init
#                max_kernel.variables[var_index].type.dfevar = True
                dataflow_indices += [loop.variable.name]
        max_kernel.body = Transformer(loop_map).visit(max_kernel.body)

        # Replace conditionals by conditional statements
        # TODO: This does not handle nested conditionals!
        cond_map = {}
        for cnt, cond in enumerate(FindNodes(Conditional).visit(max_kernel.body)):
            body = []

            # Extract conditions as separate variables
            cond_vars = []
            cond_type = SymbolType(DataType.LOGICAL, dfevar=False)
            for i, c in enumerate(cond.conditions):
                cond_vars += [Variable(name='cond_%d_%d' % (cnt, i), type=cond_type,
                                       scope=max_kernel.symbols)]
                body += [Statement(target=cond_vars[-1], expr=c)]
            max_kernel.variables += cond_vars

            # Build list of dicts with all the statements from all bodies of the conditional
            stmts = []
            for cond_body in cond.bodies:
                body_stmts = OrderedDict()
                for stmt in FindNodes(Statement).visit(cond_body):
                    body_stmts[stmt.target] = body_stmts.get(stmt.target, []) + [stmt]
                stmts += [body_stmts]

            else_stmts = OrderedDict()
            for stmt in FindNodes(Statement).visit(cond.else_body):
                else_stmts[stmt.target] = else_stmts.get(stmt.target, []) + [stmt]

            # Collect all the statements grouped by their target
            targets = set([t for slist in (stmts + [else_stmts]) for t in slist.keys()])
            target_stmts = {t: [slist.get(t, []) for slist in stmts] for t in targets}

            # Hacky heuristic: We use the first body to hangle us along the order of statements
            # TODO: Do this in a better way!
            for stmt in FindNodes(Statement).visit(cond.bodies[0]):
                t = stmt.target
                cond_stmt = else_stmts[t].pop(0).expr if else_stmts.get(t, []) else t
                for var, slist in zip(reversed(cond_vars), reversed(target_stmts.get(t, []))):
                    cond_stmt = ConditionalStatement(target=t, condition=var,
                                                     expr=slist.pop(0).expr if slist else t,
                                                     else_expr=cond_stmt)
                body += [cond_stmt]

            # Add all remaining statements of all targets at the end
            for t in targets:
                while else_stmts.get(t, []) or any(target_stmts.get(t, [])):
                    cond_stmt = else_stmts[t].pop(0).expr if else_stmts.get(t, []) else t
                    for var, slist in zip(reversed(cond_vars), reversed(target_stmts.get(t, []))):
                        cond_stmt = ConditionalStatement(target=t, condition=var,
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
            return {isinstance(expr, (Scalar, Array)) and expr.type.dfevar is True}
        dfevar_mapper = ExpressionCallbackMapper(callback=is_dfevar, combine=lambda v: {any(v)})
        node_fields = {Statement: ('expr',),
                       ConditionalStatement: ('condition', 'expr', 'else_expr')}

        for stmt in FindNodes(tuple(node_fields.keys())).visit(max_kernel.body):
            is_dfe = any(dfevar_mapper(getattr(stmt, attr)).pop()
                         for attr in node_fields[stmt.__class__])
            stmt.target.type.dfevar = stmt.target.type.dfevar or is_dfe

        # Replace array access by stream inflow
        if len(dataflow_indices) > 0:
            vmap = {}
            stream_counter = 0
            for v in FindVariables(unique=False).visit(max_kernel.body):
                if isinstance(v, Array) and v.dimensions is not None:
                    dfe_dims = {d: d.name in dataflow_indices for d in retrieve_variables(v)}
                    if not any(dfe_dims.values()) or v in vmap:
                        continue
                    if len(v.dimensions) > 1:
                        raise NotImplementedError('Cannot yet handle >1 dataflow dim!')
                    dim = dataflow_indices[0]
                    v_type = v.type.clone(shape=None, dfestream=True)
                    if v.dimensions[0].name == dim:
                        v_init = None 
                        v_name = v.name
                    else:
                        parameters = (Literal(v.name),
                                      v.dimensions[0] - Variable(name=dim,
                                                                 scope=max_kernel.symbols))
                        v_init = InlineCall('stream.offset', parameters=parameters)
                        v_name = 's_%s_%s' % (v.name, stream_counter)
                        stream_counter += 1
                    vmap[v] = v.clone(name=v_name, dimensions=None, type=v_type, initial=v_init)
            max_kernel.body = SubstituteExpressions(vmap).visit(max_kernel.body)

            # Replace old declarations by stream args
            obs_args = {v.name for v in vmap.keys()}
            new_args = list(set(vmap.values()))
            max_kernel.arguments = [v for v in max_kernel.arguments if v.name not in obs_args]
            max_kernel.arguments += new_args
            max_kernel.variables = [v for v in max_kernel.variables if v.name not in obs_args]
            max_kernel.variables += new_args

        # TODO: This has to be communicated back to the host interface
        # Find out which streams are actually unneeded (e.g., because they got obsolete with the
        # removal of the dataflow loop or due to 'inout' being split up into 'in' and 'out'
#        dfevar_mapper = ExpressionCallbackMapper(callback=lambda expr: {expr},
#                                                 combine=lambda v: reduce(operator.or_, v, set()))
#        depmap = {}
#        for stmt in FindNodes(tuple(node_fields.keys())).visit(max_kernel.body):
#            deps = [dfevar_mapper(getattr(stmt, attr)) for attr in node_fields[stmt.__class__]]
#            deps = reduce(operator.or_, deps, set())
#            depmap[stmt.target] = depmap.get(stmt.target, {}) or deps
#
#        deps = {v: len(depmap.get(v, {})) > 0 for v in max_kernel.arguments}
#        deps = {k: any([k in d for d in depmap.values()]) or v for k, v in deps.items()}
#        max_kernel.arguments = [k for k, v in deps.items() if v]
#        obs_args = [k for k, v in deps.items() if not v]
#        max_kernel.variables = [v for v in max_kernel.variables if v not in obs_args]

        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        self._invert_array_indices(max_kernel, **kwargs)
        self._shift_to_zero_indexing(max_kernel, **kwargs)
#        self._replace_intrinsics(kernel, **kwargs)

        return max_kernel

    def generate_host_interface(self, routine, variable_map, **kwargs):
        """
        Generate the necessary call for the host interface.
        """
        # Create a copy of the spec for the host interface
        spec = Transformer({}).visit(routine.spec)
        kernel = Subroutine(name='%s_c' % routine.name, spec=spec)

        # The arguments for the host interface subroutine are the same as for the original
        # kernel plus some additional ones. Variables are just the arguments
        arguments = [arg.clone(scope=kernel.symbols) for arg in routine.arguments]
        variables = [v for v in arguments]

        # First, add an argument for ticks
        size_t_type = SymbolType(DataType.INTEGER, intent='in')  # TODO: make this size_t
        ticks_argument = Variable(name='ticks', type=size_t_type, scope=kernel.symbols)
        arguments = [ticks_argument] + arguments
        variables = [ticks_argument] + variables

        # Secondly, the DFE wants to know how big arrays are, so we replace array arguments by
        # pairs of (arg, arg_size)
        arg_pairs = {a.name: (a, Variable(name='%s_size' % a.name, type=size_t_type,
                                          scope=kernel.symbols))
                     if isinstance(a, Array) else a for a in arguments}
        arguments = flatten([arg_pairs[a.name] for a in arguments])

        # For the transformed arguments we can reuse the existing size-arguments
        var_pairs = {v: (v, arg_pairs[variable_map[v.name]][1])
                     if isinstance(v, Array) else v for v in variables}
        call_arguments = flatten([var_pairs[v] for v in variables])

        # Remove initial values from inout-arguments
        for v in variables:
            if isinstance(v, Array) and v.type.intent == 'inout':
                v.initial = None

        # The entire body is just a call to the SLiC interface
        kernel.body = (Call(name=routine.name, arguments=call_arguments),)
        kernel.arguments = arguments
        kernel.variables = variables

        self._resolve_omni_size_indexing(kernel, **kwargs)

        return kernel

    @staticmethod
    def iso_c_intrinsic_kind(_type):
        if _type.dtype == DataType.INTEGER:
            return 'c_int'
        elif _type.dtype == DataType.REAL:
            kind = str(_type.kind)
            if kind.lower() in ('real32', 'c_float'):
                return 'c_float'
            elif kind.lower() in ('real64', 'jprb', 'selected_real_kind(13, 300)', 'c_double'):
                return 'c_double'
            else:
                return None
        else:
            return None

    def generate_c_interface_routine(self, routine):
        # Create a copy of the routine
        kernel = Subroutine(name=routine.name)
        kernel.spec = Section(body=[])
        bmap = {v: v.clone(scope=kernel.symbols) for v in FindVariables().visit(routine.body)}
        kernel.body = SubstituteExpressions(bmap).visit(Transformer({}).visit(routine.body))
        kernel.arguments = [a.clone(scope=kernel.symbols) for a in routine.arguments]
        kernel.variables = kernel.arguments 

        # Force pointer on reference-passed arguments (i.e., all except input scalars)
        for arg in kernel.arguments:
            if not (arg.type.intent.lower() == 'in' and isinstance(arg, Scalar)):
                arg.type.pointer = True

        return kernel

    def generate_iso_c_interface(self, routine, c_structs, parent):
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
                                  body=None, bind=routine.name)

        # Generate variables and types for argument declarations
        for arg in routine.arguments:
            if arg.type.dtype == DataType.DERIVED_TYPE:
                ctype = SymbolType(DataType.DERIVED_TYPE, variables=None, shape=arg.type.shape,
                                   name=c_structs[arg.type.name.lower()].name)
            else:
                # Only scalar, intent(in) arguments are pass by value
                # Pass by reference for array types
                value = isinstance(arg, Scalar) and arg.type.intent.lower() == 'in'
                ctype = SymbolType(arg.type.dtype, value=value,
                                   kind=self.iso_c_intrinsic_kind(arg.type))
            dimensions = arg.dimensions if isinstance(arg, Array) else None
            var = Variable(name=arg.name, dimensions=dimensions, type=ctype,
                           scope=intf_routine.symbols)
            intf_routine.variables += [var]
            intf_routine.arguments += [var]

        return Interface(body=(intf_routine, ))

    def generate_iso_c_wrapper_routine(self, routine, c_structs):
        # Create initial object to have a scope
        wrapper = Subroutine(name='%s_fmax' % routine.name)

        interface = self.generate_iso_c_interface(routine, c_structs, wrapper)

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
            if arg.type.dtype == DataType.DERIVED_TYPE:
                ctype = SymbolType(DataType.DERIVED_TYPE, variables=None,
                                   name=c_structs[arg.type.name.lower()].name)
                cvar = Variable(name='%s_c' % arg.name, type=ctype, scope=wrapper.symbols)
                cast_in = InlineCall('transfer', parameters=(arg,), kw_parameters={'mold': cvar})
                casts_in += [Statement(target=cvar, expr=cast_in)]

                cast_out = InlineCall('transfer', parameters=(cvar,), kw_parameters={'mold': arg})
                casts_out += [Statement(target=arg, expr=cast_out)]
                local_arg_map[arg.name] = cvar

        arguments = [local_arg_map[a.name]
                     if a.name in local_arg_map else a.clone(scope=wrapper.symbols)
                     for a in routine.arguments]
        wrapper_body = casts_in
        wrapper_body += [Call(name=interface.body[0].name, arguments=arguments)]
        wrapper_body += casts_out
        wrapper.__init__(name='%s_fmax' % routine.name, spec=wrapper_spec, body=wrapper_body,
                         symbols=wrapper.symbols, types=wrapper.types)

        # Copy internal argument and declaration definitions
        wrapper.arguments = [arg.clone(scope=wrapper.symbols) for arg in routine.arguments]
        wrapper.variables = wrapper.arguments + [v for v in local_arg_map.values()]
        return wrapper

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
        Shift each array indices to adjust to Java indexing conventions
        """
        vmap = {}
        for v in FindVariables(unique=True).visit(kernel.body):
            if isinstance(v, Array):
                new_dims = as_tuple(d - 1 for d in v.dimensions)
                vmap[v] = v.clone(dimensions=new_dims)
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)
