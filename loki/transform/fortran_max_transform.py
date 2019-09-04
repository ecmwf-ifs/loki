from collections import OrderedDict
from pathlib import Path
from sympy import evaluate

from loki.transform.transformation import BasicTransformation
from loki.backend import maxjgen, maxjcgen, maxjmanagergen
from loki.expression import (Array, FindVariables, InlineCall, Literal, RangeIndex,
                             SubstituteExpressions, Variable)
from loki.ir import (Call, Import, Interface, Intrinsic, Loop, Section, Statement,
                     Conditional, ConditionalStatement)
from loki.module import Module
from loki.sourcefile import SourceFile
from loki.subroutine import Subroutine
from loki.tools import as_tuple, flatten
from loki.types import BaseType, DerivedType
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

            # Apply some common transformations
            arguments, argument_map = self.transform_arguments(source)
            body = self.transform_body(source)
            host_interface = self.generate_host_interface(source, arguments, argument_map)

            # Generate Fortran wrapper routine
            wrapper = self.generate_iso_c_wrapper_routine(host_interface, c_structs)
            self.wrapperpath = (self.maxj_src / wrapper.name.lower()).with_suffix('.f90')
            self.write_to_file(wrapper, filename=self.wrapperpath, module_wrap=True)

            # Generate C host code
            c_interface = self.generate_c_interface_routine(host_interface)
            self.c_path = (self.maxj_src / host_interface.name).with_suffix('.c')
            SourceFile.to_file(source=maxjcgen(c_interface), path=self.c_path)

            # Replicate the kernel to strip the Fortran-specific boilerplate
            spec = Section(body=[])
            variables = [a for a in arguments if a not in source.variables] + source.variables
            kernel = Subroutine(name=source.name, spec=spec, body=body, cache=source._cache)
            kernel.arguments = arguments
            kernel.variables = variables

            # Generate kernel manager
            self.maxj_manager_path = Path('%sManager.maxj' % (self.maxj_src / kernel.name))
            SourceFile.to_file(source=maxjmanagergen(kernel), path=self.maxj_manager_path)

            # Finally, generate MaxJ kernel that is to be run on the FPGA
            maxj_kernel = self.generate_maxj_kernel(kernel)
            self.maxj_kernel_path = (self.maxj_src / maxj_kernel.name).with_suffix('.maxj')
            SourceFile.to_file(source=maxjgen(maxj_kernel), path=self.maxj_kernel_path)

            # Generate C host code
            c_kernel = self.generate_c_kernel(source)
            self.c_path = (self.maxj_src / c_kernel.name).with_suffix('.c')
            SourceFile.to_file(source=maxjcgen(c_kernel), path=self.c_path)

            # Generate maxj kernel that is to be run on the FPGA
            maxj_kernel = self.generate_maxj_kernel(source)
            self.maxj_kernel_path = (self.maxj_src / maxj_kernel.name).with_suffix('.maxj')
            SourceFile.to_file(source=maxjgen(maxj_kernel), path=self.maxj_kernel_path)

            # Generate matching kernel manager
            self.maxj_manager_path = Path('%sManager.maxj' % (self.maxj_src / maxj_kernel.name))
            SourceFile.to_file(source=maxjmanagergen(source), path=self.maxj_manager_path)

        else:
            raise RuntimeError('Can only translate Module or Subroutine nodes')

    def transform_arguments(self, routine):
        """
        Copies all arguments and splits up 'inout' arguments into a new 'in'-argument and
        the original 'inout' argument with the new 'in'-argument as initial value assigned.
        """
        arguments = []
        argument_map = {}
        for arg in routine.arguments:
            if arg.type.intent.lower() == 'inout':
                arg_in = arg.clone(name='%s_in' % arg.name, initial=arg,
                                   type=arg.type.clone(pointer=False, intent='in'))
                arguments += [arg_in, arg]
                argument_map.update({arg_in: arg, arg: arg})
            else:
                arguments += [arg]
                argument_map.update({arg: arg})

        # In the SLiC interface, scalars are kept in-order apparently, followed by
        # instreams (alphabetically) and then outstreams (alphabetically)
        scalar_arguments = [arg for arg in arguments
                            if arg.type.intent.lower() == 'in' and arg.is_Scalar]
        in_arguments = [arg for arg in arguments
                        if arg.type.intent.lower() == 'in' and arg.is_Array]
        out_arguments = [arg for arg in arguments if arg.type.intent.lower() in ('inout', 'out')]
        in_arguments.sort(key=lambda a: a.name)
        out_arguments.sort(key=lambda a: a.name)
        arguments = scalar_arguments + in_arguments + out_arguments

        return arguments, argument_map

    def transform_body(self, routine):
        # Replicate the body to strip the Fortran-specific boilerplate
        body = Transformer({}).visit(routine.body)
        body = as_tuple(body)

        # Force all variables to lower-case, as Java and C are case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(body)
                if (v.is_Scalar or v.is_Array) and not v.name.islower()}
        body = SubstituteExpressions(vmap).visit(body)

        return body

    def generate_maxj_kernel(self, kernel, **kwargs):
        # Assign matching 'in' argument as initial value to 'inout' arguments
        arguments = []
        for arg in kernel.arguments:
            if arg.type.intent.lower() == 'inout':
                arg_in = next(a for a in kernel.arguments if a.name == '%s_in' % arg.name)
                arguments += [arg.clone(name=arg.name, initial=arg_in)]
            else:
                arguments += [arg]
        kernel.arguments = arguments

#        # Force pointer on reference-passed arguments
#        for arg in kernel.arguments:
#            if not (arg.type.intent.lower() == 'in' and arg.is_Scalar):
#                arg.type.pointer = True
#
#        # Propagate that reference pointer to all variables
#        arg_map = {a.name: a for a in kernel.arguments}
#        for v in FindVariables(unique=False).visit(kernel.body):
#            if v.name in arg_map:
#                if v.type:
#                    v.type.pointer = arg_map[v.name].type.pointer
#                else:
#                    v._type = arg_map[v.name].type

        self._resolve_vector_notation(kernel, **kwargs)
        self._resolve_omni_size_indexing(kernel, **kwargs)

        # Remove dataflow loops
        loop_map = {}
        vmap = {}
        for loop in FindNodes(Loop).visit(kernel.body):
            if (loop.pragma is not None and loop.pragma.keyword == 'loki' and
                    'dataflow' in loop.pragma.content):
                loop_map[loop] = loop.body
                vmap[loop.variable] = \
                    loop.variable.clone(dimensions=None,
                                        initial=InlineCall(name='control.count.simpleCounter',
                                                           arguments=[Literal(32)]))
        dataflow_indices = as_tuple(vmap.keys())  # [loop.variable for loop in loop_map.keys()]
        kernel.body = Transformer(loop_map).visit(kernel.body)

        # Replace conditionals by conditional statements
        # TODO: This does not handle nested conditionals!
        cond_map = {}
        for cnt, cond in enumerate(FindNodes(Conditional).visit(kernel.body)):
            body = []

            # Extract conditions as separate variables
            cond_vars = []
            for i, c in enumerate(cond.conditions):
                cond_vars += [Variable(name='cond_%d_%d' % (cnt, i))]
                body += [Statement(target=cond_vars[-1], expr=c)]
            kernel.variables += cond_vars

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
        kernel.body = Transformer(cond_map).visit(kernel.body)

        # Replace array access by stream inflow
        # TODO: this only works for vectors so far
        # TODO: this doesn't work at all!
        if dataflow_indices:
            dim = dataflow_indices[0]
            emap = {}
            for v in FindVariables(unique=True).visit(kernel.body):
                if isinstance(v, Array) and v.find(dim):
                    stream_v = v.clone(dimensions=None, shape=None)
                    if dim == v.dimensions[0]:
                        emap[v] = stream_v
                        vmap[v] = stream_v
                    else:
                        new_dim = v.dimensions[0] - dim
                        emap[v] = InlineCall(name='stream.offset', arguments=[stream_v, new_dim])
            kernel.body = SubstituteExpressions(vmap).visit(kernel.body)
            kernel.arguments = [vmap.get(v, v) for v in kernel.arguments]
            kernel.variables = [vmap.get(v, v) for v in kernel.variables]

        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        self._invert_array_indices(kernel, **kwargs)
        self._shift_to_zero_indexing(kernel, **kwargs)
#        self._replace_intrinsics(kernel, **kwargs)

        return kernel

    def generate_host_interface(self, routine, variables, variable_map, **kwargs):
        """
        Generate the necessary call for the host interface.
        """
        # The arguments for the host interface subroutine are the same as for the original
        # kernel plus some additional ones.
        arguments = routine.arguments

        # First, add an argument for ticks
        size_t_type = BaseType('INTEGER', intent='in')  # TODO: make this size_t
        ticks_argument = Variable(name='ticks', type=size_t_type)
        arguments = [ticks_argument] + arguments
        variables = [ticks_argument] + variables

        # Secondly, the DFE wants to know how big arrays are, so we replace array arguments by
        # pairs of (arg, arg_size)
        arg_pairs = {a: (a, Variable(name='%s_size' % a.name, type=size_t_type))
                     if a.is_Array else a for a in arguments}
        arguments = flatten([arg_pairs[a] for a in arguments])

        # For the transformed arguments we can reuse the existing size-arguments
        var_pairs = {v: (v, arg_pairs[variable_map[v]][1])
                     if v.is_Array else v for v in variables}
        call_arguments = flatten([var_pairs[v] for v in variables])

        # Remove initial values from inout-arguments
        for v in variables:
            if v.is_Array and v.type.intent == 'inout':
                v.initial = None

        # The entire body is just a call to the SLiC interface
        spec = Transformer().visit(routine.spec)
        body = (Call(name=routine.name, arguments=call_arguments),)
        kernel = Subroutine(name='%s_c' % routine.name, spec=spec, body=body)
        kernel.arguments = arguments
        kernel.variables = variables

        self._resolve_omni_size_indexing(kernel, **kwargs)

        return kernel

    def generate_c_interface_routine(self, routine):

        # Force pointer on reference-passed arguments
        for arg in routine.arguments:
            if not (arg.type.intent.lower() == 'in' and arg.is_Scalar):
                arg.type.pointer = True

        # Remove imports and other declarations
        routine.spec = Section(body=[])

        return routine

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
                                  body=None, bind=routine.name)

        # Generate variables and types for argument declarations
        for arg in routine.arguments:
            if isinstance(arg.type, DerivedType):
                ctype = DerivedType(name=c_structs[arg.type.name.lower()].name, variables=None)
            else:
                ctype = arg.type.dtype.isoctype
                # Only scalar, intent(in) arguments are pass by value
                ctype.value = arg.is_Scalar and arg.type.intent.lower() == 'in'
                # Pass by reference for array types
            dimensions = arg.dimensions if arg.is_Array else None
            shape = arg.shape if arg.is_Array else None
            var = intf_routine.Variable(name=arg.name, dimensions=dimensions,
                                        shape=shape, type=ctype)
            intf_routine.variables += [var]
            intf_routine.arguments += [var]

        return Interface(body=(intf_routine, ))

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
        wrapper = Subroutine(name='%s_fmax' % routine.name, spec=wrapper_spec, body=wrapper_body)

        # Copy internal argument and declaration definitions
        wrapper.variables = routine.arguments + [v for _, v in local_arg_map.items()]
        wrapper.arguments = routine.arguments
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
                with evaluate(False):
                    new_dims = as_tuple(d - 1 for d in v.dimensions)
                    vmap[v] = v.clone(dimensions=new_dims)
        kernel.body = SubstituteExpressions(vmap).visit(kernel.body)
