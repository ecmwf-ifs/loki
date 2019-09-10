from collections import OrderedDict
from pathlib import Path
from sympy import evaluate, Mul

from loki.transform.transformation import BasicTransformation
from loki.backend import maxjgen, maxjcgen, maxjmanagergen
from loki.expression import (Array, FindVariables, InlineCall, Literal, RangeIndex,
                             SubstituteExpressions, Variable)
from loki.ir import Call, Import, Interface, Intrinsic, Loop, Section, Statement
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

            # Generate Fortran wrapper routine
            host_interface = self.generate_host_interface(source)
            wrapper = self.generate_iso_c_wrapper_routine(host_interface, c_structs)
            self.wrapperpath = (self.maxj_src / wrapper.name.lower()).with_suffix('.f90')
            self.write_to_file(wrapper, filename=self.wrapperpath, module_wrap=True)

            # Generate C host code
            c_interface = self.generate_c_interface_routine(host_interface)
            self.c_path = (self.maxj_src / host_interface.name).with_suffix('.c')
            SourceFile.to_file(source=maxjcgen(c_interface), path=self.c_path)

            # Generate maxj kernel that is to be run on the FPGA
            maxj_kernel = self.generate_maxj_kernel(source)
            self.maxj_kernel_path = (self.maxj_src / maxj_kernel.name).with_suffix('.maxj')
            SourceFile.to_file(source=maxjgen(maxj_kernel), path=self.maxj_kernel_path)

            # Generate matching kernel manager
            maxj_manager = self.generate_maxj_manager(source)
            self.maxj_manager_path = Path('%sManager.maxj' % (self.maxj_src / maxj_kernel.name))
            SourceFile.to_file(source=maxjmanagergen(maxj_manager), path=self.maxj_manager_path)

        else:
            raise RuntimeError('Can only translate Module or Subroutine nodes')

    def generate_maxj_manager(self, routine, **kwargs):
        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = Section(body=[])
        body = as_tuple(Transformer({}).visit(routine.body))

        # Force all variables to lower-case, as Java is case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(body)
                if (v.is_Scalar or v.is_Array) and not v.name.islower()}
        body = SubstituteExpressions(vmap).visit(body)

        kernel = Subroutine(name=routine.name, spec=spec, body=body, cache=routine._cache)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        self._resolve_omni_size_indexing(kernel, **kwargs)

        return kernel

    def generate_maxj_kernel(self, routine, **kwargs):
        # Replicate the kernel to strip the Fortran-specific boilerplate
        spec = Section(body=[])
        body = Transformer({}).visit(routine.body)
        body = as_tuple(body)

        # Force all variables to lower-case, as Java is case-sensitive
        vmap = {v: v.clone(name=v.name.lower()) for v in FindVariables().visit(body)
                if (v.is_Scalar or v.is_Array) and not v.name.islower()}
        body = SubstituteExpressions(vmap).visit(body)

        kernel = Subroutine(name=routine.name, spec=spec, body=body, cache=routine._cache)
        kernel.arguments = routine.arguments
        kernel.variables = routine.variables

        # Force pointer on reference-passed arguments
        for arg in kernel.arguments:
            if not (arg.type.intent.lower() == 'in' and arg.is_Scalar):
                arg.type.pointer = True

        # Propagate that reference pointer to all variables
        arg_map = {a.name: a for a in kernel.arguments}
        for v in FindVariables(unique=False).visit(kernel.body):
            if v.name in arg_map:
                if v.type:
                    v.type.pointer = arg_map[v.name].type.pointer
                else:
                    v._type = arg_map[v.name].type

        self._resolve_vector_notation(kernel, **kwargs)
        self._resolve_omni_size_indexing(kernel, **kwargs)

        # TODO: Resolve reductions (eg. SUM(myvar(:)))
        self._invert_array_indices(kernel, **kwargs)
        self._shift_to_zero_indexing(kernel, **kwargs)
#        self._replace_intrinsics(kernel, **kwargs)

        return kernel

    def generate_host_interface(self, routine, **kwargs):
        """
        Generate the necessary call for the host interface.
        """
        # The arguments for the host interface subroutine are the same as for the original kernel
        # plus some additional ones.
        arguments = routine.arguments

        # First, add an argument for ticks
        size_t_type = BaseType('INTEGER', intent='in')  # TODO: make this size_t
        ticks_argument = Variable(name='ticks', type=size_t_type)
        arguments = [ticks_argument] + arguments

        # Copy all arguments and split up inout arguments
        # Apparently, scalars are kept in-order, followed by instreams (alphabetically) and
        # then outstreams (alphabetically)
        variables = flatten([(arg, arg.clone(name='%s_in' % arg.name, initial=arg,
                                             type=arg.type.clone(pointer=False, intent='in')))
                              if arg.type.intent.lower() == 'inout' else arg
                              for arg in arguments])
        scalar_variables = [arg for arg in variables
                            if arg.type.intent.lower() == 'in' and arg.is_Scalar]
        in_variables = [arg for arg in variables
                        if arg.type.intent.lower() == 'in' and arg.is_Array]
        out_variables = [arg for arg in variables if arg.type.intent.lower() in ('inout', 'out')]
        in_variables.sort(key=lambda a: a.name)
        out_variables.sort(key=lambda a: a.name)
        variables = scalar_variables + in_variables + out_variables

        # The DFE wants to know how big arrays are, so we replace array arguments by
        # pairs of (arg, arg_size)
        arg_pairs = {a: (a, Variable(name='%s_size' % a.name, type=size_t_type)) if a.is_Array
                     else a for a in variables}
        arguments = flatten([arg_pairs[a] for a in arguments])
        var_pairs = {a: (a, Variable(name='%s_byte_size' % a.name, type=size_t_type,
                                     initial=arg_pairs[a][1]))
                     if a.is_Array else a for a in variables}
        variables = flatten([var_pairs[a] for a in variables])

        # The entire body is just a call to the SLiC interface
        spec = Transformer().visit(routine.spec)
        body = (Call(name=routine.name, arguments=variables),)
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

        # Remove pointer properties
        #for arg in routine.arguments:
        #    if arg.type.pointer:
        #        arg.type.pointer = False

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
