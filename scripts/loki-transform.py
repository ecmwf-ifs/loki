#!/usr/bin/env python

"""
Loki head script for source-to-source transformations concerning ECMWF
physics, including "Single Column" (SCA) and CLAW transformations.
"""

from collections import OrderedDict, defaultdict
from pathlib import Path
import click

from loki import (
    SourceFile, Transformer, FindNodes, FindVariables,
    SubstituteExpressions, as_tuple, Loop, Variable, Array,
    CallStatement, Pragma, DataType, SymbolType, RangeIndex,
    ArraySubscript, LoopRange, Transformation,
    DependencyTransformation, FortranCTransformation, Frontend, OMNI,
    OFP, fgen, SubstituteExpressionsMapper
)


def get_typedefs(typedef, xmods=None, frontend=OFP):
    """
    Read derived type definitions from typedef modules.
    """
    definitions = {}
    for tfile in typedef:
        source = SourceFile.from_file(tfile, xmods=xmods, frontend=frontend)
        definitions.update(source.modules[0].typedefs)
    return definitions


class DerivedArgsTransformation(Transformation):
    """
    Pipeline to remove derived types from subroutine signatures by replacing
    the relevant derived arguments with the sub-variables used in the called
    routine. The equivalent change is also applied to all callers of the
    transformed subroutines.

    Note, due to the dependency between caller and callee, this transformation
    should be applied atomically to sets of subroutine, if further transformations
    depend on the accurate signatures and call arguments.
    """

    def transform_subroutine(self, routine, **kwargs):
        # Determine role in bulk-processing use case
        task = kwargs.get('task', None)
        role = kwargs.get('role') if task is None else task.config['role']

        # Apply argument transformation, caller first!
        self.flatten_derived_args_caller(routine)
        if role == 'kernel':
            self.flatten_derived_args_routine(routine)


    @staticmethod
    def _derived_type_arguments(routine):
        """
        Find all derived-type arguments used in a given routine.

        :return: A map of ``arg => [type_vars]``, where ``type_var``
                 is a :class:`Variable` for each derived sub-variable
                 defined in the original compound type.
        """
        # Get all variables used in the kernel that have parents
        variables = FindVariables(unique=True).visit(routine.ir)
        variables = [v for v in variables if hasattr(v, 'parent') and v.parent is not None]
        candidates = defaultdict(list)

        for arg in routine.arguments:
            if arg.type.dtype == DataType.DERIVED_TYPE:
                # Skip derived types with no array members
                if all(not v.type.pointer and not v.type.allocatable
                       for v in arg.type.variables.values()):
                    continue

                # Add candidate type variables, preserving order from the typedef
                arg_member_vars = set(v.basename.lower() for v in variables
                                      if v.parent.name.lower() == arg.name.lower())
                candidates[arg] += [v for v in arg.type.variables.values()
                                    if v.basename.lower() in arg_member_vars]
        return candidates

    def flatten_derived_args_caller(self, caller):
        """
        Flatten all derived-type call arguments used in the target
        :class:`Subroutine` for all active :class:`CallStatement` nodes.

        The convention used is: ``derived%var => derived_var``.

        :param caller: The calling :class:`Subroutine`.
        """
        call_mapper = {}
        for call in FindNodes(CallStatement).visit(caller.body):
            if call.context is not None and call.context.active:
                candidates = self._derived_type_arguments(call.context.routine)

                # Simultaneously walk caller and subroutine arguments
                new_arguments = list(call.arguments)
                for d_arg, k_arg in zip(call.arguments, call.context.routine.arguments):
                    if k_arg in candidates:
                        # Found derived-type argument, unroll according to candidate map
                        new_args = []
                        for type_var in candidates[k_arg]:
                            # Insert `:` range dimensions into newly generated args
                            new_dims = tuple(RangeIndex((None, None)) for _ in type_var.type.shape or [])
                            new_dims = ArraySubscript(new_dims)
                            new_type = type_var.type.clone(parent=d_arg)
                            new_arg = type_var.clone(dimensions=new_dims, type=new_type,
                                                     parent=d_arg, scope=d_arg.scope)
                            new_args += [new_arg]

                        # Replace variable in dummy signature
                        # TODO: There's no cache anymore, maybe this can be changed?
                        # TODO: This is hacky, but necessary, as the variables
                        # from caller and callee don't cache, so we
                        # need to compare their string representation.
                        new_arg_strs = [str(a) for a in new_arguments]
                        i = new_arg_strs.index(str(d_arg))
                        new_arguments[i:i+1] = new_args

                # Set the new call signature on the IR ndoe
                call_mapper[call] = call.clone(arguments=as_tuple(new_arguments))

        # Rebuild the caller's IR tree
        caller.body = Transformer(call_mapper).visit(caller.body)

    def flatten_derived_args_routine(self, routine):
        """
        Unroll all derived-type arguments used in the subroutine
        signature, declarations and body.

        The convention used is: ``derived%var => derived_var``
        """
        candidates = self._derived_type_arguments(routine)

        # Callee: Establish replacements for declarations and dummy arguments
        new_arguments = list(routine.arguments)
        new_variables = list(routine.variables)
        for arg, type_vars in candidates.items():
            new_vars = []
            for type_var in type_vars:
                # Create a new variable with a new type mimicking the old one
                new_type = SymbolType(type_var.type.dtype, kind=type_var.type.kind,
                                      intent=arg.type.intent, shape=type_var.type.shape)
                new_name = '%s_%s' % (arg.name, type_var.basename)
                new_dimensions = ArraySubscript(new_type.shape) if new_type.shape else None
                new_var = Variable(name=new_name, type=new_type, dimensions=new_dimensions,
                                   scope=routine.symbols)
                new_vars += [new_var]

            # Replace variable in subroutine argument list
            i = new_arguments.index(arg)
            new_arguments[i:i+1] = new_vars

            # Also replace the variable in the variable list to
            # trigger the re-generation of the according declaration.
            i = new_variables.index(arg)
            new_variables[i:i+1] = new_vars

        # Apply replacements to routine by setting the properties
        routine.arguments = new_arguments
        routine.variables = new_variables

        # Create a variable substitution mapper and apply to body
        argnames = [arg.name.lower() for arg in candidates.keys()]
        variables = FindVariables(unique=False).visit(routine.body)
        variables = [v for v in variables
                     if hasattr(v, 'parent') and str(v.parent).lower() in argnames]
        # Note: The ``type=None`` prevents this clone from overwriting the type
        # we just derived above, as it would otherwise use whaterever type we
        # had derived previously (ie. the pointer type from the struct definition.)
        vmap = {v: v.clone(name=v.name.replace('%', '_'), parent=None, type=None)
                for v in variables}

        routine.body = SubstituteExpressions(vmap).visit(routine.body)


class Dimension:
    """
    Dimension that defines a one-dimensional iteration space.

    :param name: Name of the dimension, as used in data array declarations
    :param variable: Name of the iteration variable used in loops in this
                     dimension.
    :param iteration: Tuple defining the start/end variable names or values for
                      loops in this dimension.
    """

    def __init__(self, name=None, aliases=None, variable=None, iteration=None):
        self.name = name
        self.aliases = aliases
        self.variable = variable
        self.iteration = iteration

    @property
    def variables(self):
        return (self.name, self.variable) + self.iteration

    @property
    def size_expressions(self):
        """
        Return a list of expression strings all signifying "dimension size".
        """
        iteration = ['%s - %s + 1' % (self.iteration[1], self.iteration[0])]
        # Add ``1:x`` size expression for OMNI (it will insert an explicit lower bound)
        iteration += ['1:%s - %s + 1' % (self.iteration[1], self.iteration[0])]
        iteration += ['1:%s' % self.name]
        iteration += ['1:%s' % alias for alias in self.aliases]
        return [self.name] + self.aliases + iteration

    @property
    def index_expressions(self):
        """
        Return a list of expression strings all signifying potential
        dimension indices, including range accesses like `START:END`.
        """
        i_range = ['%s:%s' % (self.iteration[0], self.iteration[1])]
        # A somewhat strange expression used in VMASS bracnhes
        i_range += ['%s-%s+1' % (self.variable, self.iteration[0])]
        return [self.variable] + i_range


class SCATransformation(Transformation):
    """
    Pipeline to transform kernel into SCA format and insert CLAW directives.

    Note, this requires preprocessing with the `DerivedArgsTransformation`.
    """

    def __init__(self, dimension):
        self.dimension = dimension

    def transform_subroutine(self, routine, **kwargs):
        task = kwargs.get('task', None)
        role = kwargs['role'] if task is None else task.config['role']

        if role == 'driver':
            self.hoist_dimension_from_call(routine, target=self.dimension, wrap=True)

        elif role == 'kernel':
            self.hoist_dimension_from_call(routine, target=self.dimension, wrap=False)
            self.remove_dimension(routine, target=self.dimension)

        if routine.members is not None:
            for member in routine.members:
                self.apply(member, **kwargs)

    @staticmethod
    def remove_dimension(routine, target):
        """
        Remove all loops and variable indices of a given target dimension
        from the given routine.
        """
        size_expressions = target.size_expressions

        # Remove all loops over the target dimensions
        loop_map = OrderedDict()
        for loop in FindNodes(Loop).visit(routine.body):
            if str(loop.variable).upper() == target.variable:
                loop_map[loop] = loop.body

        routine.body = Transformer(loop_map).visit(routine.body)

        # Drop declarations for dimension variables (eg. loop counter or sizes)
        # Note that this also removes arguments and their declarations!
        routine.variables = [v for v in routine.variables if str(v).upper() not in target.variables]

        # Establish the new dimensions and shapes first, before cloning the variables
        # The reason for this is that shapes of all variable instances are linked
        # via caching, meaning we can easily void the shape of an unprocessed variable.
        variables = list(routine.variables)
        variables += list(FindVariables(unique=False).visit(routine.body))

        # We also include the member routines in the replacement process, as they share
        # declarations.
        for m in as_tuple(routine.members):
            variables += list(FindVariables(unique=False).visit(m.body))
        variables = [v for v in variables if isinstance(v, Array) and v.shape is not None]
        shape_map = {v.name: v.shape for v in variables}

        # Now generate a mapping of old to new variable symbols
        vmap = {}
        for v in variables:
            old_shape = shape_map[v.name]
            new_shape = as_tuple(s for s in old_shape if fgen(s).upper() not in size_expressions)
            new_dims = as_tuple(d for d, s in zip(v.dimensions.index_tuple, old_shape)
                                if fgen(s).upper() not in size_expressions)
            new_dims = None if len(new_dims) == 0 else ArraySubscript(new_dims)
            if len(old_shape) != len(new_shape):
                new_type = v.type.clone(shape=new_shape)
                vmap[v] = v.clone(dimensions=new_dims, type=new_type)

        # Apply vmap to variable and argument list and subroutine body
        routine.variables = [vmap.get(v, v) for v in routine.variables]

        # Apply substitution map to replacements to capture nesting
        mapper = SubstituteExpressionsMapper(vmap)
        vmap2 = {k: mapper(v) for k, v in vmap.items()}

        routine.body = SubstituteExpressions(vmap2).visit(routine.body)
        for m in as_tuple(routine.members):
            m.body = SubstituteExpressions(vmap2).visit(m.body)

    @staticmethod
    def hoist_dimension_from_call(caller, target, wrap=True):
        """
        Remove all indices and variables of a target dimension from
        caller (driver) and callee (kernel) routines, and insert the
        necessary loop over the target dimension into the driver.

        Note: In order for this routine to see the target dimensions
        in the argument declarations of the kernel, it must be applied
        before they are stripped from the kernel itself.
        """
        size_expressions = target.size_expressions
        replacements = {}

        for call in FindNodes(CallStatement).visit(caller.body):
            if call.context is not None and call.context.active:
                routine = call.context.routine
                argmap = {}

                # Replace target dimension with a loop index in arguments
                for arg, val in call.context.arg_iter(call):
                    if not isinstance(arg, Array) or not isinstance(val, Array):
                        continue

                    # TODO: Properly construct the vmap with updated dims for the call
                    new_dims = None

                    # Insert ':' for all missing dimensions in argument
                    if arg.shape is not None and len(val.dimensions.index_tuple) == 0:
                        new_dims = tuple(RangeIndex((None, None)) for _ in arg.shape)

                    # Remove target dimension sizes from caller-side argument indices
                    if val.shape is not None:
                        v_dims = val.dimensions.index_tuple or new_dims
                        new_dims = tuple(Variable(name=target.variable, scope=caller.symbols)
                                         if str(tdim).upper() in size_expressions else ddim
                                         for ddim, tdim in zip(v_dims, val.shape))

                    if new_dims is not None:
                        argmap[val] = val.clone(dimensions=ArraySubscript(new_dims))

                # Apply argmap to the list of call arguments
                arguments = [argmap.get(a, a) for a in call.arguments]
                kwarguments = as_tuple((k, argmap.get(a, a)) for k, a in call.kwarguments)

                # Collect caller-side expressions for dimension sizes and bounds
                dim_lower = None
                dim_upper = None
                for arg, val in call.context.arg_iter(call):
                    if str(arg).upper() == target.iteration[0]:
                        dim_lower = val
                    if str(arg).upper() == target.iteration[1]:
                        dim_upper = val

                # Remove call-side arguments (in-place)
                arguments = tuple(darg for darg, karg in zip(arguments, routine.arguments)
                                  if str(karg).upper() not in target.variables)
                kwarguments = list((darg, karg) for darg, karg in kwarguments
                                   if str(karg).upper() not in target.variables)
                new_call = call.clone(arguments=arguments, kwarguments=kwarguments)

                # Create and insert new loop over target dimension
                if wrap:
                    loop = Loop(variable=Variable(name=target.variable, scope=caller.symbols),
                                bounds=LoopRange((dim_lower, dim_upper, None)),
                                body=as_tuple([new_call]))
                    replacements[call] = loop
                else:
                    replacements[call] = new_call

        caller.body = Transformer(replacements).visit(caller.body)

        # Finally, we add the declaration of the loop variable
        if wrap and target.variable not in [str(v) for v in caller.variables]:
            # TODO: Find a better way to define raw data type
            dtype = SymbolType(DataType.INTEGER, kind='JPIM')
            caller.variables += (Variable(name=target.variable, type=dtype, scope=caller.symbols),)


def insert_claw_directives(routine, driver, claw_scalars, target):
    """
    Insert the necessary pragmas and directives to instruct the CLAW.

    Note: Must be run after generic SCA conversion.
    """
    from loki import FortranCodegen  # pylint: disable=import-outside-toplevel

    # Insert loop pragmas in driver (in-place)
    for loop in FindNodes(Loop).visit(driver.body):
        if str(loop.variable).upper() == target.variable:
            pragma = Pragma(keyword='claw', content='sca forward create update')
            loop._update(pragma=pragma)

    # Generate CLAW directives and insert into routine spec
    segmented_scalars = FortranCodegen(chunking=6).segment([str(s) for s in claw_scalars])
    directives = [Pragma(keyword='claw', content='define dimension jl(1:nproma) &'),
                  Pragma(keyword='claw', content='sca &'),
                  Pragma(keyword='claw', content='scalar(%s)\n\n\n' % segmented_scalars)]
    routine.spec.append(directives)


def remove_omp_do(routine):
    """
    Utility routine that strips existing !$opm do pragmas from driver code.
    """
    mapper = {}
    for p in FindNodes(Pragma).visit(routine.body):
        if p.keyword.lower() == 'omp':
            if p.content.startswith('do') or p.content.startswith('end do'):
                mapper[p] = None
    routine.body = Transformer(mapper).visit(routine.body)


@click.group()
def cli():
    pass


@cli.command('idem')
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated source files.')
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--flatten-args/--no-flatten-args', default=True,
              help='Flag to trigger derived-type argument unrolling')
@click.option('--openmp/--no-openmp', default=False,
              help='Flag to force OpenMP pragmas onto existing horizontal loops')
@click.option('--frontend', default='ofp', type=click.Choice(['ofp', 'omni']),
              help='Frontend parser to use (default OFP)')
def idempotence(out_path, source, driver, header, xmod, include, flatten_args, openmp, frontend):
    """
    Idempotence: A "do-nothing" debug mode that performs a parse-and-unparse cycle.
    """
    driver_name = 'CLOUDSC_DRIVER'
    kernel_name = 'CLOUDSC'

    frontend = Frontend[frontend.upper()]
    typedefs = get_typedefs(header, xmods=xmod, frontend=OFP)
    kernel = SourceFile.from_file(source, xmods=xmod, includes=include,
                                  frontend=frontend, typedefs=typedefs,
                                  builddir=out_path)
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=frontend, builddir=out_path)
    # Ensure that the kernel calls have all meta-information
    driver[driver_name].enrich_calls(routines=kernel[kernel_name])

    class IdemTransformation(Transformation):
        """
        Define a custom transformation pipeline that optionally inserts
        experimental OpenMP pragmas for horizontal loops.
        """

        def transform_subroutine(self, routine, **kwargs):
            # Define the horizontal dimension
            horizontal = Dimension(name='KLON', aliases=['NPROMA', 'KDIM%KLON'],
                                   variable='JL', iteration=('KIDIA', 'KFDIA'))

            if openmp:
                # Experimental OpenMP loop pragma insertion
                for loop in FindNodes(Loop).visit(routine.body):
                    if loop.variable == horizontal.variable:
                        # Update the loop in-place with new OpenMP pragmas
                        pragma = Pragma(keyword='omp', content='do simd')
                        pragma_nowait = Pragma(keyword='omp',
                                               content='end do simd nowait')
                        loop._update(pragma=pragma, pragma_post=pragma_nowait)

    if flatten_args:
        # Unroll derived-type arguments into multiple arguments
        # Caller must go first, as it needs info from routine
        driver.apply(DerivedArgsTransformation(), role='driver')
        kernel.apply(DerivedArgsTransformation(), role='kernel')

    # Now we instantiate our pipeline and apply the "idempotence" changes
    kernel.apply(IdemTransformation())
    driver.apply(IdemTransformation())

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_IDEM', mode='module', module_suffix='_MOD')
    kernel.apply(dependency, role='kernel')
    kernel.write(path=Path(out_path)/kernel.path.with_suffix('.idem.F90').name)

    # Re-generate the driver that mimicks the original source file,
    # but imports and calls our re-generated kernel.
    driver.apply(dependency, role='driver', targets=kernel_name)
    driver.write(path=Path(out_path)/driver.path.with_suffix('.idem.F90').name)


@cli.command()
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated souce files.')
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--header', '-h', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--strip-omp-do', is_flag=True, default=False,
              help='Removes existing !$omp do loop pragmas')
@click.option('--mode', '-m', default='sca',
              type=click.Choice(['sca', 'claw']))
@click.option('--frontend', default='ofp', type=click.Choice(['ofp', 'omni']),
              help='Frontend parser to use (default OFP)')
def convert(out_path, source, driver, header, xmod, include, strip_omp_do, mode, frontend):
    """
    Single Column Abstraction (SCA): Convert kernel into single-column
    format and adjust driver to apply it over in a horizontal loop.

    Optionally, this can also insert CLAW directives that may be use
    for further downstream transformations.
    """
    driver_name = 'CLOUDSC_DRIVER'
    kernel_name = 'CLOUDSC'

    frontend = Frontend[frontend.upper()]
    typedefs = get_typedefs(header, xmods=xmod, frontend=OFP)
    kernel = SourceFile.from_file(source, xmods=xmod, includes=include,
                                  frontend=frontend, typedefs=typedefs,
                                  builddir=out_path)
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=frontend, builddir=out_path)
    # Ensure that the kernel calls have all meta-information
    driver[driver_name].enrich_calls(routines=kernel[kernel_name])

    if mode == 'claw':
        claw_scalars = [v.name.lower() for v in kernel[kernel_name].variables
                        if isinstance(v, Array) and len(v.dimensions.index_tuple) == 1]

    # First, remove all derived-type arguments; caller first!
    driver.apply(DerivedArgsTransformation(), role='driver')
    kernel.apply(DerivedArgsTransformation(), role='kernel')

    # Define the target dimension to strip from kernel and caller
    horizontal = Dimension(name='KLON', aliases=['NPROMA', 'KDIM%KLON'],
                           variable='JL', iteration=('KIDIA', 'KFDIA'))

    # Now we instantiate our SCA pipeline and apply the changes
    sca_transform = SCATransformation(dimension=horizontal)
    driver.apply(sca_transform, role='driver')
    kernel.apply(sca_transform, role='kernel')

    if mode == 'claw':
        insert_claw_directives(kernel[kernel_name], driver[driver_name],
                               claw_scalars, target=horizontal)

    if strip_omp_do:
        remove_omp_do(driver[driver_name])

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_{}'.format(mode.upper()),
                                          mode='module', module_suffix='_MOD')
    kernel.apply(dependency, role='kernel')
    kernel.write(path=Path(out_path)/kernel.path.with_suffix('.%s.F90' % mode).name)

    # Re-generate the driver that mimicks the original source file,
    # but imports and calls our re-generated kernel.
    driver.apply(dependency, role='driver', targets=kernel_name)
    driver.write(path=Path(out_path)/driver.path.with_suffix('.%s.F90' % mode).name)


@cli.command()
@click.option('--out-path', '-out', type=click.Path(),
              help='Path for generated souce files.')
@click.option('--header', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s).')
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--driver', '-d', type=click.Path(),
              help='Driver file to convert.')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
def transpile(out_path, header, source, driver, xmod, include):
    """
    Convert kernels to C and generate ISO-C bindings and interfaces.
    """
    driver_name = 'CLOUDSC_DRIVER'
    kernel_name = 'CLOUDSC'

    # Parse original driver and kernel routine, and enrich the driver
    typedefs = get_typedefs(header, xmods=xmod, frontend=OFP)
    kernel = SourceFile.from_file(source, xmods=xmod, includes=include,
                                  frontend=OMNI, typedefs=typedefs,
                                  builddir=out_path)
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=OMNI, builddir=out_path)
    # Ensure that the kernel calls have all meta-information
    driver[driver_name].enrich_calls(routines=kernel[kernel_name])

    # First, remove all derived-type arguments; caller first!
    driver.apply(DerivedArgsTransformation(), role='driver')
    kernel.apply(DerivedArgsTransformation(), role='kernel')

    typepaths = [Path(h) for h in header]
    typemods = [SourceFile.from_file(tp, frontend=OFP)[tp.stem] for tp in typepaths]
    for typemod in typemods:
        FortranCTransformation().apply(source=typemod, path=out_path)

    # Now we instantiate our pipeline and apply the changes
    transformation = FortranCTransformation(header_modules=typemods)
    transformation.apply(kernel, path=out_path)

    # Housekeeping: Inject our re-named kernel and auto-wrapped it in a module
    dependency = DependencyTransformation(suffix='_FC', mode='module', module_suffix='_MOD')
    kernel.apply(dependency, role='kernel')
    kernel.write(path=Path(out_path)/kernel.path.with_suffix('.c.F90').name)

    # Re-generate the driver that mimicks the original source file,
    # but imports and calls our re-generated kernel.
    driver.apply(dependency, role='driver', targets=kernel_name)
    driver.write(path=Path(out_path)/driver.path.with_suffix('.c.F90').name)


class InferArgShapeTransformation(Transformation):
    """
    Uses IPA context information to infer the shape of arguments from
    the caller.
    """

    def transform_subroutine(self, source, **kwargs):  # pylint: disable=arguments-differ

        for call in FindNodes(CallStatement).visit(source.body):
            if call.context is not None and call.context.active:
                routine = call.context.routine

                # Create a variable map with new shape information from source
                vmap = {}
                for arg, val in call.context.arg_iter(call):
                    if isinstance(arg, Array) and len(arg.shape) > 0:
                        # Only create new shapes for deferred dimension args
                        if all(str(d) == ':' for d in arg.shape):
                            if len(val.shape) == len(arg.shape):
                                # We're passing the full value array, copy shape
                                vmap[arg] = arg.clone(shape=val.shape)
                            else:
                                # Passing a sub-array of val, find the right index
                                new_shape = [s for s, d in zip(val.shape, val.dimensions)
                                             if str(d) == ':']
                                vmap[arg] = arg.clone(shape=new_shape)

                # TODO: The derived call-side dimensions can be undefined in the
                # called routine, so we need to add them to the call signature.

                # Propagate the updated variables to variable definitions in routine
                routine.variables = [vmap.get(v, v) for v in routine.variables]

                # And finally propagate this to the variable instances
                vname_map = {k.name.lower(): v for k, v in vmap.items()}
                vmap_body = {}
                for v in FindVariables(unique=False).visit(routine.body):
                    if v.name.lower() in vname_map:
                        new_shape = vname_map[v.name.lower()].shape
                        vmap_body[v] = v.clone(shape=new_shape)
                routine.body = SubstituteExpressions(vmap_body).visit(routine.body)


if __name__ == "__main__":
    cli()
