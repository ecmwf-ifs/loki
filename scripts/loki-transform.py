#!/usr/bin/env python

"""
Loki head script for source-to-source transformations concerning ECMWF
physics, including "Single Column" (SCA) and CLAW transformations.
"""

from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
import click
import toml

from loki import (
    SourceFile, Transformer, TaskScheduler,
    FindNodes, FindVariables, SubstituteExpressions,
    info, as_tuple, Loop, Variable,
    Array, CallStatement, Pragma, DataType,
    SymbolType, Import, RangeIndex, ArraySubscript, LoopRange,
    AbstractTransformation, BasicTransformation, FortranCTransformation,
    Frontend, OMNI, OFP, fgen, SubstituteExpressionsMapper
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


class DerivedArgsTransformation(AbstractTransformation):
    """
    Pipeline to remove derived types from subroutine signatures by replacing
    the relevant derived arguments with the sub-variables used in the called
    routine. The equivalent change is also applied to all callers of the
    transformed subroutines.

    Note, due to the dependency between caller and callee, this transformation
    should be applied atomically to sets of subroutine, if further transformations
    depend on the accurate signatures and call arguments.
    """

    def _pipeline(self, source, **kwargs):
        # Determine role in bulk-processing use case
        task = kwargs.get('task', None)
        role = 'kernel' if task is None else task.config['role']

        # Apply argument transformation, caller first!
        self.flatten_derived_args_caller(source)
        if role == 'kernel':
            self.flatten_derived_args_routine(source)

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


class SCATransformation(AbstractTransformation):
    """
    Pipeline to transform kernel into SCA format and insert CLAW directives.

    Note, this requires preprocessing with the `DerivedArgsTransformation`.
    """

    def __init__(self, dimension):
        self.dimension = dimension

    def _pipeline(self, source, **kwargs):
        task = kwargs.get('task', None)
        role = kwargs['role'] if task is None else task.config['role']

        if role == 'driver':
            self.hoist_dimension_from_call(source, target=self.dimension, wrap=True)

        elif role == 'kernel':
            self.hoist_dimension_from_call(source, target=self.dimension, wrap=False)
            self.remove_dimension(source, target=self.dimension)

        if source.members is not None:
            for member in source.members:
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
        # TODO: Careful with order her, as removing the variables first
        # can invalidate the .arguments property! This needs more testing!
        routine.arguments = [a for a in routine.arguments if str(a).upper() not in target.variables]
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
        routine.arguments = [vmap.get(v, v) for v in routine.arguments]
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
            pragma = Pragma(keyword='claw', content='parallelize forward create update')
            loop._update(pragma=pragma)

    # Generate CLAW directives and insert into routine spec
    segmented_scalars = FortranCodegen(chunking=6).segment([str(s) for s in claw_scalars])
    directives = [Pragma(keyword='claw', content='define dimension jl(1:nproma) &'),
                  Pragma(keyword='claw', content='parallelize &'),
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
    frontend = Frontend[frontend.upper()]
    typedefs = get_typedefs(header, xmods=xmod, frontend=OFP)
    routine = SourceFile.from_file(source, xmods=xmod, includes=include,
                                   frontend=frontend, typedefs=typedefs)['cloudsc']
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=frontend)['cloudsc_driver']
    driver.enrich_calls(routines=routine)

    # Prepare output paths
    out_path = Path(out_path)
    source_out = (out_path/routine.name.lower()).with_suffix('.idem.F90')
    driver_out = (out_path/driver.name.lower()).with_suffix('.idem.F90')

    class IdemTransformation(BasicTransformation):
        """
        Here we define a custom transformation pipeline to re-generate
        separate kernel and driver versions, adding some optional changes
        like derived-type argument removal or an experimental low-level
        OpenMP wrapping.
        """

        def _pipeline(self, source, **kwargs):
            # Define the horizontal dimension
            horizontal = Dimension(name='KLON', aliases=['NPROMA', 'KDIM%KLON'],
                                   variable='JL', iteration=('KIDIA', 'KFDIA'))

            if openmp:
                # Experimental OpenMP loop pragma insertion
                for loop in FindNodes(Loop).visit(source.body):
                    if loop.variable == horizontal.variable:
                        # Update the loop in-place with new OpenMP pragmas
                        pragma = Pragma(keyword='omp', content='do simd')
                        pragma_nowait = Pragma(keyword='omp',
                                               content='end do simd nowait')
                        loop._update(pragma=pragma, pragma_post=pragma_nowait)

            # Perform necessary housekeeking tasks
            self.rename_routine(source, suffix='IDEM')
            self.write_to_file(source, **kwargs)

    if flatten_args:
        # Unroll derived-type arguments into multiple arguments
        # Caller must go first, as it needs info from routine
        DerivedArgsTransformation().apply(driver)
        DerivedArgsTransformation().apply(routine)

    # Now we instantiate our pipeline and apply the changes
    transformation = IdemTransformation()
    transformation.apply(routine, filename=source_out)

    # Insert new module import into the driver and re-generate
    # TODO: Needs internalising into `BasicTransformation.module_wrap()`
    driver.spec.prepend(Import(module='%s_MOD' % routine.name.upper(),
                               symbols=[routine.name.upper()]))
    transformation.rename_calls(driver, suffix='IDEM')
    transformation.write_to_file(driver, filename=driver_out, module_wrap=False)


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
    frontend = Frontend[frontend.upper()]
    typedefs = get_typedefs(header, xmods=xmod, frontend=OFP)
    routine = SourceFile.from_file(source, xmods=xmod, includes=include,
                                   frontend=frontend, typedefs=typedefs)['cloudsc']
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=frontend)['cloudsc_driver']
    driver.enrich_calls(routines=routine)

    # Prepare output paths
    out_path = Path(out_path)
    source_out = (out_path/routine.name.lower()).with_suffix('.%s.F90' % mode)
    driver_out = (out_path/driver.name.lower()).with_suffix('.%s.F90' % mode)

    if mode == 'claw':
        claw_scalars = [v.name.lower() for v in routine.variables
                        if isinstance(v, Array) and len(v.dimensions.index_tuple) == 1]

    # Debug addition: detect calls to `ref_save` and replace with `ref_error`
    for call in FindNodes(CallStatement).visit(routine.body):
        if call.name.lower() == 'ref_save':
            call.name = 'ref_error'

    # First, remove all derived-type arguments; caller first!
    DerivedArgsTransformation().apply(driver)
    DerivedArgsTransformation().apply(routine)

    # Define the target dimension to strip from kernel and caller
    horizontal = Dimension(name='KLON', aliases=['NPROMA', 'KDIM%KLON'],
                           variable='JL', iteration=('KIDIA', 'KFDIA'))

    # Now we instantiate our SCA pipeline and apply the changes
    transformation = SCATransformation(dimension=horizontal)
    transformation.apply(driver, role='driver', filename=driver_out)
    transformation.apply(routine, role='kernel', filename=source_out)

    if mode == 'claw':
        insert_claw_directives(routine, driver, claw_scalars, target=horizontal)

    if strip_omp_do:
        remove_omp_do(driver)

    # And finally apply the necessary housekeeping changes
    BasicTransformation().apply(routine, suffix=mode.upper(), filename=source_out)

    # TODO: Needs internalising into `BasicTransformation.module_wrap()`
    driver.spec.prepend(Import(module='%s_MOD' % routine.name.upper(),
                               symbols=[routine.name.upper()]))
    BasicTransformation().rename_calls(driver, suffix=mode.upper())
    BasicTransformation().write_to_file(driver, filename=driver_out, module_wrap=False)


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

    # Parse original driver and kernel routine, and enrich the driver
    typedefs = get_typedefs(header, xmods=xmod, frontend=OFP)
    routine = SourceFile.from_file(source, typedefs=typedefs, xmods=xmod,
                                   includes=include, frontend=OMNI).subroutines[0]
    driver = SourceFile.from_file(driver, xmods=xmod, includes=include,
                                  frontend=OMNI).subroutines[0]
    driver.enrich_calls(routines=routine)

    # Prepare output paths
    out_path = Path(out_path)
    source_out = (out_path / routine.name.lower()).with_suffix('.c.F90')
    driver_out = (out_path / driver.name.lower()).with_suffix('.c.F90')

    # Unroll derived-type arguments into multiple arguments
    # Caller must go first, as it needs info from routine
    DerivedArgsTransformation().apply(driver)
    DerivedArgsTransformation().apply(routine)

    typepaths = [Path(h) for h in header]
    typemods = [SourceFile.from_file(tp, frontend=OFP)[tp.stem] for tp in typepaths]
    for typemod in typemods:
        FortranCTransformation().apply(source=typemod, path=out_path)

    # Now we instantiate our pipeline and apply the changes
    transformation = FortranCTransformation(header_modules=typemods)
    transformation.apply(routine, filename=source_out, path=out_path)

    # Insert new module import into the driver and re-generate
    # TODO: Needs internalising into `BasicTransformation.module_wrap()`
    driver.spec.prepend(Import(module='%s_fc_mod' % routine.name,
                               symbols=['%s_fc' % routine.name]))
    transformation.rename_calls(driver, suffix='fc')
    transformation.write_to_file(driver, filename=driver_out, module_wrap=False)


class InferArgShapeTransformation(AbstractTransformation):
    """
    Uses IPA context information to infer the shape of arguments from
    the caller.
    """

    def _pipeline(self, source, **kwargs):

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
                routine.arguments = [vmap.get(v, v) for v in routine.arguments]
                routine.variables = [vmap.get(v, v) for v in routine.variables]

                # And finally propagate this to the variable instances
                vname_map = {k.name.lower(): v for k, v in vmap.items()}
                vmap_body = {}
                for v in FindVariables(unique=False).visit(routine.body):
                    if v.name.lower() in vname_map:
                        new_shape = vname_map[v.name.lower()].shape
                        vmap_body[v] = v.clone(shape=new_shape)
                routine.body = SubstituteExpressions(vmap_body).visit(routine.body)


class RapsTransformation(BasicTransformation):
    """
    Dedicated housekeeping pipeline for dealing with module wrapping
    and dependency management for RAPS.
    """

    def __init__(self, raps_deps=None, loki_deps=None, basepath=None):
        self.raps_deps = raps_deps
        self.loki_deps = loki_deps
        self.basepath = basepath

    def _pipeline(self, source, **kwargs):
        task = kwargs.get('task')
        mode = task.config['mode']
        role = task.config['role']
        # TODO: Need enrichment for Imports to get rid of this!
        processor = kwargs.get('processor', None)

        info('Processing %s (role=%s, mode=%s)', source.name, role, mode)

        original = source.name
        if role == 'kernel':
            self.rename_routine(source, suffix=mode.upper())

        self.rename_calls(source, suffix=mode.upper())
        self.adjust_imports(source, mode, processor)

        filename = task.path.with_suffix('.%s.F90' % mode)
        modules = as_tuple(task.file.modules)
        if len(modules) > 0:
            # If module imports are used, we inject the mode-specific name
            assert len(modules) == 1
            module = modules[0]
            modname = ''.join(module.name.lower().split('_mod')[:-1]).upper()
            module.name = ('%s_%s_MOD' % (modname, mode)).upper()
            self.write_to_file(modules, filename=filename, module_wrap=False)
        else:
            self.write_to_file(source, filename=filename, module_wrap=False)

        self.adjust_dependencies(original=original, task=task, processor=processor)

        # Re-generate interfaces and interface blocks after adjusting call signature
        for incl in processor.includes:
            # TODO: This header-searching madness should be improved with loki.build!
            for ending in ['.intfb.h', '.h']:
                intfb_path = Path(incl)/task.path.with_suffix(ending).name
                if (intfb_path).exists():
                    new_intfb_path = Path(incl)/task.path.with_suffix('.%s%s' % (mode, ending)).name
                    SourceFile.to_file(source=fgen(source.interface), path=Path(new_intfb_path))

    @staticmethod
    def adjust_imports(routine, mode, processor):
        """
        Utility routine to rename all calls to relevant subroutines and
        adjust the relevant imports.
        """
        replacements = {}

        # Update all relevant interface abd module imports
        # Note: C-style header imports live in routine.body!
        endings = ['.h', '.intfb', '_mod']
        for im in FindNodes(Import).visit(routine.ir):
            modname = im.module.lower()
            for ending in endings:
                modname = modname.split(ending)[0]

            if modname in processor.item_map:
                if im.c_import:
                    new_modname = im.module.replace(modname, '%s.%s' % (modname, mode))
                    replacements[im] = im.clone(module=new_modname)
                else:
                    # THE DIFFERENCES ARE VERY SUBTLE!
                    modname = modname.upper()  # Fortran
                    new_modname = im.module.replace(modname, '%s_%s' % (modname, mode.upper()))
                    replacements[im] = im.clone(module=new_modname)

        # Insert new declarations and transform existing ones
        routine.spec = Transformer(replacements).visit(routine.spec)
        routine.body = Transformer(replacements).visit(routine.body)

    def adjust_dependencies(self, original, task, processor):
        """
        Utility routine to generate Loki-specific build dependencies from
        RAPS-generated dependency files.

        Hack alert: The two dependency files are stashed on the transformation
        and modified on-the-fly. This is required to get selective replication
        until we have some smarter dependency generation and globbing support.
        """
        from raps_deps import Dependency  # pylint: disable=import-outside-toplevel

        mode = task.config['mode']
        whitelist = task.config['whitelist']
        original = original.lower()
        sourcepath = task.path

        # Adjust the object entry in the dependency file
        f_path = sourcepath.relative_to(self.basepath)
        f_mode_path = f_path.with_suffix('.%s.F90' % mode)
        o_path = f_path.with_suffix('.o')
        o_mode_path = o_path.with_suffix('.%s.o' % mode)

        # Re-generate dependencies for objects
        r_deps = deepcopy(self.raps_deps.content_map[str(o_path)])
        r_deps.replace(str(o_path), str(o_mode_path))
        r_deps.replace(str(f_path), str(f_mode_path))
        self.loki_deps.content += [r_deps]

        # Replicate the dependencies for header files
        for incl in processor.includes:
            for ending in ['.h', '.intfb.h']:
                h_path = Path(incl)/('%s%s' % (original, ending))
                if h_path.exists():
                    h_path = h_path.relative_to(self.basepath)
                    ok_path = h_path.with_suffix('.ok')

                    h_deps = deepcopy(self.raps_deps.content_map[str(ok_path)])
                    h_deps.replace(str(h_path),
                                   str(h_path).replace(original, '%s.%s' % (original, mode)))
                    h_deps.replace(str(ok_path),
                                   str(ok_path).replace(original, '%s.%s' % (original, mode)))
                    self.loki_deps.content += [h_deps]

        # Run through all previous dependencies and inject
        # the transformed object/header names
        for d in self.loki_deps.content:
            # We're depended on by an auto-generated header
            if isinstance(d, Dependency) and '%s.intfb.ok' % original in str(d.deps):
                intfb = d.find('%s.intfb.ok' % original)
                if intfb is not None:
                    intfb_new = intfb.replace(original, '%s.%s' % (original, mode))
                    d.replace(intfb, intfb_new)

            # We're depended on by an natural header
            if isinstance(d, Dependency) and '%s.ok' % original in str(d.deps):
                intfb = d.find('%s.ok' % original)
                if intfb is not None:
                    intfb_new = intfb.replace(original, '%s.%s' % (original, mode))
                    d.replace(intfb, intfb_new)

            if isinstance(d, Dependency) and str(o_path) in d.deps:
                d.replace(str(o_path), str(o_mode_path))

        # Inject new object into the final binary libs
        objs_ifsloki = self.loki_deps.content_map['OBJS_ifsloki']
        if original in whitelist:
            # Add new dependency inplace, next to the old one
            objs_ifsloki.append_inplace(str(o_path), str(o_mode_path))
        elif str(o_path) in objs_ifsloki.objects:
            # Replace old dependency to avoid ghosting where possible
            objs_ifsloki.replace(str(o_path), str(o_mode_path))
        else:
            objs_ifsloki.objects += [str(o_mode_path)]


@cli.command('physics')
@click.option('--config', '-cfg', type=click.Path(),
              help='Path to configuration file.')
@click.option('--basepath', type=click.Path(),
              help='Basepath of the IFS/RAPS installation directory.')
@click.option('--source', '-s', type=click.Path(), multiple=True,
              help='Path to source files to transform.')
@click.option('--xmod', '-M', type=click.Path(), multiple=True,
              help='Path for additional module file(s)')
@click.option('--include', '-I', type=click.Path(), multiple=True,
              help='Path for additional header file(s)')
@click.option('--typedef', '-t', type=click.Path(), multiple=True,
              help='Path for additional source file(s) containing type definitions')
@click.option('--raps-dependencies', '-deps', type=click.Path(), default=None,
              help='Path to RAPS-generated dependency file')
@click.option('--frontend', default='ofp', type=click.Choice(['ofp', 'omni']),
              help='Frontend parser to use (default OFP)')
@click.option('--callgraph', '-cg', is_flag=True, default=False,
              help='Generate and display the subroutine callgraph.')
def physics(config, basepath, source, xmod, include, typedef, raps_dependencies,
            frontend, callgraph):
    """
    Physics bulk-processing option that employs a :class:`TaskScheduler` to apply
    source-to-source transformations, such as the Single Column Abstraction (SCA),
    to large sets of interdependent subroutines.
    """
    from raps_deps import RapsDependencyFile, Rule  # pylint: disable=import-outside-toplevel

    frontend = Frontend[frontend.upper()]
    typedefs = get_typedefs(typedef, xmods=xmod, frontend=OFP)

    # Load configuration file and process options
    with Path(config).open('r') as f:
        config = toml.load(f)

    # Convert 'routines' to an ordered dictionary
    config['routines'] = OrderedDict((r['name'], r) for r in config['routine'])

    # Create and setup the scheduler for bulk-processing
    scheduler = TaskScheduler(paths=source, config=config, xmods=xmod,
                              includes=include, typedefs=typedefs,
                              frontend=frontend)
    scheduler.append(config['routines'].keys())

    # Add explicitly blacklisted subnodes
    if 'blacklist' in config['default']:
        scheduler.blacklist += [b.upper() for b in config['default']['blacklist']]
    for opts in config['routines'].values():
        if 'blacklist' in opts:
            scheduler.blacklist += [b.upper() for b in opts['blacklist']]

    scheduler.populate()

    raps_deps = None
    loki_deps = None
    if raps_dependencies:
        # Load RAPS dependency file for injection into the build system
        raps_deps = RapsDependencyFile.from_file(raps_dependencies)

        # Create new deps file with lib dependencies and a build rule
        objs_ifsloki = deepcopy(raps_deps.content_map['OBJS_ifs'])
        objs_ifsloki.target = 'OBJS_ifsloki'
        rule_ifsloki = Rule(target='$(BMDIR)/libifsloki.a', deps=['$(OBJS_ifsloki)'],
                            cmds=['/bin/rm -f $@', 'ar -cr $@ $^', 'ranlib $@'])
        loki_deps = RapsDependencyFile(content=[objs_ifsloki, rule_ifsloki])

    # Create the RapsTransformation to manage dependency injection
    raps_transform = RapsTransformation(raps_deps, loki_deps, basepath=basepath)

    mode = config['default']['mode']
    if mode == 'sca':
        # Define the target dimension to strip from kernel and caller
        horizontal = Dimension(name='KLON', aliases=['NPROMA', 'KDIM%KLON'],
                               variable='JL', iteration=('KIDIA', 'KFDIA'))

        # First, remove derived-type arguments
        scheduler.process(transformation=DerivedArgsTransformation())
        # Backward insert argument shapes (for surface routines)
        scheduler.process(transformation=InferArgShapeTransformation())
        # And finally, remove the horizontal dimension
        scheduler.process(transformation=SCATransformation(dimension=horizontal))

    # Finalize by applying the RapsTransformation
    scheduler.process(transformation=raps_transform)
    if raps_dependencies:
        # Write new mode-specific dependency rules file
        loki_config_path = raps_deps.path.with_suffix('.loki.def')
        raps_transform.loki_deps.write(path=loki_config_path)

    # Output the resulting callgraph
    if callgraph:
        scheduler.graph.render('callgraph', view=False)


if __name__ == "__main__":
    cli()
