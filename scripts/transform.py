import click
import toml
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

from loki import (FortranSourceFile, Visitor, ExpressionVisitor,
                  Transformer, FindNodes, FindVariables, info,
                  as_tuple, Loop, Variable, Declaration, Call, Pragma,
                  BaseType, DerivedType, Import, Index,
                  AbstractTransformation, BasicTransformation)

from raps_deps import RapsDependencyFile, Dependency, Rule
from scheduler import TaskScheduler


def get_typedefs(typedef):
    """
    Read derived type definitions from typedef modules.
    """
    definitions = {}
    for tfile in typedef:
        module = FortranSourceFile(tfile).modules[0]
        definitions.update(module.typedefs)
    return definitions


class VariableTransformer(ExpressionVisitor, Visitor):
    """
    Utility :class:`Transformer` that applies string replacements to
    :class:`Variable`s in-place.
    """

    def __init__(self, argnames):
        super(VariableTransformer, self).__init__()
        self.argnames = argnames

    def visit_Variable(self, o):
        if o.name in self.argnames and o.subvar is not None:
            # HACK: In-place merging of var with subvar
            o.name = '%s_%s' % (o.name, o.subvar.name)
            o._type = o.subvar._type
            o._shape = o.subvar._shape
            o.dimensions = o.subvar.dimensions
            o.initial = o.subvar.initial
            o.subvar = o.subvar.subvar


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

    def _pipeline(self, routine, **kwargs):
        # Determine role in bulk-processing use case
        task = kwargs.get('task', None)
        role = 'kernel' if task is None else task.config['role']

        # Apply argument transformation, caller first!
        self.flatten_derived_args_caller(routine)
        if role == 'kernel':
            self.flatten_derived_args_routine(routine)

    def _derived_type_arguments(self, routine):
        """
        Find all derived-type arguments used in a given routine.

        :return: A map of ``arg => [type_vars]``, where ``type_var``
                 is a :class:`Variable` for each derived sub-variable
                 defined in the original compound type.
        """
        variables = FindVariables().visit(routine.ir)
        candidates = defaultdict(list)
        for arg in routine.arguments:
            if isinstance(arg.type, DerivedType):
                # Add candidate type variables, preserving order from the typedef
                argvars = [v for v in variables if v.name == arg.name]
                argsubvars = set(v.subvar.name for v in argvars if v.subvar is not None)
                candidates[arg] += [v for v in arg.type.variables.values()
                                    if v.name in argsubvars]

        return candidates

    def flatten_derived_args_caller(self, caller):
        """
        Flatten all derived-type call arguments used in the target
        :class:`Subroutine` for all active :class:`Call` nodes.

        The convention used is: ``derived%var => derived_var``.

        :param caller: The calling :class:`Subroutine`.
        """
        call_mapper = {}
        for call in FindNodes(Call).visit(caller.ir):
            if call.context is not None and call.context.active:
                candidates = self._derived_type_arguments(call.context.routine)

                # Simultaneously walk caller and subroutine arguments
                new_arguments = list(deepcopy(call.arguments))
                for d_arg, k_arg in zip(call.arguments, call.context.routine.arguments):
                    if k_arg in candidates:
                        # Found derived-type argument, unroll according to candidate map
                        new_args = []
                        for type_var in candidates[k_arg]:
                            # Insert `:` range dimensions into newly generated args
                            new_dims = tuple(Index(name=':') for _ in type_var.dimensions)
                            new_arg = deepcopy(d_arg)
                            new_arg.subvar = Variable(name=type_var.name, dimensions=new_dims,
                                                      shape=type_var.dimensions)
                            new_args += [new_arg]

                        # Replace variable in dummy signature
                        i = new_arguments.index(d_arg)
                        new_arguments[i:i+1] = new_args

                # Set the new call signature on the IR ndoe
                call_mapper[call] = call.clone(arguments=as_tuple(new_arguments))

        # Rebuild the caller's IR tree
        caller._ir = Transformer(call_mapper).visit(caller.ir)

    def flatten_derived_args_routine(self, routine):
        """
        Unroll all derived-type arguments used in the subroutine
        signature, declarations and body.

        The convention used is: ``derived%var => derived_var``
        """
        candidates = self._derived_type_arguments(routine)
        declarations = FindNodes(Declaration).visit(routine.ir)

        # Callee: Establish replacements for declarations and dummy arguments
        decl_mapper = defaultdict(list)
        for arg, type_vars in candidates.items():
            old_decl = [d for d in declarations if arg in d.variables][0]
            new_names = []

            for type_var in type_vars:
                # Create new name and add to variable mapper
                new_name = '%s_%s' % (arg.name, type_var.name)
                new_names += [new_name]

                # Create new declaration and add to declaration mapper
                new_type = BaseType(name=type_var.type.name,
                                    kind=type_var.type.kind,
                                    intent=arg.type.intent)
                new_var = Variable(name=new_name, type=new_type,
                                   dimensions=as_tuple(type_var.dimensions),
                                   shape=as_tuple(type_var.dimensions))
                decl_mapper[old_decl] += [Declaration(variables=[new_var], type=new_type)]

            # Replace variable in dummy signature
            i = routine.argnames.index(arg)
            routine._argnames[i:i+1] = new_names

        # Replace variable occurences in-place (derived%v => derived_v)
        argnames = [arg.name for arg in candidates.keys()]
        VariableTransformer(argnames=argnames).visit(routine.ir)

        # Replace `Declaration` nodes (re-generates the IR tree)
        routine._ir = Transformer(decl_mapper).visit(routine.ir)


class Dimension(object):
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
        iteration = ['%s-%s+1' % (self.iteration[1], self.iteration[0])]
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

    def _pipeline(self, routine, **kwargs):
        task = kwargs.get('task', None)
        role = kwargs['role'] if task is None else task.config['role']

        if role == 'driver':
            self.hoist_dimension_from_call(routine, target=self.dimension, wrap=True)

        elif role == 'kernel':
            self.hoist_dimension_from_call(routine, target=self.dimension, wrap=False)
            self.remove_dimension(routine, target=self.dimension)

    def remove_dimension(self, routine, target):
        """
        Remove all loops and variable indices of a given target dimension
        from the given routine.
        """
        replacements = {}
        size_expressions = target.size_expressions
        index_expressions = target.index_expressions

        # Remove all loops over the target dimensions
        loop_map = {}
        for loop in FindNodes(Loop).visit(routine.ir):
            if loop.variable == target.variable:
                loop_map[loop] = loop.body
        routine._ir = Transformer(loop_map).visit(routine.ir)

        # Drop declarations for dimension variables (eg. loop counter or sizes)
        for decl in FindNodes(Declaration).visit(routine.ir):
            new_vars = tuple(v for v in decl.variables
                             if str(v) not in target.variables)

            # Strip target dimension from declaration-level dimensions
            if decl.dimensions is not None and len(decl.dimensions) > 0:
                # TODO: This is quite hacky, as we rely on the first
                # variable in the declaration to provide the correct shape.
                assert len(decl.dimensions) == len(decl.variables[0].shape)
                new_dims = tuple(d for d, s in zip(decl.dimensions, decl.variables[0].shape)
                                 if str(s) not in size_expressions)
                if len(new_dims) == 0:
                    new_dims = None
            else:
                new_dims = decl.dimensions

            if len(new_vars) == 0:
                # Drop the declaration if it becomes empty
                replacements[decl] = None
            else:
                replacements[decl] = decl.clone(variables=new_vars, dimensions=new_dims)

        # Remove all variable indices representing the target dimension (in-place)
        for v in FindVariables(unique=False).visit(routine.ir):
            if v.dimensions is not None and v.shape is not None:
                # Filter index variables against index expressions
                # and shape dimensions against size expressions.
                filtered = [(d, s) for d, s in zip(v.dimensions, v.shape)
                            if str(s) not in size_expressions and d not in index_expressions]

                # Reconstruct variable dimensions and shape from filtered
                if len(filtered) > 0:
                    v.dimensions, v._shape = zip(*(filtered))
                else:
                    v.dimensions, v._shape = (), None

        # Remove dimension size expressions from variable declarations (in-place)
        # Note: We do this last, because changing the declaration affects
        # the variable_map used above.
        for decl in FindNodes(Declaration).visit(routine.ir):
            for v in decl.variables:
                if v.dimensions is not None:
                    v.dimensions = as_tuple(d for d in v.dimensions
                                            if str(d) not in size_expressions)

        # Remove dummy variables from subroutine signature (in-place)
        routine._argnames = tuple(arg for arg in routine.argnames
                                  if arg not in target.variables)

        routine._ir = Transformer(replacements).visit(routine.ir)

    def hoist_dimension_from_call(self, caller, target, wrap=True):
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

        for call in FindNodes(Call).visit(caller.ir):
            if call.context is not None and call.context.active:
                routine = call.context.routine

                # Replace target dimension with a loop index in arguments
                for arg, val in call.context.arg_iter(call):
                    if not isinstance(val, Variable):
                        continue

                    # Skip to the innermost variable of derived types
                    while val.subvar is not None:
                        val = val.subvar

                    # Insert ':' for all missing dimensions in argument
                    if arg.shape is not None and len(val.dimensions) == 0:
                        val.dimensions = tuple(Index(name=':') for _ in arg.shape)

                    # Remove target dimension sizes from caller-side argument indices
                    if val.shape is not None:
                        val.dimensions = tuple(Index(name=target.variable)
                                               if str(tdim) in size_expressions else ddim
                                               for ddim, tdim in zip(val.dimensions, val.shape))

                # Collect caller-side expressions for dimension sizes and bounds
                dim_lower = None
                dim_upper = None
                for arg, val in call.context.arg_iter(call):
                    if arg == target.iteration[0]:
                        dim_lower = val
                    if arg == target.iteration[1]:
                        dim_upper = val

                # Remove call-side arguments (in-place)
                arguments = tuple(darg for darg, karg in zip(call.arguments, routine.arguments)
                                  if karg not in target.variables)
                kwarguments = list((darg, karg) for darg, karg in call.kwarguments
                                   if karg not in target.variables)
                new_call = call.clone(arguments=arguments, kwarguments=kwarguments)

                # Create and insert new loop over target dimension
                if wrap:
                    loop = Loop(variable=Variable(name=target.variable),
                                bounds=(dim_lower, dim_upper), body=as_tuple([new_call]))
                    replacements[call] = loop
                else:
                    replacements[call] = new_call

        caller._ir = Transformer(replacements).visit(caller.ir)

        # Finally, we add the declaration of the loop variable
        if wrap and target.variable not in caller.variables:
            caller.spec.append(Declaration(variables=Variable(name=target.variable),
                                           type=BaseType(name='INTEGER', kind='JPIM')))


def insert_claw_directives(routine, driver, claw_scalars, target):
    """
    Insert the necessary pragmas and directives to instruct the CLAW.

    Note: Must be run after generic SCA conversion.
    """
    from loki import FortranCodegen

    # Insert loop pragmas in driver (in-place)
    for loop in FindNodes(Loop).visit(driver.ir):
        if loop.variable == target.variable:
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
    for p in FindNodes(Pragma).visit(routine.ir):
        if p.keyword.lower() == 'omp':
            if p.content.startswith('do') or p.content.startswith('end do'):
                mapper[p] = None
    routine._ir = Transformer(mapper).visit(routine.ir)


@click.group()
def cli():
    pass


@cli.command('idem')
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--source-out', '-so', type=click.Path(),
              help='Path for generated source output.')
@click.option('--driver', '-d', type=click.Path(), default=None,
              help='Driver file to convert.')
@click.option('--driver-out', '-do', type=click.Path(), default=None,
              help='Path for generated driver output.')
@click.option('--typedef', '-t', type=click.Path(), multiple=True,
              help='Path for additional soUrce file(s) containing type definitions')
@click.option('--flatten-args/--no-flatten-args', default=True,
              help='Flag to trigger derived-type argument unrolling')
@click.option('--openmp/--no-openmp', default=False,
              help='Flag to force OpenMP pragmas onto existing horizontal loops')
def idempotence(source, source_out, driver, driver_out, typedef, flatten_args, openmp):
    """
    Idempotence: A "do-nothing" debug mode that performs a parse-and-unparse cycle.
    """
    typedefs = get_typedefs(typedef)

    # Parse original driver and kernel routine, and enrich the driver
    routine = FortranSourceFile(source, typedefs=typedefs).subroutines[0]
    driver = FortranSourceFile(driver).subroutines[0]
    driver.enrich_calls(routines=routine)

    class IdemTransformation(BasicTransformation):
        """
        Here we define a custom transformation pipeline to re-generate
        separate kernel and driver versions, adding some optional changes
        like derived-type argument removal or an experimental low-level
        OpenMP wrapping.
        """

        def _pipeline(self, routine, **kwargs):
            # Define the horizontal dimension
            horizontal = Dimension(name='KLON', aliases=['NPROMA', 'KDIM%KLON'],
                                   variable='JL', iteration=('KIDIA', 'KFDIA'))

            if openmp:
                # Experimental OpenMP loop pragma insertion
                for loop in FindNodes(Loop).visit(routine.ir):
                    if loop.variable == horizontal.variable:
                        # Update the loop in-place with new OpenMP pragmas
                        pragma = Pragma(keyword='omp', content='do simd')
                        pragma_nowait = Pragma(keyword='omp',
                                               content='end do simd nowait')
                        loop._update(pragma=pragma, pragma_post=pragma_nowait)

            # Perform necessary housekeeking tasks
            self.rename_routine(routine, suffix='IDEM')
            self.write_to_file(routine, **kwargs)

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
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--source-out', '-so', type=click.Path(),
              help='Path for generated source output.')
@click.option('--driver', '-d', type=click.Path(), default=None,
              help='Driver file to convert.')
@click.option('--driver-out', '-do', type=click.Path(), default=None,
              help='Path for generated driver output.')
@click.option('--typedef', '-t', type=click.Path(), multiple=True,
              help='Path for additional source file(s) containing type definitions')
@click.option('--strip-omp-do', is_flag=True, default=False,
              help='Removes existing !$omp do loop pragmas')
@click.option('--mode', '-m', type=click.Choice(['sca', 'claw']), default='sca')
def convert(source, source_out, driver, driver_out, typedef, strip_omp_do, mode):
    """
    Single Column Abstraction (SCA): Convert kernel into single-column
    format and adjust driver to apply it over in a horizontal loop.

    Optionally, this can also insert CLAW directives that may be use
    for further downstream transformations.
    """
    typedefs = get_typedefs(typedef)

    # Parse original kernel routine and inject type definitions
    routine = FortranSourceFile(source, typedefs=typedefs).subroutines[0]
    driver = FortranSourceFile(driver, typedefs=typedefs).subroutines[0]
    driver.enrich_calls(routines=routine)

    if mode == 'claw':
        claw_scalars = [v.name.lower() for v in routine.variables
                        if len(v.dimensions) == 1]

    # Debug addition: detect calls to `ref_save` and replace with `ref_error`
    for call in FindNodes(Call).visit(routine.ir):
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


class InferArgShapeTransformation(AbstractTransformation):
    """
    Uses IPA context information to infer the shape of arguments from
    the caller.
    """

    def _pipeline(self, routine, **kwargs):

        for call in FindNodes(Call).visit(routine.ir):
            if call.context is not None and call.context.active:

                # Insert shapes of call values into routine arguments
                for arg, val in call.context.arg_iter(call):
                    if arg.dimensions is not None and len(arg.dimensions) > 0:
                        if all(d == ':' for d in arg.dimensions):
                            if len(val.shape) == len(arg.dimensions):
                                # We're passing the full value array, copy shape
                                arg.dimensions = val.shape
                            else:
                                # Passing a sub-array of val, find the right index
                                idx = [s for s, d in zip(val.shape, val.dimensions)
                                       if d == ':']
                                arg.dimensions = as_tuple(idx)

                # TODO: The derived call-side dimensions can be undefined in the
                # called routine, so we need to add them to the call signature.

                # Propagate the new shape through the IR
                call.context.routine._derive_variable_shape()


class RapsTransformation(BasicTransformation):
    """
    Dedicated housekeeping pipeline for dealing with module wrapping
    and dependency management for RAPS.
    """

    def __init__(self, raps_deps=None, loki_deps=None, basepath=None):
        self.raps_deps = raps_deps
        self.loki_deps = loki_deps
        self.basepath = basepath

    def _pipeline(self, routine, **kwargs):
        task = kwargs.get('task')
        mode = task.config['mode']
        role = task.config['role']
        # TODO: Need enrichment for Imports to get rid of this!
        processor = kwargs.get('processor', None)

        info('Processing %s (role=%s, mode=%s)' % (routine.name, role, mode))

        original = routine.name
        if role == 'kernel':
            self.rename_routine(routine, suffix=mode.upper())

        self.rename_calls(routine, suffix=mode.upper())
        self.adjust_imports(routine, mode, processor)

        filename = routine.sourcefile.path.with_suffix('.%s.F90' % mode)
        if role == 'driver':
            self.write_to_file(routine, filename=filename, module_wrap=False)
        else:
            self.write_to_file(routine, filename=filename, module_wrap=True)

        self.adjust_dependencies(original=original, task=task, processor=processor)

    def adjust_imports(self, routine, mode, processor):
        """
        Utility routine to rename all calls to relevant subroutines and
        adjust the relevant imports.

        Note: This will always assume that any non-blacklisted routines
        will be re-generated wrapped in a module with the naming
        convention ``<KERNEL>_<MODE>_MOD``.
        """
        replacements = {}

        # Update all relevant interface abd module imports
        new_imports = []
        for im in FindNodes(Import).visit(routine.ir):
            for r in processor.routines:
                if im.c_import and r.name.lower() == im.module.split('.')[0]:
                    replacements[im] = None  # Drop old C-style import
                    new_imports += [Import(module='%s_%s_MOD' % (r.name.upper(), mode.upper()),
                                           symbols=['%s_%s' % (r.name.upper(), mode.upper())])]
                elif not im.c_import and r.name.lower() == im.module.lower():
                    # Hacky-ish: The above use of 'in' assumes we always use _MOD in original
                    replacements[im] = Import(module='%s_%s_MOD' % (r.name.upper(), mode.upper()),
                                              symbols=['%s_%s' % (r.name.upper(), mode.upper())])

        # Insert new declarations and transform existing ones
        routine.spec.prepend(new_imports)
        routine._ir = Transformer(replacements).visit(routine.ir)

    def adjust_dependencies(self, original, task, processor):
        """
        Utility routine to generate Loki-specific build dependencies from
        RAPS-generated dependency files.

        Hack alert: The two dependency files are stashed on the transformation
        and modified on-the-fly. This is required to get selective replication
        until we have some smarter dependency generation and globbing support.
        """
        mode = task.config['mode']
        whitelist = task.config['whitelist']
        original = original.lower()
        sourcepath = task.path

        # Adjust the object entry in the dependency file
        f_path = sourcepath.relative_to(self.basepath)
        f_mode_path = f_path.with_suffix('.%s.F90' % mode)
        o_path = f_path.with_suffix('.o')
        o_mode_path = o_path.with_suffix('.%s.o' % mode)

        r_deps = deepcopy(self.raps_deps.content_map[str(o_path)])
        r_deps.replace(str(o_path), str(o_mode_path))
        r_deps.replace(str(f_path), str(f_mode_path))
        self.loki_deps.content += [r_deps]

        # Run through all previous dependencies and replace any
        # interface entries (.ok) or previous module entries ('.o')
        # for the current target with a dependency on the newly
        # created module.
        # TODO: Inverse traversal might help here..?
        for d in self.loki_deps.content:
            if isinstance(d, Dependency) and str(o_mode_path) not in d.target:
                intfb = d.find('%s.intfb.ok' % original)
                if intfb is not None:
                    d.replace(intfb[0], str(o_mode_path))
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
@click.option('--typedef', '-t', type=click.Path(), multiple=True,
              help='Path for additional source file(s) containing type definitions')
@click.option('--raps-dependencies', '-deps', type=click.Path(), default=None,
              help='Path to RAPS-generated dependency file')
@click.option('--callgraph', '-cg', is_flag=True, default=False,
              help='Generate and display the subroutine callgraph.')
def physics(config, basepath, source, typedef, raps_dependencies, callgraph):
    """
    Physics bulk-processing option that employs a :class:`TaskScheduler` to apply
    source-to-source transformations, such as the Single Column Abstraction (SCA),
    to large sets of interdependent subroutines.
    """
    # Get external derived-type definitions
    typedefs = get_typedefs(typedef)

    # Load configuration file and process options
    with Path(config).open('r') as f:
        config = toml.load(f)

    # Convert 'routines' to an ordered dictionary
    config['routines'] = OrderedDict((r['name'], r) for r in config['routine'])

    # Create and setup the scheduler for bulk-processing
    scheduler = TaskScheduler(paths=source, config=config, typedefs=typedefs)
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
