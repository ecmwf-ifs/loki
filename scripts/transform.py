import click
import toml
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

from loki import (FortranSourceFile, Visitor, Loop, Variable,
                  Declaration, FindNodes, Call, Pragma, fgen,
                  BaseType, Module, info, DerivedType,
                  ExpressionVisitor, Transformer, Import, as_tuple,
                  FindVariables, Index, Allocation)

from raps_deps import RapsDependencyFile, Rule
from scheduler import Scheduler


class VariableTransformer(ExpressionVisitor, Visitor):
    """
    In-place transformer that applies string replacements
    :class:`Variable`s.
    """

    def __init__(self, argnames):
        super(VariableTransformer, self).__init__()
        self.argnames = argnames

    def visit_Variable(self, o):
        if o.name in self.argnames and o.subvar is not None:
            # HACK: In-place merging of var with subvar
            o.name = '%s_%s' % (o.name, o.subvar.name)
            o._type = o.subvar._type
            o.dimensions = o.subvar.dimensions
            o.initial = o.subvar.initial
            o.subvar = o.subvar.subvar


def get_typedefs(typedef):
    # Read derived type definitions from typedef modules
    definitions = {}
    for tfile in typedef:
        module = FortranSourceFile(tfile).modules[0]
        definitions.update(module.typedefs)
    return definitions


def flatten_derived_arguments(routine, driver, candidate_routines):
    """
    Unroll all derived-type arguments used in the subroutine signature,
    declarations and body, as well as all call arguments for a
    particular driver/kernel pair.

    The convention used is simply: ``derived%var => derived_var``
    """
    variables = FindVariables().visit(routine.ir)

    # Establish candidate sub-variables based on usage in the
    # kernel routine. These are stored in a map of
    # ``arg => [type_vars]``, where ``arg%type_var`` is to be
    # replaced by ``arg_type_var``
    candidates = defaultdict(list)
    for arg in routine.arguments:
        if isinstance(arg.type, DerivedType):
            for type_var in arg.type.variables.values():
                combined = '%s%%%s' % (arg.name, type_var.name)
                if any(combined in str(v) for v in variables):
                    candidates[arg] += [type_var]

    # Caller: Tandem-walk the argument lists of the kernel for each call
    call_mapper = {}
    for call in FindNodes(Call).visit(driver._ir):
        if call.name.lower() in (r.lower() for r in candidate_routines):

            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            # Simultaneously walk driver and kernel arguments
            new_arguments = list(deepcopy(call.arguments))
            for d_arg, k_arg in zip(call.arguments, routine.arguments):
                if k_arg in candidates:
                    # Found derived-type argument, unroll according to candidate map
                    new_args = []
                    for type_var in candidates[k_arg]:
                        # Insert `:` range dimensions into newly generated args
                        new_dims = tuple(Index(name=':') for _ in type_var.dimensions)
                        new_arg = deepcopy(d_arg)
                        new_arg.subvar = Variable(name=type_var.name, dimensions=new_dims)
                        new_args += [new_arg]

                    # Replace variable in dummy signature
                    i = new_arguments.index(d_arg)
                    new_arguments[i:i+1] = new_args

            # Set the new call signature on the IR ndoe
            call_mapper[call] = call.clone(arguments=new_arguments)

    driver._ir = Transformer(call_mapper).visit(driver.ir)

    # Callee: Establish replacements for declarations and dummy arguments
    declarations = FindNodes(Declaration).visit(routine.ir)
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
                               dimensions=type_var.dimensions)
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


# Define the target dimension to strip from kernel and caller
target = Dimension(name='KLON', aliases=['NPROMA'],
                   variable='JL', iteration=('KIDIA', 'KFDIA'))
target_routine = 'CLOUDSC'


def remove_dimension(routine, target):
    """
    Remove all loops and variable indices of a given target dimension
    from the given routine.
    """
    size_expressions = target.size_expressions
    index_expressions = target.index_expressions

    # Remove all loops over the target dimensions
    loop_map = {}
    for loop in FindNodes(Loop).visit(routine.ir):
        if loop.variable == target.variable:
            loop_map[loop] = loop.body
    routine._ir = Transformer(loop_map).visit(routine.ir)

    # Remove all variable indices representing the target dimension (in-place)
    variable_map = routine.variable_map
    for v in FindVariables(unique=False).visit(routine.ir):
        if v.dimensions is not None:
            v.dimensions = as_tuple(d for d in v.dimensions
                                    if str(d) not in index_expressions)

        # Remove implicit ranges (`:`) by checking against the allocated dimensions
        if v.name not in variable_map:
            continue
        declared_dims = variable_map[v.name].dimensions
        if declared_dims is not None:
            v.dimensions = as_tuple(dim for dd, dim in zip(declared_dims, v.dimensions)
                                    if not (dim == ':' and str(dd) in size_expressions))

    # Remove dimension size expressions from variable declarations (in-place)
    # Note: We do this last, because changing the declaration affects
    # the variable_map used above.
    for decl in FindNodes(Declaration).visit(routine.ir):
        for v in decl.variables:
            if v.dimensions is not None:
                v.dimensions = as_tuple(d for d in v.dimensions
                                        if str(d) not in size_expressions)

    # Drop the declaration for the loop counter variable
    for decl in FindNodes(Declaration).visit(routine.ir):
        if target.variable in decl.variables:
            decl.variables.remove(target.variable)
        if len(decl.variables) == 0:
            # Drop the loop counter declaration
            routine._ir = Transformer({decl: None}).visit(routine.ir)


def hoist_dimension_from_call(routine, driver, candidate_routines, wrap=True):
    """
    Remove all indices and variables of a target dimension from
    caller (driver) and callee (kernel) routines, and insert the
    necessary loop over the target dimension into the driver.

    Note: In order for this routine to see the target dimensions
    in the argument declarations of the kernel, it must be applied
    before they are stripped from the kernel itself.
    """
    size_expressions = target.size_expressions
    vmap = driver.variable_map
    replacements = {}

    # Create map of variable names to their "true dimensions",
    # that is either their declared or the allocated dimensions.
    vdims = {}
    for v in driver.variables:
        if v.dimensions is not None and len(v.dimensions) > 0:
            vdims[v.name] = v.dimensions
        # Quick and dirty hack...
        if isinstance(v.type, DerivedType):
            for tv in v.type.variables.values():
                vdims[tv.name] = tv.dimensions
    for alloc in FindNodes(Allocation).visit(driver.ir):
        vdims[alloc.variable.name] = alloc.variable.dimensions

    for call in FindNodes(Call).visit(driver.ir):
        if call.name.lower() in (r.lower() for r in candidate_routines):

            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            # Replace target dimension with a loop index in arguments
            for darg, karg in zip(call.arguments, routine.arguments):
                if not isinstance(darg, Variable):
                    continue

                if darg.name in vdims:
                    # The "template" is the list of dimensions originally
                    # declared or allocated for this (sub-)variable.
                    template = vdims[darg.name]

                    # Skip to the innermost compound variable
                    while darg.subvar is not None:
                        type_vars = vmap[darg.name.upper()].type.variables
                        template = type_vars[darg.subvar.name].dimensions
                        darg = darg.subvar

                    # Remove dimension from caller-side argument indices
                    new_dims = tuple(Index(name=target.variable)
                                     if str(tdim) in size_expressions else ddim
                                     for ddim, tdim in zip(darg.dimensions, template))
                    darg.dimensions = new_dims

                # Super-hacky: infer dimensions for compound variables
                if darg.subvar is not None and darg.subvar.name in vdims:
                    darg = darg.subvar
                    template = vdims[darg.name]

                    # Remove dimension from caller-side argument indices
                    new_dims = tuple(Index(name=target.variable)
                                     if str(tdim) in size_expressions else ddim
                                     for ddim, tdim in zip(darg.dimensions, template))
                    darg.dimensions = new_dims


            # Collect caller-side expressions for dimension sizes and bounds
            dim_lower = None
            dim_upper = None
            for darg, karg in zip(call.arguments, routine.arguments):
                if karg == target.iteration[0]:
                    dim_lower = darg
                if karg == target.iteration[1]:
                    dim_upper = darg

            # Remove call-side arguments (in-place)
            arguments = tuple(darg for darg, karg in zip(call.arguments, routine.arguments)
                              if karg not in target.variables)
            new_call = call.clone(arguments=arguments)

            # Create and insert new loop over target dimension
            if wrap:
                loop = Loop(variable=Variable(name=target.variable),
                        bounds=(dim_lower, dim_upper), body=new_call)
                replacements[call] = loop
            else:
                replacements[call] = new_call

            # Remove kernel-side arguments from signature and declarations
            routine._argnames = tuple(arg for arg in routine.argnames
                                      if arg not in target.variables)
            routine_replacements = {}
            for decl in FindNodes(Declaration).visit(routine.ir):
                decl.variables = tuple(v for v in decl.variables
                                       if v not in target.variables)
                if len(decl.variables) == 0:
                    routine_replacements[decl] = None
            routine._ir = Transformer(routine_replacements).visit(routine.ir)

    # Finally, add declaration of loop variable (a little hacky!)
    if wrap and target.variable not in driver.variables:
        decls = FindNodes(Declaration).visit(driver.ir)
        new_decl = Declaration(variables=[Variable(name=target.variable)],
                               type=BaseType(name='INTEGER', kind='JPIM'))
        replacements[decls[-1]] = [deepcopy(decls[-1]), new_decl]
    driver._ir = Transformer(replacements).visit(driver.ir)


def insert_claw_directives(routine, driver, claw_scalars, target):
    """
    Insert the necessary pragmas and directives to instruct the CLAW.

    Note: Must be run after generic SCA conversion.
    """

    # Insert loop pragmas in driver (in-place)
    for loop in FindNodes(Loop).visit(driver.ir):
        if loop.variable == target.variable:
            loop.pragma = Pragma(keyword='claw', content='parallelize forward create update')

    # Generate CLAW directives
    directives = [Pragma(keyword='claw', content='define dimension jl(1:nproma) &'),
                  Pragma(keyword='claw', content='parallelize &'),
                  Pragma(keyword='claw', content='scalar(%s)\n\n\n' % ', '.join(claw_scalars))]

    # Insert directives into driver (HACK!)
    decls = FindNodes(Declaration).visit(routine.ir)
    replacements = {decls[-1]: [deepcopy(decls[-1])] + directives}
    routine._ir = Transformer(replacements).visit(routine.ir)


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
def idempotence(source, source_out, driver, driver_out, typedef, flatten_args):
    """
    Idempotence: A "do-nothing" debug mode that performs a parse-and-unparse cycle.
    """
    typedefs = get_typedefs(typedef)

    # Parse original kernel routine and update the name
    f_source = FortranSourceFile(source, typedefs=typedefs)
    routine = f_source.subroutines[0]
    routine.name = '%s_IDEM' % routine.name

    # Parse the original driver (caller)
    f_driver = FortranSourceFile(driver)
    driver = f_driver.subroutines[0]

    # Unroll derived-type arguments into multiple arguments
    if flatten_args:
        flatten_derived_arguments(routine, driver, candidate_routines=[target_routine])

    # Replace the non-reference call in the driver for evaluation
    for call in FindNodes(Call).visit(driver._ir):
        if call.name == target_routine:
            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            call.name = '%s_IDEM' % call.name

    # Re-generate the target routine with the updated name
    module = Module(name='%s_MOD' % routine.name.upper(), routines=[routine])
    f_source.write(source=fgen(module), filename=source_out)

    # Insert new module import into the driver and re-generate
    new_import = Import(module='%s_MOD' % routine.name.upper(),
                        symbols=[routine.name.upper()])
    driver._ir = tuple([new_import] + list(driver._ir))
    f_driver.write(source=fgen(driver), filename=driver_out)


@cli.command()
@click.option('--source', '-s', type=click.Path(),
              help='Source file to convert.')
@click.option('--source-out', '-so', type=click.Path(),
              help='Path for generated source output.')
@click.option('--driver', '-d', type=click.Path(), default=None,
              help='Driver file to convert.')
@click.option('--driver-out', '-do', type=click.Path(), default=None,
              help='Path for generated driver output.')
@click.option('--interface', '-intfb', type=click.Path(), default=None,
              help='Path to auto-generate and interface file')
@click.option('--typedef', '-t', type=click.Path(), multiple=True,
              help='Path for additional source file(s) containing type definitions')
@click.option('--mode', '-m', type=click.Choice(['sca', 'claw']), default='sca')
def convert(source, source_out, driver, driver_out, interface, typedef, mode):
    """
    Single Column Abstraction: Convert kernel into SCA format and adjust driver.
    """
    # convert_sca(source, source_out, driver, driver_out, interface, typedef, mode)
    typedefs = get_typedefs(typedef)

    # Parse original kernel routine and inject type definitions
    f_source = FortranSourceFile(source, typedefs=typedefs)
    routine = f_source.subroutines[0]

    # Parse the original driver (caller)
    f_driver = FortranSourceFile(driver, typedefs=typedefs)
    driver = f_driver.subroutines[0]

    if mode == 'claw':
        claw_scalars = [v.name.lower() for v in routine.arguments
                        if len(v.dimensions) == 1]

    # Remove horizontal dimension from kernels and hoist loop to
    # driver to transform a subroutine invocation to SCA format.
    flatten_derived_arguments(routine, driver, candidate_routines=[target_routine])
    hoist_dimension_from_call(routine, driver, candidate_routines=[target_routine])
    remove_dimension(routine=routine, target=target)

    if mode == 'claw':
        insert_claw_directives(routine, driver, claw_scalars, target)

    # Update the name of the kernel routine
    routine.name = '%s_%s' % (routine.name, mode.upper())

    # Update the names of all non-reference calls in the driver
    for call in FindNodes(Call).visit(driver._ir):
        if call.name == target_routine:
            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            call.name = '%s_%s' % (call.name, mode.upper())

    # Re-generate the target routine with the updated name
    module = Module(name='%s_MOD' % routine.name.upper(), routines=[routine])
    f_source.write(source=fgen(module), filename=source_out)

    # Insert new module import into the driver and re-generate
    new_import = Import(module='%s_MOD' % routine.name.upper(),
                        symbols=[routine.name.upper()])
    driver._ir = tuple([new_import] + list(driver._ir))
    f_driver.write(source=fgen(driver), filename=driver_out)


def adjust_calls_and_imports(routine, mode, processor):
    """
    Utility routine to rename all calls to relevant subroutines and
    adjust the relevant imports.

    Note: This will always assume that any non-blacklisted routines
    will be re-generated wrapped in a module with the naming
    convention ``<KERNEL>_<MODE>_MOD``.
    """
    replacements = {}

    # Replace the non-reference call in the driver for evaluation
    for call in FindNodes(Call).visit(routine.ir):
        if call.name.lower() in (r.name for r in processor.routines):
            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            # Re-generate the call IR node with a new name
            replacements[call] = call.clone(name='%s_%s' % (call.name, mode.upper()))

    # Update all relevant interface imports
    new_imports = []
    for im in FindNodes(Import).visit(routine.ir):
        for r in processor.routines:
            if im.c_import and r.name.lower() == im.module.split('.')[0]:
                replacements[im] = None  # Drop old C-style import
                new_imports += [Import(module='%s_%s_MOD' % (r.name.upper(), mode.upper()),
                                       symbols=['%s_%s' % (r.name.upper(), mode.upper())])]

    # A hacky way to insert new imports at the end of module imports
    last_module = [im for im in FindNodes(Import).visit(routine.ir)
                   if not im.c_import][-1]
    replacements[last_module] = [deepcopy(last_module)] + new_imports

    routine._ir = Transformer(replacements).visit(routine.ir)


def adjust_dependencies(original, config, processor):
    """
    Utility routine to generate Loki-specific build dependencies from
    RAPS-generated dependency files.

    Hack alert: The two dependency files are stashed in the `config` and
    modified on-the-fly. This is required to get selective replication
    until we have some smarter dependency generation and globbing support.
    """
    mode = config['default']['mode']
    whitelist = config['default']['whitelist']
    original = original.lower()

    # Adjust the object entry in the dependency file
    orig_path = 'ifs/phys_ec/%s.o' % original
    r_deps = deepcopy(config['raps_deps'].content_map[orig_path])
    r_deps.replace('ifs/phys_ec/%s.o' % original,
                   'ifs/phys_ec/%s.%s.o' % (original, mode))
    r_deps.replace('ifs/phys_ec/%s.F90' % original,
                   'ifs/phys_ec/%s.%s.F90' % (original, mode))
    for r in processor.routines:
        routine = r.name.lower()
        r_deps.replace('flexbuild/raps17/intfb/ifs/%s.intfb.ok' % routine,
                       'ifs/phys_ec/%s.%s.o' % (routine, mode))
    config['loki_deps'].content += [r_deps]

    # Inject new object into the final binary libs
    objs_ifsloki = config['loki_deps'].content_map['OBJS_ifsloki']
    if original in whitelist:
        # Add new dependency inplace, next to the old one
        objs_ifsloki.append_inplace('ifs/phys_ec/%s.o' % original,
                                    'ifs/phys_ec/%s.%s.o' % (original, mode))
    else:
        # Replace old dependency to avoid ghosting where possible
        objs_ifsloki.replace('ifs/phys_ec/%s.o' % original,
                             'ifs/phys_ec/%s.%s.o' % (original, mode))


def physics_idem_kernel(source_file, config=None, processor=None):
    """
    Processing method for kernel routines that recreates a mode-specific
    version of the kernel and swaps out all subroutine calls.
    """
    mode = config['default']['mode']
    routine = source_file.subroutines[0]

    info('Processing kernel: %s, mode=%s' % (routine.name, mode))

    # Rename kernel routine to mode-specific variant
    original = routine.name
    routine.name = '%s_%s' % (routine.name, mode.upper())

    # Housekeeping for injecting re-generated routines into the build
    adjust_calls_and_imports(routine, mode, processor)

    # Generate mode-specific kernel subroutine with module wrappers
    module = Module(name='%s_MOD' % routine.name.upper(), routines=[routine])
    source_file.write(source=fgen(module),
                      filename=source_file.path.with_suffix('.idem.F90'))

    # Add dependencies for newly created source files into RAPS build
    adjust_dependencies(original=original, config=config, processor=processor)


def physics_idem_driver(source, config, processor):
    """
    Processing method for driver (root) routines that creates a clone
    of the driver routine in the usual style and swaps out all
    relevant subroutine calls.

    Note: We do not change the driver routine's name and we do not
    module-wrap it either. This way it can be slotted into a build
    as a straight replacement via object dependencies.
    """
    mode = config['default']['mode']
    driver = source.subroutines[0]

    info('Processing driver: %s, mode=%s' % (driver.name, mode))

    # Housekeeping for injecting re-generated routines into the build
    adjust_calls_and_imports(driver, mode, processor)

    # Re-generate updated driver subroutine (do not module-wrap!)
    source.write(source=fgen(driver),
                 filename=source.path.with_suffix('.%s.F90' % mode))

    # Add dependencies for newly created source files into RAPS build
    adjust_dependencies(original=driver.name, config=config, processor=processor)


def physics_sca_kernel(source_file, config=None, processor=None):
    """
    Processing method to convert kernel routines into Single Column Abstract (SCA)
    format by removing the horizontal dimension (KLON/NPROMA) from the subroutine.
    """
    mode = config['default']['mode']
    routine = source_file.subroutines[0]

    info('Processing kernel: %s, mode=%s' % (routine.name, mode))

    # Rename kernel routine to mode-specific variant
    original = routine.name
    routine.name = '%s_%s' % (routine.name, mode.upper())

    # Perform caller-side transformations for all children in the callgraph
    for call in FindNodes(Call).visit(routine.ir):
        if call.name.lower() in (r.name for r in processor.routines):
            child = processor.item_map[call.name.lower()].routine

            # Apply inter-procedural part of the conversion
            flatten_derived_arguments(routine=child, driver=routine,
                                      candidate_routines=[r.name for r in processor.routines])

            # Hoist dimension, but do not wrap in new loop
            hoist_dimension_from_call(routine=child, driver=routine, wrap=False,
                                      candidate_routines=[r.name for r in processor.routines])

    # Remove the target dimension from all loops and variables
    remove_dimension(routine=routine, target=target)

    # Housekeeping for injecting re-generated routines into the build
    adjust_calls_and_imports(routine, mode, processor)

    # Generate mode-specific kernel subroutine with module wrappers
    module = Module(name='%s_MOD' % routine.name.upper(), routines=[routine])
    source_file.write(source=fgen(module),
                      filename=source_file.path.with_suffix('.sca.F90'))

    # Add dependencies for newly created source files into RAPS build
    adjust_dependencies(original=original, config=config, processor=processor)


def physics_sca_driver(source, config, processor):
    """
    Processing method for driver (root) routines for SCA
    """
    mode = config['default']['mode']
    driver = source.subroutines[0]

    info('Processing driver: %s, mode=%s' % (driver.name, mode))

    for call in FindNodes(Call).visit(driver.ir):
        if call.name.lower() in (r.name for r in processor.routines):
            routine = processor.item_map[call.name.lower()].routine

            # Apply inter-procedural part of the conversion
            flatten_derived_arguments(routine, driver,
                                      candidate_routines=[r.name for r in processor.routines])
            hoist_dimension_from_call(routine, driver, wrap=True,
                                      candidate_routines=[r.name for r in processor.routines])

    # Housekeeping for injecting re-generated routines into the build
    adjust_calls_and_imports(driver, mode, processor)

    # Re-generate updated driver subroutine (do not module-wrap!)
    source.write(source=fgen(driver),
                 filename=source.path.with_suffix('.%s.F90' % mode))

    # Add dependencies for newly created source files into RAPS build
    adjust_dependencies(original=driver.name, config=config, processor=processor)


@cli.command('physics')
@click.option('--config', '-cfg', type=click.Path(),
              help='Path to configuration file.')
@click.option('--source', '-s', type=click.Path(),
              help='Path to source files to transform.')
@click.option('--typedef', '-t', type=click.Path(), multiple=True,
              help='Path for additional source file(s) containing type definitions')
@click.option('--raps-dependencies', '-deps', type=click.Path(), default=None,
              help='Path to RAPS-generated dependency file')
@click.option('--callgraph', '-cg', is_flag=True, default=False,
              help='Generate and display the subroutine callgraph.')
def physics(config, source, typedef, raps_dependencies, callgraph):

    kernel_map = {'noop': {'driver': None, 'kernel': None},
                  'idem': {'driver': physics_idem_driver,
                           'kernel': physics_idem_kernel},
                  'sca': {'driver': physics_sca_driver,
                          'kernel': physics_sca_kernel}}

    # Get external derived-type definitions
    typedefs = get_typedefs(typedef)

    # Load configuration file and process options
    with Path(config).open('r') as f:
        config = toml.load(f)

    # Convert 'routines' to an ordered dictionary
    config['routines'] = OrderedDict((r['name'], r) for r in config['routine'])

    if raps_dependencies:
        # Load RAPS dependency file for injection into the build system
        config['raps_deps'] = RapsDependencyFile.from_file(raps_dependencies)

        # Create new deps file with lib dependencies and a build rule
        objs_ifsloki = deepcopy(config['raps_deps'].content_map['OBJS_ifs'])
        objs_ifsloki.target = 'OBJS_ifsloki'
        rule_ifsloki = Rule(target='$(BMDIR)/libifsloki.a', deps=['$(OBJS_ifsloki)'],
                            cmds=['/bin/rm -f $@', 'ar -cr $@ $^', 'ranlib $@'])
        config['loki_deps'] = RapsDependencyFile(content=[objs_ifsloki, rule_ifsloki])

    # Create and setup the bulk source processor
    processor = Scheduler(path=source, config=config, kernel_map=kernel_map, typedefs=typedefs)
    processor.append(config['routines'].keys())

    # Add explicitly blacklisted subnodes
    if 'blacklist' in config['default']:
        processor.blacklist += [b.upper() for b in config['default']['blacklist']]
    for opts in config['routines'].values():
        if 'blacklist' in opts:
            processor.blacklist += [b.upper() for b in opts['blacklist']]

    # Execute and extract the resulting callgraph
    processor.process(discovery=False)
    if callgraph:
        processor.graph.render('callgraph', view=False)

    if raps_dependencies:
        # Write new mode-specific dependency rules file
        loki_config_path = config['raps_deps'].path.with_suffix('.loki.def')
        config['loki_deps'].write(path=loki_config_path)


if __name__ == "__main__":
    cli()
