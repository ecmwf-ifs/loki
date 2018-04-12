import click
import toml
import re
from collections import OrderedDict, defaultdict, Iterable, deque
from copy import deepcopy
import sys
from pathlib import Path
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

from loki import (FortranSourceFile, Visitor, flatten, chunks, Loop,
                  Variable, TypeDef, Declaration, FindNodes,
                  Statement, Call, Pragma, fgen, BaseType, Source,
                  Module, info, DerivedType, ExpressionVisitor,
                  Transformer, Import, warning, as_tuple, error, debug,
                  FindVariables, Index, Allocation, BaseType, Pragma)

from raps_deps import RapsDependencyFile, Rule


class SourceProcessor(object):
    """
    Work queue manager to enqueue and process individual source
    routines/modules with a given kernel.

    Note: The processing module can create a callgraph and perform
    automated discovery, to enable easy bulk-processing of large
    numbers of source files.
    """

    blacklist = ['DR_HOOK', 'ABOR1']

    def __init__(self, path, config=None, kernel_map=None):
        self.path = Path(path)
        self.config = config
        self.kernel_map = kernel_map

        self.queue = deque()
        self.processed = []

        if Digraph is not None:
            self.graph = Digraph(format='pdf', strict=True)
        else:
            self.graph = None

    @property
    def routines(self):
        return list(self.processed) + list(self.queue)

    def append(self, sources):
        """
        Add names of source routines or modules to find and process.
        """
        sources = as_tuple(sources)
        self.queue.extend(s for s in sources if s not in self.routines)

    def process(self, discovery=False):
        """
        Process all enqueued source modules and routines with the
        stored kernel.
        """
        while len(self.queue) > 0:
            source = self.queue.popleft()
            source_path = (self.path / source).with_suffix('.F90')

            if source_path.exists():
                try:
                    config = self.config['default'].copy()
                    if source in self.config['routines']:
                        config.update(self.config['routines'][source])

                    # Re-generate target routine and interface block with updated name
                    source_file = FortranSourceFile(source_path, preprocess=True)
                    routine = source_file.subroutines[0]

                    debug("Source: %s, config: %s" % (source, config))

                    if self.graph:
                        if routine.name.lower() in config['whitelist']:
                            self.graph.node(routine.name, color='black', shape='diamond',
                                            fillcolor='limegreen', style='rounded,filled')
                        else:
                            self.graph.node(routine.name, color='black',
                                            fillcolor='limegreen', style='filled')

                    for call in FindNodes(Call).visit(routine.ir):
                        # Yes, DR_HOOK is that(!) special
                        if self.graph and call.name not in ['DR_HOOK', 'ABOR1']:
                            self.graph.edge(routine.name, call.name)
                            if call.name.upper() in self.blacklist:
                                self.graph.node(call.name, color='black',
                                                fillcolor='orangered', style='filled')
                            elif call.name.lower() not in self.processed:
                                self.graph.node(call.name, color='black',
                                                fillcolor='lightblue', style='filled')

                        if call.name.upper() in self.blacklist:
                            continue

                        if config['expand']:
                            self.append(call.name.lower())

                    # Apply the user-defined kernel
                    kernel = self.kernel_map[config['mode']][config['role']]

                    if kernel is not None:
                        kernel(source_file, config=self.config, processor=self)

                    self.processed.append(source)

                except Exception as e:
                    if self.graph:
                        self.graph.node(source.upper(), color='red', style='filled')
                    warning('Could not parse %s:' % source)
                    if config['strict']:
                        raise e
                    else:
                        error(e)

            else:
                if self.graph:
                    self.graph.node(source.upper(), color='lightsalmon', style='filled')
                info("Could not find source file %s; skipping..." % source)


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


def flatten_derived_arguments(routine, driver):
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
    for call in FindNodes(Call).visit(driver._ir):
        if call.name == target_routine:
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
            call.arguments = tuple(new_arguments)

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


def hoist_dimension_from_call(routine, driver):
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
    for alloc in FindNodes(Allocation).visit(driver.ir):
        vdims[alloc.variable.name] = alloc.variable.dimensions

    for call in FindNodes(Call).visit(driver.ir):
        if call.name == target_routine:
            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            # Replace target dimension with a loop index in arguments
            for darg, karg in zip(call.arguments, routine.arguments):
                if isinstance(darg, Variable) and darg.name in vdims:
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

            # Collect caller-side expressions for dimension sizes and bounds
            dim_lower = None
            dim_upper = None
            for darg, karg in zip(call.arguments, routine.arguments):
                if karg == target.iteration[0]:
                    dim_lower = darg
                if karg == target.iteration[1]:
                    dim_upper = darg

            # Create and insert new loop over target dimension
            loop = Loop(variable=Variable(name=target.variable),
                        bounds=(dim_lower, dim_upper), body=call)
            replacements[call] = loop

            # Remove call-side arguments (in-place)
            call.arguments = tuple(darg for darg, karg in zip(call.arguments, routine.arguments)
                                   if karg not in target.variables)

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
        flatten_derived_arguments(routine, driver)

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
@click.option('--mode', '-m', type=click.Choice(['sca', 'claw', 'idem']), default='sca')
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
    flatten_derived_arguments(routine, driver)
    hoist_dimension_from_call(routine, driver)
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


def adjust_dependencies(original, config, processor, interface=False):
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
    r_deps.replace('ifs/phys_ec/%s.F90' % original,
                   'ifs/phys_ec/%s.%s.F90' % (original, mode))
    for r in processor.routines:
        routine = r.lower()
        r_deps.replace('flexbuild/raps17/intfb/ifs/%s.intfb.ok' % routine,
                       'flexbuild/raps17/intfb/ifs/%s.%s.intfb.ok' % (routine, mode))
    config['loki_deps'].content += [r_deps]

    # Add build rule for interface block
    if interface:
        intfb_path = 'flexbuild/raps17/intfb/ifs/%s.intfb.ok' % original
        intfb_deps = deepcopy(config['raps_deps'].content_map[intfb_path])
        intfb_deps.replace('ifs/phys_ec/%s.F90' % original,
                           'ifs/phys_ec/%s.%s.F90' % (original, mode))
        config['loki_deps'].content += [intfb_deps]

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

    # Modify calls to other subroutines in our list
    for call in FindNodes(Call).visit(routine.ir):
        if call.name.lower() in (r.lower() for r in processor.routines):
            call.name = '%s_%s' % (call.name, mode.upper())

    # Update all relevant interface imports
    for im in FindNodes(Import).visit(routine.ir):
        for r in processor.routines:
            if im.c_import and r == im.module.split('.')[0]:
                im.module = im.module.replace('.intfb', '.idem.intfb')

    # Generate mode-specific kernel subroutine
    source_file.write(source=fgen(routine), filename=source_file.path.with_suffix('.idem.F90'))

    # Generate updated interface block
    intfb_path = (Path(config['interface']) / source_file.path.stem).with_suffix('.idem.intfb.h')
    source_file.write(source=fgen(routine.interface), filename=intfb_path)

    # Add dependencies for newly created source files into RAPS build
    adjust_dependencies(original=original, config=config,
                        processor=processor, interface=True)


def physics_driver(source, config, processor):
    """
    Processing method for driver (root) routines that recreates the
    driver and swaps out all subroutine calls.
    """
    mode = config['default']['mode']
    driver = source.subroutines[0]

    info('Processing driver: %s, mode=%s' % (driver.name, mode))

    # Replace the non-reference call in the driver for evaluation
    for call in FindNodes(Call).visit(driver.ir):
        if call.name.lower() in (r.lower() for r in processor.routines):
            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            call.name = '%s_%s' % (call.name, mode.upper())

    # Update all relevant interface imports
    for im in FindNodes(Import).visit(driver.ir):
        for r in processor.routines:
            if im.c_import and r == im.module.split('.')[0]:
                im.module = im.module.replace('.intfb', '.idem.intfb')

    # Re-generate updated driver subroutine
    source.write(source=fgen(driver), filename=source.path.with_suffix('.idem.F90'))

    # Add dependencies for newly created source files into RAPS build
    adjust_dependencies(original=driver.name, config=config,
                        processor=processor, interface=False)


@cli.command('physics')
@click.option('--config', '-cfg', type=click.Path(),
            help='Path to configuration file.')
@click.option('--source', '-s', type=click.Path(),
            help='Path to source files to transform.')
@click.option('--typedef', '-t', type=click.Path(), multiple=True,
            help='Path for additional source file(s) containing type definitions')
@click.option('--interface', '-intfb', type=click.Path(), default=None,
            help='Path to auto-generate interface file(s)')
@click.option('--dependency-file', '-deps', type=click.Path(), default=None,
              help='Path to RAPS-generated dependency file')
@click.option('--callgraph', '-cg', is_flag=True, default=False,
            help='Generate and display the subroutine callgraph.')
def physics(config, source, typedef, interface, dependency_file, callgraph):

    kernel_map = {'idem' : {'driver': physics_driver,
                            'kernel': physics_idem_kernel},
                  'noop' : {'driver': None,
                            'kernel': None},
    }

    # Load configuration file and process options
    with Path(config).open('r') as f:
        config = toml.load(f)

    mode = config['default']['mode']
    config['interface'] = interface
    # Convert 'routines' to an ordered dictionary
    config['routines'] = OrderedDict((r['name'], r) for r in config['routine'])

    # Load RAPS dependency file for injection into the build system
    raps_deps = RapsDependencyFile.from_file(dependency_file)
    config['raps_deps'] = RapsDependencyFile.from_file(dependency_file)

    # Create new deps file with lib dependencies and a build rule
    objs_ifsloki = deepcopy(config['raps_deps'].content_map['OBJS_ifs'])
    objs_ifsloki.target = 'OBJS_ifsloki'
    rule_ifsloki = Rule(target='$(BMDIR)/libifsloki.a', deps=['$(OBJS_ifsloki)'],
                        cmds=[ '/bin/rm -f $@', 'ar -cr $@ $^', 'ranlib $@'])
    config['loki_deps'] = RapsDependencyFile(content=[objs_ifsloki, rule_ifsloki])

    # Create and setup the bulk source processor
    processor = SourceProcessor(path=source, config=config, kernel_map=kernel_map)
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

    # Write new mode-specific dependency rules file
    loki_config_path = config['raps_deps'].path.with_suffix('.loki.def')
    config['loki_deps'].write(path=loki_config_path)


@cli.command('noop')
@click.argument('routines', nargs=-1)
@click.option('--source', '-s', type=click.Path(),
            help='Path to source files to transform.')
@click.option('--discovery', '-d', is_flag=True, default=False,
            help='Automatically attempt to discover new subroutines.')
@click.option('--callgraph', '-cg', is_flag=True, default=False,
            help='Generate and display the subroutine callgraph.')
def noop(routines, source, discovery, callgraph):
    """
    Do-nothing mode to test the parsing and bulk-traversal capabilities.
    """
    processor = SourceProcessor(kernel=None, path=source)
    processor.append(routines)
    processor.process(discovery=discovery)

    if callgraph:
        processor.graph.render('callgraph', view=True)

if __name__ == "__main__":
    cli()
