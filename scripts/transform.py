import click
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
                  Transformer, Import, warning, as_tuple, error)


def generate_signature(name, arguments):
    """
    Generate subroutine signature from a given list of arguments
    """
    arg_names = list(chunks([a.name for a in arguments], 6))
    dummies = ', &\n & '.join(', '.join(c) for c in arg_names)
    return 'SUBROUTINE %s &\n & (%s)\n' % (name, dummies)


def generate_interface(filename, name, arguments, imports):
    """
    Generate the interface file for a given Subroutine.
    """
    signature = generate_signature(name, arguments)
    interface = 'INTERFACE\n%s' % signature

    base_types = ['REAL', 'INTEGER', 'LOGICAL', 'COMPLEX']

    # Collect unknown symbols that we might need to import
    undefined = set()
    anames = [a.name for a in arguments]
    for a in arguments:
        # Add potentially unkown TYPE and KIND symbols to 'undefined'
        if a.type.name.upper() not in base_types:
            undefined.add(a.type.name)
        if a.type.kind and not a.type.kind.isdigit():
            undefined.add(a.type.kind)
        # Add (pure) variable dimensions that might be defined elsewhere
        undefined.update([str(d) for d in a.dimensions
                          if isinstance(d, Variable) and d not in anames])

    # Write imports for undefined symbols from external modules
    for use in imports:
        symbols = [s for s in use.symbols if s in undefined]
        if len(symbols) > 0:
            interface += 'USE %s, ONLY: %s\n' % (use.module, ', '.join(symbols))

    # Add type declarations for all arguments
    for arg in arguments:
        interface += '%s%s, INTENT(%s) %s:: %s\n' % (
            'TYPE(%s)' % arg.type.name if arg.type.name not in base_types else arg.type.name,
            ('(KIND=%s)' % arg.type.kind) if arg.type.kind else '',
            arg.type.intent.upper(), ', OPTIONAL ' if arg.type.optional else '', str(arg))
    interface += 'END SUBROUTINE %s\nEND INTERFACE\n' % name

    # And finally dump the generated string to file
    print("Writing interface to %s" % filename)
    with open(filename, 'w') as file:
        file.write(interface)


class SourceProcessor(object):
    """
    Work queue manager to enqueue and process individual source
    routines/modules with a given kernel.

    Note: The processing module can create a callgraph and perform
    automated discovery, to enable easy bulk-processing of large
    numbers of source files.
    """

    blacklist = ['DR_HOOK', 'ABOR1']

    def __init__(self, kernel, path):
        self.kernel = kernel
        self.path = Path(path)

        self.queue = deque()
        self.processed = []

        if Digraph is not None:
            self.graph = Digraph(format='pdf', strict=True)
        else:
            self.graph = None

    def append(self, sources):
        """
        Add names of source routines or modules to find and process.
        """
        self.queue.extend(as_tuple(sources))

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
                    # Re-generate target routine and interface block with updated name
                    source_file = FortranSourceFile(source_path, preprocess=True)
                    routine = source_file.subroutines[0]

                    # TODO: Apply the user-defined kernel
                    if self.kernel is not None:
                        self.kernel(source_file, processor=self)

                    if self.graph:
                        self.graph.node(routine.name, color='lightseagreen', style='filled')

                    for call in FindNodes(Call).visit(routine.ir):
                        if call.name in self.blacklist:
                            continue

                        if self.graph:
                            self.graph.node(call.name, color='lightblue', style='filled')
                            self.graph.edge(routine.name, call.name)

                        if discovery:
                            self.append(call.name.lower())

                except Exception as e:
                    if self.graph:
                        self.graph.node(source.upper(), color='red', style='filled')
                    warning('Could not parse %s:' % source)
                    error(e)

            else:
                if self.graph:
                    self.graph.node(source.upper(), color='lightsalmon', style='filled')
                info("Could not find source file %s; skipping..." % source)


class FindVariables(ExpressionVisitor, Visitor):

    default_retval = set

    def visit_tuple(self, o):
        vars = set()
        for c in o:
            vars.update(flatten(self.visit(c)))
        return  vars

    visit_list = visit_tuple

    def visit_Variable(self, o):
        return set([o])

    def visit_Expression(self, o):
        vars = set()
        for c in o.children:
            vars.update(flatten(self.visit(c)))
        return  vars

    def visit_Statement(self, o, **kwargs):
        vars = self.visit(o.expr, **kwargs)
        vars.update(flatten(self.visit(o.target)))
        return vars


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
                        new_arg = deepcopy(d_arg)
                        new_arg.subvar = Variable(name=type_var.name, dimensions=None)
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
            decl_mapper[old_decl] += [Declaration(variables=[new_var])]

        # Replace variable in dummy signature
        i = routine.argnames.index(arg.name)
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


# Define the target dimension to strip from kernel and caller
target = Dimension(name='KLON', aliases=['NPROMA'],
                   variable='JL', iteration=('KIDIA', 'KFDIA'))
target_routine = 'CLOUDSC'


def process_driver(driver, driver_out, routine, derived_arg_var, mode):
    f_driver = FortranSourceFile(driver)
    driver_routine = f_driver.subroutines[0]

    driver_routine._infer_variable_dimensions()

    # Process individual calls to our target routine
    for call in FindNodes(Call).visit(driver_routine._ir):
        if call.name == target_routine:
            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            new_args = []
            for d_arg, k_arg in zip(call.arguments, routine.arguments):
                if k_arg.name in target.variables:
                    continue
                elif k_arg in derived_arg_var:
                    # Found derived-type argument, unroll according to mapping
                    for arg_var in derived_arg_var[k_arg]:
                        new_dims = tuple(target.variable if d == target.name else ':'
                                         for d in arg_var.dimensions)
                        new_args.append(Variable(name='%s%%%s' % (d_arg, arg_var.name),
                                                 type=d_arg.type, dimensions=new_dims))
                elif len(d_arg.dimensions) == 0:
                    # Driver-side relies on implicit dimensions, unroll using ':'
                    new_dims = tuple(target.variable if d == target.name else ':'
                                     for d in k_arg.dimensions)
                    new_args.append(Variable(name=d_arg.name, type=d_arg.type,
                                             dimensions=new_dims))
                elif len(d_arg.dimensions) == len(k_arg.dimensions):
                    # Driver-side already has dimensions, just replace target
                    new_dims = tuple(target.variable if d == target.name else d
                                     for d in k_arg.dimensions)
                    new_args.append(Variable(name=d_arg.name, type=d_arg.type,
                                             dimensions=new_dims))
                elif len(d_arg.dimensions) > len(k_arg.dimensions):
                    # Driver-side has additional outer dimensions (eg. for blocking)
                    d_var = driver_routine.variable_map[d_arg.name]
                    new_dims = tuple(target.variable if d_v in target.aliases else d_c
                                     for d_v, d_c in zip(d_var.dimensions, d_arg.dimensions))
                    new_args.append(Variable(name=d_arg.name, type=d_arg.type,
                                             dimensions=new_dims))
                else:
                    # Something has gone terribly wrong if we reach this
                    raise ValueError('Unknown driver-side call argument')

            # Create new call and wrap it in a loop. The arguments used for
            # the bound variables in the kernel code are used as bounds here.
            new_bounds = tuple(d for d, k in zip(call.arguments, routine.arguments)
                               if k in target.iteration)
            new_call = Call(name='%s_%s' % (call.name, mode.upper()), arguments=new_args)
            if mode == 'claw':
                pragma_string = '!$claw parallelize forward create update'
                new_pragma = Pragma(keyword='parallelize',
                                    source=Source(string=pragma_string, lines=None))
            else:
                new_pragma = None
            new_loop = Loop(body=[new_call], variable=target.variable,
                            bounds=new_bounds, pragma=new_pragma)
            driver_routine.body.replace(call._source.string, fgen(new_loop, chunking=4))

    # Finally, add the loop counter variable to declarations
    new_var = Variable(name=target.variable, type=BaseType(name='INTEGER', kind='JPIM'))
    new_decl = Declaration(variables=[new_var])
    driver_routine.declarations._source += '\n%s\n\n' % fgen(new_decl)

    if mode == 'sca':
        # Add the include statement for the new header
        new_include = '#include "%s.%s.intfb.h"\n\n' % (routine.name.lower(), mode.lower())
        driver_routine.declarations._source += new_include
    elif mode == 'claw':
        kernel_name = '%s_%s' % (routine.name.upper(), mode.upper())
        new_import = 'USE %s_%s_MOD, ONLY: %s\n' % (routine.name.upper(), mode.upper(), kernel_name)
        driver_routine.declarations._source = new_import + driver_routine.declarations._source

    print("Writing to %s" % driver_out)
    f_driver.write(filename=driver_out)


def convert_sca(source, source_out, driver, driver_out, interface, typedef, mode):

    typedefs = get_typedefs(typedef)

    # Read the primary source routine
    f_source = FortranSourceFile(source, typedefs=typedefs)
    routine = f_source.subroutines[0]
    new_routine_name = '%s_%s' % (routine.name, mode.upper())

    ####  Remove target loops  ####
    # It's important to do this first, as the IR on the `routine`
    # object is not updated when the source changes...
    # TODO: Fully integrate IR with source changes...
    for loop in FindNodes(Loop).visit(routine._ir):
        if loop.variable == target.variable:
            # Get loop body and drop two leading chars for unindentation
            lines = loop._source.string.splitlines(keepends=True)[1:-1]
            lines = ''.join([line.replace('  ', '', 1) for line in lines])
            routine.body.replace(loop._source.string, lines)

    ####  Signature and interface adjustments  ####

    # We deep-copy arguments to make sure we are not affecting the
    # variable dimensions used in the later parts for regex replacement.
    arguments = deepcopy(routine.arguments)

    # Detect argument variables with derived types in the signature
    # that use the target diemnsion. For those types we explicitly unroll
    # the subtypes used in the signature and adjust caller and callee.
    derived_arg_map = OrderedDict()
    derived_arg_repl = {}
    derived_arg_var = defaultdict(list)
    # Note: This is getting tedious; there must be a better way...
    for arg in arguments:
        if arg.type.name.upper() in typedefs:
            new_vars = []
            # TODO: Need to define __key/__hash__ for Variables and (Derived)Types
            derived = typedefs[arg.type.name.upper()]
            for type_var in derived.variables:
                # Check if variable has the target dimension and is used in routine
                t_str = '%s%%%s' % (arg.name, type_var.name)
                if target.name in type_var.dimensions and t_str in routine.body._source:
                    derived_arg_var[arg].append(type_var)
                    new_name = '%s_%s' % (arg.name, type_var.name)
                    new_type = BaseType(name=type_var.type.name, kind=type_var.type.kind,
                                        intent=arg.type.intent)
                    new_vars.append(Variable(name=new_name, type=new_type,
                                             dimensions=type_var.dimensions))

                    # Record the string-replacement for the body
                    derived_arg_repl[t_str] = new_name

            # Derive index on-the-fly (multi-element insertions change indices!)
            # and update the argument list with unrolled arguments.
            idx = arguments.index(arg)
            # Store replacement for later declaration adjustment
            derived_arg_map[arg] = new_vars
            arguments[idx:idx+1] = new_vars

    # Now we replace the declarations for the previously derived arguments
    # Note: Re-generation from AST would probably be cleaner...
    declarations = FindNodes(Declaration).visit(routine.ir)
    for derived_arg, new_args in derived_arg_map.items():
        for decl in declarations:
            if derived_arg in decl.variables:
                # A simple sanity check...
                # decl.variables.remove(derived_arg)
                # if len(decl.variables) > 0:
                #     raise NotImplementedError('More than one derived argument per declaration found!')

                # Replace derived argument declaration with new declarations
                new_string = fgen([Declaration(variables=[arg]) for arg in new_args]) + '\n'
                routine.declarations.replace(decl._source.string, new_string)

    # And finally, replace all occurences of derived sub-types with unrolled ones
    routine.body.replace(derived_arg_repl)

    # Strip the target dimension from arguments
    arguments = [a for a in arguments if a.name not in target.variables]

    # Remove the target dimensions from our input arguments
    for a in arguments:
        a.dimensions = tuple(d for d in a.dimensions if target.name not in str(d))

    if interface:
        # Generate the interface file associated with this routine
        generate_interface(filename=interface, name=new_routine_name,
                           arguments=arguments, imports=routine.imports)

    # Generate new signature and replace the old one in file
    re_sig = re.compile('SUBROUTINE\s+%s.*?\(.*?\)' % routine.name, re.DOTALL)
    signature = re_sig.findall(routine._source)[0]
    new_signature = generate_signature(new_routine_name, arguments)
    routine.header.replace(signature, new_signature)

    routine._post.replace('END SUBROUTINE %s' % routine.name,
                          'END SUBROUTINE %s' % new_routine_name)

    # Strip target sizes from declarations
    for decl in FindNodes(Declaration).visit(routine.ir):
        if len(decl.variables) == 1 and decl.variables[0].name in target.variables:
            routine.declarations.replace(decl._source.string, '')

    # Strip target loop variable
    for decl in FindNodes(Declaration).visit(routine.ir):
        if target.variable in decl.variables:
            line = decl._source.string
            new_line = line.replace('%s, ' % target.variable, '')
            routine.declarations.replace(line, new_line)

    ####  Index replacements  ####

    # Strip all target iteration indices
    routine.body.replace({'(%s,' % target.variable: '(',
                          '(%s)' % target.variable: ''})

    # Find all variables affected by the transformation
    # Note: We assume here that the target dimension is matched
    # exactly in v.dimensions!
    variables = [v for v in routine.variables if target.name in v.dimensions]
    for v in variables:
        # Target is a vector, we now promote it to a scalar
        promote_to_scalar = len(v.dimensions) == 1
        new_dimensions = list(v.dimensions)
        new_dimensions.remove(target.name)

        # Strip target dimension from declarations and body (for ALLOCATEs)
        old_dims = '(%s)' % ','.join(str(d).replace(' ', '') for d in v.dimensions)
        new_dims = '' if promote_to_scalar else '(%s)' % ','.join(str(d).replace(' ', '')
                                                                  for d in new_dimensions)
        routine.declarations.replace(old_dims, new_dims)
        routine.body.replace(old_dims, new_dims)

        # Strip all colon indices for leading dimensions
        # TODO: Could do this in a smarter, more generic way...
        if promote_to_scalar:
            routine.body.replace('%s(:)' % v.name, '%s' % v.name)
        else:
            routine.body.replace('%s(:,' % v.name, '%s(' % v.name)

        if v.type.allocatable:
            routine.declarations.replace('%s(:,' % v.name, '%s(' % v.name)

    ####  Hacks that for specific annoyances in the CLOUDSC dwarf  ####

    variables = [v for v in routine.variables
                 if 'KFDIA' in ','.join(str(d) for d in v.dimensions)
                 or 'KLON' in ','.join(str(d) for d in v.dimensions)]
    for v in variables:
        routine.declarations.replace({'%s(KFDIA-KIDIA+1)' % v.name: '%s' % v.name,
                                      '%s(KFDIA-KIDIA+1,' % v.name: '%s(' % v.name,
                                      '%s(2*(KFDIA-KIDIA+1))' % v.name: '%s(2)' % v.name,
                                      '%s(2*KLON)' % v.name: '%s(2)' % v.name,
                                  })
        # TODO: This one is hacky and assumes we always process FULL BLOCKS!
        # We effectively treat block_start:block_end v.nameiables as (:)
        routine.body.replace({'%s(JL-KIDIA+1,' % v.name: '%s(' % v.name,
                              '%s(JL-KIDIA+1)' % v.name: '%s' % v.name,
                              '%s(KIDIA:KFDIA,' % v.name: '%s(' % v.name,
                              '%s(KIDIA:KFDIA)' % v.name: '%s' % v.name,
                              '%s(KIDIA,' % v.name: '%s(' % v.name,
                              '%s(KIDIA)' % v.name: '%s' % v.name,
                         })
    # And finally we have no shame left... :(
    routine.body.replace({'Z_TMPK(1,JK)': 'Z_TMPK(JK)',
                          'CALL CUADJTQ': 'CALL CUADJTQ_SCA',
                          '& (KIDIA,    KFDIA,   KLON,     KLEV,    IK,&':
                          '& (KLEV,    IK,&',
                          'CALL CUADJTQ(YDEPHLI,KIDIA,KFDIA,KLON,':
                          'CALL CUADJTQ(YDEPHLI,',
                          'JLEN=KFDIA-KIDIA+1': 'JLEN=1',
                          'KFDIA-KIDIA+1)': '1)',
                      })

    ####  CLAW-specific modifications  ####

    if mode == 'claw':
        # Prepend CLAW directives to subroutine body
        scalars = [v.name.lower() for v in routine.arguments
                   if len(v.dimensions) == 1]
        directives = '!$claw define dimension jl(1:nproma) &\n'
        directives += '!$claw parallelize &\n'
        directives += '!$claw scalar(%s)\n\n\n' % ', '.join(scalars)
        routine.body._source = directives + routine.body._source

        # Wrap subroutine in a module
        new_module = '%s_%s_MOD' % (routine.name.upper(), mode.upper())
        routine.header._source = 'MODULE %s\ncontains\n' % new_module + routine.header._source
        routine._post._source += 'END MODULE'

    print("Writing to %s" % source_out)
    f_source.write(filename=source_out)

    # Now let's process the driver/caller side
    if driver is not None:
        process_driver(driver, driver_out, routine, derived_arg_var, mode)

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
            help='Path for additional source file(s) containing type definitions')
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
    convert_sca(source, source_out, driver, driver_out, interface, typedef, mode)


__macro_template = """
#define LOKI_ROOT_CALL() %s(%s)
"""


def adjust_dependencies(routines, dependency_file):
    """
    Utility script to override the RAPS-generated dependencies
    following subroutine transformation.
    """

    deps_path = Path(dependency_file)
    with deps_path.open('r') as f:
        deps = f.read()

    def deps_replace(deps, base, routine, mode, suffix):
        oldpath = '%s/%s%s' % (base, routine, suffix)
        newpath = '%s/%s.%s%s' % (base, routine, mode, suffix)
        return deps.replace(oldpath, newpath)

    # Performn string substitutions on dependency list
    mode = 'idem'
    for rname in routines:
        deps = deps_replace(deps, 'ifs/phys_ec', rname, mode, suffix='.o')
        deps = deps_replace(deps, 'ifs/phys_ec', rname, mode, suffix='.F90')
        deps = deps_replace(deps, 'flexbuild/raps17/intfb/ifs', rname, mode, suffix='.intfb.h')
        deps = deps_replace(deps, 'flexbuild/raps17/intfb/ifs', rname, mode, suffix='.intfb.ok')

    info('Writing dependencies: %s' % deps_path)
    with deps_path.open('w') as f:
        f.write(deps)


@cli.command('physics')
@click.argument('routines', nargs=-1)
@click.option('--source', '-s', type=click.Path(),
            help='Path to source files to transform.')
@click.option('--typedef', '-t', type=click.Path(), multiple=True,
            help='Path for additional source file(s) containing type definitions')
@click.option('--interface', '-intfb', type=click.Path(), default=None,
            help='Path to auto-generate interface file(s)')
@click.option('--root-macro', '-m', type=click.Path(),
            help='Path to root macro for insertion of transformed call-tree')
@click.option('--dependency-file', '-deps', type=click.Path(), default=None,
              help='Path to RAPS-generated dependency file')
@click.option('--callgraph', '-cg', is_flag=True, default=False,
            help='Generate and display the subroutine callgraph.')
def physics(routines, source, typedef, root_macro, interface, dependency_file, callgraph):

    def physics_idem_kernel(source_file, processor):
        routine = source_file.subroutines[0]
        routine.name = '%s_IDEM' % routine.name

        # Modify calls to other subroutines in our list
        for call in FindNodes(Call).visit(routine.ir):
            if call.name.lower() in (r.lower() for r in routines):
                call.name += '_IDEM'

        for im in FindNodes(Import).visit(routine.ir):
            for r in routines:
                if im.c_import and r == im.module.split('.')[0]:
                    im.module = im.module.replace('.intfb', '.idem.intfb')

        source_file.write(source=fgen(routine), filename=source_file.path.with_suffix('.idem.F90'))

        intfb_path = (Path(interface) / source_file.path.stem).with_suffix('.idem.intfb.h')
        source_file.write(source=fgen(routine.interface), filename=intfb_path)

    processor = SourceProcessor(kernel=physics_idem_kernel, path=source)
    processor.append(routines)
    processor.process(discovery=False)

    if callgraph:
        processor.graph.render('callgraph', view=False)

    adjust_dependencies(routines, dependency_file)

    # Insert the root of the transformed call-tree into the root macro
    # TODO: To get argument naming right, we need driver information!
    # root_args = ', '.join(arg.name.upper() for arg in routine.arguments)
    # macro = __macro_template % (routine.name, root_args)
    # info('Writing root macro: %s' % root_macro)
    # with open(root_macro, 'w') as f:
    #     f.write(macro)


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
