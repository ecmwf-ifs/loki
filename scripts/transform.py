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
                  FindVariables, Index, Allocation)

from raps_deps import RapsDependencyFile, Rule


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


def hoist_dimension_from_call(driver):
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

            for darg in call.arguments:

                # Replace target dimension with a loop index in arguments
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
                    d_var = driver_routine.variable_map[d_arg.name.upper()]
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
    new_decl = Declaration(variables=[new_var], type=new_var.type)
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
                new_string = fgen([Declaration(variables=[arg], type=new_args[0].type)
                                   for arg in new_args]) + '\n'
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

    # Remove horizontal dimension from kernels and hoist loop to
    # driver to transform a subroutine invocation to SCA format.
    flatten_derived_arguments(routine, driver)
    hoist_dimension_from_call(driver)
    remove_dimension(routine=routine, target=target)

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


__macro_template = """
#define LOKI_ROOT_CALL() %s(%s)
"""


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
