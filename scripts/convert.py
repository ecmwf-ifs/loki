import click
import re
from collections import OrderedDict, defaultdict, Iterable
from copy import deepcopy
import sys

from ecir import (FortranSourceFile, Visitor, flatten, chunks, Loop,
                  Variable, Type, DerivedType, Declaration, FindNodes,
                  Statement, Call, Pragma, fgen)


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
    f_driver = FortranSourceFile(driver, preprocess=False)
    driver_routine = f_driver.subroutines[0]

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
                    d_var = driver_routine._variables[d_arg.name]
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
            new_pragma = Pragma(keyword='parallelize',
                                source='!$claw parallelize forward create update') if mode =='claw' else None
            new_loop = Loop(body=[new_call], variable=target.variable,
                            bounds=new_bounds, pragma=new_pragma)
            driver_routine.body.replace(call._source, fgen(new_loop, chunking=4))

    # Finally, add the loop counter variable to declarations
    new_var = Variable(name=target.variable, type=Type(name='INTEGER', kind='JPIM'))
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

    # Read additional derived types from typedef modules
    derived_types = {}
    for tfile in typedef:
        module = FortranSourceFile(tfile).modules[0]
        for derived in module._spec:
            if isinstance(derived, DerivedType):
                # TODO: Need better __hash__ for (derived) types
                derived_types[derived.name.upper()] = derived

    # Read the primary source routine
    f_source = FortranSourceFile(source, preprocess=False)
    routine = f_source.subroutines[0]
    new_routine_name = '%s_%s' % (routine.name, mode.upper())

    ####  Remove target loops  ####
    # It's important to do this first, as the IR on the `routine`
    # object is not updated when the source changes...
    # TODO: Fully integrate IR with source changes...
    for loop in FindNodes(Loop).visit(routine._ir):
        if loop.variable == target.variable:
            # Get loop body and drop two leading chars for unindentation
            lines = loop._source.splitlines(keepends=True)[1:-1]
            lines = ''.join([line.replace('  ', '', 1) for line in lines])
            routine.body.replace(loop._source, lines)

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
        if arg.type.name.upper() in derived_types:
            new_vars = []
            # TODO: Need to define __key/__hash__ for Variables and (Derived)Types
            derived = derived_types[arg.type.name.upper()]
            for type_var in derived.variables:
                # Check if variable has the target dimension and is used in routine
                t_str = '%s%%%s' % (arg.name, type_var.name)
                if target.name in type_var.dimensions and t_str in routine.body._source:
                    derived_arg_var[arg].append(type_var)
                    new_name = '%s_%s' % (arg.name, type_var.name)
                    new_type = Type(name=type_var.type.name, kind=type_var.type.kind,
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
                decl.variables.remove(derived_arg)
                if len(decl.variables) > 0:
                    raise NotImplementedError('More than one derived argument per declaration found!')

                # Replace derived argument declaration with new declarations
                new_string = fgen([Declaration(variables=[arg]) for arg in new_args]) + '\n'
                routine.declarations.replace(decl._source, new_string)

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
    for v in routine.arguments:
        if v.name in target.variables:
            routine.declarations.replace(v._source, '')

    # Strip target loop variable
    line = routine._variables[target.variable]._source
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
@click.option('--interface', '-intfb', type=click.Path(), default=None,
            help='Path to auto-generate and interface file')
@click.option('--conservative/--no-conservative', default=False,
            help='Force conservative re-generation')
def idempotence(source, source_out, driver, driver_out, interface, conservative):
    """
    Idempotence: A "do-nothing" debug mode that performs a parse-and-unparse cycle.
    """
    f_source = FortranSourceFile(source, preprocess=True)
    routine = f_source.subroutines[0]
    routine.name = '%s_IDEM' % routine.name

    # Generate the interface file associated with this routine
    generate_interface(filename=interface, name=routine.name,
                       arguments=routine.arguments, imports=routine.imports)

    print("Writing to %s" % source_out)
    if conservative:
        # Debug hack to get 'conservative' mode happening
        routine._source = routine._source.replace('SUBROUTINE CLOUDSC', 'SUBROUTINE CLOUDSC_IDEM')
        f_source.write(source=fgen(routine, conservative=True), filename=source_out)
    else:
        f_source.write(source=fgen(routine, conservative=False), filename=source_out)

    # Replace the non-reference call in the driver for evaluation
    f_driver = FortranSourceFile(driver, preprocess=False)
    driver_routine = f_driver.subroutines[0]
    for call in FindNodes(Call).visit(driver_routine._ir):
        if call.name == target_routine:
            # Skip calls marked for reference data collection
            if call.pragma is not None and call.pragma.keyword == 'reference':
                continue

            call.name = '%s_IDEM' % call.name
            driver_routine.body.replace(call._source, fgen(call, chunking=4))

    print("Writing to %s" % driver_out)
    f_driver.write(filename=driver_out)


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

if __name__ == "__main__":
    cli()
