import click as cli
import re
from collections import OrderedDict, Iterable

from ecir import FortranSourceFile, GenericVisitor, flatten


class FindLoops(GenericVisitor):

    def __init__(self, target_var):
        super(FindLoops, self).__init__()

        self.target_var = target_var

    def visit_Loop(self, o):
        lines = o._source.splitlines(keepends=True)
        if self.target_var in lines[0]:
            # Loop is over target dimension
            return (o, )
        elif o._children is not None:
            # Recurse over children to find target
            children = tuple(self.visit(c) for c in flatten(o._children))
            children = tuple(c for c in flatten(children) if c is not None)
            return children
        else:
            return ()


def remove_dummy_variables(signature, variables):
    re_sig_args = re.compile('\((?P<args>.*?)\)', re.DOTALL)
    sig_args = re_sig_args.search(signature).groupdict()['args']
    arguments = sig_args.split(',')
    for var in variables:
        for arg in arguments:
            if var == arg.strip():
                arguments.remove(arg)
    return signature.replace(sig_args, ','.join(arguments))


@cli.command()
@cli.option('--source', '-s', help='Source file to convert.')
@cli.option('--source-out', '-so', help='Path for generated source output.')
@cli.option('--driver', '-d', default=None, help='Driver file to convert.')
@cli.option('--driver-out', '-do', default=None, help='Path for generated driver output.')
@cli.option('--mode', '-m', type=cli.Choice(['onecol', 'claw']), default='onecol')
@cli.option('--strip-signature/--no-strip-signature', default=True)
def convert(source, source_out, driver, driver_out, mode, strip_signature):

    f_source = FortranSourceFile(source)
    routine = f_source.routines[0]

    tdim = 'KLON'  # Name of the target dimension
    tvar = 'JL'  # Name of the target iteration variable

    dummy_variables = ['KIDIA', 'KFDIA', 'KLON']  # Variables to strip from signatures

    ####  Remove target loops  ####

    # It's important to do this first, as the IR on the `routine`
    # object is not updated when the source changes...
    # TODO: Fully integrate IR with source changes...
    finder = FindLoops(target_var=tvar)
    for loop in flatten(routine._ir):
        target_loops = finder.visit(loop)
        for target in target_loops:
            # Get loop body and drop two leading chars for unindentation
            lines = target._source.splitlines(keepends=True)[1:-1]
            lines = ''.join([line.replace('  ', '', 1) for line in lines])
            routine.body._source = routine.body._source.replace(target._source, lines)

    ####  Signature adjustments  ####

    if strip_signature:
        # Strip dummy variables from signature
        re_sig = re.compile('SUBROUTINE\s+%s.*?\(.*?\)' % routine.name, re.DOTALL)
        signature = re_sig.findall(routine._source)[0]
        new_signature = remove_dummy_variables(signature, dummy_variables)
        routine.declarations._source = routine.declarations._source.replace(signature, new_signature)

        # Strip dummy variables from declarations
        for v in routine.arguments:
            if v.name in dummy_variables:
                routine.declarations._source = routine.declarations._source.replace(v._line, '')

        # Strip target loop variable
        line = routine._variables[tvar]._line
        new_line = line.replace('%s, ' % tvar, '')
        routine.declarations._source = routine.declarations._source.replace(line, new_line)

    ####  Index replacements  ####

    # Strip all target iteration indices
    routine.body.replace({'(%s,' % tvar: '(', '(%s)' % tvar: ''})

    # Find all variables affected by the transformation
    # Note: We assume here that the target dimenion is match exactly
    # in v.dimensions!
    variables = [v for v in routine.variables if tdim in v.dimensions]
    for v in variables:
        # Target is a vector, we now promote it to a scalar
        promote_to_scalar = len(v.dimensions) == 1
        new_dimensions = list(v.dimensions)
        new_dimensions.remove(tdim)

        # Strip target dimension from declarations and body (for ALLOCATEs)
        old_dims = '(%s)' % ','.join(v.dimensions)
        new_dims = '' if promote_to_scalar else '(%s)' % ','.join(new_dimensions)
        routine.declarations.replace({old_dims: new_dims})
        routine.body.replace({old_dims: new_dims})

        # Strip all colon indices for leading dimensions
        # TODO: Could do this in a smarter, more generic way...
        if promote_to_scalar:
            routine.body.replace({'%s(:)' % v.name: '%s' % v.name})
        else:
            routine.body.replace({'%s(:,' % v.name: '%s(' % v.name})

        if v.allocatable:
            routine.declarations.replace({'%s(:,' % v.name: '%s(' % v.name})

    ####  Hacks that for specific annoyances in the CLOUDSC dwarf  ####

    variables = [v for v in routine.variables
                 if 'KFDIA' in ','.join(v.dimensions)
                 or 'KLON' in ','.join(v.dimensions)]
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
                          '& (KIDIA,    KFDIA,   KLON,     KLEV,    IK,&':
                          '& (    1,        1,      1,     KLEV,    IK,&',
                          'JLEN=KFDIA-KIDIA+1': 'JLEN=1',
                          'KFDIA-KIDIA+1)': '1)',
                      })

    ####  CLAW-specific modifications  ####

    if mode == 'claw':
        # Prepend CLAW directives to subroutine body
        scalars = [v.name.lower() for v in routine.arguments
                   if len(v.dimensions) == 1]
        directives = '!$claw define dimension jl(1:klon) &\n'
        directives += '!$claw parallelize &\n'
        directives += '!$claw scalar(%s)\n\n\n' % ', '.join(scalars)
        routine.body._source = directives + routine.body._source

        # Wrap subroutine in a module
        f_source._pre._source += 'MODULE cloudsc_mod\ncontains\n'
        f_source._post._source += 'END MODULE'

    print("Writing to %s" % source_out)
    f_source.write(source_out)

    # Now let's process the driver/caller side
    if driver is not None:
        f_driver = FortranSourceFile(driver)

        # # Process individual calls to our target routine
        # re_call = re.compile('CALL %s[\s\&\(].*?\)\s*?\n' % routine.name, re.DOTALL)
        # for call in re_call.findall(f_driver._raw_source):
        #     # Create the outer loop from the first two arguments
        #     pass
            
        print("Writing to %s" % driver_out)
        f_driver.write(driver_out)

if __name__ == "__main__":
    convert()
