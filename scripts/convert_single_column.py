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


@cli.command()
@cli.option('--source', '-s', help='Source file to convert.')
@cli.option('--source-out', '-so', help='Path for generated source output.')
@cli.option('--driver', '-d', default=None, help='Driver file to convert.')
@cli.option('--driver-out', '-do', default=None, help='Path for generated driver output.')
@cli.option('--mode', '-m', type=cli.Choice(['onecol', 'claw']), default='onecol')
def convert(source, source_out, driver, driver_out, mode):

    f_source = FortranSourceFile(source)
    routine = f_source.routines[0]

    tdim = 'KLON'  # Name of the target dimension
    tvar = 'JL'  # Name of the target iteration variable

    dummy_variables = ['KIDIA', 'KFDIA', 'KLON']

    # First, let's strip the target loops. It's important to do this
    # first, as the IR on the `routine` object is not updated when the
    # source changes...
    # TODO: Fully integrate IR with source changes...
    finder = FindLoops(target_var=tvar)
    for loop in flatten(routine._ir):
        target_loops = finder.visit(loop)
        for target in target_loops:
            # Get loop body and drop two leading chars for unindentation
            lines = target._source.splitlines(keepends=True)[1:-1]
            lines = ''.join([line.replace('  ', '', 1) for line in lines])
            routine.body._source = routine.body._source.replace(target._source, lines)

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
        routine.body.replace({'%s(:,' % v.name: '%s(' % v.name,
                              '%s(:)' % v.name: '%s' % v.name})

        if v.allocatable:
            routine.declarations.replace({'%s(:,' % v.name: '%s(' % v.name})

    # Hacks, hacks, nothing but hacks... (for CUADJTQ and CLOUDSC)
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
                              'Z_TMPK(1,JK)': 'Z_TMPK(JK)'  # <= Dirty little hobbitses!!!
                         })

    # Strip dummy variables from signature
    re_sig = re.compile('SUBROUTINE\s+%s.*?\(.*?\)' % routine.name, re.DOTALL)
    signature = re_sig.findall(routine._source)[0]
    new_signature = signature[:]
    re_sig_args = re.compile('\((?P<args>.*?)\)', re.DOTALL)
    sig_args = re_sig_args.search(signature).groupdict()['args']
    dummy_args = sig_args.split(',')
    for dummy in dummy_variables:
        for arg in dummy_args:
            if dummy in arg:
                dummy_args.remove(arg)
    # Reconstruct args in new_signature and replace in the routine source
    new_signature = new_signature.replace(sig_args, ','.join(dummy_args))
    routine.declarations._source = routine.declarations._source.replace(signature, new_signature)

    # Strip dummy variables from declarations
    for v in routine.arguments:
        if v.name in dummy_variables:
            routine.declarations._source = routine.declarations._source.replace(v._line, '')

    if mode == 'claw':
        # Prepend CLAW directives to subroutine body
        scalars = [v.name.lower() for v in routine.arguments
                   if len(v.dimensions) == 1]
        directives = '!$claw define dimension jl(1:klon) &\n'
        directives += '!$claw parallelize &\n'
        directives += '!$claw scalar(%s)\n\n\n' % ', '.join(scalars)
        routine.body._source = directives + routine.body._source

    print("Writing to %s" % source_out)
    f_source.write(source_out)

    # Now let's process the driver/caller side
    if driver is not None:
        f_driver = FortranSourceFile(driver)

        print("Writing to %s" % driver_out)
        f_driver.write(driver_out)

if __name__ == "__main__":
    convert()
