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
@cli.option('--file', '-f', help='Source file to convert.')
@cli.option('--output', '-o', help='Source file to convert.')
def convert(file, output):

    print('Processing %s ...' % file)
    source = FortranSourceFile(file)
    routine = source.routines[0]

    tdim = 'KLON'  # Name of the target dimension
    tvar = 'JL'  # Name of the target iteration variable

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

    # Note: We assume that KLON is always the leading dimension(!)
    # Strip target dimension from declarations and body (for ALLOCATEs)
    routine.declarations.replace({'(%s,' % tdim: '(', '(%s)' % tdim: ''})
    routine.body.replace({'(%s,' % tdim: '(', '(%s)' % tdim: ''})

    # Strip all target iteration indices
    routine.body.replace({'(%s,' % tvar: '(', '(%s)' % tvar: ''})

    # Find all variables affected by the transformation
    variables = [v for v in routine.variables if tdim in ','.join(v.dimensions)]
    for v in variables:
        # Strip all colon indices for leading dimensions
        routine.body.replace({'%s(:,' % v.name: '%s(' % v.name,
                              '%s(:)' % v.name: '%s' % v.name})
        # TODO: This one is hacky and assumes we always process FULL BLOCKS!
        # We effectively treat block_start:block_end v.nameiables as (:)
        routine.body.replace({'%s(JL-KIDIA+1,' % v.name: '%s(' % v.name,
                              '%s(JL-KIDIA+1)' % v.name: '%s' % v.name,
                              '%s(KIDIA:KFDIA,' % v.name: '%s(' % v.name,
                              '%s(KIDIA:KFDIA)' % v.name: '%s' % v.name,
                              '%s(KIDIA,' % v.name: '%s(' % v.name,
                              '%s(KIDIA)' % v.name: '%s' % v.name,
                         })

        if v.allocatable:
            routine.declarations.replace({'%s(:,' % v.name: '%s(' % v.name})

    print("Writing to %s" % output)
    source.write(output)


if __name__ == "__main__":
    convert()
