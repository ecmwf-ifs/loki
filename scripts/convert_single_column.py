import click as cli
import re
from collections import OrderedDict, Iterable

from ecir import FortranSourceFile, GenericVisitor

def flatten(l):
    """Flatten a hierarchy of nested lists into a plain list."""
    newlist = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist


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

    # Get list of all declarations involving the target dimension
    decls = [l for l in routine.declarations.longlines if tdim in l]
    decls = [l for l in decls if not l.startswith('!')]  # Strip comments

    # Get list of all allocations involving the target dimension
    re_allocs = re.compile('ALLOCATE\(.*?\)\)')
    allocs = [l for l in re_allocs.findall(routine.body._source) if tdim in l]

    # Extract all variables that use the target dimensions from declarations
    re_vars = re.compile('::(?P<vars>.*)')
    varlists = [re_vars.search(l).groupdict()['vars'] for l in decls]
    re_vname = re.compile('(?P<name>[a-zA-Z0-9\_]*)\(.*?\)')
    vnames = flatten([re_vname.findall(vl) for vl in varlists])
    vnames = [v for v in vnames if len(v) > 0]

    # Add all ALLOCATABLE variables that use target dimension
    re_valloc = re.compile('ALLOCATE\((?P<var>[a-zA-Z0-9\_]*)\(')
    vnames += [re_valloc.search(l).groupdict()['var'] for l in allocs]

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

    for var in vnames:
        # Strip all colon indices for leading dimensions
        routine.body.replace({'%s(:,' % var: '%s(' % var,
                              '%s(:)' % var: '%s' % var})
        # TODO: This one is hacky and assumes we always process FULL BLOCKS!
        # We effectively treat block_start:block_end variables as (:)
        routine.body.replace({'%s(JL-KIDIA+1,' % var: '%s(' % var,
                              '%s(JL-KIDIA+1)' % var: '%s' % var,
                              '%s(KIDIA:KFDIA,' % var: '%s(' % var,
                              '%s(KIDIA:KFDIA)' % var: '%s' % var,
                              '%s(KIDIA,' % var: '%s(' % var,
                              '%s(KIDIA)' % var: '%s' % var,
                         })

    # Strip a dimension from all affected ALLOCATABLEs
    allocatables = [ll for ll in routine.declarations.longlines
                    if 'ALLOCATABLE' in ll]
    for allocatable in allocatables:
        for vname in vnames:
            if vname in allocatable:
                routine.declarations.replace({'%s(:,' % vname: '%s(' % vname})

    print("Writing to %s" % output)
    source.write(output)


if __name__ == "__main__":
    convert()
