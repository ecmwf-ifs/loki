import click as cli
import re
from collections import OrderedDict, Iterable

from ecir import FortranSourceFile

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

    # Super-hacky regex replacement for the target loops,
    # assuming that we only ever replace the inner (fast) loop!
    re_target_loop = re.compile('[^\n]*DO %s.*?ENDDO' % tvar, re.DOTALL)
    re_loop_body = re.compile('DO %s.*?\n(.*?)\n\W*ENDDO' % tvar, re.DOTALL)
    for loop in re_target_loop.findall(routine.body._source):
        # Get loop body and drop two leading chars for unindentation
        body = re_loop_body.findall(loop)[0]
        body = '\n'.join([line.replace('  ', '', 1) for line in body.split('\n')])
        # Manually perform the replacement, as we're going accross lines
        routine.body._source = routine.body._source.replace(loop, body)

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
