import click as cli
import re
from collections import OrderedDict

from ecir import FortranSourceFile

@cli.command()
@cli.option('--file', '-f', help='Source file to convert.')
@cli.option('--output', '-o', help='Source file to convert.')
def convert(file, output):

    print('Processing %s ...' % file)
    source = FortranSourceFile(file)

    tdim = 'KLON'  # Name of the target dimension
    tvar = 'JL'  # Name of the target iteration variable

    # Get list of all declarations involving the target dimension
    decls = [l for l in source.declarations.longlines if tdim in l]
    decls = [l for l in decls if not l.startswith('!')]  # Strip comments

    # Extract names of all variables that have the target dimensions
    re_vnames = re.compile('::\W*(?P<name>[a-zA-Z\_]*)')
    vnames = [re_vnames.search(l).groupdict()['name'] for l in decls]

    # Strip the target dimension from all declarations
    # Note: We assume that KLON is always the leading dimension(!)
    newdecls = [d.replace('(%s,' % tdim, '(') for d in decls]
    newdecls = [d.replace('(%s)' % tdim, '') for d in newdecls]
    source.declarations.replace(dict(zip(decls, newdecls)))

    # Replace all target iteration indices
    index_mapping = OrderedDict()
    for var in vnames:
        index_mapping.update({'(%s,' % tvar: '(', '(%s)' % tvar: ''})
    source.body.replace(index_mapping)

    # Super-hacky regex replacement for the target loops,
    # assuming that we only ever replace the inner (fast) loop!
    re_target_loop = re.compile('[^\n]*DO %s.*?ENDDO' % tvar, re.DOTALL)
    re_loop_body = re.compile('DO %s.*?\n(.*?)\n\W*ENDDO' % tvar, re.DOTALL)
    for loop in re_target_loop.findall(source.body._source):
        # Get loop body and drop two leading chars for unindentation
        body = re_loop_body.findall(loop)[0]
        body = '\n'.join([line.replace('  ', '', 1) for line in body.split('\n')])
        # Manually perform the replacement, as we're going accross lines
        source.body._source = source.body._source.replace(loop, body)

    print("Writing to %s" % output)
    source.write(output)


if __name__ == "__main__":
    convert()
