import click as cli
import re

from ecir import FortranSourceFile

@cli.command()
@cli.option('--file', '-f', help='Source file to convert.')
@cli.option('--output', '-o', help='Source file to convert.')
def convert(file, output):

    print('Processing %s ...' % file)
    source = FortranSourceFile(file)

    target = 'KLON'  # Name of the target dimension

    # Get list of all declarations involving the target dimension
    decls = [l for l in source.declarations.longlines if target in l]
    decls = [l for l in decls if not l.startswith('!')]  # Strip comments

    # Extract names of all variables that have the target dimensions
    re_vnames = re.compile('::\W*(?P<name>[a-zA-Z\_]*)')
    vnames = [re_vnames.search(l).groupdict()['name'] for l in decls]

    # Strip the target dimension from all declarations
    # Note: We assume that KLON is always the leading dimension(!)
    newdecls = [d.replace('(KLON,', '(') for d in decls]
    newdecls = [d.replace('(KLON)', '') for d in newdecls]
    
    source.declarations.replace(dict(zip(decls, newdecls)))
    # from IPython import embed; embed()

    print("Writing to %s" % output)
    source.write(output)


if __name__ == "__main__":
    convert()
