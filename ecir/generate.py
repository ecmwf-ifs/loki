from open_fortran_parser import parse

from ecir.visitors import IRGenerator

__all__ = ['generate']


def generate(filename, verbosity=100):
    """
    Generate an IR (internal representation) tree from Fortran source code.
    """

    # Get raw Fortran AST in XML format from parser
    ast = parse(filename, verbosity=verbosity)

    # Create our own internal representation of the code
    return IRGenerator().visit(ast)
