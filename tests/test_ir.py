import pytest
import click as cli
from os import path
import open_fortran_parser as ofp

from ecir import generate, pprint


@pytest.fixture
def filename():
    return 'tests/claw_loop1.f90'


def test_parsing(filename):
    with open(filename) as file:
        print(file.read())
    ir = generate(filename)
    pprint(ir)


@cli.command()
@cli.option('--filename', '-f', help='Source file to parse.')
def parse_ofp(filename):
    with open(filename) as file:
        source = file.read()
    ast = ofp.parse(filename)
    ir = generate(ast, source)

    pprint(ir, verbose=True)


if __name__ == "__main__":
    parse_ofp()
