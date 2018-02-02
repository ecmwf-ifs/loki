import pytest
from os import path

from ecir import generate, pprint


@pytest.fixture
def filename():
    return 'tests/claw_loop1.f90'


def test_parsing(filename):
    with open(filename) as file:
        print(file.read())
    ir = generate(filename)
    pprint(ir)
