import pytest
from pathlib import Path

from loki import clean, compile_and_load, OFP, OMNI, FP
from conftest import generate_identity


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'types.f90'


@pytest.fixture(scope='module')
def reference(refpath):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    pymod = compile_and_load(refpath, cwd=str(refpath.parent))
    return getattr(pymod, refpath.stem)


def test_data_type():
    """
    Tests the conversion of strings to `DataType`.
    """
    from loki.types import DataType
    from random import choice

    fortran_type_map = {'LOGICAL': DataType.LOGICAL, 'INTEGER': DataType.INTEGER,
                        'REAL': DataType.REAL, 'CHARACTER': DataType.CHARACTER,
                        'COMPLEX': DataType.COMPLEX}

    # Randomly change case of single letters (FORTRAN is not case-sensitive)
    test_map = {''.join(choice((str.upper, str.lower))(c) for c in s): t
                for s, t in fortran_type_map.items()}

    assert all([t == DataType.from_fortran_type(s) for s, t in test_map.items()])
    assert all([t == DataType.from_str(s) for s, t in test_map.items()])

    c99_type_map = {'bool': DataType.LOGICAL, '_Bool': DataType.LOGICAL,
                    'short': DataType.INTEGER, 'unsigned short': DataType.INTEGER,
                    'signed short': DataType.INTEGER, 'int': DataType.INTEGER,
                    'unsigned int': DataType.INTEGER, 'signed int': DataType.INTEGER,
                    'long': DataType.INTEGER, 'unsigned long': DataType.INTEGER,
                    'signed long': DataType.INTEGER, 'long long': DataType.INTEGER,
                    'unsigned long long': DataType.INTEGER, 'signed long long': DataType.INTEGER,
                    'float': DataType.REAL, 'double': DataType.REAL, 'long double': DataType.REAL,
                    'char': DataType.CHARACTER, 'float _Complex': DataType.COMPLEX,
                    'double _Complex': DataType.COMPLEX, 'long double _Complex': DataType.COMPLEX}

    assert all([t == DataType.from_c99_type(s) for s, t in c99_type_map.items()])
    assert all([t == DataType.from_str(s) for s, t in c99_type_map.items()])


def test_symbol_type():
    """
    Tests the attachment, lookup and deletion of arbitrary attributes from
    class:``SymbolType``
    """
    from loki.types import SymbolType, DataType

    _type = SymbolType('integer', a='a', b=True, c=None)
    assert _type.dtype == DataType.INTEGER
    assert _type.a == 'a'
    assert _type.b
    assert _type.c is None
    assert _type.foo is None

    _type.foo = 'bar'
    assert _type.foo == 'bar'

    delattr(_type, 'foo')
    assert _type.foo is None
