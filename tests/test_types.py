from random import choice
import pytest

from loki import OFP, OMNI, FP, SourceFile, DataType, SymbolType, FCodeMapper


def test_data_type():
    """
    Tests the conversion of strings to `DataType`.
    """
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


def test_symbol_type_compare():
    """
    Test dedicated `type.compare` methods that allows certain
    attributes to be excluded from comparison.
    """
    someint = SymbolType('integer', a='a', b=True, c=None)
    another = SymbolType('integer', a='a', b=False, c=None)
    somereal = SymbolType('real', a='a', b=True, c=None)

    assert not someint.compare(another)
    assert not another.compare(someint)
    assert someint.compare(another, ignore='b')
    assert another.compare(someint, ignore=['b'])
    assert not someint.compare(somereal)


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Segfault with pragmas in derived types')),
    FP
])
def test_pragmas(frontend):
    """
    Test detection of `!$loki dimension` pragmas to indicate intended shapes.
    """
    fcode = """
module types

  integer, parameter :: jprb = selected_real_kind(13,300)

  type pragma_type
    !$loki dimension(3,3)
    real(kind=jprb), dimension(:,:), pointer :: matrix
    !$loki dimension(klon,klat,2)
    real(kind=jprb), pointer :: tensor(:, :, :)
  end type pragma_type

contains

  subroutine alloc_pragma_type(item)
    type(pragma_type), intent(inout) :: item
    allocate(item%matrix(5,5))
    allocate(item%tensor(3,4,5))
  end subroutine

  subroutine free_pragma_type(item)
    type(pragma_type), intent(inout) :: item
    deallocate(item%matrix)
    deallocate(item%tensor)
  end subroutine

end module types
"""
    fsymgen = FCodeMapper()

    source = SourceFile.from_source(fcode, frontend=frontend)
    pragma_type = source['types'].types['pragma_type']

    assert fsymgen(pragma_type.variables['matrix'].shape) == '(3, 3)'
    assert fsymgen(pragma_type.variables['tensor'].shape) == '(klon, klat, 2)'
