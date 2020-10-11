from random import choice
import pytest

from loki import (
    OFP, OMNI, FP, SourceFile, Module, Subroutine, DataType,
    SymbolType, DerivedType, Array, Scalar, FCodeMapper
)


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
    pytest.param(OFP, marks=pytest.mark.xfail(reason='OFP needs preprocessing to support contiguous keyword')),
    OMNI,
    FP
])
def test_type_declaration_attributes(frontend):
    """
    Test recognition of different declaration attributes.
    """
    fcode = """
subroutine test_type_declarations(b, c)
    integer, parameter :: a = 4
    integer, intent(in) :: b
    real(kind=a), target, intent(inout) :: c(:)
    real(kind=a), allocatable :: d(:)
    real(kind=a), pointer, contiguous :: e(:)

end subroutine test_type_declarations
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert routine.symbols['a'].parameter
    assert routine.symbols['b'].intent == 'in'
    assert routine.symbols['c'].target
    assert routine.symbols['c'].intent == 'inout'
    assert routine.symbols['d'].allocatable
    assert routine.symbols['e'].pointer
    assert routine.symbols['e'].contiguous


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

    assert fsymgen(pragma_type.dtype.variable_map['matrix'].shape) == '(3, 3)'
    assert fsymgen(pragma_type.dtype.variable_map['tensor'].shape) == '(klon, klat, 2)'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_type_derived_type(frontend):
    """
    Test the detection of known derived type definitions.
    """

    fcode = """
module test_type_derived_type_mod
  implicit none
  integer, parameter :: a_kind = 4

  type my_struct
    real(kind=a_kind) :: a(:), b(:,:)
  end type my_struct

  contains
  subroutine test_type_derived_type(a, b, c)
    type(my_struct), target, intent(inout) :: a
    type(my_struct), allocatable :: b(:)
    type(my_struct), pointer :: c

  end subroutine test_type_derived_type
end module test_type_derived_type_mod
"""
    module = Module.from_source(fcode, frontend=frontend)
    routine = module['test_type_derived_type']

    a, b, c = routine.variables
    assert isinstance(a, Scalar)
    assert isinstance(a.type.dtype, DerivedType)
    assert a.type.target
    assert isinstance(b, Array)
    assert isinstance(b.type.dtype, DerivedType)
    assert b.type.allocatable
    assert isinstance(c, Scalar)
    assert isinstance(c.type.dtype, DerivedType)
    assert c.type.pointer

    # Ensure derived types have links to type definition and correct scope
    assert len(a.type.dtype.variables) == 2
    assert len(b.type.dtype.variables) == 2
    assert len(c.type.dtype.variables) == 2
    assert all(v.scope == routine.symbols for v in a.type.dtype.variables)
    assert all(v.scope == routine.symbols for v in b.type.dtype.variables)
    assert all(v.scope == routine.symbols for v in c.type.dtype.variables)

    # Ensure all member variable have an entry in the local symbol table
    assert routine.symbols['a%a'].shape == (':',)
    assert routine.symbols['a%b'].shape == (':',':')
    assert routine.symbols['b%a'].shape == (':',)
    assert routine.symbols['b%b'].shape == (':',':')
    assert routine.symbols['c%a'].shape == (':',)
    assert routine.symbols['c%b'].shape == (':',':')


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='OMNI cannot deal with deferred type info')),
    FP])
def test_type_module_imports(frontend):
    """
    Test the detection of known / unknown symbols types from module imports.
    """
    fcode = """
subroutine test_type_module_imports(arg_a, arg_b)
  use my_types_mod, only: a_kind, a_dim, a_type
  implicit none

  real(kind=a_kind), intent(in) :: arg_a(a_dim)
  type(a_type), intent(in) :: arg_b
end subroutine test_type_module_imports
"""
    # Ensure types are deferred without a-priori context info
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert routine.symbols['a_kind'].dtype == DataType.DEFERRED
    assert routine.symbols['a_dim'].dtype == DataType.DEFERRED
    assert routine.symbols['a_type'].dtype == DataType.DEFERRED

    # Ensure local variable info is correct, as far as known
    arg_a, arg_b = routine.variables
    assert arg_a.type.kind.type == routine.symbols['a_kind']
    assert arg_a.dimensions.index_tuple[0].type == routine.symbols['a_dim']
    assert isinstance(arg_b.type.dtype, DerivedType)
    assert arg_b.type.dtype.typedef == DataType.DEFERRED
