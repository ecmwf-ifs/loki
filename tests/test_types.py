from random import choice
import pytest

from conftest import available_frontends
from loki import (
    OFP, OMNI, Sourcefile, Module, Subroutine, BasicType,
    SymbolAttributes, DerivedType, TypeDef, Array, Scalar, FCodeMapper,
    DataType, fgen
)


def test_basic_type():
    """
    Tests the conversion of strings to `BasicType`.
    """
    assert all(t == BasicType(t.value) for t in BasicType)
    assert all(isinstance(t, DataType) for t in BasicType)

    assert all(t == BasicType.from_name(t.name) for t in BasicType)
    assert all(t == BasicType.from_str(t.name) for t in BasicType)

    fortran_type_map = {'LOGICAL': BasicType.LOGICAL, 'INTEGER': BasicType.INTEGER,
                        'REAL': BasicType.REAL, 'CHARACTER': BasicType.CHARACTER,
                        'COMPLEX': BasicType.COMPLEX}

    # Randomly change case of single letters (FORTRAN is not case-sensitive)
    test_map = {''.join(choice((str.upper, str.lower))(c) for c in s): t
                for s, t in fortran_type_map.items()}

    assert all(t == BasicType.from_fortran_type(s) for s, t in test_map.items())
    assert all(t == BasicType.from_str(s) for s, t in test_map.items())

    c99_type_map = {'bool': BasicType.LOGICAL, '_Bool': BasicType.LOGICAL,
                    'short': BasicType.INTEGER, 'unsigned short': BasicType.INTEGER,
                    'signed short': BasicType.INTEGER, 'int': BasicType.INTEGER,
                    'unsigned int': BasicType.INTEGER, 'signed int': BasicType.INTEGER,
                    'long': BasicType.INTEGER, 'unsigned long': BasicType.INTEGER,
                    'signed long': BasicType.INTEGER, 'long long': BasicType.INTEGER,
                    'unsigned long long': BasicType.INTEGER, 'signed long long': BasicType.INTEGER,
                    'float': BasicType.REAL, 'double': BasicType.REAL, 'long double': BasicType.REAL,
                    'char': BasicType.CHARACTER, 'float _Complex': BasicType.COMPLEX,
                    'double _Complex': BasicType.COMPLEX, 'long double _Complex': BasicType.COMPLEX}

    assert all(t == BasicType.from_c99_type(s) for s, t in c99_type_map.items())
    assert all(t == BasicType.from_str(s) for s, t in c99_type_map.items())


def test_symbol_attributes():
    """
    Tests the attachment, lookup and deletion of arbitrary attributes from
    :any:`SymbolAttributes`
    """
    _type = SymbolAttributes('integer', a='a', b=True, c=None)
    assert _type.dtype == BasicType.INTEGER
    assert _type.a == 'a'
    assert _type.b
    assert _type.c is None
    assert _type.foofoo is None

    _type.foofoo = 'bar'
    assert _type.foofoo == 'bar'

    delattr(_type, 'foofoo')
    assert _type.foofoo is None


def test_symbol_attributes_compare():
    """
    Test dedicated `type.compare` methods that allows certain
    attributes to be excluded from comparison.
    """
    someint = SymbolAttributes('integer', a='a', b=True, c=None)
    another = SymbolAttributes('integer', a='a', b=False, c=None)
    somereal = SymbolAttributes('real', a='a', b=True, c=None)

    assert not someint.compare(another)
    assert not another.compare(someint)
    assert someint.compare(another, ignore='b')
    assert another.compare(someint, ignore=['b'])
    assert not someint.compare(somereal)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[
  (OFP, 'OFP needs preprocessing to support contiguous keyword'
)]))
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
    assert routine.symbol_attrs['a'].parameter
    assert routine.symbol_attrs['b'].intent == 'in'
    assert routine.symbol_attrs['c'].target
    assert routine.symbol_attrs['c'].intent == 'inout'
    assert routine.symbol_attrs['d'].allocatable
    assert routine.symbol_attrs['e'].pointer
    assert routine.symbol_attrs['e'].contiguous


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Segfault with pragmas in derived types')]))
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

    source = Sourcefile.from_source(fcode, frontend=frontend)
    pragma_type = source['types'].symbol_attrs['pragma_type'].dtype

    assert pragma_type.typedef is source['types'].typedefs['pragma_type']
    assert fsymgen(pragma_type.typedef.variables[0].shape) == '(3, 3)'
    assert fsymgen(pragma_type.typedef.variables[1].shape) == '(klon, klat, 2)'


@pytest.mark.parametrize('frontend', available_frontends())
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
    for var_getter in [lambda v: v.type.dtype.typedef.variables, lambda v: v.variables]:
        assert len(var_getter(a)) == 2
        assert len(var_getter(b)) == 2
        assert len(var_getter(c)) == 2
    assert all(v.scope is routine for v in a.variables)
    assert all(v.scope is routine for v in b.variables)
    assert all(v.scope is routine for v in c.variables)

    # Ensure all member variable have an entry in the local symbol table
    assert routine.symbol_attrs['a%a'].shape == (':',)
    assert routine.symbol_attrs['a%b'].shape == (':',':')
    assert routine.symbol_attrs['b%a'].shape == (':',)
    assert routine.symbol_attrs['b%b'].shape == (':',':')
    assert routine.symbol_attrs['c%a'].shape == (':',)
    assert routine.symbol_attrs['c%b'].shape == (':',':')


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot deal with deferred type info')]))
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
    assert routine.symbol_attrs['a_kind'].dtype == BasicType.DEFERRED
    assert routine.symbol_attrs['a_dim'].dtype == BasicType.DEFERRED
    assert routine.symbol_attrs['a_type'].dtype == BasicType.DEFERRED

    # Ensure local variable info is correct, as far as known
    arg_a, arg_b = routine.variables
    assert arg_a.type.kind.type.compare(routine.symbol_attrs['a_kind'], ignore=('imported'))
    assert arg_a.dimensions[0].type.compare(routine.symbol_attrs['a_dim'])
    assert isinstance(arg_b.type.dtype, DerivedType)
    assert arg_b.type.dtype.typedef == BasicType.DEFERRED

    fcode_module = """
module my_types_mod
  implicit none

  integer, parameter :: a_kind = 4
  integer(kind=a_kind) :: a_dim

  type a_type
    real(kind=a_kind), allocatable :: a(:), b(:,:)
  end type a_type
end module my_types_mod
"""
    module = Module.from_source(fcode_module, frontend=frontend)
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend)

    # Check that module variables and types have been imported
    assert routine.symbol_attrs['a_kind'].dtype == BasicType.INTEGER
    assert routine.symbol_attrs['a_kind'].parameter
    assert routine.symbol_attrs['a_kind'].initial == 4
    assert routine.symbol_attrs['a_dim'].dtype == BasicType.INTEGER
    assert routine.symbol_attrs['a_dim'].kind == 'a_kind'
    assert isinstance(routine.symbol_attrs['a_type'].dtype.typedef, TypeDef)

    # Check that external type definition has been linked
    assert isinstance(routine.variable_map['arg_b'].type.dtype.typedef, TypeDef)
    assert routine.variable_map['arg_b'].type.dtype.typedef.symbol_attrs != routine.symbol_attrs

    # Check that we correctly re-scoped the member variable
    a, b = routine.variable_map['arg_b'].variables
    assert ','.join(str(d) for d in a.dimensions) == ':'
    assert ','.join(str(d) for d in b.dimensions) == ':,:'
    assert a.type.kind == b.type.kind == 'a_kind'
    assert a.scope is routine
    assert b.scope is routine

    # Ensure all member variable have an entry in the local symbol table
    assert routine.symbol_attrs['arg_b%a'].shape == (':',)
    assert routine.symbol_attrs['arg_b%b'].shape == (':',':')


@pytest.mark.parametrize('frontend', available_frontends())
def test_type_char_length(frontend):
    """
    Test the various beautiful ways of how Fortran allows to specify
    character lengths
    """
    fcode = f"""
subroutine test_type_char_length
    implicit none
    character*80  :: kill_it_with_fire
    character(60) :: if_you_insist
    character(len=21) :: okay
    character(len=*) :: oh_dear
    character(len=:) :: come_on
    character :: you_gotta_be_kidding_me*20
    character(*) :: whatever(5)
    character(10, 1) :: this_is_getting_silly
    {'character(11, kind=1) :: i_mean' if frontend != OMNI else ''}
    character(len=12, kind=1) :: WHAT
    character(kind=1) :: DO_YOU_WANT
    character(kind=1, len=13) :: FROM_ME
    character*(*) :: where_do_I_begin
    character :: and_how_does_it_end*(*)
end subroutine test_type_char_length
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert routine.variable_map['kill_it_with_fire'].type.length == '80'
    assert routine.variable_map['if_you_insist'].type.length == '60'
    assert routine.variable_map['okay'].type.length == '21'
    assert routine.variable_map['oh_dear'].type.length == '*'
    assert routine.variable_map['come_on'].type.length == ':'
    assert routine.variable_map['you_gotta_be_kidding_me'].type.length == '20'
    assert routine.variable_map['whatever'].type.length == '*'
    assert routine.variable_map['whatever'].shape == ('5',)
    assert routine.variable_map['this_is_getting_silly'].type.length == '10'
    if frontend != OMNI:
        # OMNI swallows this one
        assert routine.variable_map['this_is_getting_silly'].type.kind == '1'
    if frontend != OMNI:
        # OMNI fails with syntax error on this one
        assert routine.variable_map['i_mean'].type.length == '11'
        assert routine.variable_map['i_mean'].type.kind == '1'
    assert routine.variable_map['what'].type.length == '12'
    assert routine.variable_map['what'].type.kind == '1'
    assert routine.variable_map['do_you_want'].type.length is None
    if frontend != OMNI:
        # OMNI swallows that one, too
        assert routine.variable_map['do_you_want'].type.kind == '1'
    assert routine.variable_map['from_me'].type.length == '13'
    if frontend != OMNI:
        # And that one
        assert routine.variable_map['from_me'].type.kind == '1'
    assert routine.variable_map['and_how_does_it_end'].type.length == '*'

    code = routine.to_fortran()
    for length in ('80', '60', '21', '*', ':', '20'):
        assert f'CHARACTER(LEN={length}) ::' in code

    if frontend == OMNI:
        for length in (10, 13):
            assert f'CHARACTER(LEN={length!s}) :: ' in code
        assert 'CHARACTER(LEN=12, KIND=1) :: ' in code

    else:
        for length in range(10, 14):
            assert f'CHARACTER(LEN={length!s}, KIND=1) :: ' in code
        assert 'CHARACTER(KIND=1) :: ' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_type_kind_value(frontend):
    """
    Test the various way how kind parameters can be specified
    """
    fcode = """
subroutine test_type_kind_value
    implicit none

    integer, parameter :: jprb = selected_real_kind(13,300)
    integer, parameter :: jpim = selected_int_kind(9)

    integer*8 :: int_8_s
    integer(8) :: int_8_p
    integer(kind=8) :: int_8_k

    integer(jpim) :: int_jpim_p
    integer(kind=jpim) :: int_jpim_k

    real*16 :: real_16_s
    real(16) :: real_16_p
    real(kind=16) :: real_16_k

    real(jprb) :: real_jprb_p
    real(kind=jprb) :: real_jprb_k
end subroutine test_type_kind_value
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    if frontend == OMNI:
        int_kinds = ('8', 'selected_int_kind(9)')
        real_kinds = ('16', 'selected_real_kind(13, 300)')
    else:
        int_kinds = ('8', 'jpim')
        real_kinds = ('16', 'jprb')

    for kind in int_kinds:
        for var in routine.variables:
            if var.name.lower().startswith(f'int_{kind}'):
                assert var.type.kind == kind and f'INTEGER(KIND={kind.upper()})' in str(fgen(var.type)).upper()

    for kind in real_kinds:
        for var in routine.variables:
            if var.name.lower().startswith(f'real_{kind}'):
                assert var.type.kind == kind and f'REAL(KIND={kind.upper()})' in str(fgen(var.type)).upper()
