from pathlib import Path
import math
import sys
import pytest
import numpy as np

import pymbolic.primitives as pmbl

from conftest import (
    jit_compile, clean_test, stdchannel_redirected, stdchannel_is_captured
)
from loki import (
    OFP, OMNI, FP, Sourcefile, fgen, Cast, RangeIndex, Assignment, Intrinsic, Variable,
    Nullify, IntLiteral, FloatLiteral, IntrinsicLiteral, InlineCall, Subroutine,
    FindVariables, FindNodes, SubstituteExpressions, Scope, BasicType, SymbolAttributes,
    parse_fparser_expression, Sum, DerivedType, ProcedureType
)
from loki.expression import symbols
from loki.tools import gettempdir, filehash


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_arithmetic(here, frontend):
    """
    Test simple floating point arithmetic expressions (+,-,*,/,**).
    """
    fcode = """
subroutine arithmetic_expr(v1, v2, v3, v4, v5, v6)
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: v1, v2, v3, v4
  real(kind=jprb), intent(out) :: v5, v6

  v5 = (v1 + v2) * (v3 - v4)
  v6 = (v1 ** v2) - (v3 / v4)
end subroutine arithmetic_expr
"""
    filepath = here/('expression_arithmetic_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='arithmetic_expr')

    v5, v6 = function(2., 3., 10., 5.)
    assert v5 == 25. and v6 == 6.
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_math_intrinsics(here, frontend):
    """
    Test supported math intrinsic functions (min, max, exp, abs, sqrt, log)
    """
    fcode = """
subroutine math_intrinsics(v1, v2, vmin, vmax, vabs, vexp, vsqrt, vlog)
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: v1, v2
  real(kind=jprb), intent(out) :: vmin, vmax, vabs, vexp, vsqrt, vlog

  vmin = min(v1, v2)
  vmax = max(v1, v2)
  vabs = abs(v1 - v2)
  vexp = exp(v1 + v2)
  vsqrt = sqrt(v1 + v2)
  vlog = log(v1 + v2)
end subroutine math_intrinsics
"""
    filepath = here/('expression_math_intrinsics_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='math_intrinsics')

    vmin, vmax, vabs, vexp, vsqrt, vlog = function(2., 4.)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vexp == np.exp(6.) and vsqrt == np.sqrt(6.) and vlog == np.log(6.)
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_logicals(here, frontend):
    """
    Test logical expressions (and, or, not, tru, false, equal, not nequal).
    """
    fcode = """
subroutine logicals(t, f, vand_t, vand_f, vor_t, vor_f, vnot_t, vnot_f, vtrue, vfalse, veq, vneq)
  logical, intent(in) :: t, f
  logical, intent(out) :: vand_t, vand_f, vor_t, vor_f, vnot_t, vnot_f, vtrue, vfalse, veq, vneq

  vand_t = t .and. t
  vand_f = t .and. f
  vor_t = t .or. f
  vor_f = f .or. f
  vnot_t = .not. f
  vnot_f = .not. t
  vtrue = .true.
  vfalse = .false.
  veq = 3 == 4
  vneq = 3 /= 4
end subroutine logicals
"""
    filepath = here/('expression_logicals_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='logicals')

    vand_t, vand_f, vor_t, vor_f, vnot_t, vnot_f, vtrue, vfalse, veq, vneq = function(True, False)
    assert vand_t and vor_t and vnot_t and vtrue and vneq
    assert not(vand_f and vor_f and vnot_f and vfalse and veq)
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_literals(here, frontend):
    """
    Test simple literal values.
    """
    fcode = """
subroutine literals(v1, v2, v3, v4, v5, v6)
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(out) :: v1, v2, v3
  real(kind=selected_real_kind(13,300)), intent(out) :: v4, v5, v6

  v1 = 66
  v2 = 66.0
  v3 = 2.3
  v4 = 2.4_jprb
  v5 = real(6, kind=jprb) + real(1, kind=selected_real_kind(13,300))
  v6 = real(3.5,jprb)
  v6 = int(3.5)
end subroutine literals
"""
    filepath = here/('expression_literals_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='literals')

    v1, v2, v3, v4, v5, v6 = function()
    assert v1 == 66. and v2 == 66. and v4 == 2.4 and v5 == 7.0 and v6 == 3.0
    assert math.isclose(v3, 2.3, abs_tol=1.e-6)
    clean_test(filepath)

    # In addition to value testing, let's make sure
    # that we created the correct expression types
    stmts = FindNodes(Assignment).visit(routine.body)
    assert isinstance(stmts[0].rhs, IntLiteral)
    assert isinstance(stmts[1].rhs, FloatLiteral)
    assert isinstance(stmts[2].rhs, FloatLiteral)
    assert isinstance(stmts[3].rhs, FloatLiteral)
    assert stmts[3].rhs.kind in ['jprb']
    assert isinstance(stmts[4].rhs, Sum)
    for expr in stmts[4].rhs.children:
        assert isinstance(expr, Cast)
        assert str(expr.kind).lower() in ['selected_real_kind(13, 300)', 'jprb']
    assert isinstance(stmts[5].rhs, Cast)
    assert str(stmts[5].rhs.kind).lower() in ['selected_real_kind(13, 300)', 'jprb']
    assert isinstance(stmts[6].rhs, Cast)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_boz_literals(here, frontend):
    """
    Test boz literal values.
    """
    fcode = """
subroutine boz_literals(n1, n2, n3, n4, n5, n6)
  integer, intent(out) :: n1, n2, n3, n4, n5, n6

  n1 = B'00000'
  n2 = b"101010"
  n3 = O'737'
  n4 = o"007"
  n5 = Z'CAFE'
  n6 = z"babe"
end subroutine boz_literals
"""
    filepath = here/('expression_boz_literals_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='boz_literals')

    n1, n2, n3, n4, n5, n6 = function()
    clean_test(filepath)
    assert n1 == 0 and n2 == 42 and n3 == 479 and n4 == 7 and n5 == 51966 and n6 == 47806

    # In addition to value testing, let's make sure that we created the correct expression types
    if frontend is not OMNI:
        # Note: Omni evaluates BOZ constants, so it creates IntegerLiteral instead...
        # Note: FP converts constants to upper case
        stmts = FindNodes(Assignment).visit(routine.body)
        assert isinstance(stmts[0].rhs, IntrinsicLiteral) and stmts[0].rhs.value == "B'00000'"
        assert isinstance(stmts[1].rhs, IntrinsicLiteral) and stmts[1].rhs.value == 'b"101010"'
        assert isinstance(stmts[2].rhs, IntrinsicLiteral) and stmts[2].rhs.value == "O'737'"
        assert isinstance(stmts[3].rhs, IntrinsicLiteral) and stmts[3].rhs.value == 'o"007"'
        assert isinstance(stmts[4].rhs, IntrinsicLiteral) and stmts[4].rhs.value == "Z'CAFE'"
        assert isinstance(stmts[5].rhs, IntrinsicLiteral) and stmts[5].rhs.value == 'z"babe"'


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='They are represented too stupid in OFP parse tree')),
    OMNI,
    FP
])
def test_complex_literals(here, frontend):
    """
    Test complex literal values.
    """
    fcode = """
subroutine complex_literals(c1, c2, c3)
  complex, intent(out) :: c1, c2, c3

  c1 = (1.0, -1.0)
  c2 = (3, 2E8)
  c3 = (21_2, 4._8)
end subroutine complex_literals
"""
    filepath = here/('expression_complex_literals_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='complex_literals')

    c1, c2, c3 = function()
    clean_test(filepath)
    assert c1 == (1-1j) and c2 == (3+2e8j) and c3 == (21+4j)

    # In addition to value testing, let's make sure that we created the correct expression types
    stmts = FindNodes(Assignment).visit(routine.body)
    assert isinstance(stmts[0].rhs, IntrinsicLiteral) and stmts[0].rhs.value == '(1.0, -1.0)'
    # Note: Here, for inconsistency, FP converts the exponential letter 'e' to lower case...
    assert isinstance(stmts[1].rhs, IntrinsicLiteral) and stmts[1].rhs.value.lower() == '(3, 2e8)'
    assert isinstance(stmts[2].rhs, IntrinsicLiteral)
    try:
        assert stmts[2].rhs.value == '(21_2, 4._8)'
    except AssertionError as excinfo:
        if frontend == OMNI:
            pytest.xfail('OMNI wrongfully assigns the same kind to real and imaginary part')
        raise excinfo


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_casts(here, frontend):
    """
    Test data type casting expressions.
    """
    fcode = """
subroutine casts(v1, v2, v3, v4, v5)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: v1
  real(kind=jprb), intent(in) :: v2, v3
  real(kind=jprb), intent(out) :: v4, v5

  v4 = real(v1, kind=jprb)  ! Test a plain cast
  v5 = real(v1, kind=jprb) * max(v2, v3)  ! Cast as part of expression
end subroutine casts
"""
    filepath = here/('expression_casts_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='casts')

    v4, v5 = function(2, 1., 4.)
    assert v4 == 2. and v5 == 8.
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_logical_array(here, frontend):
    """
    Test logical arrays for masking.
    """
    fcode = """
subroutine logical_array(dim, arr, out)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: dim
  real(kind=jprb), intent(in) :: arr(dim)
  real(kind=jprb), intent(out) :: out(dim)
  logical :: mask(dim)
  integer :: i

  mask(:) = .true.
  mask(1) = .false.
  mask(2) = .false.

  do i=1, dim
    ! Use a logical array and a relational
    ! containing an array in a single expression
    if (mask(i) .and. arr(i) > 1.) then
      out(i) = 3.
    else
      out(i) = 1.
    end if
  end do
end subroutine logical_array
"""
    filepath = here/('expression_logical_array_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='logical_array')

    out = np.zeros(6)
    function(6, [0., 2., -1., 3., 0., 2.], out)
    assert (out == [1., 1., 1., 3., 1., 3.]).all()
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Precedence not honoured')),
    FP
])
def test_parenthesis(frontend):
    """
    Test explicit parethesis in provided source code.

    Note, that this test is very niche, as it ensures that mathematically
    insignificant (and hence sort of wrong) bracketing is still honoured.
    The reason is that, if sub-expressions are sufficiently complex,
    this can still cause round-off deviations and hence destroy
    bit-reproducibility.

    Also note, that the OMNI-frontend parser will resolve precedence and
    hence we cannot honour these precedence cases (for now).
    """

    fcode = """
subroutine parenthesis(v1, v2, v3)
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: v1, v2
  real(kind=jprb), intent(out) :: v3

  v3 = (v1**1.23_jprb) * 1.3_jprb + (1_jprb - v2**1.26_jprb)
end subroutine parenthesis
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    stmt = FindNodes(Assignment).visit(routine.body)[0]

    # Check that the reduntant bracket around the minus
    # and the first exponential are still there.
    # assert str(stmt.expr) == '1.3*(v1**1.23) + (1 - v2**1.26)'
    assert fgen(stmt) == 'v3 = (v1**1.23_jprb)*1.3_jprb + (1_jprb - v2**1.26_jprb)'

    # Now perform a simple substitutions on the expression
    # and make sure we are still parenthesising as we should!
    v2 = [v for v in FindVariables().visit(stmt) if v.name == 'v2'][0]
    v4 = v2.clone(name='v4')
    stmt2 = SubstituteExpressions({v2: v4}).visit(stmt)
    # assert str(stmt2.expr) == '1.3*(v1**1.23) + (1 - v4**1.26)'
    assert fgen(stmt2) == 'v3 = (v1**1.23_jprb)*1.3_jprb + (1_jprb - v4**1.26_jprb)'


@pytest.mark.parametrize('frontend', [OFP, FP, OMNI])
def test_commutativity(frontend):
    """
    Verifies the strict adherence to ordering of commutative terms,
    which can introduce round-off errors if not done conservatively.
    """
    fcode = """
subroutine commutativity(v1, v2, v3)
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), pointer, intent(in) :: v1(:), v2
  real(kind=jprb), pointer, intent(out) :: v3(:)

  v3(:) = 1._jprb + v2*v1(:) - v2 - v3(:)
end subroutine commutativity
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    stmt = FindNodes(Assignment).visit(routine.body)[0]

    # assert str(stmt.expr) == '1.0 + v2*v1(:) - v2 - v3(:)'
    assert fgen(stmt) in ('v3(:) = 1.0_jprb + v2*v1(:) - v2 - v3(:)',
                          'v3(:) = 1._jprb + v2*v1(:) - v2 - v3(:)')


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_index_ranges(frontend):
    """
    Test index range expressions for array accesses.
    """
    fcode = """
subroutine index_ranges(dim, v1, v2, v3, v4, v5)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: dim
  real(kind=jprb), intent(in) :: v1(:), v2(0:), v3(0:4), v4(dim)
  real(kind=jprb), intent(out) :: v5(1:dim)

  v5(:) = v2(1:dim)*v1(::2) - v3(0:4:2)
end subroutine index_ranges
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    vmap = routine.variable_map

    assert str(vmap['v1']) == 'v1(:)'
    assert str(vmap['v2']) == 'v2(0:)'
    assert str(vmap['v3']) == 'v3(0:4)'
    # OMNI will insert implicit lower=1 into shape declarations,
    # we simply have to live with it... :(
    assert str(vmap['v4']) == 'v4(dim)' or str(vmap['v4']) == 'v4(1:dim)'
    assert str(vmap['v5']) == 'v5(1:dim)'

    vmap_body = {v.name: v for v in FindVariables().visit(routine.body)}
    assert str(vmap_body['v1']) == 'v1(::2)'
    assert str(vmap_body['v2']) == 'v2(1:dim)'
    assert str(vmap_body['v3']) == 'v3(0:4:2)'
    assert str(vmap_body['v5']) == 'v5(:)'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_strings(here, frontend, capsys):
    """
    Test recognition of literal strings.
    """

    # This tests works only if stdout/stderr is not captured by pytest
    if stdchannel_is_captured(capsys):
        pytest.skip('pytest executed without "--show-capture"/"-s"')

    fcode = """
subroutine strings()
  print *, 'Hello world!'
  print *, "42!"
end subroutine strings
"""
    filepath = here/('expression_strings_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)

    function = jit_compile(routine, filepath=filepath, objname='strings')
    output_file = gettempdir()/filehash(str(filepath), prefix='', suffix='.log')
    with capsys.disabled():
        with stdchannel_redirected(sys.stdout, output_file):
            function()

    clean_test(filepath)

    with open(output_file, 'r') as f:
        output_str = f.read()

    assert output_str == ' Hello world!\n 42!\n'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_very_long_statement(here, frontend):
    """
    Test a long statement with line breaks.
    """
    fcode = """
subroutine very_long_statement(scalar, res)
  integer, intent(in) :: scalar
  integer, intent(out) :: res

  res = 5 * scalar + scalar - scalar + scalar - scalar + (scalar - scalar &
      & + scalar - scalar) - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8 - (9 + 10      &
        - 9) + 10 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1
end subroutine very_long_statement
"""
    filepath = here/('expression_very_long_statement_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='very_long_statement')

    scalar = 1
    result = function(scalar)
    assert result == 5
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_output_intrinsics(frontend):
    """
    Some collected intrinsics or other edge cases that failed in cloudsc.
    """
    fcode = """
subroutine output_intrinsics
     integer, parameter :: jprb = selected_real_kind(13,300)
     integer :: numomp, ngptot
     real(kind=jprb) :: tdiff

     numomp = 1
     ngptot = 2
     tdiff = 1.2

1002 format(1x, 2i10, 1x, i4, ' : ', i10)
     write(0, 1002) numomp, ngptot, - 1, int(tdiff * 1000.0_jprb)
end subroutine output_intrinsics
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    ref = ['format(1x, 2i10, 1x, i4, \' : \', i10)',
           'write(0, 1002) numomp, ngptot, - 1, int(tdiff * 1000.0_jprb)']

    if frontend == OMNI:
        ref[0] = ref[0].replace("'", '"')
        ref[1] = ref[1].replace('0, 1002', 'unit=0, fmt=1002')
        ref[1] = ref[1].replace(' * ', '*')
        ref[1] = ref[1].replace('- 1', '-1')

    intrinsics = FindNodes(Intrinsic).visit(routine.body)
    assert len(intrinsics) == 2
    assert intrinsics[0].text.lower() == ref[0]
    assert intrinsics[1].text.lower() == ref[1]
    assert fgen(intrinsics).lower() == '{} {}\n{}'.format('1002', *ref)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_nested_call_inline_call(here, frontend):
    """
    The purpose of this test is to highlight the differences between calls in expression
    (such as `InlineCall`, `Cast`) and call nodes in the IR.
    """
    fcode = """
subroutine simple_expr(v1, v2, v3, v4, v5, v6)
  ! simple floating point arithmetic
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: v1, v2, v3, v4
  real(kind=jprb), intent(out) :: v5, v6

  v5 = (v1 + v2) * (v3 - v4)
  v6 = (v1 ** v2) - (v3 / v4)
end subroutine simple_expr

subroutine very_long_statement(scalar, res)
  integer, intent(in) :: scalar
  integer, intent(out) :: res

  res = 5 * scalar + scalar - scalar + scalar - scalar + (scalar - scalar &
        + scalar - scalar) - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8 - (9 + 10      &
        - 9) + 10 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1
end subroutine very_long_statement

subroutine nested_call_inline_call(v1, v2, v3)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: v1
  real(kind=jprb), intent(out) :: v2
  integer, intent(out) :: v3
  real(kind=jprb) :: tmp1, tmp2

  tmp1 = real(1, kind=jprb)
  call simple_expr(tmp1, abs(-2.0_jprb), 3.0_jprb, real(v1, jprb), v2, tmp2)
  v2 = abs(tmp2 - v2)
  call very_long_statement(int(v2), v3)
end subroutine nested_call_inline_call
"""
    filepath = here/('expression_nested_call_inline_call_%s.f90' % frontend)
    routine = Sourcefile.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='nested_call_inline_call')

    v2, v3 = function(1)
    assert v2 == 8.
    assert v3 == 40
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Not implemented')),
    OMNI,
    FP
])
def test_character_concat(here, frontend):
    """
    Concatenation operator ``//``
    """
    fcode = """
subroutine character_concat(string)
  character(10) :: tmp_str1, tmp_str2
  character(len=12), intent(out) :: string

  tmp_str1 = "Hel" // "lo"
  tmp_str2 = "wor" // "l" // "d"
  string = trim(tmp_str1) // " " // trim(tmp_str2)
  string = trim(string) // "!"
end subroutine character_concat
"""
    filepath = here/('expression_character_concat_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='character_concat')

    result = function()
    assert result == b'Hello world!'
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Inline WHERE not implemented')),
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Not implemented')),
    FP
])
def test_masked_statements(here, frontend):
    """
    Masked statements (WHERE(...) ... [ELSEWHERE ...] ENDWHERE)
    """
    fcode = """
subroutine masked_statements(length, vec1, vec2, vec3)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: length
  real(kind=jprb), intent(inout), dimension(length) :: vec1, vec2, vec3

  where (vec1(:) > 5.0_jprb)
    vec1(:) = 7.0_jprb
    vec1(:) = 5.0_jprb
  endwhere

  where (vec2(:) < 0.d0)
    vec2(:) = 0.0_jprb
  elsewhere
    vec2(:) = 1.0_jprb
  endwhere

  where (0.0_jprb < vec3(:) .and. vec3(:) < 3.0_jprb) vec3(:) = 1.0_jprb
end subroutine masked_statements
"""
    filepath = here/('expression_masked_statements_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='masked_statements')

    # Reference solution
    length = 11
    ref1 = np.append(np.arange(0, 6, dtype=np.float64),
                     5 * np.ones(length - 6, dtype=np.float64))
    ref2 = np.append(np.zeros(5, dtype=np.float64),
                     np.ones(length - 5, dtype=np.float64))
    ref3 = np.append(np.arange(-2, 1, dtype=np.float64), np.ones(2, dtype=np.float64))
    ref3 = np.append(ref3, np.arange(3, length - 2, dtype=np.float64))

    vec1 = np.arange(0, length, dtype=np.float64)
    vec2 = np.arange(-5, length - 5, dtype=np.float64)
    vec3 = np.arange(-2, length - 2, dtype=np.float64)
    function(length, vec1, vec2, vec3)
    assert np.all(ref1 == vec1)
    assert np.all(ref2 == vec2)
    assert np.all(ref3 == vec3)
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Not implemented')),
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Not implemented')),
    pytest.param(FP, marks=pytest.mark.xfail(reason='Not implemented')),
])
def test_data_declaration(here, frontend):
    """
    Variable initialization with DATA statements
    """
    fcode = """
subroutine data_declaration(data_out)
  implicit none
  integer, dimension(5, 4), intent(out) :: data_out
  integer, dimension(5, 4) :: data1, data2
  integer, dimension(3) :: data3
  integer :: i, j

  data data1 /20*5/

  data ((data2(i,j), i=1,5), j=1,4) /20*3/

  data data3(1), data3(3), data3(2) /1, 2, 3/

  data_out(:,:) = data1(:,:) + data2(:,:)
  data_out(1:3,1) = data3
end subroutine data_declaration
"""
    filepath = here/('expression_data_declaration_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='data_declaration')

    expected = np.ones(shape=(5, 4), dtype=np.int32, order='F') * 8
    expected[[0, 1, 2], 0] = [1, 3, 2]
    result = np.zeros(shape=(5, 4), dtype=np.int32, order='F')
    function(result)
    assert np.all(result == expected)
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Not implemented')),
    FP
])
def test_pointer_nullify(here, frontend):
    """
    POINTERS and their nullification via '=> NULL()'
    """
    fcode = """
subroutine pointer_nullify()
  implicit none
  character(len=64), dimension(:), pointer :: charp => NULL()
  character(len=64), pointer :: pp => NULL()
  allocate(charp(3))
  charp(:) = "_ptr_"
  pp => charp(1)
  pp = "_other_ptr_"
  nullify(pp)
  deallocate(charp)
  charp => NULL()
end subroutine pointer_nullify
"""
    filepath = here/('expression_pointer_nullify_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert np.all(v.type.pointer for v in routine.variables)
    assert np.all(isinstance(v.initial, InlineCall) and v.type.initial.name.lower() == 'null'
                  for v in routine.variables)
    nullify_stmts = FindNodes(Nullify).visit(routine.body)
    assert len(nullify_stmts) == 1
    assert nullify_stmts[0].variables[0].name == 'pp'
    assert [stmt.ptr for stmt in FindNodes(Assignment).visit(routine.body)].count(True) == 2

    # Execute the generated identity (to verify it is valid Fortran)
    function = jit_compile(routine, filepath=filepath, objname='pointer_nullify')
    function()
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_parameter_stmt(here, frontend):
    """
    PARAMETER(...) statement
    """
    fcode = """
subroutine parameter_stmt(out1)
  implicit none
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb) :: param
  parameter(param=2.0)
  real(kind=jprb), intent(out) :: out1

  out1 = param
end subroutine parameter_stmt
"""
    filepath = here/('expression_parameter_stmt_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='parameter_stmt')

    out1 = function()
    assert out1 == 2.0
    clean_test(filepath)


def test_string_compare():
    """
    Test that we can identify symbols and expressions by equivalent strings.

    Note that this only captures comparsion of a canonical string representation,
    not full symbolic equivalence.
    """
    # Utility objects for manual expression creation
    scope = Scope()
    type_int = SymbolAttributes(dtype=BasicType.INTEGER)
    type_real = SymbolAttributes(dtype=BasicType.REAL)

    i = Variable(name='i', scope=scope, type=type_int)
    j = Variable(name='j', scope=scope, type=type_int)

    # Test a scalar variable
    u = Variable(name='u', scope=scope, type=SymbolAttributes(dtype=BasicType.REAL))
    assert all(u == exp for exp in ['u', 'U', 'u ', 'U '])
    assert not all(u == exp for exp in ['u()', '_u', 'U()', '_U'])

    # Test an array variable
    v = Variable(name='v', dimensions=(i, j), scope=scope, type=type_real)
    assert all(v == exp for exp in ['v(i,j)', 'v(i, j)', 'v (i , j)', 'V(i,j)', 'V(I, J)'])
    assert not all(v == exp for exp in ['v(i,j())', 'v(i,_j)', '_V(i,j)'])

    # Test array variable dimensions (ArraySubscript)
    assert all(v.dimensions == exp for exp in ['(i,j)', '(i, j)', ' (i ,  j)', '(i,J)', '(I, J)'])  # pylint: disable=no-member
    assert not all(v.dimensions == exp for exp in ['i, j', '(j, i)', '[i, j]'])  # pylint: disable=no-member

    # Test a standard array dimension range
    r = RangeIndex(children=(i, j))
    w = Variable(name='w', dimensions=(r,), scope=scope, type=type_real)
    assert all(w == exp for exp in ['w(i:j)', 'w (i : j)', 'W(i:J)', ' w( I:j)'])

    # Test simple arithmetic expressions
    assert all(symbols.Sum((i, u)) == exp for exp in ['i+u', 'i + u', 'i +  U', ' I + u'])
    assert all(symbols.Product((i, u)) == exp for exp in ['i*u', 'i * u', 'i *  U', ' I * u'])
    assert all(symbols.Quotient(i, u) == exp for exp in ['i/u', 'i / u', 'i /  U', ' I / u'])
    assert all(symbols.Power(i, u) == exp for exp in ['i**u', 'i ** u', 'i **  U', ' I ** u'])
    assert all(symbols.Comparison(i, '==', u) == exp for exp in ['i==u', 'i == u', 'i ==  U', ' I == u'])
    assert all(symbols.LogicalAnd((i, u)) == exp for exp in ['i AND u', 'i and u', 'i and  U', ' I and u'])
    assert all(symbols.LogicalOr((i, u)) == exp for exp in ['i OR u', 'i or u', 'i or  U', ' I oR u'])
    assert all(symbols.LogicalNot(u) == exp for exp in ['not u', ' nOt u', 'not  U', ' noT u'])

    # Test literal behaviour
    assert symbols.Literal(41) == 41
    assert symbols.Literal(41) == '41'
    assert symbols.Literal(41) != symbols.Literal(41, kind='jpim')
    assert symbols.Literal(66.6) == 66.6
    assert symbols.Literal(66.6) == '66.6'
    assert symbols.Literal(66.6) != symbols.Literal(66.6, kind='jprb')
    assert symbols.Literal('u') == 'u'
    assert symbols.Literal('u') != 'U'
    assert symbols.Literal('u') != u  # The `Variable(name='u', ...) from above


@pytest.mark.parametrize('source, ref', [
    ('1 + 1', '1 + 1'),
    ('1+2+3+4', '1 + 2 + 3 + 4'),
    ('5*4 - 3*2 - 1', '5*4 - 3*2 - 1'),
    ('1*(2 + 3)', '1*(2 + 3)'),
    ('5*a +3*7**5 - 4/b', '5*a + 3*7**5 - 4 / b'),
    ('5 + (4 + 3) - (2*1)', '5 + (4 + 3) - (2*1)'),
    ('a*(b*(c+(d+e)))', 'a*(b*(c + (d + e)))'),
])
def test_parse_fparser_expression(source, ref):
    """
    Test the utility function that parses simple expressions.
    """
    scope = Scope()
    ir = parse_fparser_expression(source, scope)
    assert isinstance(ir, pmbl.Expression)
    assert str(ir) == ref


@pytest.mark.parametrize('kwargs,reftype', [
    ({}, symbols.DeferredTypeSymbol),
    ({'type': SymbolAttributes(BasicType.DEFERRED)}, symbols.DeferredTypeSymbol),
    ({'type': SymbolAttributes(BasicType.INTEGER)}, symbols.Scalar),
    ({'type': SymbolAttributes(BasicType.REAL)}, symbols.Scalar),
    ({'type': SymbolAttributes(DerivedType('t'))}, symbols.Scalar),
    ({'type': SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(3),))}, symbols.Array),
    ({'type': SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(3),)),
      'dimensions': (symbols.Literal(1),)}, symbols.Array),
    ({'type': SymbolAttributes(BasicType.INTEGER), 'dimensions': (symbols.Literal(1),)}, symbols.Array),
    ({'type': SymbolAttributes(BasicType.DEFERRED), 'dimensions': (symbols.Literal(1),)}, symbols.Array),
    ({'type': SymbolAttributes(ProcedureType('routine'))}, symbols.ProcedureSymbol),
])
def test_variable_factory(kwargs, reftype):
    """
    Test the factory class :any:`Variable` and the dispatch to correct classes.
    """
    scope = Scope()
    assert isinstance(symbols.Variable(name='var', scope=scope, **kwargs), reftype)


def test_variable_factory_invalid():
    """
    Test invalid variable instantiations
    """
    with pytest.raises(KeyError):
        _ = symbols.Variable()


@pytest.mark.parametrize('initype,inireftype,newtype,newreftype', [
    # From deferred type to other type
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(BasicType.INTEGER), symbols.Scalar),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(BasicType.REAL), symbols.Scalar),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(DerivedType('t')), symbols.Scalar),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(4),)), symbols.Array),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(ProcedureType('routine')), symbols.ProcedureSymbol),
    (None, symbols.DeferredTypeSymbol, SymbolAttributes(BasicType.INTEGER), symbols.Scalar),
    # From Scalar to other type
    (SymbolAttributes(BasicType.INTEGER), symbols.Scalar,
     SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol),
    (SymbolAttributes(BasicType.INTEGER), symbols.Scalar,
     SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(3),)), symbols.Array),
    (SymbolAttributes(BasicType.INTEGER), symbols.Scalar,
     SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol),
    # From Array to other type
    (SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(4),)), symbols.Array,
     SymbolAttributes(BasicType.INTEGER), symbols.Scalar),
    (SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(4),)), symbols.Array,
     SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol),
    (SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(4),)), symbols.Array,
     SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol),
    # From ProcedureSymbol to other type
    (SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol,
     SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol),
    (SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol,
     SymbolAttributes(BasicType.INTEGER), symbols.Scalar),
    (SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol,
     SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(5),)), symbols.Array),
])
def test_variable_rebuild(initype, inireftype, newtype, newreftype):
    """
    Test that rebuilding a variable object changes class according to symmbol type
    """
    scope = Scope()
    var = symbols.Variable(name='var', scope=scope, type=initype)
    assert isinstance(var, inireftype)
    assert 'var' in scope.symbols
    scope.symbols['var'] = newtype
    assert isinstance(var, inireftype)
    var = var.clone()  # pylint: disable=no-member
    assert isinstance(var, newreftype)


@pytest.mark.parametrize('initype,inireftype,newtype,newreftype', [
    # From deferred type to other type
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(BasicType.INTEGER), symbols.Scalar),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(BasicType.REAL), symbols.Scalar),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(DerivedType('t')), symbols.Scalar),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(4),)), symbols.Array),
    (SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol,
     SymbolAttributes(ProcedureType('routine')), symbols.ProcedureSymbol),
    (None, symbols.DeferredTypeSymbol, SymbolAttributes(BasicType.INTEGER), symbols.Scalar),
    # From Scalar to other type
    (SymbolAttributes(BasicType.INTEGER), symbols.Scalar,
     SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol),
    (SymbolAttributes(BasicType.INTEGER), symbols.Scalar,
     SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(3),)), symbols.Array),
    (SymbolAttributes(BasicType.INTEGER), symbols.Scalar,
     SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol),
    # From Array to other type
    (SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(4),)), symbols.Array,
     SymbolAttributes(BasicType.INTEGER), symbols.Scalar),
    (SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(4),)), symbols.Array,
     SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol),
    (SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(4),)), symbols.Array,
     SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol),
    # From ProcedureSymbol to other type
    (SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol,
     SymbolAttributes(BasicType.DEFERRED), symbols.DeferredTypeSymbol),
    (SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol,
     SymbolAttributes(BasicType.INTEGER), symbols.Scalar),
    (SymbolAttributes(ProcedureType('foo')), symbols.ProcedureSymbol,
     SymbolAttributes(BasicType.INTEGER, shape=(symbols.Literal(5),)), symbols.Array),
])
def test_variable_clone(initype, inireftype, newtype, newreftype):
    """
    Test that cloning a variable object changes class according to symbol type
    """
    scope = Scope()
    var = symbols.Variable(name='var', scope=scope, type=initype)
    assert isinstance(var, inireftype)
    assert 'var' in scope.symbols
    var = var.clone(type=newtype)  # pylint: disable=no-member
    assert isinstance(var, newreftype)


def test_variable_without_scope():
    """
    Test that creating variables without scope works and scopes can be
    attached and detached
    """
    # Create a plain variable without type or scope
    var = symbols.Variable(name='var')
    assert isinstance(var, symbols.DeferredTypeSymbol)
    assert var.type and var.type.dtype is BasicType.DEFERRED
    # Attach a scope with a data type for this variable
    scope = Scope()
    scope.symbols['var'] = SymbolAttributes(BasicType.INTEGER)
    assert isinstance(var, symbols.DeferredTypeSymbol)
    assert var.type and var.type.dtype is BasicType.DEFERRED
    var.scope = scope
    assert isinstance(var, symbols.DeferredTypeSymbol)
    assert var.type.dtype is BasicType.INTEGER
    # Rebuild the variable
    var = var.clone()
    assert isinstance(var, symbols.Scalar)
    assert var.type.dtype is BasicType.INTEGER
    # Change the data type
    var.type = SymbolAttributes(BasicType.REAL)
    assert isinstance(var, symbols.Scalar)
    assert var.type.dtype is BasicType.REAL
    assert scope.symbols['var'].dtype is BasicType.REAL
    # Detach the scope
    var.scope = None
    assert isinstance(var, symbols.Scalar)
    assert var.type.dtype is BasicType.DEFERRED
    assert scope.symbols['var'].dtype is BasicType.REAL
    # Rebuild the variable
    var = var.clone()
    assert isinstance(var, symbols.DeferredTypeSymbol)
    assert var.type.dtype is BasicType.DEFERRED
    # Assign a data type locally
    var.type = SymbolAttributes(BasicType.LOGICAL)
    assert isinstance(var, symbols.DeferredTypeSymbol)
    assert var.type.dtype is BasicType.LOGICAL
    # Rebuild the variable
    var = var.clone()
    assert isinstance(var, symbols.Scalar)
    assert var.type.dtype is BasicType.LOGICAL
    # Re-attach the scope
    var.scope = scope
    assert isinstance(var, symbols.Scalar)
    assert var.type.dtype is BasicType.REAL
