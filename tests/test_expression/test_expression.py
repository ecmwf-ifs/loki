from pathlib import Path
import math
import pytest
import numpy as np

from loki import (
    clean, compile_and_load, OFP, OMNI, FP, SourceFile, fgen, as_tuple,
    Cast, Statement, Intrinsic, CallStatement, Nullify,
    IntLiteral, FloatLiteral, InlineCall, Subroutine,
    FindVariables, FindNodes, SubstituteExpressions)
from conftest import generate_identity, jit_compile, clean_test


@pytest.fixture(scope='module', name='here')
def here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent / 'expression.f90'


@pytest.fixture(scope='module', name='reference')
def fixture_reference(refpath):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    return compile_and_load(refpath, cwd=str(refpath.parent))


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
    """
    fcode = """
subroutine literals(v1, v2, v3, v4, v5, v6)
  ! simple literal values
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(out) :: v1, v2, v3, v4, v5, v6

  v1 = 66
  v2 = 66.0
  v3 = 2.3
  v4 = 2.4_jprb
  v5 = real(7, kind=jprb)
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
    stmts = FindNodes(Statement).visit(routine.body)
    assert isinstance(stmts[0].expr, IntLiteral)
    assert isinstance(stmts[1].expr, FloatLiteral)
    assert isinstance(stmts[2].expr, FloatLiteral)
    assert isinstance(stmts[3].expr, FloatLiteral)
    assert stmts[3].expr.kind in ['jprb']
    assert isinstance(stmts[4].expr, Cast)
    assert str(stmts[4].expr.kind) in ['selected_real_kind(13, 300)', 'jprb']
    assert isinstance(stmts[5].expr, Cast)
    assert str(stmts[5].expr.kind) in ['selected_real_kind(13, 300)', 'jprb']
    assert isinstance(stmts[6].expr, Cast)


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
    stmt = list(routine.body)[0]

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
    stmt = list(routine.body)[0]

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
def test_strings(here, frontend):
    """
    character(len=64), intent(inout) :: str1
    character(len=8) :: str2

    str2 = " world!"
    str1 = str1 // str2
    """
    fcode = """
subroutine strings()
  print *, 'Hello world!'
  print *, "42!"
end subroutine strings
"""
    filepath = here/('expression_strings_%s.f90' % frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='strings')

    function()
    # TODO: Need a better way to capture output of this
    assert True
    clean_test(filepath)

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
        + scalar - scalar) - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8 - (9 + 10      &
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


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Format stmt labels not implemented')),
    OMNI,
    FP
])
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

    assert isinstance(routine.body[-2], Intrinsic)
    assert isinstance(routine.body[-1], Intrinsic)
    assert routine.body[-2].text.strip('\n').lower() in ["format(1x, 2i10, 1x, i4, ' : ', i10)",
                                                         'format(1x, 2i10, 1x, i4, " : ", i10)']
    assert fgen(routine.body[-2]).lower() in ["1002 format(1x, 2i10, 1x, i4, ' : ', i10)",
                                              '1002 format(1x, 2i10, 1x, i4, " : ", i10)']
    assert routine.body[-1].text.strip('\n').lower() in \
        ['write(0, 1002) numomp, ngptot, - 1, int(tdiff * 1000.0_jprb)',
         'write(unit=0, fmt=1002) numomp, ngptot, -1, int(tdiff*1000.0_jprb)']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_nested_call_inline_call(refpath, reference, frontend):
    """
    The purpose of this test is to highlight the differences between calls in expression
    (such as `InlineCall`, `Cast`) and call nodes in the IR.
    """
    # Test the reference solution
    v2, v3 = reference.nested_call_inline_call(1)
    assert v2 == 8.
    assert v3 == 40

    # Test the generated identity
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine_names = ['nested_call_inline_call', 'simple_expr', 'very_long_statement']
    routines = []
    for routine in source.subroutines:
        if routine.name in routine_names:
            routine.name += '_%s' % frontend
            for call in FindNodes(CallStatement).visit(routine.body):
                call.name += '_%s' % frontend
            routines.append(routine)
    testname = refpath.parent / ('%s_nested_call_inline_call_%s.f90' % (refpath.stem, frontend))
    source.write(source=fgen(as_tuple(routines)), filename=testname)
    pymod = compile_and_load(testname, cwd=str(refpath.parent), use_f90wrap=True)
    function = getattr(pymod, 'nested_call_inline_call_%s' % frontend)
    v2, v3 = function(1)
    assert v2 == 8.
    assert v3 == 40


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
    FP
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
    assert FindNodes(Nullify).visit(routine.body)[0].variable.name == 'pp'
    assert [stmt.ptr for stmt in FindNodes(Statement).visit(routine.body)].count(True) == 2

    # Execute the generated identity (to verify it is valid Fortran)
    function = jit_compile(routine, filepath=filepath, objname='pointer_nullify')
    function()
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Not implemented')),
    OMNI,
    pytest.param(FP, marks=pytest.mark.xfail(reason='Order in spec not preserved')),
])
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
