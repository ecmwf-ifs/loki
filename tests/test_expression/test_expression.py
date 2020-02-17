import pytest
import numpy as np
from pathlib import Path
import math

from loki import (clean, compile_and_load, OFP, OMNI, FP, SourceFile, fgen, Cast)
from conftest import generate_identity


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'expression.f90'


@pytest.fixture(scope='module')
def reference(refpath):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    return compile_and_load(refpath, cwd=str(refpath.parent))


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_simple_expr(refpath, reference, frontend):
    """
    v5 = (v1 + v2) * (v3 - v4)
    v6 = (v1 ** v2) - (v3 / v4)
    """
    # Test the reference solution
    v5, v6 = reference.simple_expr(2., 3., 10., 5.)
    assert v5 == 25. and v6 == 6.

    # Test the generated identity
    test = generate_identity(refpath, 'simple_expr', frontend=frontend)
    function = getattr(test, 'simple_expr_%s' % frontend)
    v5, v6 = function(2., 3., 10., 5.)
    assert v5 == 25. and v6 == 6.


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_intrinsic_functions(refpath, reference, frontend):
    """
    vmin = min(v1, v2)
    vmax = max(v1, v2)
    vabs = abs(v1 - v2)
    vexp = exp(v1 + v2)
    vsqrt = sqrt(v1 + v2)
    vlog = log(v1 + v2)
    """
    # Test the reference solution
    vmin, vmax, vabs, vexp, vsqrt, vlog = reference.intrinsic_functions(2., 4.)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vexp == np.exp(6.) and vsqrt == np.sqrt(6.) and vlog == np.log(6.)

    # Test the generated identity
    test = generate_identity(refpath, 'intrinsic_functions', frontend=frontend)
    function = getattr(test, 'intrinsic_functions_%s' % frontend)
    vmin, vmax, vabs, vexp, vsqrt, vlog = function(2., 4.)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vexp == np.exp(6.) and vsqrt == np.sqrt(6.) and vlog == np.log(6.)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_logical_expr(refpath, reference, frontend):
    """
    vand_t = t .and. t
    vand_f = t .and. f
    vor_t = t .or. f
    vor_f = f .or. f
    vnot_t = .not. f
    vnot_f = .not. t
    veq = 3 == 4
    vneq = 3 /= 4
    """
    # Test the reference solution
    vand_t, vand_f, vor_t, vor_f, vnot_t, vnot_f, vtrue, vfalse, veq, vneq = reference.logical_expr(True, False)
    assert vand_t and vor_t and vnot_t and vtrue and vneq
    assert not(vand_f and vor_f and vnot_f and vfalse and veq)

    # Test the generated identity
    test = generate_identity(refpath, 'logical_expr', frontend=frontend)
    function = getattr(test, 'logical_expr_%s' % frontend)
    vand_t, vand_f, vor_t, vor_f, vnot_t, vnot_f, vtrue, vfalse, veq, vneq = function(True, False)
    assert vand_t and vor_t and vnot_t and vtrue and vneq
    assert not(vand_f and vor_f and vnot_f and vfalse and veq)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_literal_expr(refpath, reference, frontend):
    """
    v1 = 1
    v2 = 1.0
    v3 = 2.3
    v4 = 2.4_jprb
    v5 = real(7, kind=jprb)
    v6 = int(3.5)
    """
    from loki import SourceFile, FindNodes, Statement

    # Test the reference solution
    v1, v2, v3, v4, v5, v6 = reference.literal_expr()
    assert v1 == 66. and v2 == 66. and v4 == 2.4 and v5 == 7.0 and v6 == 3.0
    # Fortran will default this to single precision
    # so we need to give a significant range of error
    assert math.isclose(v3, 2.3, abs_tol=1.e-6)

    # Test the generated identity
    test = generate_identity(refpath, 'literal_expr', frontend=frontend)
    function = getattr(test, 'literal_expr_%s' % frontend)
    v1, v2, v3, v4, v5, v6 = function()
    assert v1 == 66. and v2 == 66. and v4 == 2.4 and v5 == 7.0 and v6 == 3.0
    assert math.isclose(v3, 2.3, abs_tol=1.e-6)

    # In addition to value testing, let's make sure
    # that we created the correct expression types
    from loki.expression.symbol_types import IntLiteral, FloatLiteral
    source = SourceFile.from_file(refpath, frontend=frontend)
    stmts = FindNodes(Statement).visit(source['literal_expr'].body)
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
def test_cast_expr(refpath, reference, frontend):
    """
    v4 = real(v1, kind=jprb)
    v5 = real(v1, kind=jprb) * max(v2, c3)
    """
    # Test the reference solution
    v4, v5 = reference.cast_expr(2, 1., 4.)
    assert v4 == 2. and v5 == 8.

    # Test the generated identity
    test = generate_identity(refpath, 'cast_expr', frontend=frontend)
    function = getattr(test, 'cast_expr_%s' % frontend)
    v4, v5 = function(2, 1., 4.)
    assert v4 == 2. and v5 == 8.


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_logical_array(refpath, reference, frontend):
    """
    mask(1:2) = .false.
    mask(3:) = .true.

    do i=1, dim
      ! Use a logical array and a relational
      ! containing an array in a single expression
      if (mask(i) .and. in(i) > 1.) then
        out(i) = 3.
      else
        out(i) = 1.
      end if
    end do
    """
    # Test the reference solution
    out = np.zeros(6)
    reference.logical_array(6, [0., 2., -1., 3., 0., 2.], out)
    assert (out == [1., 1., 1., 3., 1., 3.]).all()

    # Test the generated identity
    out = np.zeros(6)
    test = generate_identity(refpath, 'logical_array', frontend=frontend)
    function = getattr(test, 'logical_array_%s' % frontend)
    function(6, [0., 2., -1., 3., 0., 2.], out)
    assert (out == [1., 1., 1., 3., 1., 3.]).all()


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Precedence not honoured')),
    FP
])
def test_parenthesis(refpath, reference, frontend):
    """
    v3 = (v1**1.23_jprb) * 1.3_jprb + (1_jprb - (v2**1.26_jprb))

    Note, that this test is very niche, as it ensures that mathematically
    insignificant (and hence sort of wrong) bracketing is still honoured.
    The reason is that, if sub-expressions are sufficiently complex,
    this can still cause round-off deviations and hence destroy
    bit-reproducibility.

    Also note, that the OMNI-frontend parser will resolve precedence and
    hence we cannot honour these precedence cases (for now).
    """
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['parenthesis']
    stmt = list(routine.body)[0]

    # Check that the reduntant bracket around the minus
    # and the first exponential are still there.
    # assert str(stmt.expr) == '1.3*(v1**1.23) + (1 - v2**1.26)'
    assert fgen(stmt) == 'v3 = (v1**1.23_jprb)*1.3_jprb + (1_jprb - v2**1.26_jprb)'

    # Now perform a simple substitutions on the expression
    # and make sure we are still parenthesising as we should!
    from loki import SubstituteExpressions, FindVariables
    v2 = [v for v in FindVariables().visit(stmt) if v.name == 'v2'][0]
    v4 = v2.clone(name='v4')
    stmt2 = SubstituteExpressions({v2: v4}).visit(stmt)
    # assert str(stmt2.expr) == '1.3*(v1**1.23) + (1 - v4**1.26)'
    assert fgen(stmt2) == 'v3 = (v1**1.23_jprb)*1.3_jprb + (1_jprb - v4**1.26_jprb)'


@pytest.mark.parametrize('frontend', [OFP, FP, OMNI])
def test_commutativity(refpath, reference, frontend):
    """
    v3 = 1._jprb + v2*v1 - v2 - v3

    Verifies the strict adherence to ordering of commutative terms,
    which can introduce round-off errors if not done conservatively.
    """
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['commutativity']
    stmt = list(routine.body)[0]

    # assert str(stmt.expr) == '1.0 + v2*v1(:) - v2 - v3(:)'
    assert fgen(stmt) in ('v3(:) = 1.0_jprb + v2*v1(:) - v2 - v3(:)',
                          'v3(:) = 1._jprb + v2*v1(:) - v2 - v3(:)')


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_index_ranges(refpath, reference, frontend):
    """
    real(kind=jprb), intent(in) :: v1(:), v2(0:), v3(0:4), v4(dim)
    real(kind=jprb), intent(out) :: v5(1:dim)

    v5(:) = v2(1:dim)*v1(::2) - v3(0:4:2)
    """
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['index_ranges']
    vmap = routine.variable_map

    assert str(vmap['v1']) == 'v1(:)'
    assert str(vmap['v2']) == 'v2(0:)'
    assert str(vmap['v3']) == 'v3(0:4)'
    # OMNI will insert implicit lower=1 into shape declarations,
    # we simply have to live with it... :(
    assert str(vmap['v4']) == 'v4(dim)' or str(vmap['v4']) == 'v4(1:dim)'
    assert str(vmap['v5']) == 'v5(1:dim)'

    from loki import FindVariables
    vmap_body = {v.name: v for v in FindVariables().visit(routine.body)}
    assert str(vmap_body['v1']) == 'v1(::2)'
    assert str(vmap_body['v2']) == 'v2(1:dim)'
    assert str(vmap_body['v3']) == 'v3(0:4:2)'
    assert str(vmap_body['v5']) == 'v5(:)'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_strings(refpath, reference, frontend):
    """
    character(len=64), intent(inout) :: str1
    character(len=8) :: str2

    str2 = " world!"
    str1 = str1 // str2
    """

    # Test the generated identity
    test = generate_identity(refpath, 'strings', frontend=frontend)
    _ = getattr(test, 'strings_%s' % frontend)
    assert True


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_very_long_statement(refpath, reference, frontend):
    """
    Some long statement with line breaks.
    """
    # Test the reference solution
    scalar = 1
    result = reference.very_long_statement(scalar)
    assert result == 5

    # Test the generated identity
    test = generate_identity(refpath, 'very_long_statement', frontend=frontend)
    function = getattr(test, 'very_long_statement_%s' % frontend)
    result = function(scalar)
    assert result == 5


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Format stmt labels not implemented')),
    OMNI,
    FP
])
def test_intrinsics(refpath, reference, frontend):
    """
    Some collected intrinsics or other edge cases that failed in cloudsc.
    """
    from loki import Intrinsic, fgen

    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['intrinsics']

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
    from loki import fgen, FindNodes, CallStatement, as_tuple
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
def test_character_concat(refpath, reference, frontend):
    """
    Concatenation operator ``//``
    """
    # Test the reference solution
    ref = reference.character_concat()
    assert ref == b'Hello world!'

    # Test the generated identity
    test = generate_identity(refpath, 'character_concat', frontend=frontend)
    function = getattr(test, 'character_concat_%s' % frontend)
    result = function()
    assert result == b'Hello world!'


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Inline WHERE not implemented')),
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Not implemented')),
    FP
])
def test_masked_statements(refpath, reference, frontend):
    """
    Masked statements (WHERE(...) ... [ELSEWHERE ...] ENDWHERE)
    """
    # Reference solution
    length = 11
    ref1 = np.append(np.arange(0, 6, dtype=np.float64),
                     5 * np.ones(length - 6, dtype=np.float64))
    ref2 = np.append(np.zeros(5, dtype=np.float64),
                     np.ones(length - 5, dtype=np.float64))
    ref3 = np.append(np.arange(-2, 1, dtype=np.float64), np.ones(2, dtype=np.float64))
    ref3 = np.append(ref3, np.arange(3, length - 2, dtype=np.float64))

    # Test the reference solution
    vec1 = np.arange(0, length, dtype=np.float64)
    vec2 = np.arange(-5, length - 5, dtype=np.float64)
    vec3 = np.arange(-2, length - 2, dtype=np.float64)
    reference.masked_statements(length, vec1, vec2, vec3)
    assert np.all(ref1 == vec1)
    assert np.all(ref2 == vec2)
    assert np.all(ref3 == vec3)

    # Test the generated identity
    test = generate_identity(refpath, 'masked_statements', frontend=frontend)
    function = getattr(test, 'masked_statements_%s' % frontend)
    vec1 = np.arange(0, length, dtype=np.float64)
    vec2 = np.arange(-5, length - 5, dtype=np.float64)
    vec3 = np.arange(-2, length - 2, dtype=np.float64)
    function(length, vec1, vec2, vec3)
    assert np.all(ref1 == vec1)
    assert np.all(ref2 == vec2)
    assert np.all(ref3 == vec3)


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Not implemented')),
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Not implemented')),
    FP
])
def test_data_declaration(refpath, reference, frontend):
    """
    Variable initialization with DATA statements
    """
    expected = np.ones(shape=(5, 4), dtype=np.int32, order='F') * 8
    expected[[0, 1, 2], 0] = [1, 3, 2]
    # Test the reference solution
    ref = np.zeros(shape=(5, 4), dtype=np.int32, order='F')
    reference.data_declaration(ref)
    assert np.all(ref == expected)

    # Test the generated identity
    test = generate_identity(refpath, 'data_declaration', frontend=frontend)
    function = getattr(test, 'data_declaration_%s' % frontend)
    result = np.zeros(shape=(5, 4), dtype=np.int32, order='F')
    function(result)
    assert np.all(result == ref)


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Not implemented')),
    FP
])
def test_pointer_nullify(refpath, reference, frontend):
    """
    POINTERS and their nullification via '=> NULL()'
    """
    from loki import FindNodes, Nullify, Statement, InlineCall

    # Execute the reference solution (does not return anything but should not fail
    reference.pointer_nullify()

    # Create the AST and perform some checks
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['pointer_nullify']
    routine.name += '_%s' % frontend

    assert np.all(v.type.pointer for v in routine.variables)
    assert np.all(isinstance(v.initial, InlineCall) and v.type.initial.name.lower() == 'null'
                  for v in routine.variables)
    assert FindNodes(Nullify).visit(routine.body)[0].variable.name == 'pp'
    assert [stmt.ptr for stmt in FindNodes(Statement).visit(routine.body)].count(True) == 2

    # Generate the identitiy
    testname = refpath.parent / ('%s_pointer_nullify_%s.f90' % (refpath.stem, frontend))
    source.write(source=fgen(routine), filename=testname)
    pymod = compile_and_load(testname, cwd=str(refpath.parent), use_f90wrap=True)
    function = getattr(pymod, 'pointer_nullify_%s' % frontend)

    # Execute the generated identity (to verify it is valid Fortran)
    function()
