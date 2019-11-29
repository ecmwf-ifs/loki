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


@pytest.mark.parametrize('frontend', [OFP, FP])
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


@pytest.mark.parametrize('frontend', [OFP, OMNI])  # FP fails to read format statements
def test_intrinsics(refpath, reference, frontend):
    """
    Some collected intrinsics or other edge cases that failed in cloudsc.
    """
    from loki.ir import Intrinsic

    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = source['intrinsics']

    assert isinstance(routine.body[-2], Intrinsic)
    assert isinstance(routine.body[-1], Intrinsic)
    assert routine.body[-2].text.strip('\n') in ["1002  format(1x,2i10,1xi4,' : ',i10)"]
    assert routine.body[-1].text.strip('\n') in \
        ['write(0,1002) numomp,ngptot,-1,int(tdiff*1000.0_jprb)',
         'write(unit=0, fmt=1002) numomp, ngptot, -1, int(tdiff*1000.0_jprb)']
