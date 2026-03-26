# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Sourcefile, Subroutine, Module, config_override
from loki.expression import symbols as sym, parse_expr
from loki.frontend import available_frontends, OMNI, SourceStatus
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, FindTypedSymbols,
    SubstituteExpressions, SubstituteExpressionsSkipLHS,
    SubstituteStringExpressions,
    FindLiterals, FindRealLiterals
)


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_finder_retrieval_function(frontend, tmp_path):
    """
    Verify that expression finder visitors work as intended and remain
    functional if re-used
    """
    fcode = """
module some_mod
    implicit none
contains
    function some_func() result(ret)
        integer :: ret
        ret = 1
    end function some_func

    subroutine other_routine
        integer :: var, tmp
        var = 5 + some_func()
    end subroutine other_routine
end module some_mod
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    expected_ts = {'var', 'some_func'}
    expected_vars = ('var',)

    # Instantiate the first expression finder and make sure it works as expected
    find_ts = FindTypedSymbols()
    assert find_ts.visit(source['other_routine'].body) == expected_ts

    # Verify that it works also on a repeated invocation
    assert find_ts.visit(source['other_routine'].body) == expected_ts

    # Instantiate the second expression finder and make sure it works as expected
    find_vars = FindVariables(unique=False)
    assert find_vars.visit(source['other_routine'].body) == expected_vars

    # Make sure the first expression finder still works
    assert find_ts.visit(source['other_routine'].body) == expected_ts


@pytest.mark.parametrize('frontend', available_frontends())
def test_find_variables(frontend, tmp_path):
    """ Test that :any:`FindVariables` finds all symbol uses. """

    fcode_external = """
module external_mod
implicit none
contains
subroutine rick(dave, never)
  real(kind=8), intent(inout) :: dave, never
end subroutine rick
end module external_mod
    """
    fcode = """
module test_mod
  use external_mod, only: rick
  implicit none

  type my_type
    real(kind=8) :: never
    real(kind=8), pointer :: give_you(:)
  end type my_type

contains

  subroutine test_routine(n, a, b, gonna)
    integer, intent(in) :: n
    real(kind=8), intent(inout) :: a, b(n)
    type(my_type), intent(inout) :: gonna
    integer :: i

    associate(will=>gonna%never, up=>n)
    do i=1, n
      b(i) = b(i) + a
    end do

    call rick(will, never=gonna%give_you(up))
    end associate
  end subroutine test_routine
end module test_mod
    """
    _ = Sourcefile.from_source(fcode_external, frontend=frontend, xmods=[tmp_path])
    source = Sourcefile.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = source['test_routine']

    # Test unique=True|False using the spec
    expected = ['n', 'a', 'gonna', 'i', 'b(n)']
    spec_vars = FindVariables(unique=True).visit(routine.spec)
    assert len(spec_vars) == 5
    assert all(v in spec_vars for v in expected)

    spec_vars = FindVariables(unique=False).visit(routine.spec)
    assert len(spec_vars) == 6
    assert all(v in spec_vars for v in expected)
    assert len([v for v in spec_vars if v == 'n']) == 2  # two occurences of 'n'

    # Test retrieval with associates and keyword arg calls
    expected = [
        'will', 'gonna', 'gonna%never', 'up', 'n', 'i', 'b(i)', 'a',
        'rick', 'gonna%give_you(up)'
    ]
    body_vars = FindVariables(unique=True).visit(routine.body)
    assert len(body_vars) == 10
    assert all(v in body_vars for v in expected)

@pytest.mark.parametrize('frontend', available_frontends())
def test_find_literals(frontend):
    """
    Test that :any:`FindLiterals` finds all literals
    and :any:`FindRealLiterals` all real/float literals.
    """
    fcode = """
subroutine test_find_literals()
  implicit none
  integer :: n, n1
  real(kind=8) :: x

  n = 1 + 5 + 42
  x = 1.0 / 10.5
  n1 = int(B'00000')
  if (.TRUE.) then
    call some_func(x, some_string='string_kwarg')
  endif

end subroutine test_find_literals
"""
    expected_int_literals = ('1', '5', '42')
    expected_real_literals = ('1.0', '10.5')
    # Omni evaluates BOZ constants, so it creates IntegerLiteral instead...
    expected_intrinsic_literals = ("B'00000'",) if frontend != OMNI else ('0',)
    expected_logic_literals = ('True',)
    expected_string_literals = ('string_kwarg',)
    expected_literals = expected_int_literals + expected_real_literals +\
            expected_intrinsic_literals + expected_logic_literals +\
            expected_string_literals
    routine = Subroutine.from_source(fcode, frontend=frontend)
    literals = FindLiterals().visit(routine.body)
    assert sorted(list(expected_literals)) == sorted([str(literal.value) for literal in literals])
    real_literals = FindRealLiterals().visit(routine.body)
    assert sorted(list(expected_real_literals)) == sorted([str(literal.value) for literal in real_literals])
    real_literals_isinstance = [literal for literal in literals if isinstance(literal, sym.FloatLiteral)]
    assert sorted(list(expected_real_literals)) == sorted([str(literal.value) for literal in real_literals_isinstance])


@pytest.mark.parametrize('frontend', available_frontends())
def test_substitute_expressions(frontend):
    """ Test symbol replacement with :any:`Expression` symbols. """

    fcode = """
subroutine test_routine(n, a, b)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a, b(n)
  real(kind=8) :: c(n)
  integer :: i

  associate(d => a)
  do i=1, n
    c(i) = b(i) + a
  end do

  call another_routine(n, a, c(:), a2=d)

  end associate
end subroutine test_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assoc = FindNodes(ir.Associate).visit(routine.body)[0]
    assert calls[0].arguments == ('n', 'a', 'c(:)')
    assert calls[0].kwarguments == (('a2', 'd'),)

    n = routine.variable_map['n']
    i = routine.variable_map['i']
    a = routine.variable_map['a']
    b_i = parse_expr('b(i)', scope=routine)
    c_r = parse_expr('c(:)', scope=routine)
    d = parse_expr('d', scope=assoc)
    expr_map = {
        n: sym.Sum((n, sym.Product((-1, sym.Literal(1))))),
        b_i: b_i.clone(dimensions=sym.Sum((i, sym.Literal(1)))),
        c_r: c_r.clone(dimensions=sym.Range((sym.Literal(1), sym.Literal(2)))),
        a: d,
        d: a,
    }
    routine.body = SubstituteExpressions(expr_map).visit(routine.body)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert loops[0].bounds == '1:n-1'
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert assigns[0].lhs == 'c(i)' and assigns[0].rhs == 'b(i+1) + d'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert calls[0].arguments == ('n - 1', 'd', 'c(1:2)')
    assert calls[0].kwarguments == (('a2', 'a'),)


@pytest.mark.parametrize('frontend', available_frontends())
def test_substitute_string_expressions(frontend):
    """ Test symbol replacement with symbol string mappping. """

    fcode = """
subroutine test_routine(n, a, b)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a, b(n)
  real(kind=8) :: c(n)
  integer :: i

  associate(d => a)
  do i=1, n
    c(i) = b(i) + a
  end do

  call another_routine(n, a, c(:), a2=d)

  end associate
end subroutine test_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assoc = FindNodes(ir.Associate).visit(routine.body)[0]
    assert calls[0].arguments == ('n', 'a', 'c(:)')
    assert calls[0].kwarguments == (('a2', 'd'),)

    expr_map = {
        'n': 'n - 1',
        'b(i)': 'b(i+1)',
        'c(:)': 'c(1:2)',
        'a': 'd',
        'd': 'a',
    }
    # Note that we need to use the associate block here, as it defines 'd'
    routine.body = SubstituteStringExpressions(expr_map, scope=assoc).visit(routine.body)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert loops[0].bounds == '1:n-1'
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert assigns[0].lhs == 'c(i)' and assigns[0].rhs == 'b(i+1) + d'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert calls[0].arguments == ('n - 1', 'd', 'c(1:2)')
    assert calls[0].kwarguments == (('a2', 'a'),)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('use_string', [True, False])
def test_substitute_expression_source_invalidation(use_string, frontend, tmp_path):
    """ Test source invalidation when using symbol or string substitution """

    fcode_type = """
module type_mod
  integer, parameter :: jprb = 8
end module type_mod
"""

    fcode = """
subroutine test_routine(n, a, b)
  use type_mod, only: jprb
  implicit none
  integer, intent(in) :: n
  real(kind=jprb), intent(inout) :: a, b
  real(kind=jprb) :: c(n)
  integer :: i

  associate(d => b)
  do i=1, n
    if (i > 2) then
      c(i) = b(i) + a
    else
      c(i) = 42.0
    end if

    if (a > 0.5) then
      c(i) = 66.6
    end if
  end do

  if (c(1) > 0.5) then
    call another_routine(n, a, c(:), a2=d)
  end if

  end associate
end subroutine test_routine
"""
    Module.from_source(fcode_type, frontend=frontend, xmods=[tmp_path])
    with config_override({'frontend-store-source': True}):
        routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assoc = FindNodes(ir.Associate).visit(routine.body)[0]
    assert calls[0].arguments == ('n', 'a', 'c(:)')
    assert calls[0].kwarguments == (('a2', 'd'),)

    if use_string:
        expr_map = {'a': 'd'}
        routine.body = SubstituteStringExpressions(expr_map, scope=assoc).visit(routine.body)
    else:
        a = routine.variable_map['a']
        expr_map = {a: parse_expr('d', scope=assoc)}
        routine.body = SubstituteExpressions(expr_map).visit(routine.body)


    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].variable == 'i' and loops[0].bounds == '1:n'
    assert loops[0].source.status == SourceStatus.INVALID_CHILDREN

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert all(a.lhs == 'c(i)' for a in assigns)
    assert assigns[0].rhs == 'b(i) + d'
    assert assigns[0].source.status == SourceStatus.INVALID_NODE
    assert assigns[1].source.status == SourceStatus.VALID
    assert assigns[2].source.status == SourceStatus.VALID

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('n', 'd', 'c(:)')
    assert calls[0].kwarguments == (('a2', 'd'),)
    assert calls[0].source.status == SourceStatus.INVALID_NODE

    conds = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds) == 3
    assert conds[0].condition == 'i > 2'
    assert conds[1].condition == 'd > 0.5'
    assert conds[2].condition == 'c(1) > 0.5'
    assert conds[0].source.status == SourceStatus.INVALID_CHILDREN
    assert conds[1].source.status == SourceStatus.INVALID_NODE
    assert conds[2].source.status == SourceStatus.INVALID_CHILDREN

    # Now test replacing the kind attribute in imports and declarations
    if use_string:
        expr_map = {'jprb': 'dbl'}
        routine.spec = SubstituteStringExpressions(expr_map, scope=assoc).visit(routine.spec)
    else:
        a = routine.imported_symbol_map['jprb']
        expr_map = {a: parse_expr('dbl', scope=assoc)}
        routine.spec = SubstituteExpressions(expr_map).visit(routine.spec)

    imports = FindNodes(ir.Import).visit(routine.spec)
    assert len(imports) == 1
    assert imports[0].module == 'type_mod' and imports[0].symbols == ('dbl',)
    assert imports[0].source.status == SourceStatus.INVALID_NODE

    # OMNI changes declarations too much
    if not frontend == OMNI:
        decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
        assert len(decls) == 4
        assert decls[0].symbols == ('n',) and decls[0].symbols[0].type.intent == 'in'
        assert decls[1].symbols == ('a', 'b') and decls[1].symbols[0].type.kind == 'dbl'
        assert decls[2].symbols == ('c(n)',) and decls[2].symbols[0].type.kind == 'dbl'
        assert decls[3].symbols == ('i',) and decls[3].symbols[0].type.intent is None
        assert decls[0].source.status == SourceStatus.VALID
        assert decls[1].source.status == SourceStatus.INVALID_NODE
        assert decls[2].source.status == SourceStatus.INVALID_NODE
        assert decls[3].source.status == SourceStatus.VALID


@pytest.mark.parametrize('frontend', available_frontends())
def test_substitute_expressions_skip_lhs(frontend):
    """
    Test that :any:`SubstituteExpressionsSkipLHS` applies substitutions
    only to the RHS of assignments, leaving the LHS unchanged, while
    still substituting in non-assignment nodes such as loop bounds and
    call arguments.
    """

    fcode = """
subroutine test_routine(n, a, b)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a, b(n)
  real(kind=8) :: c(n)
  integer :: i

  do i=1, n
    a = a + b(i)
    c(i) = b(i) + a
  end do

  call another_routine(n, a, c(:))
end subroutine test_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    a = routine.variable_map['a']
    n = routine.variable_map['n']
    new_a = parse_expr('a + 1', scope=routine)
    new_n = sym.Sum((n, sym.Product((-1, sym.Literal(1)))))
    expr_map = {a: new_a, n: new_n}

    # Apply substitution skipping LHS
    routine.body = SubstituteExpressionsSkipLHS(expr_map).visit(routine.body)

    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # First assignment: a = a + b(i)
    # LHS 'a' must remain 'a', not become 'a + 1'
    assert assigns[0].lhs == 'a'
    assert assigns[0].rhs == 'a + 1 + b(i)'

    # Second assignment: c(i) = b(i) + a
    # LHS 'c(i)' must remain 'c(i)', RHS 'a' becomes 'a + 1'
    assert assigns[1].lhs == 'c(i)' and assigns[1].rhs == 'b(i) + a + 1'

    # Loop bounds should still be substituted (n -> n-1)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert loops[0].bounds == '1:n - 1'

    # Call arguments should still be substituted
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert calls[0].arguments == ('n - 1', 'a + 1', 'c(:)')


@pytest.mark.parametrize('frontend', available_frontends())
def test_substitute_expressions_skip_lhs_array(frontend):
    """
    Test that :any:`SubstituteExpressionsSkipLHS` leaves the entire
    LHS of an assignment unchanged, including array subscripts.

    This verifies that when ``x(i)`` appears in the expression map
    (e.g. ``x(i) -> MAX(x(i), TINY(1.0))``), the LHS ``x(i)``
    is not modified while the RHS is.
    """

    fcode = """
subroutine test_routine(n, x, y)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: x(n), y(n)
  integer :: i

  do i=1, n
    x(i) = 1.0 / x(i) + y(i)
  end do
end subroutine test_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    x_i = assigns[0].lhs  # x(i) as it appears on the LHS

    # Map x(i) to a wrapped expression (simulating safe-denominator logic)
    wrapped = parse_expr('max(x(i), tiny(1.0))', scope=routine)
    expr_map = {x_i: wrapped}

    routine.body = SubstituteExpressionsSkipLHS(expr_map).visit(routine.body)

    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # LHS must remain x(i), NOT MAX(x(i), TINY(1.0))
    assert str(assigns[0].lhs).lower() == 'x(i)'

    # RHS should have the substitution applied
    rhs_str = str(assigns[0].rhs).lower()
    assert 'max' in rhs_str
    assert 'tiny' in rhs_str


@pytest.mark.parametrize('frontend', available_frontends())
def test_substitute_expressions_skip_lhs_noop(frontend):
    """
    Test that :any:`SubstituteExpressionsSkipLHS` preserves assignment
    nodes when the expression map has no matching entries.
    """

    fcode = """
subroutine test_routine(a, b)
  implicit none
  real(kind=8), intent(inout) :: a, b

  a = a + b
end subroutine test_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assigns_before = FindNodes(ir.Assignment).visit(routine.body)
    orig_lhs = str(assigns_before[0].lhs)
    orig_rhs = str(assigns_before[0].rhs)

    # Map with a variable that does not appear in the routine
    expr_map = {sym.Variable(name='zzz_nonexistent'): sym.Literal(42)}
    routine.body = SubstituteExpressionsSkipLHS(expr_map).visit(routine.body)

    assigns_after = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns_after) == 1
    # The assignment should be unchanged
    assert str(assigns_after[0].lhs) == orig_lhs
    assert str(assigns_after[0].rhs) == orig_rhs
