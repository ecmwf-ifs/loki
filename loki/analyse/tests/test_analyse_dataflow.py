# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import (
    Subroutine, FindNodes, Assignment, Loop, Conditional, Pragma, fgen, Sourcefile,
    CallStatement, MultiConditional, MaskedStatement, ProcedureSymbol, WhileLoop,
    Associate, Module
)
from loki.analyse import (
    dataflow_analysis_attached, read_after_write_vars, loop_carried_dependencies
)
from loki.frontend import available_frontends


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_live_symbols(frontend):
    fcode = """
subroutine analyse_live_symbols(v1, v2, v3)
  integer, intent(in) :: v1
  integer, intent(inout) :: v2
  integer, intent(out) :: v3
  integer :: i, j, n=10, tmp, a

  do i=1,n
    do j=1,n
      tmp = j + 1
    end do
    a = v2 + tmp
  end do

  v3 = v1 + v2
  v2 = a
end subroutine analyse_live_symbols
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    ref_fgen = fgen(routine)

    assignments = FindNodes(Assignment).visit(routine.body)
    assert len(assignments) == 4

    with pytest.raises(RuntimeError):
        for assignment in assignments:
            _ = assignment.live_symbols

    ref_live_symbols = {
        'tmp': {'i', 'j', 'n', 'v1', 'v2'},
        'a': {'i', 'tmp', 'n', 'v1', 'v2'},
        'v3': {'tmp', 'a', 'n', 'v1', 'v2'},
        'v2': {'tmp', 'a', 'n', 'v1', 'v2', 'v3'}
    }

    with dataflow_analysis_attached(routine):
        assert routine.body

        for assignment in assignments:
            live_symbols = {str(s).lower() for s in assignment.live_symbols}
            assert live_symbols == ref_live_symbols[str(assignment.lhs).lower()]

    assert routine.body
    assert fgen(routine) == ref_fgen

    with pytest.raises(RuntimeError):
        for assignment in assignments:
            _ = assignment.live_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_defines_uses_symbols(frontend):
    fcode = """
subroutine analyse_defines_uses_symbols(a, j, m, n)
  integer, intent(out) :: a, j
  integer, intent(in) :: m, n
  integer :: i
  j = n
  a = 1
  do i=m-1,n
    if (i > a) then
      a = a + 1
      if (i < n) exit
    end if
    j = j - 1
  end do
end subroutine analyse_defines_uses_symbols
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    ref_fgen = fgen(routine)

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 2
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1

    with pytest.raises(RuntimeError):
        for cond in conditionals:
            _ = cond.defines_symbols
        for cond in conditionals:
            _ = cond.uses_symbols

    with dataflow_analysis_attached(routine):
        assert fgen(routine) == ref_fgen
        assert len(FindNodes(Conditional).visit(routine.body)) == 2
        assert len(FindNodes(Loop).visit(routine.body)) == 1

        assert {str(s) for s in routine.body.uses_symbols} == {'m', 'n'}
        assert {str(s) for s in loops[0].uses_symbols} == {'m', 'n', 'a', 'j'}
        assert {str(s) for s in conditionals[0].uses_symbols} == {'i', 'a', 'n'}
        assert {str(s) for s in conditionals[1].uses_symbols} == {'i', 'n'}
        assert not conditionals[1].body[0].uses_symbols

        assert {str(s) for s in routine.body.defines_symbols} == {'j', 'a'}
        assert {str(s) for s in loops[0].defines_symbols} == {'j', 'a'}
        assert {str(s) for s in conditionals[0].defines_symbols} == {'a'}
        assert not conditionals[1].defines_symbols
        assert not conditionals[1].body[0].defines_symbols

    assert fgen(routine) == ref_fgen

    with pytest.raises(RuntimeError):
        for cond in conditionals:
            _ = cond.defines_symbols
        for cond in conditionals:
            _ = cond.uses_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_read_after_write_vars(frontend):
    fcode = """
subroutine analyse_read_after_write_vars
  integer :: a, b, c, d, e, f

  a = 1
!$loki A
  b = 2
!$loki B
  c = a + 1
!$loki C
  d = b + 1
!$loki D
  e = c + d
!$loki E
  e = 3
  f = e
end subroutine analyse_read_after_write_vars
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    variable_map = routine.variable_map

    vars_at_inspection_node = {
        'A': {variable_map['a']},
        'B': {variable_map['a'], variable_map['b']},
        'C': {variable_map['b'], variable_map['c']},
        'D': {variable_map['c'], variable_map['d']},
        'E': set(),
    }

    pragmas = FindNodes(Pragma).visit(routine.body)
    assert len(pragmas) == 5

    with dataflow_analysis_attached(routine):
        for pragma in pragmas:
            assert read_after_write_vars(routine.body, pragma) == vars_at_inspection_node[pragma.content]


@pytest.mark.parametrize('frontend', available_frontends())
def test_read_after_write_vars_conditionals(frontend):
    fcode = """
subroutine analyse_read_after_write_vars_conditionals(a, b, c, d, e, f)
  integer, intent(in) :: a
  integer, intent(out) :: b, c, d, e, f

  b = 1
  d = 0
!$loki A
  if (a < 3) then
    d = b
!$loki B
  endif
!$loki C
  c = 2 + d
!$loki D
  if (a < 5) then
    e = a
  else
    e = c
  endif
!$loki E
  f = e
end subroutine analyse_read_after_write_vars_conditionals
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    variable_map = routine.variable_map

    vars_at_inspection_node = {
        'A': {variable_map['b'], variable_map['d']},
        'B': {variable_map['d']},
        'C': {variable_map['d']},
        'D': {variable_map['c']},
        'E': {variable_map['e']},
    }

    pragmas = FindNodes(Pragma).visit(routine.body)
    assert len(pragmas) == len(vars_at_inspection_node)

    with dataflow_analysis_attached(routine):
        for pragma in pragmas:
            assert read_after_write_vars(routine.body, pragma) == vars_at_inspection_node[pragma.content]


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_carried_dependencies(frontend):
    fcode = """
subroutine analyse_loop_carried_dependencies(a, b, c)
  integer, intent(inout) :: a, b, c
  integer :: i, tmp

  do i = 1,a
    b = b + i
    tmp = c
    c = 5 + tmp
  end do
end subroutine analyse_loop_carried_dependencies
    """.strip()


    routine = Subroutine.from_source(fcode, frontend=frontend)
    variable_map = routine.variable_map

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1

    with dataflow_analysis_attached(routine):
        assert loop_carried_dependencies(loops[0]) == {variable_map['b'], variable_map['c']}

@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_interface(frontend):
    fcode = """
subroutine random_call(v_out,v_in,v_inout)
implicit none

  real,intent(in)  :: v_in
  real,intent(out)  :: v_out
  real,intent(inout)  :: v_inout


end subroutine random_call

subroutine test(v_out,v_in,v_inout)
implicit none
interface
  subroutine random_call(v_out,v_in,v_inout)
     real,intent(in)  :: v_in
     real,intent(out)  :: v_out
     real,intent(inout)  :: v_inout
  end subroutine random_call
end interface

real,intent(in   )  :: v_in
real,intent(out  )  :: v_out
real,intent(inout)  :: v_inout

end subroutine test
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['test']

    with dataflow_analysis_attached(routine):
        assert len(routine.body.defines_symbols) == 0
        assert len(routine.body.uses_symbols) == 0
        assert len(routine.spec.uses_symbols) == 0
        assert len(routine.spec.defines_symbols) == 1
        assert isinstance(list(routine.spec.defines_symbols)[0], ProcedureSymbol)
        assert 'random_call' in routine.spec.defines_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_imports(frontend):
    fcode_module = """
module some_mod
implicit none
real :: my_global
contains
subroutine random_call(v_out,v_in,v_inout)

  real,intent(in)  :: v_in
  real,intent(out)  :: v_out
  real,intent(inout)  :: v_inout


end subroutine random_call
end module some_mod
""".strip()

    fcode = """
subroutine test()
use some_mod, only: my_global, random_call
implicit none

end subroutine test
""".strip()

    module = Module.from_source(fcode_module, frontend=frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=module)

    with dataflow_analysis_attached(routine):
        assert len(routine.spec.defines_symbols) == 1
        assert 'random_call' in routine.spec.defines_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_enriched_call(frontend):
    fcode = """
subroutine random_call(v_out,v_in,v_inout)
implicit none

  real,intent(in)  :: v_in
  real,intent(out)  :: v_out
  real,intent(inout)  :: v_inout


end subroutine random_call

subroutine test(v_out,v_in,v_inout)
implicit none

  real,intent(in   )  :: v_in
  real,intent(out  )  :: v_out
  real,intent(inout)  :: v_inout

  call random_call(v_out,v_in,v_inout)

end subroutine test
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['test']
    routine.enrich(source.all_subroutines)
    call = FindNodes(CallStatement).visit(routine.body)[0]

    with dataflow_analysis_attached(routine):
        assert all(i in call.defines_symbols for i in ('v_out', 'v_inout'))
        assert all(i in call.uses_symbols for i in ('v_in', 'v_inout'))


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_unenriched_call(frontend):
    fcode = """
subroutine test(v_out,v_in,v_inout)
implicit none

  real,intent(in   )  :: v_in
  real,intent(out  )  :: v_out
  real,intent(inout)  :: v_inout

  call random_call(v_out,v_in,var=v_inout)

end subroutine test
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['test']
    call = FindNodes(CallStatement).visit(routine.body)[0]

    with dataflow_analysis_attached(routine):
        assert all(i in call.defines_symbols for i in ('v_out', 'v_inout', 'v_in'))
        assert all(i in call.uses_symbols for i in ('v_in', 'v_inout', 'v_in'))


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_allocate_statement(frontend):
    fcode = """
subroutine test(n,m)
implicit none

  integer,intent(in   ) :: n
  integer,intent(inout) :: m
  real,allocatable :: a(:,:)

  allocate(a(n,m))


  deallocate(a)

end subroutine test
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    with dataflow_analysis_attached(routine):
        assert all(i not in routine.body.defines_symbols for i in ['m', 'n'])
        assert all(i in routine.body.uses_symbols for i in ['m', 'n'])
        assert 'a' in routine.body.defines_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_import_kind(frontend):
    fcode = """
subroutine test(n,m)
use iso_fortran_env, only: real64
implicit none

  integer,intent(in   ) :: n
  integer,intent(inout) :: m
  real(kind=real64),allocatable :: a(:,:)

  a = 0._real64

end subroutine test
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    with dataflow_analysis_attached(routine):
        assert 'real64' in routine.body.uses_symbols
        assert 'real64' not in routine.body.defines_symbols
        assert 'a' in routine.body.defines_symbols
        assert 'a' not in routine.body.uses_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_query_memory_attributes(frontend):
    """
    Test that checks whether variables used only in function calls that
    query memory attributes appear in uses_symbols.
    """

    fcode = """
subroutine test(a)
implicit none

  real,intent(out) :: a(:,:)
  real             :: b(10)
  integer                     :: bsize

  if(size(a) > 0) a(:,:) = 0.
  bsize = size(b)

end subroutine test
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    with dataflow_analysis_attached(routine):
        assert not 'a' in routine.body.uses_symbols
        assert 'a' in routine.body.defines_symbols
        assert not 'b' in routine.body.uses_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_call_args_array_slicing(frontend):
    fcode = """
subroutine random_call(v)
implicit none

  integer,intent(out) :: v

  v = 1

end subroutine random_call

subroutine test(v,n,b)
implicit none

  integer,intent(out) :: v(:)
  integer,intent( in) :: n
  integer,intent( in) :: b(n)

  call random_call(v(n))
  call random_call(v(b(1)))

end subroutine test
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['test']

    calls = FindNodes(CallStatement).visit(routine.body)
    routine.enrich(source.all_subroutines)

    with dataflow_analysis_attached(routine):
        assert 'n' in calls[0].uses_symbols
        assert not 'n' in calls[0].defines_symbols
        assert 'b' in calls[1].uses_symbols
        assert not 'b' in calls[0].defines_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_multiconditional(frontend):
    fcode = """
subroutine test(ia,ib,ic)
integer, intent(in) :: ia,ib,ic
integer             :: a,b

multicond: select case (ic)
case (10) multicond
  a = 0
case (ia) multicond
  b = 0
case default multicond
  b = ib
end select multicond
end subroutine test
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    mcond = FindNodes(MultiConditional).visit(routine.body)[0]
    with dataflow_analysis_attached(routine):
        assert len(mcond.bodies) == 2
        assert len(mcond.else_body) == 1
        for b in mcond.bodies:
            assert len(b) == 1

        assert len(mcond.uses_symbols) == 3
        assert len(mcond.defines_symbols) == 2
        assert all(i in mcond.uses_symbols for i in ['ic', 'ia', 'ib'])
        assert all(i in mcond.defines_symbols for i in ['a', 'b'])

        assigns = FindNodes(Assignment).visit(routine.body)
        for assign in assigns:
            assert assign.live_symbols == {'ia', 'ib', 'ic'}


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_maskedstatement(frontend):
    fcode = """
subroutine masked_statements(n, mask, vec1, vec2)
  integer, intent(in) :: n
  integer, intent(in), dimension(n) :: mask
  real, intent(out), dimension(n) :: vec1,vec2

  where (mask(:) < -5)
    vec1(:) = -5.0
    vec1(:) = vec1(:) -5.0
  elsewhere (mask(:) > 5)
    vec1(:) =  5.0
  elsewhere
    vec1(:) = 0.0
  endwhere

end subroutine masked_statements
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    mask = FindNodes(MaskedStatement).visit(routine.body)[0]
    num_bodies = len(mask.bodies)
    with dataflow_analysis_attached(routine):
        assert len(mask.uses_symbols) == 1
        assert len(mask.defines_symbols) == 1
        assert 'mask' in mask.uses_symbols
        assert 'vec1' in mask.defines_symbols

    assert len(mask.bodies) == num_bodies


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_whileloop(frontend):
    fcode = """
subroutine while_loop(flag)
   implicit none

   logical, intent(in) :: flag
   integer :: ij
   real :: a(10)

   if(flag)then
      ij = 0
      do while(ij .lt. 10)
          ij = ij + 1
          a(ij) = 0.
      enddo
   endif

end subroutine while_loop
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loop = FindNodes(WhileLoop).visit(routine.body)[0]
    cond = FindNodes(Conditional).visit(routine.body)[0]
    with dataflow_analysis_attached(routine):
        assert len(cond.uses_symbols) == 1
        assert 'flag' in cond.uses_symbols
        assert len(loop.uses_symbols) == 1
        assert len(loop.defines_symbols) == 2
        assert 'ij' in loop.uses_symbols
        assert all(v in loop.defines_symbols for v in ('ij', 'a'))

    with dataflow_analysis_attached(cond):
        assert len(loop.uses_symbols) == 1
        assert len(loop.defines_symbols) == 2
        assert 'ij' in loop.uses_symbols
        assert all(v in loop.defines_symbols for v in ('ij', 'a'))


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_associate(frontend):

    fcode = """
subroutine associate_test(a, b, c, in_var)
   implicit none

   real, intent(in) :: in_var
   real, intent(inout) :: a, b, c

   associate(d=>a, e=>b, f=>c)
     e = in_var
     f = in_var
     associate(d0=>d)
       d0 = in_var
     end associate
   end associate

end subroutine associate_test
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    associates = FindNodes(Associate).visit(routine.body)
    assigns = FindNodes(Assignment).visit(routine.body)
    with dataflow_analysis_attached(routine):
        # check that associates use variables names in outer scope
        assert associates[0].uses_symbols == {'in_var'}
        assert associates[0].defines_symbols == {'a', 'b', 'c'}

        assert associates[1].uses_symbols == {'in_var'}
        assert associates[1].defines_symbols == {'d'}

        # check that assignments use associated symbols
        assert assigns[0].uses_symbols == {'in_var'}
        assert assigns[1].uses_symbols == {'in_var'}
        assert assigns[2].uses_symbols == {'in_var'}

        assert assigns[0].defines_symbols == {'e'}
        assert assigns[1].defines_symbols == {'f'}
        assert assigns[2].defines_symbols == {'d0'}
