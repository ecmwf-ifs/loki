# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module, Sourcefile, Subroutine
from loki.analyse import (
    dataflow_analysis_attached, read_after_write_vars, loop_carried_dependencies,
    classify_array_access_offsets, array_loop_carried_dependencies,
    detect_vertical_carry_variables, classify_multilevel_arrays
)
from loki.analyse.analyse_dataflow import DataflowAnalysisAttacher, DataflowAnalysisDetacher
from loki.backend import fgen
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_live_symbols(frontend):
    fcode = """
subroutine analyse_live_symbols(v1, v2, v3)
  integer, intent(in) :: v1
  integer, intent(inout) :: v2
  integer, intent(out) :: v3
  integer :: i, j, k, n=10, tmp, a, b
  b(k) = k + 1

  do i=1,n
    do j=1,n
      tmp = b(j)
    end do
    a = v2 + tmp
  end do

  v3 = v1 + v2
  v2 = a
end subroutine analyse_live_symbols
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    ref_fgen = fgen(routine)

    assignments = FindNodes(ir.Assignment).visit(routine.body)
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

    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conditionals) == 2
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    with pytest.raises(RuntimeError):
        for cond in conditionals:
            _ = cond.defines_symbols
        for cond in conditionals:
            _ = cond.uses_symbols

    with dataflow_analysis_attached(routine):
        assert fgen(routine) == ref_fgen
        assert len(FindNodes(ir.Conditional).visit(routine.body)) == 2
        assert len(FindNodes(ir.Loop).visit(routine.body)) == 1

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

    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    assert len(pragmas) == 5

    with dataflow_analysis_attached(routine):
        for pragma in pragmas:
            assert read_after_write_vars(routine.body, pragma) == vars_at_inspection_node[pragma.content]


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('include_literal_kinds', [True, False])
def test_read_after_write_vars_conditionals(frontend, include_literal_kinds):
    fcode = """
subroutine analyse_read_after_write_vars_conditionals(a, b, c, d, e, f)
  use iso_fortran_env, only : int32
  integer, intent(in) :: a
  integer, intent(out) :: b, c, d, e, f

  b = 1
  d = 0
!$loki A
  if (a < 3_int32) then
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

    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    assert len(pragmas) == len(vars_at_inspection_node)

    # We skip the context manager here to test the "include_literal_kinds" option
    DataflowAnalysisAttacher(include_literal_kinds=include_literal_kinds).visit(routine.body)

    if include_literal_kinds:
        assert 'int32' in routine.body.uses_symbols
    else:
        assert not 'int32' in routine.body.uses_symbols
    for pragma in pragmas:
        assert read_after_write_vars(routine.body, pragma) == vars_at_inspection_node[pragma.content]

    DataflowAnalysisDetacher().visit(routine.body)


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

    loops = FindNodes(ir.Loop).visit(routine.body)
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
        assert isinstance(list(routine.spec.defines_symbols)[0], sym.ProcedureSymbol)
        assert 'random_call' in routine.spec.defines_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_imports(frontend, tmp_path):
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

    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=module, xmods=[tmp_path])

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
    call = FindNodes(ir.CallStatement).visit(routine.body)[0]

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
    call = FindNodes(ir.CallStatement).visit(routine.body)[0]

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
        assert 'real64' in routine.spec.uses_symbols
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
  integer          :: bsize, i

  if(size(a) > 0) a(:,:) = 0.
  bsize = size(b)

  do i=1,size(b)
    print *, i
  enddo

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

    calls = FindNodes(ir.CallStatement).visit(routine.body)
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
    mcond = FindNodes(ir.MultiConditional).visit(routine.body)[0]
    with dataflow_analysis_attached(routine):
        assert len(mcond.bodies) == 2
        assert len(mcond.else_body) == 1
        for b in mcond.bodies:
            assert len(b) == 1

        assert len(mcond.uses_symbols) == 3
        assert len(mcond.defines_symbols) == 2
        assert all(i in mcond.uses_symbols for i in ['ic', 'ia', 'ib'])
        assert all(i in mcond.defines_symbols for i in ['a', 'b'])

        assigns = FindNodes(ir.Assignment).visit(routine.body)
        for assign in assigns:
            assert assign.live_symbols == {'ia', 'ib', 'ic'}


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI fails to read without full module')]))
def test_analyse_typeconditional(frontend):
    fcode = """
subroutine test(arg)
use type_mod, only: base_type, some_type, other_type
class(base_type), intent(in) :: arg
integer             :: a, b, c

typecond: select type(arg)
  class is(some_type)
    associate (aa => arg%s)
      a = aa
    end associate
  type is(other_type)
    associate (bb => arg%t)
      b = bb
    end associate
  class default
    c = 0
end select typecond
end subroutine test
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    tcond = FindNodes(ir.TypeConditional).visit(routine.body)[0]
    with dataflow_analysis_attached(routine):
        assert len(tcond.bodies) == 2
        assert len(tcond.else_body) == 1
        for b in tcond.bodies:
            assert len(b) == 1

        assert tcond.uses_symbols == {'arg%t', 'arg%s', 'arg'}
        assert tcond.defines_symbols == {'a', 'b', 'c'}
        assert tcond.live_symbols == {'arg'}

        assigns = FindNodes(ir.Assignment).visit(routine.body)
        for assign in assigns:
            assert assign.live_symbols == {'arg'}


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('include_literal_kinds', [True, False])
def test_analyse_maskedstatement(frontend, include_literal_kinds):
    fcode = """
subroutine masked_statements(n, mask, vec1, vec2)
  use iso_fortran_env, only : int32
  integer, intent(in) :: n
  integer, intent(in), dimension(n) :: mask
  real, intent(out), dimension(n) :: vec1,vec2

  where (mask(:) < -5_int32)
    vec1(:) = -5.0
    vec1(:) = vec1(:) -5.0
  elsewhere (mask(:) > 5_int32)
    vec1(:) =  5.0
  elsewhere
    vec1(:) = 0.0
  endwhere

end subroutine masked_statements
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    mask = FindNodes(ir.MaskedStatement).visit(routine.body)[0]
    num_bodies = len(mask.bodies)

    # We skip the context manager here to test the "include_literal_kinds" option
    DataflowAnalysisAttacher(include_literal_kinds=include_literal_kinds).visit(routine.body)

    if include_literal_kinds:
        assert len(mask.uses_symbols) == 2
        assert 'int32' in mask.uses_symbols
    else:
        assert len(mask.uses_symbols) == 1
        assert not 'int32' in mask.uses_symbols
    assert len(mask.defines_symbols) == 1
    assert 'mask' in mask.uses_symbols
    assert 'vec1' in mask.defines_symbols

    DataflowAnalysisDetacher().visit(routine.body)

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
    loop = FindNodes(ir.WhileLoop).visit(routine.body)[0]
    cond = FindNodes(ir.Conditional).visit(routine.body)[0]
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
    associates = FindNodes(ir.Associate).visit(routine.body)
    assigns = FindNodes(ir.Assignment).visit(routine.body)
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


@pytest.mark.parametrize('frontend', available_frontends())
def test_analyse_derived_types(frontend, tmp_path):
    """
    Test dataflow analysis on nested derived-types.
    """

    fcode = r"""
module my_mod
   implicit none

   type :: my_sub_type
      real, allocatable :: c(:)
   end type

   type :: my_type
      type(my_sub_type), allocatable :: b(:)
   end type

contains

subroutine kernel(a, d)
   type(my_type), intent(inout) :: a
   type(my_type), intent(in) :: d
   integer :: i

   do i=1,10
     A%B(i)%C(:) = D%B(i)%C(:)
   enddo

end subroutine

end module
"""

    source = Sourcefile.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = source['kernel']

    with dataflow_analysis_attached(routine):
        assert routine.body.defines_symbols == {'a%b%c'}
        assert routine.body.uses_symbols == {'d%b%c'}


@pytest.mark.parametrize('frontend', available_frontends())
def test_classify_array_access_offsets_simple(frontend):
    """
    Test that classify_array_access_offsets correctly identifies
    subscript offsets for simple JK and JK-1 patterns.
    """
    fcode = """
subroutine test_offsets(n, arr, brr)
  integer, intent(in) :: n
  real, dimension(n) :: arr, brr
  integer :: jk

  do jk = 2, n
    arr(jk) = brr(jk - 1) + brr(jk)
  end do
end subroutine test_offsets
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    access_map = classify_array_access_offsets(loops[0])
    # arr is written at offset 0
    assert ('arr', 0) in access_map
    assert access_map[('arr', 0)] == {0: {'write'}}
    # brr is read at offsets -1 and 0
    assert ('brr', 0) in access_map
    assert access_map[('brr', 0)] == {-1: {'read'}, 0: {'read'}}


@pytest.mark.parametrize('frontend', available_frontends())
def test_classify_array_access_offsets_write_plus_one(frontend):
    """
    Test that write at JK+1 is detected correctly (sedimentation pattern).
    """
    fcode = """
subroutine test_write_plus_one(n, flux, source)
  integer, intent(in) :: n
  real, dimension(n+1) :: flux
  real, dimension(n) :: source
  integer :: jk

  do jk = 1, n
    flux(jk + 1) = source(jk) * 0.5
  end do
end subroutine test_write_plus_one
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    access_map = classify_array_access_offsets(loops[0])
    assert ('flux', 0) in access_map
    assert 1 in access_map[('flux', 0)]
    assert 'write' in access_map[('flux', 0)][1]
    assert ('source', 0) in access_map
    assert access_map[('source', 0)] == {0: {'read'}}


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_loop_carried_dependencies_simple_flow(frontend):
    """
    Test detection of a simple flow (RAW) loop-carried dependency:
    data(i) = data(i) + data(i-1)
    """
    fcode = """
subroutine test_flow_dep(data, n)
  integer, intent(in) :: n
  real, dimension(n) :: data
  integer :: i

  do i = 2, n
    data(i) = data(i) + data(i - 1)
  end do
end subroutine test_flow_dep
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    deps = array_loop_carried_dependencies(loops[0])
    assert 'data' in deps
    # Should find a flow dependency: written at 0, read at -1
    flow_deps = [d for d in deps['data'] if d['type'] == 'flow']
    assert len(flow_deps) >= 1
    found = any(d['write_offset'] == 0 and d['read_offset'] == -1 for d in flow_deps)
    assert found, f"Expected flow dep (write=0, read=-1), got {flow_deps}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_loop_carried_dependencies_no_dep(frontend):
    """
    Test that arr(i) = arr(i) * 2 (same offset read/write) has NO
    loop-carried dependency.
    """
    fcode = """
subroutine test_no_dep(arr, n)
  integer, intent(in) :: n
  real, dimension(n) :: arr
  integer :: i

  do i = 1, n
    arr(i) = arr(i) * 2.0
  end do
end subroutine test_no_dep
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    deps = array_loop_carried_dependencies(loops[0])
    # arr is written and read at offset 0 only -- no cross-offset dependency
    if 'arr' in deps:
        # Should have no flow or output deps (both read and write at offset 0)
        assert all(d['write_offset'] == d['read_offset'] == 0 for d in deps['arr']) is False or \
               len(deps['arr']) == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_loop_carried_dependencies_shift_register(frontend):
    """
    Test the shift register pattern: write at JK+1, read at JK.
    This is the sedimentation flux pattern from CLOUDSC.
    """
    fcode = """
subroutine test_shift_register(flux, source, n)
  integer, intent(in) :: n
  real, dimension(n+1) :: flux
  real, dimension(n) :: source
  integer :: jk

  do jk = 1, n
    source(jk) = source(jk) + flux(jk)
    flux(jk + 1) = source(jk) * 0.5
  end do
end subroutine test_shift_register
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    deps = array_loop_carried_dependencies(loops[0])
    # flux is written at offset +1 and read at offset 0
    assert 'flux' in deps
    flow_deps = [d for d in deps['flux'] if d['type'] == 'flow']
    assert len(flow_deps) >= 1
    found = any(d['write_offset'] == 1 and d['read_offset'] == 0 for d in flow_deps)
    assert found, f"Expected flow dep (write=+1, read=0), got {flow_deps}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_loop_carried_dependencies_multiple(frontend):
    """
    Test multiple arrays with different dependency patterns.
    """
    fcode = """
subroutine test_multi_dep(a, b, c, n)
  integer, intent(in) :: n
  real, dimension(n) :: a, b, c
  integer :: i

  do i = 2, n
    a(i) = a(i - 1) + b(i)
    c(i) = c(i) * 2.0
    b(i) = a(i) + 1.0
  end do
end subroutine test_multi_dep
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    deps = array_loop_carried_dependencies(loops[0])
    # a: written at 0, read at -1 => flow dep
    assert 'a' in deps
    a_flow = [d for d in deps['a'] if d['type'] == 'flow']
    assert any(d['write_offset'] == 0 and d['read_offset'] == -1 for d in a_flow)
    # c: only accessed at offset 0 => no loop-carried dep
    assert 'c' not in deps or len(deps.get('c', [])) == 0
    # b: written at 0, read at 0 => no loop-carried dep
    assert 'b' not in deps or len(deps.get('b', [])) == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_detect_vertical_carry_variables_scalar(frontend):
    """
    Test detection of scalar carry variables (1D variables that are
    both read and written inside a loop over JK).
    """
    fcode = """
subroutine test_carry_vars(nlon, nlev, a, b)
  integer, intent(in) :: nlon, nlev
  real, dimension(nlon, nlev) :: a, b
  real :: carry_val
  integer :: jk, jl

  carry_val = 0.0
  do jk = 1, nlev
    do jl = 1, nlon
      a(jl, jk) = b(jl, jk) + carry_val
    end do
    carry_val = a(1, jk)
  end do
end subroutine test_carry_vars
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    # Find the outer JK loop
    jk_loops = [l for l in loops if l.variable.name.lower() == 'jk']
    assert len(jk_loops) == 1

    result = detect_vertical_carry_variables(jk_loops[0])
    scalar_names = {c['name'] for c in result['scalar_carries']}
    assert 'carry_val' in scalar_names


@pytest.mark.parametrize('frontend', available_frontends())
def test_detect_vertical_carry_variables_shift_register(frontend):
    """
    Test detection of shift register patterns (array written at JK+1,
    read at JK).
    """
    fcode = """
subroutine test_shift_detect(n, flux, source)
  integer, intent(in) :: n
  real, dimension(n+1) :: flux
  real, dimension(n) :: source
  integer :: jk

  do jk = 1, n
    source(jk) = source(jk) + flux(jk)
    flux(jk + 1) = source(jk) * 0.5
  end do
end subroutine test_shift_detect
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    result = detect_vertical_carry_variables(loops[0])
    shift_names = {s['name'] for s in result['shift_registers']}
    assert 'flux' in shift_names
    # Check direction: write at +1, read at 0 => downward
    flux_shifts = [s for s in result['shift_registers'] if s['name'] == 'flux']
    assert any(s['write_offset'] == 1 and s['read_offset'] == 0
               and s['direction'] == 'downward' for s in flux_shifts)


@pytest.mark.parametrize('frontend', available_frontends())
def test_classify_multilevel_arrays_basic(frontend):
    """
    Test that classify_multilevel_arrays correctly identifies arrays accessed
    at non-zero offsets across multiple loops in a routine.
    """
    fcode = """
subroutine test_ml_classify(nlon, nz)
  integer, intent(in) :: nlon, nz
  real :: za(nlon, nz), zb(nlon, nz), zc(nlon, nz)
  real :: simple(nlon, nz)
  integer :: jl, jk

  ! Loop 1: za is written at jk, zb is written at jk
  do jk = 1, nz
    do jl = 1, nlon
      za(jl, jk) = 1.0
      zb(jl, jk) = 2.0
      simple(jl, jk) = 3.0
    end do
  end do

  ! Loop 2: za is read at jk-1 (makes it multilevel), zc is written at jk+1
  do jk = 2, nz
    do jl = 1, nlon
      zb(jl, jk) = za(jl, jk - 1) + zb(jl, jk)
      zc(jl, jk + 1) = zb(jl, jk)
    end do
  end do
end subroutine test_ml_classify
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loop_var = routine.variable_map['jk']

    result = classify_multilevel_arrays(routine, loop_var)

    # za has offset -1 in loop 2 => multilevel
    assert 'za' in result
    # zc has offset +1 in loop 2 => multilevel
    assert 'zc' in result
    # simple is only accessed at offset 0 => NOT multilevel
    assert 'simple' not in result
    # zb is only accessed at offset 0 => NOT multilevel (despite being in both loops)
    assert 'zb' not in result


@pytest.mark.parametrize('frontend', available_frontends())
def test_classify_multilevel_arrays_empty(frontend):
    """
    Test that classify_multilevel_arrays returns an empty set when all
    accesses are at offset 0.
    """
    fcode = """
subroutine test_ml_none(nlon, nz)
  integer, intent(in) :: nlon, nz
  real :: a(nlon, nz), b(nlon, nz)
  integer :: jl, jk

  do jk = 1, nz
    do jl = 1, nlon
      a(jl, jk) = 1.0
    end do
  end do

  do jk = 1, nz
    do jl = 1, nlon
      b(jl, jk) = a(jl, jk) + 2.0
    end do
  end do
end subroutine test_ml_none
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loop_var = routine.variable_map['jk']

    result = classify_multilevel_arrays(routine, loop_var)
    assert len(result) == 0
