import os
import pytest
import importlib
import subprocess

from loki import (
   Sourcefile, Subroutine, FindVariables, FindNodes, Assignment, Array, Scalar, CallStatement
   )

loki_intent_lint = importlib.import_module("scripts.loki_intent_lint",package="../../")

findvarsnotdims = loki_intent_lint.findvarsnotdims
finddimsnotvars = loki_intent_lint.finddimsnotvars

def clean_intent_lint_test():
    if os.path.exists('output.dat'):
        os.remove('output.dat')

    if os.path.exists('summary.dat'):
        os.remove('summary.dat')

    if os.path.exists('test_intent_lint_tmp.F90'):
        os.remove('test_intent_lint_tmp.F90')

@pytest.fixture(scope='module', name='fcode_call')
def fixture_fcode_call():
    fcode = """
subroutine first_random_call(v_out,v_in,v_inout)
implicit none

  real,intent(inout)  :: v_in
  real,intent(in   )  :: v_out
  real,intent(in   )  :: v_inout


end subroutine first_random_call

subroutine second_random_call(v_out,v_in,v_inout)
implicit none

  real,intent(in)  :: v_in
  real,intent(in)  :: v_out
  real,intent(in)  :: v_inout


end subroutine second_random_call

subroutine third_random_call(v_out,v_in,v_inout)
implicit none

  real,intent(in )  :: v_in
  real,intent(out)  :: v_out
  real,intent(out)  :: v_inout


end subroutine third_random_call

subroutine test(v_out,v_in,v_inout)
implicit none

  real,intent(in   )  :: v_in
  real,intent(out  )  :: v_out
  real,intent(inout)  :: v_inout

  call first_random_call(v_out,v_in,v_inout)
  call second_random_call(v_out,v_in,v_inout)

  v_out = 0.
  v_inout = 0.

  call third_random_call(v_out,v_in,v_inout)

end subroutine test
"""
    return fcode

@pytest.fixture(scope='module', name='fcode')
def fixture_fcode():
    fcode = """
subroutine test(a,b,n)
implicit none

  integer,intent(in)          :: n
  real,intent(in)  :: b(n,n)
  real,intent(out) :: a(n,n)

  integer                     :: i,j


  do j=1,n
    do i=1,n
       a(i,j) = b(i,j)
    enddo
  enddo

end subroutine test
"""
    return fcode

def test_intent_lint_findvarsnotdims(fcode):

    routine = Subroutine.from_source(fcode)
    assign = FindNodes(Assignment).visit(routine.body)[0]

    var = findvarsnotdims(assign)

    for v in var:
        assert isinstance(v, Array)

def test_intent_lint_finddimsnotvars(fcode):

    routine = Subroutine.from_source(fcode)
    assign = FindNodes(Assignment).visit(routine.body)[0]

    var = finddimsnotvars(assign)

    for v in var:
        assert isinstance(v, Scalar)

def test_intent_lint_inline_conditional_mem_check():

    fcode = """
subroutine test(a)
implicit none

  real,intent(out) :: a(:,:)

  if(size(a)>0) a(:,:) = 0.

end subroutine test
"""

    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat")
    assert result == 0, clean_intent_lint_test()

    with open('output.dat','r') as reader:
        for line in reader:
            assert not 'intent(out) rule broken' in line, clean_intent_lint_test()
    
    clean_intent_lint_test()

def test_intent_lint_conditional_out_check():

    fcode = """
subroutine test(a)
implicit none

  real,intent(out) :: a(:,:)

  if(a(1,1) == 0.) a(2,2) = a(1,1)

end subroutine test
"""
    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat")
    assert result == 0, clean_intent_lint_test()

    with open('output.dat','r') as reader:
        assert any('intent(out) rule broken for a' in line for line in reader), clean_intent_lint_test()
    
    clean_intent_lint_test()

def test_intent_lint_disable_call():

    fcode = """
subroutine test(a)
implicit none

  real,intent(out) :: a

  call random_call(a)

end subroutine test
"""
    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat --disable random_call")
    assert result == 0, clean_intent_lint_test()

    with open('output.dat','r') as reader:
        assert not any('intent(out) var a unused' in line for line in reader), clean_intent_lint_test()
    
    clean_intent_lint_test()

def test_intent_lint_specification_only():

    fcode = """
subroutine test(n,m,a)
implicit none

  integer               :: n
  integer,intent(inout) :: m
  real,intent(out) :: a(n,m)

  a = 0.

end subroutine test
"""

    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat --disable random_call")
    assert result == 0, clean_intent_lint_test()

    with open('output.dat', 'r') as reader:
        assert not any('unused' in line for line in reader), clean_intent_lint_test()
    with open('output.dat', 'r') as reader:
        assert any('used only as intent(in)' in line for line in reader), clean_intent_lint_test()
    with open('output.dat', 'r') as reader:
        assert any('Dummy argument n has no declared intent' in line for line in reader), clean_intent_lint_test()
    
    clean_intent_lint_test()

def test_intent_lint_allocation_only():

    fcode = """
subroutine test(n,m)
implicit none

  integer,intent(in   ) :: n
  integer,intent(inout) :: m
  real,allocatable :: a(:,:)

  allocate(a(n,m))


  deallocate(a)

end subroutine test
"""

    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat --disable random_call")
    assert result == 0, clean_intent_lint_test()

    with open('output.dat','r') as reader:
        assert not any('unused' in line for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert any('used only as intent(in)' in line for line in reader), clean_intent_lint_test()
    
    clean_intent_lint_test()

def test_intent_lint_loop_bounds_only():

    fcode = """
subroutine test(n,m)
implicit none

  integer,intent(in   ) :: n
  integer,intent(inout) :: m

  integer               :: i,j


  do j=1,m
    do i=1,n
    enddo
  enddo

end subroutine test
"""

    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat --disable random_call")
    assert result == 0, clean_intent_lint_test()

    with open('output.dat','r') as reader:
        assert not any('unused' in line for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert any('used only as intent(in)' in line for line in reader), clean_intent_lint_test()
    
    clean_intent_lint_test()

def test_intent_lint_violation_count_check(fcode_call):

    source = Sourcefile.from_source(fcode_call)
    source.write(path='test_intent_lint_tmp.F90')

    routines = source.all_subroutines
    for routine in routines:
        if routine.name == 'test':
            calls = FindNodes(CallStatement).visit(routine.body)

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat --summary summary.dat")
    assert result == 0, clean_intent_lint_test() 

    with open('summary.dat','r') as reader:
        assert any('Intent unused:9' == line.strip() for line in reader), clean_intent_lint_test()
    with open('summary.dat','r') as reader:
        assert any('Intent violated:5' == line.strip() for line in reader), clean_intent_lint_test()

    clean_intent_lint_test()

def test_intent_lint_call_consistency_check(fcode_call):

    source = Sourcefile.from_source(fcode_call)
    source.write(path='test_intent_lint_tmp.F90')

    routines = source.all_subroutines
    for routine in routines:
        if routine.name == 'test':
            calls = FindNodes(CallStatement).visit(routine.body)

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat")
    assert result == 0, clean_intent_lint_test() 


#   check first call
    with open('output.dat','r') as reader:
        assert not any(f'intent inconsistency in {calls[0]} for arg v_inout' == line.strip() for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert any(f'intent inconsistency in {calls[0]} for arg v_out' == line.strip() for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert any(f'intent inconsistency in {calls[0]} for arg v_in' == line.strip() for line in reader), clean_intent_lint_test()

#   check second call
    with open('output.dat','r') as reader:
        assert not any(f'intent inconsistency in {calls[1]} for arg v_inout' == line.strip() for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert any(f'intent inconsistency in {calls[1]} for arg v_out' == line.strip() for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert not any(f'intent inconsistency in {calls[1]} for arg v_in' == line.strip() for line in reader), clean_intent_lint_test()

#   check third call
    with open('output.dat','r') as reader:
        assert any(f'intent inconsistency in {calls[2]} for arg v_inout' == line.strip() for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert any(f'intent inconsistency in {calls[2]} for arg v_out' == line.strip() for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert not any(f'intent inconsistency in {calls[2]} for arg v_in' == line.strip() for line in reader), clean_intent_lint_test()
    
    clean_intent_lint_test()

def test_intent_lint_call_kwargs():

    fcode = """
subroutine random_call(c,e,d,a,b)
implicit none

  integer,intent(in) :: a,b,c,d,e

end subroutine random_call

subroutine test(a,b,c,d,e)
implicit none

  integer,intent(in) :: a,b,c,d,e

  call random_call(a=a,b=b,c=c,d=d,e=e)

end subroutine test
"""

    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat")
    assert result == 0, clean_intent_lint_test() 

    clean_intent_lint_test()

def test_intent_lint_internal_routine():

    fcode = """
subroutine test(a,b,c,n,m)
implicit none

  integer,intent(in) :: n,m
  real,intent(out) :: a(n)
  real,intent(out) :: b(n)
  real,intent(in ) :: c(n)

  call internal_routine(c,a0=a,b=b)

contains
   subroutine internal_routine(c,b,a0)

       real,intent(in)  ::  a0(n)
       real,intent(in)  ::  c(n)
       real,intent(out) ::  b(n)

       integer                     ::  i

       b=a0
       a0=c

       do i=1,m
       enddo

   end subroutine internal_routine

end subroutine test
"""
    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat")

    with open('output.dat','r') as reader:
        assert any(f'intent inconsistency in Call:: internal_routine for arg a' == line.strip() for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert any(f'intent(out) rule broken for a' == line.strip() for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert not any('unused' in line for line in reader), clean_intent_lint_test()

    assert result == 0, clean_intent_lint_test()

    clean_intent_lint_test()

def test_intent_lint_resolve_associations():

    fcode = """
subroutine test(n,a_target,b_target)
use iso_fortran_env, only: real64
implicit none

  integer,intent(in) :: n
  real(kind=real64),intent(out),target :: a_target(n)
  real(kind=real64),intent(out),target :: b_target(n)

  real(kind=real64),pointer :: a_pointer

  a_pointer => a_target
  associate(temp_a=>a_pointer)
      do i=1,n
         temp_a(i) = 0. 
      enddo
  end associate

  a_pointer=>NULL()

  a_pointer => b_target(:)
  a_pointer = a_target(:)

end subroutine test
"""
    source = Sourcefile.from_source(fcode)
    source.write(path='test_intent_lint_tmp.F90')

    result = os.system("loki-intent-lint.py --mode rule-unused --setup manual --path test_intent_lint_tmp.F90 --output output.dat")

    assert result == 0, clean_intent_lint_test()

    with open('output.dat','r') as reader:
        assert not any('unused' in line for line in reader), clean_intent_lint_test()
    with open('output.dat','r') as reader:
        assert not any('broken' in line for line in reader), clean_intent_lint_test()

    clean_intent_lint_test()
