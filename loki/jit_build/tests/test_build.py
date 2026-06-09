# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import os
import time
import pytest
import numpy as np

from loki import Subroutine
from loki.jit_build import Obj, Lib, Builder, run_isolated, jit_compile_and_run
from loki.jit_build.compiler import  (
    Compiler, GNUCompiler, NvidiaCompiler, get_compiler_from_env,
    _default_compiler
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='function', name='builder')
def fixture_builder(here, tmp_path):
    yield Builder(source_dirs=here, build_dir=tmp_path)
    Obj.clear_cache()


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


def _isolated_add(a, b):
    return a + b


def _isolated_write_file(path):
    Path(path).write_text('child process wrote this')


def _isolated_raise():
    raise ValueError('isolated failure')


def _isolated_exit(status):
    os._exit(status)  # pylint: disable=protected-access


def _isolated_sleep(seconds):
    time.sleep(seconds)


def test_run_isolated_returns_result():
    """
    Test that run_isolated returns picklable target results.
    """
    assert run_isolated(_isolated_add, 2, 5) == 7


def test_run_isolated_process_side_effect(tmp_path):
    """
    Test that run_isolated executes the target in a child process.
    """
    path = tmp_path/'isolated.txt'
    assert run_isolated(_isolated_write_file, path) is None
    assert path.read_text() == 'child process wrote this'


def test_run_isolated_exit_after_result():
    """
    Test that run_isolated can bypass child interpreter shutdown after success.
    """
    assert run_isolated(_isolated_add, 2, 5, exit_after_result=True) == 7


def test_run_isolated_exception():
    """
    Test that run_isolated reports Python exceptions from the child process.
    """
    with pytest.raises(RuntimeError) as excinfo:
        run_isolated(_isolated_raise)
    assert 'ValueError: isolated failure' in str(excinfo.value)


def test_run_isolated_nonzero_exit():
    """
    Test that run_isolated reports child process crashes or explicit exits.
    """
    with pytest.raises(RuntimeError) as excinfo:
        run_isolated(_isolated_exit, 7)
    assert 'exit code 7' in str(excinfo.value)


def test_run_isolated_timeout():
    """
    Test that run_isolated terminates child processes that exceed a timeout.
    """
    with pytest.raises(RuntimeError) as excinfo:
        run_isolated(_isolated_sleep, 5, timeout=0.1)
    assert 'timed out after 0.1 seconds' in str(excinfo.value)


def test_jit_compile_and_run_in_process(tmp_path):
    """
    Test that jit_compile_and_run supports explicit in-process execution.
    """
    source = """
subroutine fill_array(a, n)
  integer, intent(inout) :: a(n)
  integer, intent(in) :: n
  integer :: j
  do j=1,n
    a(j) = j
  end do
end subroutine fill_array
"""
    routine = Subroutine.from_source(source)
    n = 5
    a = np.zeros(shape=(n,), dtype=np.int32)
    jit_compile_and_run(routine, filepath=tmp_path/'fill_array.F90', isolated=False, a=a, n=n)
    assert np.all(a == range(1, n+1))


def test_jit_compile_and_run_isolated(tmp_path):
    """
    Test that jit_compile_and_run isolates compile-and-execute cycles by default.
    """
    source = """
subroutine fill_array(a, n)
  integer, intent(inout) :: a(n)
  integer, intent(in) :: n
  integer :: j
  do j=1,n
    a(j) = j
  end do
end subroutine fill_array
"""
    routine = Subroutine.from_source(source)
    n = 5
    a = np.zeros(shape=(n,), dtype=np.int32)
    result = jit_compile_and_run(
        routine, filepath=tmp_path/'fill_array_isolated.F90', a=a, n=n
    )
    assert result is None
    assert np.all(a == range(1, n+1))


def test_build_clean(builder):
    """
    Test basic `make clean`-style functionality.
    """
    # Mess up the build dir before cleaning it...
    (builder.build_dir/'xxx_a.o').touch(exist_ok=True)
    (builder.build_dir/'xxx_b.o').touch(exist_ok=True)
    (builder.build_dir/'xxx_a.mod').touch(exist_ok=True)
    (builder.build_dir/'xxx_a.so').touch(exist_ok=True)
    (builder.build_dir/'f90wrap_xxx_a.f90').touch(exist_ok=True)

    builder.clean('*.o *.mod *.so f90wrap*.f90')
    for f in builder.build_dir.iterdir():
        assert 'xxx' not in str(f)


def test_build_object(here, testdir, builder):
    """
    Test basic object compilation and wrapping via f90wrap.
    """
    obj = Obj(source_path=here/'base.f90')
    obj.build(builder=builder)
    assert (builder.build_dir/'base.o').exists()

    base = obj.wrap(builder=builder, kind_map=testdir/'kind_map')
    assert base.Base.a_times_b_plus_c(a=2, b=3, c=1) == 7


@pytest.mark.parametrize('workers', [None, 1, 3])
def test_build_lib(here, tmp_path, testdir, workers):
    """
    Test basic library compilation and wrapping via f90wrap
    from a specific list of source objects.
    """
    builder = Builder(source_dirs=here, build_dir=tmp_path, workers=workers)

    # Create library with explicit dependencies
    base = Obj(source_path=here/'base.f90')
    extension = Obj(source_path=here/'extension.f90')
    # Note: Need to compile statically to avoid LD_LIBRARY_PATH lookup
    lib = Lib(name='library', objs=[base, extension], shared=False)
    lib.build(builder=builder)
    assert (builder.build_dir/'liblibrary.a').exists()

    test = lib.wrap(modname='test', sources=[here/'extension.f90'], builder=builder,
                    kind_map=testdir/'kind_map')
    assert test.extended_fma(2., 3., 1.) == 7.


def test_build_lib_with_c(here, testdir, builder):
    """
    Test basic library compilation and wrapping via f90wrap
    from a specific list of source objects.
    """
    # Create library with explicit dependencies
    # objects = ['wrapper.f90', 'c_util.c']
    wrapper = Obj(source_path=here/'wrapper.f90')
    c_util = Obj(source_path=here/'c_util.c')
    lib = Lib(name='library', objs=[wrapper, c_util], shared=False)
    lib.build(builder=builder)
    assert (builder.build_dir/'liblibrary.a').exists()

    wrap = lib.wrap(modname='wrap', sources=[here/'wrapper.f90'], builder=builder,
                    kind_map=testdir/'kind_map')
    assert wrap.wrapper.mult_add_external(2., 3., 1.) == 7.


def test_build_obj_dependencies():
    """
    Test dependency resolution in a non-trivial module tree.
    """
    # # Wrap obj without specifying dependencies
    # test = builder.Obj('extension.f90').wrap()
    # assert test.library_test(1, 2, 3) == 12


def test_build_binary(builder):
    """
    Test basic binary compilation from objects and libs.
    """
    assert builder


@pytest.mark.parametrize('env,cls,attrs', [
    # Overwrite completely custom
    (
        {'CC': 'my-weird-compiler', 'FC': 'my-other-weird-compiler', 'F90': 'weird-fortran', 'FCFLAGS': '-my-flag  '},
        Compiler,
        {'CC': 'my-weird-compiler', 'FC': 'my-other-weird-compiler', 'F90': 'weird-fortran', 'FCFLAGS': ['-my-flag']},
    ),
    # GNUCompiler
    ({'CC': 'gcc'}, GNUCompiler, {'CC': 'gcc', 'FC': 'gfortran', 'F90': 'gfortran'}),
    ({'CC': 'gcc-13'}, GNUCompiler, None),
    ({'CC': '/path/to/my/gcc'}, GNUCompiler, None),
    ({'CC': '../../relative/path/to/my/gcc-11'}, GNUCompiler, None),
    ({'CC': 'C:\\windows\\path\\to\\gcc'}, GNUCompiler, None),
    ({'FC': 'gfortran'}, GNUCompiler, None),
    ({'FC': 'gfortran-13', 'FCFLAGS': '-O3 -g'}, GNUCompiler, {'FC': 'gfortran-13', 'FCFLAGS': ['-O3', '-g']}),
    ({'FC': '/path/to/my/gfortran'}, GNUCompiler, None),
    ({'FC': '../../relative/path/to/my/gfortran'}, GNUCompiler, None),
    ({'FC': 'C:\\windows\\path\\to\\gfortran'}, GNUCompiler, None),
    # NvidiaCompiler
    ({'FC': 'nvfortran'}, NvidiaCompiler, {'CC': 'nvc', 'FC': 'nvfortran', 'F90': 'nvfortran'}),
    ({'CC': 'nvc'}, NvidiaCompiler, None),
    ({'CC': '/path/to/my/nvc'}, NvidiaCompiler, None),
    ({'CC': '../../relative/path/to/my/nvc'}, NvidiaCompiler, None),
    ({'CC': 'C:\\windows\\path\\to\\nvc'}, NvidiaCompiler, None),
    ({'FC': 'pgf90'}, NvidiaCompiler, None),
    ({'FC': 'pgf95'}, NvidiaCompiler, None),
    ({'FC': 'pgfortran'}, NvidiaCompiler, None),
    ({'FC': '/path/to/my/nvfortran'}, NvidiaCompiler, None),
    ({'FC': '../../relative/path/to/my/pgfortran'}, NvidiaCompiler, None),
    ({'FC': 'C:\\windows\\path\\to\\nvfortran'}, NvidiaCompiler, None),
])
def test_get_compiler_from_env(env, cls, attrs):
    compiler = get_compiler_from_env(env)
    assert type(compiler) == cls  # pylint: disable=unidiomatic-typecheck
    for attr, expected_value in (attrs or env).items():
        # NB: We are comparing the lower-case attribute
        # because that contains the runtime value
        assert getattr(compiler, attr.lower()) == expected_value


def test_default_compiler():
    # Check that _default_compiler corresponds to a call with None
    compiler = get_compiler_from_env()
    assert type(compiler) == type(_default_compiler)  # pylint: disable=unidiomatic-typecheck


def test_obj_dependencies(tmp_path):
    fcode = """
module import_mod
    use module_mod
    implicit none
contains
    subroutine proc1
        use  , non_intrinsic :: iso_fortran_env, only: int8, int16
        use  , intrinsic :: iso_c_binding
        use other_Mod
        use :: third_mod
        use    fourth_mod , only:some
        use,non_intrinsic::very_condensed
    end subroutine proc1
    subroutine proc2
        use, intrinsic :: iso_fortran_env, only: int8, int16
        use::fifth_mod
    end subroutine proc2
end module import_mod
    """.strip()

    filepath = tmp_path/'import_mod.f90'
    filepath.write_text(fcode)

    obj = Obj(name='import_mod', source_path=filepath)
    assert obj.dependencies == (
        'module_mod', 'iso_fortran_env', 'other_Mod', 'third_mod', 'fourth_mod',
        'very_condensed', 'fifth_mod'
    )
    Obj.clear_cache()
