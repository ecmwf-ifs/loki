# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from loki.build import (
    Obj, Lib, Builder,
    Compiler, GNUCompiler, NvidiaCompiler, get_compiler_from_env, _default_compiler
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
    builder.clean()

    obj = Obj(source_path=here/'base.f90')
    obj.build(builder=builder)
    assert (builder.build_dir/'base.o').exists()

    base = obj.wrap(builder=builder, kind_map=testdir/'kind_map')
    assert base.Base.a_times_b_plus_c(a=2, b=3, c=1) == 7


def test_build_lib(here, testdir, builder):
    """
    Test basic library compilation and wrapping via f90wrap
    from a specific list of source objects.
    """
    builder.clean()

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
    builder.clean()

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


def test_build_obj_dependencies(builder):
    """
    Test dependency resolution in a non-trivial module tree.
    """
    builder.clean()

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
