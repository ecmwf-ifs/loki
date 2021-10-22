from pathlib import Path
import pytest

from loki.build import Obj, Lib, Builder


@pytest.fixture(scope='module', name='path')
def fixture_path():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builder')
def fixture_builder(path):
    return Builder(source_dirs=path, build_dir=path/'build')


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


def test_build_object(builder):
    """
    Test basic object compilation and wrapping via f90wrap.
    """
    builder.clean()

    obj = Obj(source_path='base.f90')
    obj.build(builder=builder)
    assert (builder.build_dir/'base.o').exists

    base = obj.wrap(builder=builder)
    assert base.Base.a_times_b_plus_c(a=2, b=3, c=1) == 7


def test_build_lib(builder):
    """
    Test basic library compilation and wrapping via f90wrap
    from a specific list of source objects.
    """
    builder.clean()

    # Create library with explicit dependencies
    base = Obj(source_path='base.f90')
    extension = Obj(source_path='extension.f90')
    # Note: Need to compile statically to avoid LD_LIBRARY_PATH lookup
    lib = Lib(name='library', objs=[base, extension], shared=False)
    lib.build(builder=builder)
    assert (builder.build_dir/'liblibrary.a').exists

    test = lib.wrap(modname='test', sources=['extension.f90'], builder=builder)
    assert test.extended_fma(2., 3., 1.) == 7.


def test_build_lib_with_c(builder):
    """
    Test basic library compilation and wrapping via f90wrap
    from a specific list of source objects.
    """
    builder.clean()

    # Create library with explicit dependencies
    # objects = ['wrapper.f90', 'c_util.c']
    wrapper = Obj(source_path='wrapper.f90')
    c_util = Obj(source_path='c_util.c')
    lib = Lib(name='library', objs=[wrapper, c_util], shared=False)
    lib.build(builder=builder)
    assert (builder.build_dir/'liblibrary.so').exists

    wrap = lib.wrap(modname='wrap', sources=['wrapper.f90'], builder=builder)
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
