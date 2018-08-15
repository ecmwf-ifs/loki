import pytest
import numpy as np
from pathlib import Path

from loki import Builder


@pytest.fixture(scope='module')
def path():
    return Path(__file__).parent


@pytest.fixture(scope='module')
def builder(path):
    return Builder(source_dirs=path, build_dir=path/'build')


@pytest.mark.parametrize('rules', [
    '*.o *.mod *.so f90wrap*.f90',
    ['*.o', '*.mod', '*.so', 'f90wrap*.f90'],
    ['xxx_a.o', 'xxx_b.o', 'xxx_a.mod', 'xxx_a.so', 'f90wrap_xxx_a.f90'],
])
def test_build_clean(builder, rules):
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

    obj = builder.Obj('base.f90')
    obj.build()
    assert (builder.build_dir/'base.o').exists

    base = obj.wrap()
    assert base.Base.a_times_b_plus_c(a=2, b=3, c=1) == 7


def test_build_lib(builder):
    """
    Test basic library compilation and wrapping via f90wrap
    from a specific list of source objects.
    """
    builder.clean()

    # # Create library with explicit dependencies
    # base = builder.Obj('base.f90')
    # extension = builder.Obj('extension.f90')
    # lib = builder.Lib(name='test', source=[base, extension])
    # assert (builder.build_dir/'libtest.so').exists

    # test = lib.wrap()
    # assert test.library_test(1, 2, 3) == 12


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
    pass
