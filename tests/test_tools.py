# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Unit tests for utility functions and classes in loki.tools.
"""

import sys
import operator as op
from contextlib import contextmanager
from pathlib import Path
from subprocess import CalledProcessError
from time import sleep, perf_counter
import pytest

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from conftest import stdchannel_is_captured, stdchannel_redirected
from loki.tools import (
    JoinableStringList, truncate_string, binary_insertion_sort, is_subset,
    optional, yaml_include_constructor, execute, timeout, dict_override
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('a, b, ref', [
    ((1, 2), (0, 1, 0, 2, 3), True),
    ((1, 2), (0, 2, 0, 1, 3), False),
    ((1, 2), (1, 0, 2, 3), True),
    ((1, 2), (1, 2), True),
    ((2, 1), (1, 2), False),
    ((1, 2), (1, 0, 2), True),
    ((), (1,), False),
    ((1,), (), False),
    ((), (), False),
    ((0, 0), (0, 1, 0, 2, 0, 3), True),
    ((0, 0), (0, 1, 2, 3), False),
])
def test_is_subset_ordered(a, b, ref):
    """
    Test :any:`is_subset` with ordered data types.
    """
    assert is_subset(a, b, ordered=True) == ref


@pytest.mark.parametrize('a, b, ref', [
    ((1, 2), (0, 1, 2, 3), True),
    ((1, 2), (0, 1, 2), True),
    ((1, 2), (1, 2, 3), True),
    ((1, 2), (1, 2), True),
    ((0, 1, 2, 3), (1, 2), False),
    ((0, 1, 2), (1, 2), False),
    ((1, 2, 3), (1, 2), False),
    ([1], (0, 1, 2), True),
    ((0, 1), [0, 1, 2, 3], True),
    ((1, 0), (0, 1), False),
    ((1,), (1, 2), True),
    ((1, 2), (1, 0, 2), False),
    ((), (1,), False),
    ((1,), (), False),
    ((), (), False),
    ((0, 0), (0, 1, 0, 2, 0, 3), False),
])
def test_is_subset_ordered_subsequent(a, b, ref):
    """
    Test :any:`is_subset` with ordered data types.
    """
    assert is_subset(a, b, ordered=True, subsequent=True) == ref


@pytest.mark.parametrize('a, b, ref', [
    ((1, 2), (0, 1, 2, 3), True),
    ((1, 2), (0, 1, 2), True),
    ((1, 2), (1, 2, 3), True),
    ((1, 2), (1, 2), True),
    ((0, 1, 2, 3), (1, 2), False),
    ((0, 1, 2), (1, 2), False),
    ((1, 2, 3), (1, 2), False),
    ([1], (0, 1, 2), True),
    ((0, 1), [0, 1, 2, 3], True),
    ((1, 0), (0, 1), True),
    ((1,), (1, 2), True),
    ((1, 2), (1, 0, 2), True),
])
def test_is_subset_not_ordered(a, b, ref):
    """
    Test :any:`is_subset` with ordered data types.
    """
    assert is_subset(a, b, ordered=False) == ref


@pytest.mark.parametrize('a, b', [
    ({1, 2}, [1, 2]),
    ([1, 2], {1, 2}),
])
def test_is_subset_raises(a, b):
    with pytest.raises(ValueError):
        is_subset(a, b, ordered=True)


@pytest.mark.parametrize('items, sep, width, cont, ref', [
    ([''], ' ', 90, '\n', ''),
    ([], ' ', 90, '\n', ''),
    (('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'), '', 90, '\n', 'Hello world!'),
    (('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'), '', 7, '\n', 'Hello \nworld!'),
    (('Hello', 'world!'), ' ', 90, '\n', 'Hello world!'),
    (('Hello', 'world!'), ' ', 7, '\n', 'Hello \nworld!'),
    (('Hello', 'world!'), ' ', 5, '\n', 'Hello\n \nworld!'),
    ((JoinableStringList(['H', 'e', 'l', 'l', 'o'], '', 5, '\n'), 'world!'), ' ', 5, '\n',
     'Hell\no \nworld!'),
    (('Hello', JoinableStringList(['w', 'o', 'r', 'l', 'd', '!'], '', 8, '\n', separable=False)),
     ' ', 8, '\n', 'Hello \nworld!'),
    (('Hello', JoinableStringList(['w', 'o', 'r', 'l', 'd', '!'], '', 8, '\n', separable=True)),
     ' ', 8, '\n', 'Hello w\norld!'),
])
def test_joinable_string_list(items, sep, width, cont, ref):
    """
    Test JoinableStringList for some common scenarios.
    """
    obj = JoinableStringList(items, sep, width, cont)
    assert str(obj) == ref


def test_joinable_string_list_long():
    """
    Test JoinableStringList with some long edge cases.
    """
    attributes = ['REAL(KIND=JPRB)', 'INTENT(IN)']
    attributes = JoinableStringList(attributes, ', ', 132, ' &\n   & ')
    variables = ['PDHTLS(KPROMA, YDMODEL%YRML_PHY_G%YRDPHY%NTILES, '
                 'YDMODEL%YRML_DIAG%YRMDDH%NDHVTLS + YDMODEL%YRML_DIAG%YRMDDH%NDHFTLS)']
    variables = JoinableStringList(variables, ', ', 132, ' &\n   & ')
    items = ['  ', attributes, ' :: ', variables]
    obj = JoinableStringList(items, '', 132, ' &\n  & ')
    ref = ('  REAL(KIND=JPRB), INTENT(IN) ::  &\n'
           '  & PDHTLS(KPROMA, YDMODEL%YRML_PHY_G%YRDPHY%NTILES, '
           'YDMODEL%YRML_DIAG%YRMDDH%NDHVTLS + YDMODEL%YRML_DIAG%YRMDDH%NDHFTLS)')
    assert str(obj) == ref

    name = 'io.output'
    args = ['"tensor_out"', 'tensor_out',
            'new DFEVectorType<DFEVector<DFEVar>>(new DFEVectorType<DFEVar>(dfeFloat(11, 53), m), n)']
    args_list = JoinableStringList(args, sep=', ', width=90, cont='\n      ', separable=True)
    items = ['    ', name, '(', args_list, ');']
    items_list = JoinableStringList(items, sep='', width=90, cont='\n      ', separable=True)
    line = str(items_list)
    ref = ('    io.output("tensor_out", tensor_out, \n'
           '      new DFEVectorType<DFEVector<DFEVar>>(new DFEVectorType<DFEVar>(dfeFloat(11, 53), m)\n'
           '      , n));')
    assert line == ref

    args = ['my_long_var = 5+3*tendency_loc(ibl)%T(jl,jk)']
    obj = JoinableStringList(args, sep=' ', width=40, cont=' &\n & ')
    ref = ('my_long_var =  &\n'
           ' & 5+3*tendency_loc(ibl)%T(jl,jk)')
    assert str(obj) == ref


@pytest.mark.parametrize('string, length, continuation, ref', [
    ('short string', 16, '...', 'short string'),
    ('short string', 12, '...', 'short string'),
    ('short string', 11, '...', 'short st...'),
])
def test_truncate_string(string, length, continuation, ref):
    """
    Test string truncation for different string lengths.
    """
    assert truncate_string(string, length, continuation) == ref


def test_binary_insertion_sort():
    """
    Test binary insertion sort for some random cases.
    """
    items = [37, 23, 0, 17, 12, 72, 31, 46, 100, 88, 54]

    assert binary_insertion_sort(items) == sorted(items)
    assert binary_insertion_sort(items, lt=op.gt) == sorted(items, reverse=True)

    assert binary_insertion_sort(list(range(20))) == list(range(20))
    assert binary_insertion_sort(list(reversed(range(20)))) == list(range(20))

    assert binary_insertion_sort([1] * 5) == [1] * 5


def test_optional():
    @contextmanager
    def dummy_manager(a, b, c):
        ret = a + b + c
        try:
            yield ret
        finally:
            pass

    with optional(True, dummy_manager, 1, c=10, b=100) as val:
        assert val == 111

    with optional(False, dummy_manager, 1, c=10, b=100) as val:
        assert val is None


@pytest.mark.skipif(not HAVE_YAML, reason="Pyyaml is not installed")
def test_yaml_include(here):
    include_yaml = """
foo:
  bar:
  - abc
  - def

foobar:
  - baz:
      dummy: value
  - 42:
      dummy: other_value
    """.strip()

    include_path = here/'include.yml'
    include_path.write_text(include_yaml)

    main_yaml = f"""
include: !include {include_path}

nested_foo: !include {include_path}:["foo"]

nested_foo_list: !include {include_path}:["foo"]["bar"][1]

nested_foobar: !include {include_path}:["foobar"][0]['baz']["dummy"]
    """.strip()

    main_path = here/'main.yml'
    main_path.write_text(main_yaml)

    nested_yaml = f"""
main: !include {main_path}
    """.strip()

    yaml.add_constructor('!include', yaml_include_constructor, yaml.SafeLoader)

    included = yaml.safe_load(include_yaml)
    main = yaml.safe_load(main_yaml)

    assert main['include'] == included
    assert main['nested_foo'] == included['foo']
    assert main['nested_foo_list'] == included['foo']['bar'][1]
    assert main['nested_foobar'] == included['foobar'][0]['baz']['dummy']

    nested = yaml.safe_load(nested_yaml)
    assert nested['main'] == main

    include_path.unlink()
    main_path.unlink()


def test_execute(here, capsys):

    testfile = here/'test_execute.txt'
    if testfile.is_file():
        testfile.unlink()

    # Failure with no output
    cmd = 'false'
    if stdchannel_is_captured(capsys):
        with pytest.raises(CalledProcessError):
            execute(cmd)
    else:
        with capsys.disabled():
            with stdchannel_redirected(sys.stdout, testfile):
                with stdchannel_redirected(sys.stderr, testfile):
                    with pytest.raises(CalledProcessError):
                        execute(cmd)

        assert 'Execution of false failed' in testfile.read_text()
        assert 'Full command: false' in testfile.read_text()
        assert 'Output of the command:' not in testfile.read_text()
        testfile.unlink()

    # Failure with output
    cmd = ['cat', '/not/a/file']
    if stdchannel_is_captured(capsys):
        with pytest.raises(CalledProcessError):
            execute(cmd)
    else:
        with capsys.disabled():
            with stdchannel_redirected(sys.stdout, testfile):
                with stdchannel_redirected(sys.stderr, testfile):
                    with pytest.raises(CalledProcessError):
                        execute(cmd)

        assert 'Execution of cat failed' in testfile.read_text()
        assert f'Full command: {" ".join(cmd)}' in testfile.read_text()
        assert 'Output of the command:' in testfile.read_text()
        assert 'No such file or directory' in testfile.read_text()
        testfile.unlink()

    # Success
    cmd = 'true'
    execute(cmd)


def test_timeout():
    # Should not trigger:
    start = perf_counter()
    with timeout(5):
        sleep(.3)
    stop = perf_counter()
    assert .2 < stop - start < .4

    # Timeout disabled:
    start = perf_counter()
    with timeout(0):
        sleep(.3)
    stop = perf_counter()
    assert .2 < stop - start < .4

    # Default exception
    with pytest.raises(RuntimeError) as exc:
        start = perf_counter()
        with timeout(1):
            sleep(5)
        stop = perf_counter()
        assert .9 < stop - start < 1.1
        assert "Timeout reached after 2 second(s)" in str(exc.value)

    # Custom message
    with pytest.raises(RuntimeError) as exc:
        start = perf_counter()
        with timeout(1, message="My message"):
            sleep(5)
        stop = perf_counter()
        assert .9 < stop - start < 1.1
        assert "My message" in str(exc.value)


def test_dict_override():
    kwargs = {'rick' : 42, 'dave' : 'yeah'}
    with dict_override(kwargs, {'dave' : 'nope', 'joe' : 'huh?'}):
        assert kwargs['dave'] == 'nope'
        assert kwargs['rick'] == 42
        assert kwargs['joe'] == 'huh?'
    assert kwargs['dave'] == 'yeah'
    assert kwargs['rick'] == 42
    assert 'joe' not in kwargs
    assert len(kwargs) == 2
