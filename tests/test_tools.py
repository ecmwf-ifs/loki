"""
Unit tests for utility functions and classes in loki.tools.
"""

import operator as op
import pytest
from loki.tools import JoinableStringList, truncate_string, binary_insertion_sort


@pytest.mark.parametrize('items, sep, width, cont, ref', [
    ([''], ' ', 90, '\n', ''),
    ([], ' ', 90, '\n', ''),
    (('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'), '', 90, '\n', 'Hello world!'),
    (('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'), '', 7, '\n', 'Hello \nworld!'),
    (('Hello', 'world!'), ' ', 90, '\n', 'Hello world!'),
    (('Hello', 'world!'), ' ', 7, '\n', 'Hello \nworld!'),
    (('Hello', 'world!'), ' ', 5, '\n', 'Hello \nworld!'),
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
           '      new DFEVectorType<DFEVector<DFEVar>>(new DFEVectorType<DFEVar>(dfeFloat(11, 53), \n'
           '      m), n));')
    assert line == ref


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
