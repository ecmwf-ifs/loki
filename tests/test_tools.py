"""
Unit tests for utility functions and classes in loki.tools.
"""

import pytest
from loki.ir import Pragma
from loki.tools import JoinableStringList, truncate_string, is_loki_pragma, get_pragma_parameters


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


@pytest.mark.parametrize('keyword, content, starts_with, ref', [
    ('foo', None, None, False),
    ('foo', 'bar', None, False),
    ('foo', 'loki', None, False),
    ('foo', 'loki', 'loki', False),
    ('loki', None, None, True),
    ('loki', None, 'foo', False),
    ('loki', 'dataflow', None, True),
    ('loki', 'dataflow', 'dataflow', True),
    ('loki', 'dataflow', 'foobar', False),
    ('loki', 'fusion group(1)', None, True),
    ('loki', 'fusion group(1)', 'fusion', True),
    ('loki', 'fusion group(1)', 'group', False),
])
def test_is_loki_pragma(keyword, content, starts_with, ref):
    """
    Test correct identification of Loki pragmas.
    """
    pragma = Pragma(keyword, content)
    if starts_with is not None:
        assert is_loki_pragma(pragma, starts_with=starts_with) == ref
    else:
        assert is_loki_pragma(pragma) == ref


@pytest.mark.parametrize('content, starts_with, ref', [
    (None, None, {}),
    ('', None, {}),
    ('', 'foo', {}),
    ('dataflow', None, {'dataflow': None}),
    ('dataflow', 'dataflow', {}),
    ('dataflow group(1)', None, {'dataflow': None, 'group': '1'}),
    ('dataflow group(1)', 'dataflow', {'group': '1'}),
    ('dataflow group(1)', 'foo', {'dataflow': None, 'group': '1'}),
    ('dataflow group(1) group(2)', 'dataflow', {'group': '2'}),
    ('foo bar(^£!$%*[]:@+-_=~#/?.,<>;) baz foobar(abc_123")', 'foo',
     {'bar':'^£!$%*[]:@+-_=~#/?.,<>;', 'baz': None, 'foobar': 'abc_123"'}),
])
def test_get_pragma_parameters(content, starts_with, ref):
    """
    Test correct extraction of Loki pragma parameters.
    """
    pragma = Pragma('loki', content)
    if starts_with is None:
        assert get_pragma_parameters(pragma) == ref
    else:
        assert get_pragma_parameters(pragma, starts_with=starts_with) == ref
