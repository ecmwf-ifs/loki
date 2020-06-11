"""
Unit tests for utility functions and classes in loki.tools.
"""

import pytest
from loki.tools import JoinableStringList, truncate_string


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
    obj = JoinableStringList(items, sep, width, cont, ref)
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
