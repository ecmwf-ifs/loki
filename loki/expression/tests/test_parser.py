# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

import pymbolic.primitives as pmbl
import pymbolic.mapper as pmbl_mapper

from loki import Subroutine, Module, Scope
from loki.expression import symbols as sym, FindVariables, parse_expr
from loki.frontend import (
    available_frontends, OMNI, HAVE_FP, parse_fparser_expression
)


# utility function to test parse_expr with different case
def convert_to_case(_str, mode='upper'):
    if mode == 'upper':
        return _str.upper()
    if mode == 'lower':
        return _str.lower()
    if mode == 'random':
        # this is obviously not random, but fulfils its purpose ...
        result = ''
        for i, char in enumerate(_str):
            result += char.upper() if i%2==0 and i<3 else char.lower()
        return result
    return convert_to_case(_str)


@pytest.mark.parametrize('source, ref', [
    ('1 + 1', '1 + 1'),
    ('1+2+3+4', '1 + 2 + 3 + 4'),
    ('5*4 - 3*2 - 1', '5*4 - 3*2 - 1'),
    ('1*(2 + 3)', '1*(2 + 3)'),
    ('5*a +3*7**5 - 4/b', '5*a + 3*7**5 - 4 / b'),
    ('5 + (4 + 3) - (2*1)', '5 + (4 + 3) - (2*1)'),
    ('a*(b*(c+(d+e)))', 'a*(b*(c + (d + e)))'),
])
@pytest.mark.parametrize('parse', (
    parse_expr,
    pytest.param(parse_fparser_expression,
        marks=pytest.mark.skipif(not HAVE_FP, reason='parse_fparser_expression not available!'))
))
def test_parse_expression(parse, source, ref):
    """
    Test the utility function that parses simple expressions.
    """
    scope = Scope()
    ir = parse(source, scope)  # pylint: disable=redefined-outer-name
    assert isinstance(ir, pmbl.Expression)
    assert str(ir) == ref


@pytest.mark.parametrize('case', ('upper', 'lower', 'random'))
@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_parser(frontend, case, tmp_path):
    fcode = """
subroutine some_routine()
  implicit none
  integer :: i1, i2, i3, len1, len2, len3
  real :: a, b
  real :: arr(len1, len2, len3)
end subroutine some_routine
    """.strip()

    fcode_mod = """
module external_mod
  implicit none
contains
  function my_func(a)
    integer, intent(in) :: a
    integer :: my_func
    my_func = a
  end function my_func
end module external_mod
    """.strip()

    def to_str(_parsed):
        return str(_parsed).lower().replace(' ', '')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    module = Module.from_source(fcode_mod, frontend=frontend, xmods=[tmp_path])

    parsed = parse_expr(convert_to_case('a + b', mode=case))
    assert isinstance(parsed, sym.Sum)
    assert all(isinstance(_parsed,  sym.DeferredTypeSymbol) for _parsed in parsed.children)
    assert to_str(parsed) == 'a+b'

    parsed = parse_expr(convert_to_case('a + b', mode=case), scope=routine)
    assert isinstance(parsed, sym.Sum)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in parsed.children)
    assert all(_parsed.scope == routine for _parsed in parsed.children)
    assert to_str(parsed) == 'a+b'

    parsed = parse_expr(convert_to_case('a + b + 2 + 10', mode=case), scope=routine)
    assert isinstance(parsed, sym.Sum)
    assert to_str(parsed) == 'a+b+2+10'

    parsed = parse_expr(convert_to_case('a - b', mode=case), scope=routine)
    assert isinstance(parsed, sym.Sum)
    assert isinstance(parsed.children[0], sym.Scalar)
    assert isinstance(parsed.children[1], sym.Product)
    assert to_str(parsed) == 'a-b'

    parsed = parse_expr(convert_to_case('a * b', mode=case), scope=routine)
    assert isinstance(parsed, sym.Product)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in parsed.children)
    assert all(_parsed.scope == routine for _parsed in parsed.children)
    assert to_str(parsed) == 'a*b'

    parsed = parse_expr(convert_to_case('a / b', mode=case), scope=routine)
    assert isinstance(parsed, sym.Quotient)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.numerator, parsed.denominator])
    assert all(_parsed.scope == routine for _parsed in [parsed.numerator, parsed.denominator])
    assert to_str(parsed) == 'a/b'

    parsed = parse_expr(convert_to_case('a ** b', mode=case), scope=routine)
    assert isinstance(parsed, sym.Power)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.base, parsed.exponent])
    assert all(_parsed.scope == routine for _parsed in [parsed.base, parsed.exponent])
    assert to_str(parsed) == 'a**b'

    parsed = parse_expr(convert_to_case(':', mode=case))
    assert isinstance(parsed, sym.RangeIndex)
    assert to_str(parsed) == ':'

    parsed = parse_expr(convert_to_case('a:b', mode=case), scope=routine)
    assert isinstance(parsed, sym.RangeIndex)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.lower, parsed.upper])
    assert all(_parsed.scope == routine for _parsed in [parsed.lower, parsed.upper])
    assert to_str(parsed) == 'a:b'

    parsed = parse_expr(convert_to_case('a:b:5', mode=case), scope=routine)
    assert isinstance(parsed, sym.RangeIndex)
    assert all(isinstance(_parsed,  (sym.Scalar, sym.IntLiteral))
            for _parsed in [parsed.lower, parsed.upper, parsed.step])
    assert to_str(parsed) == 'a:b:5'

    parsed = parse_expr(convert_to_case('a == b', mode=case), scope=routine)
    assert parsed.operator == '=='
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a==b'
    parsed = parse_expr(convert_to_case('a.eq.b', mode=case), scope=routine)
    assert parsed.operator == '=='
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a==b'

    parsed = parse_expr(convert_to_case('a!=b', mode=case), scope=routine)
    assert parsed.operator == '!='
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a!=b'
    parsed = parse_expr(convert_to_case('a.ne.b', mode=case), scope=routine)
    assert parsed.operator == '!='
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a!=b'

    parsed = parse_expr(convert_to_case('a>b', mode=case), scope=routine)
    assert parsed.operator == '>'
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a>b'
    parsed = parse_expr(convert_to_case('a.gt.b', mode=case), scope=routine)
    assert parsed.operator == '>'
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a>b'

    parsed = parse_expr(convert_to_case('a>=b', mode=case), scope=routine)
    assert parsed.operator == '>='
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a>=b'
    parsed = parse_expr(convert_to_case('a.ge.b', mode=case), scope=routine)
    assert parsed.operator == '>='
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a>=b'

    parsed = parse_expr(convert_to_case('a<b', mode=case), scope=routine)
    assert parsed.operator == '<'
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a<b'
    parsed = parse_expr(convert_to_case('a.lt.b', mode=case), scope=routine)
    assert parsed.operator == '<'
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a<b'

    parsed = parse_expr(convert_to_case('a<=b', mode=case), scope=routine)
    assert parsed.operator == '<='
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a<=b'
    parsed = parse_expr(convert_to_case('a.le.b', mode=case), scope=routine)
    assert parsed.operator == '<='
    assert isinstance(parsed, sym.Comparison)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in [parsed.left, parsed.right])
    assert all(_parsed.scope == routine for _parsed in [parsed.left, parsed.right])
    assert to_str(parsed) == 'a<=b'

    parsed = parse_expr(convert_to_case('arr(i1, i2, i3)', mode=case))
    assert isinstance(parsed, sym.Array)
    assert all(isinstance(_parsed,  sym.DeferredTypeSymbol) for _parsed in parsed.dimensions)
    assert(parsed) == 'arr(i1,i2,i3)'
    parsed = parse_expr(convert_to_case('arr(i1, i2, i3)', mode=case), scope=routine)
    assert isinstance(parsed, sym.Array)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in parsed.dimensions)
    assert all(_parsed.scope == routine for _parsed in parsed.dimensions)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in parsed.shape)
    assert all(_parsed.scope == routine for _parsed in parsed.shape)
    assert to_str(parsed) == 'arr(i1,i2,i3)'

    parsed = parse_expr(convert_to_case('my_func(i1)', mode=case), scope=routine)
    assert isinstance(parsed, sym.Array)
    assert to_str(parsed) == 'my_func(i1)'
    parsed = parse_expr(convert_to_case('my_func(i1)', mode=case), scope=module)
    assert isinstance(parsed, sym.InlineCall)
    assert to_str(parsed) == 'my_func(i1)'

    parsed = parse_expr(convert_to_case('min(i1, i2)', mode=case), scope=module)
    assert isinstance(parsed, sym.InlineCall)
    assert to_str(parsed) == 'min(i1,i2)'

    parsed = parse_expr(convert_to_case('a', mode=case))
    assert isinstance(parsed, sym.DeferredTypeSymbol)
    assert to_str(parsed) == 'a'
    parsed = parse_expr(convert_to_case('a', mode=case), scope=routine)
    assert isinstance(parsed, sym.Scalar)
    assert parsed.scope == routine
    assert to_str(parsed) == 'a'
    parsed = parse_expr(convert_to_case('3.1415', mode=case))
    assert isinstance(parsed, sym.FloatLiteral)
    assert to_str(parsed) == '3.1415'

    parsed = parse_expr(convert_to_case('some_type%val', mode=case))
    assert isinstance(parsed, sym.DeferredTypeSymbol)
    assert isinstance(parsed.parent, sym.DeferredTypeSymbol)
    assert to_str(parsed) == 'some_type%val'
    parsed = parse_expr(convert_to_case('-some_type%val', mode=case))
    assert isinstance(parsed, sym.Product)
    assert isinstance(parsed.children[1].parent, sym.DeferredTypeSymbol)
    assert to_str(parsed) == '-some_type%val'
    parsed = parse_expr(convert_to_case('some_type%another_type%val', mode=case))
    assert isinstance(parsed, sym.DeferredTypeSymbol)
    assert isinstance(parsed.parent, sym.DeferredTypeSymbol)
    assert isinstance(parsed.parent.parent, sym.DeferredTypeSymbol)
    assert to_str(parsed) == 'some_type%another_type%val'
    parsed = parse_expr(convert_to_case('some_type%arr(a, b)', mode=case))
    assert isinstance(parsed, sym.Array)
    assert isinstance(parsed.parent, sym.DeferredTypeSymbol)
    assert to_str(parsed) == 'some_type%arr(a,b)'
    parsed = parse_expr(convert_to_case('some_type%some_func()', mode=case))
    assert isinstance(parsed, sym.InlineCall)
    assert isinstance(parsed.function.parent, sym.DeferredTypeSymbol)
    assert to_str(parsed) == 'some_type%some_func()'

    parsed = parse_expr(convert_to_case('"some_string_literal 42 _-*"', mode=case))
    assert isinstance(parsed, sym.StringLiteral)
    assert parsed.value.lower() == 'some_string_literal 42 _-*'
    assert to_str(parsed) == "'some_string_literal42_-*'"

    parsed = parse_expr(convert_to_case("'some_string_literal 42 _-*'", mode=case))
    assert isinstance(parsed, sym.StringLiteral)
    assert parsed.value.lower() == 'some_string_literal 42 _-*'
    assert to_str(parsed) == "'some_string_literal42_-*'"

    parsed = parse_expr(convert_to_case('MODULO(A, B)', mode=case), scope=routine)
    assert isinstance(parsed, sym.InlineCall)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in parsed.parameters)
    assert all(_parsed.scope == routine for _parsed in parsed.parameters)
    assert to_str(parsed) == 'modulo(a,b)'

    parsed = parse_expr(convert_to_case('a .and. b', mode=case))
    assert isinstance(parsed, sym.LogicalAnd)
    assert all(isinstance(_parsed,  sym.DeferredTypeSymbol) for _parsed in parsed.children)
    assert to_str(parsed) == 'aandb'
    parsed = parse_expr(convert_to_case('a .and. b', mode=case), scope=routine)
    assert isinstance(parsed, sym.LogicalAnd)
    assert all(isinstance(_parsed,  sym.Scalar) for _parsed in parsed.children)
    assert all(_parsed.scope == routine for _parsed in parsed.children)
    assert to_str(parsed) == 'aandb'
    parsed = parse_expr(convert_to_case('a .or. b', mode=case))
    assert isinstance(parsed, sym.LogicalOr)
    assert all(isinstance(_parsed,  sym.DeferredTypeSymbol) for _parsed in parsed.children)
    assert to_str(parsed) == 'aorb'
    parsed = parse_expr(convert_to_case('a .or. .not. b', mode=case))
    assert isinstance(parsed, sym.LogicalOr)
    assert isinstance(parsed.children[0], sym.DeferredTypeSymbol)
    assert isinstance(parsed.children[1], sym.LogicalNot)
    assert to_str('aornotb')

    parsed = parse_expr(convert_to_case('((a + b)/(a - b))**3 + 3.1415', mode=case), scope=routine)
    assert isinstance(parsed, sym.Sum)
    assert isinstance(parsed.children[0], sym.Power)
    assert isinstance(parsed.children[0].base, sym.Quotient)
    assert isinstance(parsed.children[0].base.numerator, sym.Sum)
    assert isinstance(parsed.children[0].base.denominator, sym.Sum)
    assert isinstance(parsed.children[1], sym.FloatLiteral)
    parsed_vars = FindVariables().visit(parsed)
    assert parsed_vars == ('a', 'b', 'a', 'b')
    assert all(parsed_var.scope == routine for parsed_var in parsed_vars)
    assert to_str(parsed) == '((a+b)/(a-b))**3+3.1415'

    parsed = parse_expr(convert_to_case('call_with_kwargs(a, val=7, end=b)', mode=case))
    assert isinstance(parsed, sym.InlineCall)
    assert parsed.parameters == ('a',)
    assert parsed.kw_parameters == {'val': 7, 'end': 'b'}
    assert to_str(parsed) == 'call_with_kwargs(a,val=7,end=b)'

    parsed = parse_expr(convert_to_case('real(6, kind=jprb)', mode=case))
    assert isinstance(parsed, sym.Cast)
    assert parsed.name.lower() == 'real'
    assert all(isinstance(_parsed, sym.IntLiteral) for _parsed in parsed.parameters)
    assert parsed.kind.name.lower() == 'jprb'
    assert to_str(parsed) == 'real(6)'

    parsed = parse_expr(convert_to_case('2.4', mode=case))
    assert isinstance(parsed, sym.FloatLiteral)
    assert parsed.kind is None
    assert to_str(parsed) == '2.4'

    parsed = parse_expr(convert_to_case('2.4_jprb', mode=case), scope=routine)
    assert isinstance(parsed, sym.FloatLiteral)
    assert parsed.kind == 'jprb'
    assert to_str(parsed) == '2.4_jprb'

    parsed = parse_expr(convert_to_case('2._8', mode=case), scope=routine)
    assert isinstance(parsed, sym.FloatLiteral)
    assert parsed.kind == '8'
    assert float(parsed.value) == 2.0
    assert to_str(parsed) == '2._8'

    parsed = parse_expr(convert_to_case('2.4e18_my_kind8', mode=case), scope=routine)
    assert isinstance(parsed, sym.FloatLiteral)
    assert parsed.kind == 'my_kind8'
    assert float(parsed.value) == 2.4e18
    assert to_str(parsed) == '2.4e18_my_kind8'

    parsed = parse_expr(convert_to_case('4_jpim', mode=case), scope=routine)
    assert isinstance(parsed, sym.IntLiteral)
    assert parsed.kind == 'jpim'
    assert int(parsed.value) == 4
    assert to_str(parsed) == '4'

    parsed = parse_expr(convert_to_case('[1, 2, 3, 4]', mode=case), scope=routine)
    assert isinstance(parsed, sym.LiteralList)
    assert all(isinstance(_parsed, sym.IntLiteral) for _parsed in parsed.elements)
    assert to_str(parsed) == '[1,2,3,4]'
    parsed = parse_expr(convert_to_case('(/ 2, 3, 4, 5 /)', mode=case), scope=routine)
    assert isinstance(parsed, sym.LiteralList)
    assert all(isinstance(_parsed, sym.IntLiteral) for _parsed in parsed.elements)
    assert to_str(parsed) == '[2,3,4,5]'

    parsed = parse_expr(convert_to_case('.TRUE.', mode=case))
    assert isinstance(parsed, sym.LogicLiteral)
    assert parsed.value is True
    assert to_str(parsed) == 'true'

    parsed = parse_expr(convert_to_case('.FALSE.', mode=case))
    assert isinstance(parsed, sym.LogicLiteral)
    assert parsed.value is False
    assert to_str(parsed) == 'false'

    parsed = parse_expr(convert_to_case('.FALSE. .OR. .TRUE. .AND. .TRUE.', mode=case))
    assert to_str(parsed) == 'falseortrueandtrue'


@pytest.mark.parametrize('case', ('upper', 'lower', 'random'))
def test_expression_parser_evaluate(case):

    test_str = '.FALSE. .OR. .TRUE. .AND. .TRUE.'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True)
    assert parsed

    context = {'a': 0.9}
    test_str = 'a .lt. 1'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed

    context = {'a': 0.9}
    test_str = 'a .lt. 1_jprb'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed

    context = {'VAR1': True, 'VAR2': False}
    test_str = 'VAR1 .AND. VAR2 .AND. .TRUE.'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed

    test_str = '(2*3)**2 - 16'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True)
    assert parsed == 20

    test_str = '(2*3)**2 - a'
    context = {'A': 6}
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 30

    test_str = '(2*3)**2 - a'
    context = {'a': 6}
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 30

    context = {'a': 6}
    test_str = 'min(a, 10)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 6
    test_str = '(2*3)**2 - min(a, 10)'
    context = {'a': 6}
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 30
    context = {'a': 6}
    test_str = 'max(a, 10)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 10

    context = {'a': 6, 'b': 2}
    test_str = 'modulo(A, B)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 0

    context = {'x': '4'}
    test_str = 'real(x)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 4.0
    assert isinstance(parsed, sym.FloatLiteral)

    context = {'a': '4.67'}
    test_str = 'int(a)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 4
    assert isinstance(parsed, sym.IntLiteral)

    context = {'x': '-4.145'}
    test_str = 'abs(x)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 4.145

    context = {'x': '9'}
    test_str = 'sqrt(x)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 3

    context = {'x': '0'}
    test_str = 'exp(x)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 1

    context = {'arr': [[1, 2], [3, 4]]}
    test_str = '1 + arr(1, 2)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 3

    context = {'a': 6}
    test_str = '1 + 1 + a + some_func(a, 10)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    with pytest.raises(pmbl_mapper.evaluator.UnknownVariableError):
        parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, strict=True, context=context)
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, strict=False, context=context)
    assert str(parsed).lower().replace(' ', '') == '8+some_func(6,10)'

    def some_func(a, b, c=None):
        if c is None:
            return a + b
        return a + b + c

    context = {'a': 6, 'some_func': some_func}
    test_str = '1 + 1 + a + some_func(a, 10)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 24

    context = {'a': 6, 'some_func': some_func}
    test_str = '1 + 1 + a + some_func(a, 10, c=2)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 26

    context = {'a': 6, 'b': 7}
    test_str = '(a + b + c + 1)/(c + 1)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert str(parsed).lower().replace(' ', '') == '(13+c+1)/(c+1)'

    class BarBarBar:
        val_barbarbar = 5

    class BarBar:
        barbarbar = BarBarBar()
        val_barbar = -3
        def barbar_func(self, a):
            return a - 1

    class Bar:
        barbar = BarBar()
        val_bar = 5
        def bar_func(self, a):
            return a**2

    class Foo:
        bar = Bar() # pylint: disable=disallowed-name
        val3 = 1
        arr = [[1, 2], [3, 4]]
        def __init__(self, _val1, _val2):
            self.val1 = _val1
            self.val2 = _val2
        def some_func(self, a, b):
            return a + b
        @staticmethod
        def static_func(a):
            return 2*a

    context = {'foo': Foo(2, 3)}
    test_str = 'foo%val1 + foo%val2 + foo%val3'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case))
    assert str(parsed).lower().replace(' ', '') == 'foo%val1+foo%val2+foo%val3'
    with pytest.raises(pmbl_mapper.evaluator.UnknownVariableError):
        parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, strict=True)
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 6
    test_str = 'foo%val1 + foo%some_func(1, 2) + foo%static_func_2(3)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert str(parsed).lower().replace(' ', '') == '5+foo%static_func_2(3)'
    with pytest.raises(pmbl_mapper.evaluator.UnknownVariableError):
        parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, strict=True)
    test_str = 'foo%val1 + foo%some_func(1, 2) + foo%static_func(3) + foo%arr(1, 2)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context, strict=True)
    assert parsed == 13
    test_str = 'foo%val1 + foo%some_func(1, b=2) + foo%static_func(a=3) + foo%arr(1, 2)'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context, strict=True)
    assert parsed == 13
    test_str = 'foo%bar%val_bar + 1'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 6
    test_str = 'foo%bar%bar_func(2) + 1'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 5
    test_str = 'foo%bar%barbar%val_barbar + 1'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == -2
    test_str = 'foo%bar%barbar%barbar_func(0) + 1'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 0
    test_str = 'foo%bar%barbar%barbarbar%val_barbarbar + 1'
    parsed = parse_expr(convert_to_case(f'{test_str}', mode=case), evaluate=True, context=context)
    assert parsed == 6
