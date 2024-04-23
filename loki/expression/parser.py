# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from sys import intern
import re
import math
import pytools.lex
import numpy as np
from pymbolic.parser import Parser as ParserBase #  , FinalizedTuple
from pymbolic.mapper import Mapper
import pymbolic.primitives as pmbl
from pymbolic.mapper.evaluator import EvaluationMapper
from pymbolic.parser import (
    _openpar, _closepar, _minus, FinalizedTuple, _PREC_UNARY,
    _PREC_TIMES, _PREC_PLUS, _PREC_CALL, _times, _plus
)
try:
    from fparser.two.Fortran2003 import Intrinsic_Name

    FORTRAN_INTRINSIC_PROCEDURES = Intrinsic_Name.function_names
    """list of intrinsic fortran procedure(s) names"""
except ImportError:
    FORTRAN_INTRINSIC_PROCEDURES = ()

from loki.tools.util import CaseInsensitiveDict
import loki.expression.symbols as sym
import loki.expression.operations as sym_ops
from loki.expression.expr_visitors import AttachScopes
from loki.scope import Scope

__all__ = ['ExpressionParser', 'parse_expr', 'FORTRAN_INTRINSIC_PROCEDURES']


class PymbolicMapper(Mapper):
    """
    Pymbolic expression to Loki expression mapper.

    Convert pymbolic expressions to Loki expressions.
    """
    # pylint: disable=abstract-method,unused-argument

    def map_product(self, expr, *args, **kwargs):
        children = tuple(self.rec(child, *args, **kwargs) for child in expr.children)
        if isinstance(expr, sym_ops.ParenthesisedMul):
            return sym_ops.ParenthesisedMul(children)
        return sym.Product(children)

    def map_sum(self, expr, *args, **kwargs):
        children = tuple(self.rec(child, *args, **kwargs) for child in expr.children)
        if isinstance(expr, sym_ops.ParenthesisedAdd):
            return sym_ops.ParenthesisedAdd(children)
        return sym.Sum(children)

    def map_power(self, expr, *args, **kwargs):
        base=self.rec(expr.base, *args, **kwargs)
        exponent=self.rec(expr.exponent, *args, **kwargs)
        if isinstance(expr, sym_ops.ParenthesisedPow):
            return sym_ops.ParenthesisedPow(base=base, exponent=exponent)
        return sym.Power(base=base, exponent=exponent)

    def map_quotient(self, expr, *args, **kwargs):
        numerator=self.rec(expr.numerator, *args, **kwargs)
        denominator=self.rec(expr.denominator, *args, **kwargs)
        if isinstance(expr, sym_ops.ParenthesisedDiv):
            return sym_ops.ParenthesisedDiv(numerator=numerator, denominator=denominator)
        return sym.Quotient(numerator=numerator, denominator=denominator)

    def map_comparison(self, expr, *args, **kwargs):
        return sym.Comparison(left=self.rec(expr.left, *args, **kwargs),
                operator=expr.operator,
                right=self.rec(expr.right, *args, **kwargs))

    def map_logical_and(self, expr, *args, **kwargs):
        return sym.LogicalAnd(tuple(self.rec(child, *args, **kwargs) for child in expr.children))

    def map_logical_or(self, expr, *args, **kwargs):
        return sym.LogicalOr(tuple(self.rec(child, *args, **kwargs) for child in expr.children))

    def map_logical_not(self, expr, *args, **kwargs):
        return sym.LogicalNot(self.rec(expr.child, *args, **kwargs))

    def map_constant(self, expr, *args, **kwargs):
        if expr == -1:
            return expr
        if isinstance(expr, (sym.FloatLiteral, sym.IntLiteral, sym.StringLiteral, sym.LogicLiteral)):
            if isinstance(expr, sym.IntLiteral) and expr.value < 0:
                return sym.Product((-1, sym.IntLiteral(abs(expr.value))))
            return expr
        if isinstance(expr, bool):
            return sym.LogicLiteral('true' if expr else 'false')
        return sym.Literal(expr)

    map_logic_literal = map_constant
    map_string_literal = map_constant
    map_intrinsic_literal = map_constant

    map_int_literal = map_constant

    map_float_literal = map_int_literal
    map_variable_symbol = map_constant
    map_deferred_type_symbol = map_constant

    def map_meta_symbol(self, expr, *args, **kwargs):
        return sym.Variable(name=str(expr.name))
    map_scalar = map_meta_symbol
    map_array = map_meta_symbol

    def map_slice(self, expr, *args, **kwargs):
        children = tuple(self.rec(child, *args, **kwargs) if child is not None else child for child in expr.children)
        if len(children) == 1 and children[0] is None:
            # this corresponds to ':' (sym.RangeIndex((None, None)))
            children = (None, None)
        return sym.RangeIndex(children)

    map_range = map_slice
    map_range_index = map_slice
    map_loop_range = map_slice

    def map_variable(self, expr, *args, **kwargs):
        parent = kwargs.pop('parent', None)
        return sym.Variable(name=expr.name, parent=parent)

    def map_algebraic_leaf(self, expr, *args, **kwargs):
        if str(expr).isnumeric():
            return self.map_constant(expr)
        if isinstance(expr, pmbl.Call):
            if expr.function.name.lower() in ('real', 'int'):
                return sym.Cast(expr.function.name, [self.rec(param, *args, **kwargs) for param in expr.parameter][0])
            if expr.function.name.upper() in FORTRAN_INTRINSIC_PROCEDURES:
                return sym.InlineCall(function=sym.Variable(name=expr.function.name),
                        parameters=tuple(self.rec(param, *args, **kwargs) for param in expr.parameters))
            parent = kwargs.pop('parent', None)
            dimensions = tuple(self.rec(param, *args, **kwargs) for param in expr.parameters)
            if not dimensions:
                return sym.InlineCall(function=sym.Variable(name=expr.function.name, parent=parent),
                        parameters=dimensions)
            return sym.Variable(name=expr.function.name, parent=parent,
                    dimensions=tuple(self.rec(param, *args, **kwargs) for param in expr.parameters))
        try:
            return self.map_variable(expr, *args, **kwargs)
        except Exception as e:
            print(f"Exception: {e}")
            return expr

    def map_call_with_kwargs(self, expr, *args, **kwargs):
        name = sym.Variable(name=expr.function.name)
        parameters = tuple(self.rec(param, *args, **kwargs) for param in expr.parameters)
        kw_parameters = {key: self.rec(value, *args, **kwargs) for key, value\
                in CaseInsensitiveDict(expr.kw_parameters).items()}
        if expr.function.name.lower() in ('real', 'int'):
            return sym.Cast(name, parameters, kind=kw_parameters['kind'])

        return sym.InlineCall(function=name, parameters=parameters, kw_parameters=kw_parameters)

    def map_tuple(self, expr, *args, **kwargs):
        return tuple(self.rec(elem, *args, **kwargs) for elem in expr)

    def map_list(self, expr, *args, **kwargs):
        return sym.LiteralList([self.rec(elem, *args, **kwargs) for elem in expr])

    def map_remainder(self, expr, *args, **kwargs):
        # this should never happen as '%' is overwritten to represent derived types
        raise NotImplementedError

    def map_lookup(self, expr, *args, **kwargs):
        # construct derived type(s) variables
        parent = kwargs.pop('parent', None)
        parent = self.rec(expr.aggregate, parent=parent)
        return self.rec(expr.name, parent=parent)


class LokiEvaluationMapper(EvaluationMapper):
    """
    A mapper for evaluating expressions, based on
    :any:`pymbolic.mapper.evaluator.EvaluationMapper`.
    
    Parameters
    ----------
    strict : bool
        Raise exception for unknown symbols/expressions (default: `False`).
    """

    @staticmethod
    def case_insensitive_getattr(obj, attr):
        """
        Case-insensitive version of `getattr`.
        """
        for elem in dir(obj):
            if elem.lower() == attr.lower():
                return getattr(obj, elem)
        return getattr(obj, attr)

    def __init__(self, strict=False, **kwargs):
        self.strict = strict
        super().__init__(**kwargs)

    def map_logic_literal(self, expr):
        return expr.value

    def map_float_literal(self, expr):
        return expr.value
    map_int_literal = map_float_literal

    def map_variable(self, expr):
        if expr.name.upper() in FORTRAN_INTRINSIC_PROCEDURES:
            return self.map_call(expr)
        if self.strict:
            return super().map_variable(expr)
        if expr.name in self.context:
            return super().map_variable(expr)
        return expr

    @staticmethod
    def _evaluate_array(arr, dims):
        """
        Evaluate arrays by converting to numpy array and
        adapting the dimensions corresponding to the different
        starting index.
        """
        return np.array(arr, order='F').item(*[dim-1 for dim in dims])

    def map_call(self, expr):
        if expr.function.name.lower() == 'min':
            return min(self.rec(par) for par in expr.parameters)
        if expr.function.name.lower() == 'max':
            return max(self.rec(par) for par in expr.parameters)
        if expr.function.name.lower() == 'modulo':
            args = [self.rec(par) for par in expr.parameters]
            return args[0]%args[1]
        if expr.function.name.lower() == 'abs':
            return abs(float([self.rec(par) for par in expr.parameters][0]))
        if expr.function.name.lower() == 'int':
            return int(float([self.rec(par) for par in expr.parameters][0]))
        if expr.function.name.lower() == 'real':
            return float([self.rec(par) for par in expr.parameters][0])
        if expr.function.name.lower() == 'sqrt':
            return math.sqrt(float([self.rec(par) for par in expr.parameters][0]))
        if expr.function.name.lower() == 'exp':
            return math.exp(float([self.rec(par) for par in expr.parameters][0]))
        if expr.function.name in self.context and not callable(self.context[expr.function.name]):
            return self._evaluate_array(self.context[expr.function.name],
                    [self.rec(par) for par in expr.parameters])
        return super().map_call(expr)

    def map_call_with_kwargs(self, expr):
        args = [self.rec(par) for par in expr.parameters]
        kwargs = {
                k: self.rec(v)
                for k, v in expr.kw_parameters.items()}
        kwargs = CaseInsensitiveDict(kwargs)
        return self.rec(expr.function)(*args, **kwargs)

    def map_lookup(self, expr):

        def rec_lookup(expr, obj, name):
            return expr.name, self.case_insensitive_getattr(obj, name)

        try:
            current_expr = expr
            obj = self.rec(expr.aggregate)
            while isinstance(current_expr.name, pmbl.Lookup):
                current_expr, obj = rec_lookup(current_expr, obj, current_expr.name.aggregate.name)
            if isinstance(current_expr.name, pmbl.Variable):
                _, obj = rec_lookup(current_expr, obj, current_expr.name.name)
                return obj
            if isinstance(current_expr.name, pmbl.Call):
                name = current_expr.name.function.name
                _, obj = rec_lookup(current_expr, obj, name)
                if callable(obj):
                    return obj(*[self.rec(par) for par in current_expr.name.parameters])
                return self._evaluate_array(obj, [self.rec(par) for par in current_expr.name.parameters])
            if isinstance(current_expr.name, pmbl.CallWithKwargs):
                name = current_expr.name.function.name
                _, obj = rec_lookup(current_expr, obj, name)
                args = [self.rec(par) for par in current_expr.name.parameters]
                kwargs = CaseInsensitiveDict(
                    (k, self.rec(v))
                    for k, v in current_expr.name.kw_parameters.items()
                )
                return obj(*args, **kwargs)
        except Exception as e:
            if self.strict:
                raise e
            return expr
        if self.strict:
            raise NotImplementedError
        return expr


class ExpressionParser(ParserBase):
    """
    String Parser based on :any:`pymbolic.parser.Parser` for
    parsing expressions from strings.

    The Loki String Parser utilises and extends pymbolic's parser to incorporate
    Fortran specific syntax and to map pymbolic expressions to Loki expressions, utilising 
    the mapper :any:`PymbolicMapper`.

    **Further**, in order to ensure correct ordering of Fortran Statements as documented
    in `'WD 1539-1 J3/23-007r1 (Draft Fortran 2023)' <https://j3-fortran.org/doc/year/10/10-007.pdf#page=155>`_,
    pymbolic's parsing logic needed to be slightly adapted.

    Pymbolic references:

    * `GitHub: pymbolic <https://github.com/inducer/pymbolic>`_
    * `pymbolic/parser.py <https://github.com/inducer/pymbolic/blob/main/pymbolic/parser.py>`_
    * `pymbolic's parser documentation <https://documen.tician.de/pymbolic/utilities.html>`_

    .. note::
       **Example:**
        Using the expression parser and possibly evaluate them

        .. code-block::

            >>> from loki import parse_expr
            >>> # parse numerical expressions
            >>> ir = parse_expr('3 + 2**4')
            >>> ir
            Sum((IntLiteral(3, None), Power(IntLiteral(2, None), IntLiteral(4, None))))
            >>> # or expressions with variables
            >>> ir = parse_expr('a*b')
            >>> ir
            Product((DeferredTypeSymbol('a', None, None, <SymbolAttributes BasicType.DEFERRED>),\
                    DeferredTypeSymbol('b', None, None, <SymbolAttributes BasicType.DEFERRED>)))
            >>> # and provide a scope e.g, with some routine defining a and b as 'real's
            >>> ir = parse_expr('a*b', scope=routine)
            >>> ir
            Product((Scalar('a', None, None, None), Scalar('b', None, None, None)))
            >>> # further, it is possible to evaluate expressions
            >>> ir = parse_expr('a*b + 1', evaluate=True, context={'a': 2, 'b': 3})
            >>> ir
            >>> IntLiteral(7, None)
            >>> # even with functions implemented in Python
            >>> def add(a, b):
            >>>     return a + b
            >>> ir = parse_expr('a + add(a, b)', evaluate=True, context={'a': 2, 'b': 3, 'add': add})
            >>> ir
            >>> IntLiteral(7, None)

    .. automethod:: __call__
    """

    _f_true = intern("f_true")
    _f_false = intern("f_false")
    _f_lessequal = intern('_f_lessequal')
    _f_less = intern('_f_less')
    _f_greaterequal = intern('_f_greaterequal')
    _f_greater = intern('_f_greater')
    _f_equal = intern('_f_equal')
    _f_notequal = intern('_f_notequal')
    _f_and = intern("and")
    _f_or = intern("or")
    _f_not = intern("not")
    _f_float = intern("f_float")
    _f_int = intern("f_int")
    _f_string = intern("f_string")
    _f_openbracket = intern("openbracket")
    _f_closebracket = intern("closebracket")
    _f_derived_type = intern("dot")

    lex_table = [
            (_f_true, pytools.lex.RE(r"\.true\.", re.IGNORECASE)),
            (_f_false, pytools.lex.RE(r"\.false\.", re.IGNORECASE)),
            (_f_lessequal, pytools.lex.RE(r"\.le\.", re.IGNORECASE)),
            (_f_less, pytools.lex.RE(r"\.lt\.", re.IGNORECASE)),
            (_f_greaterequal, pytools.lex.RE(r"\.ge\.", re.IGNORECASE)),
            (_f_greater, pytools.lex.RE(r"\.gt\.", re.IGNORECASE)),
            (_f_equal, pytools.lex.RE(r"\.eq\.", re.IGNORECASE)),
            (_f_notequal, pytools.lex.RE(r"\.ne\.", re.IGNORECASE)),
            (_f_and, pytools.lex.RE(r"\.and\.", re.IGNORECASE)),
            (_f_or, pytools.lex.RE(r"\.or\.", re.IGNORECASE)),
            (_f_not, pytools.lex.RE(r"\.not\.", re.IGNORECASE)),
            (_f_float, ("|", pytools.lex.RE(r"[0-9]+\.[0-9]*([eEdD][+-]?[0-9]+)?(_([\w$]+|[0-9]+))+$", re.IGNORECASE))),
            (_f_int, pytools.lex.RE(r"[0-9]+?(_[a-zA-Z]*)", re.IGNORECASE)),
            (_f_string, ("|", pytools.lex.RE(r'\".*\"', re.IGNORECASE),
                pytools.lex.RE(r"\'.*\'", re.IGNORECASE))),
            (_f_openbracket, pytools.lex.RE(r"\(/")),
            (_f_closebracket, pytools.lex.RE(r"/\)")),
            (_f_derived_type, pytools.lex.RE(r"\%")),
            ] + ParserBase.lex_table
    """
    Extend :any:`pymbolic.parser.Parser.lex_table` to accomodate for Fortran specifix syntax/expressions.
    """

    ParserBase._COMP_TABLE.update({
         _f_lessequal: "<=",
         _f_less: "<",
         _f_greaterequal: ">=",
         _f_greater: ">",
         _f_equal: "==",
         _f_notequal: "!="
         })

    @staticmethod
    def _parenthesise(expr):
        """
        Utility method to parenthesise specific expressions.

        E.g., from :any:`pymbolic.primitives.Sum` to 
        :any:`ParenthesisedAdd`.
        """
        if isinstance(expr, pmbl.Sum):
            return sym_ops.ParenthesisedAdd(expr.children)
        if isinstance(expr, pmbl.Product):
            return sym_ops.ParenthesisedMul(expr.children)
        if isinstance(expr, pmbl.Quotient):
            return sym_ops.ParenthesisedDiv(numerator=expr.numerator,
                    denominator=expr.denominator)
        if isinstance(expr, pmbl.Power):
            return sym_ops.ParenthesisedPow(base=expr.base, exponent=expr.exponent)
        return expr

    def parse_prefix(self, pstate):
        pstate.expect_not_end()

        if pstate.is_next(_minus):
            pstate.advance()
            left_exp = pmbl.Product((-1, self.parse_expression(pstate, _PREC_UNARY)))
            return left_exp
        if pstate.is_next(_openpar):
            pstate.advance()

            if pstate.is_next(_closepar):
                left_exp = ()
            else:
                # This is parsing expressions separated by commas, so it
                # will return a tuple. Kind of the lazy way out.
                left_exp = self.parse_expression(pstate)
                # NECESSARY to ensure correct ordering!
                left_exp = self._parenthesise(left_exp)
            pstate.expect(_closepar)
            pstate.advance()
            if isinstance(left_exp, tuple):
                # These could just be plain parentheses.

                # Finalization prevents things from being appended
                # to containers after their closing delimiter.
                left_exp = FinalizedTuple(left_exp)
            return left_exp
        return super().parse_prefix(pstate)

    def parse_postfix(self, pstate, min_precedence, left_exp):

        did_something = False
        if pstate.is_next(self._f_derived_type) and _PREC_CALL > min_precedence:
            pstate.advance()
            right_exp = self.parse_expression(pstate, _PREC_PLUS)
            left_exp = pmbl.Lookup(left_exp, right_exp)
            did_something = True
        elif pstate.is_next(_times) and _PREC_TIMES > min_precedence:
            pstate.advance()
            right_exp = self.parse_expression(pstate, _PREC_PLUS)
            # NECESSARY to ensure correct ordering!
            # pylint: disable=unidiomatic-typecheck
            if type(right_exp) is pmbl.Quotient:
                left_exp = pmbl.Quotient(numerator=pmbl.Product((left_exp, right_exp.numerator)),
                        denominator=right_exp.denominator)
            # pylint: disable=unidiomatic-typecheck
            elif type(right_exp) is pmbl.Product:
                left_exp = pmbl.Product((sym.Product((left_exp, right_exp.children[0])), right_exp.children[1]))
            else:
                left_exp = pmbl.Product((left_exp, right_exp))
            did_something = True
        elif pstate.is_next(_plus) and _PREC_PLUS > min_precedence:
            pstate.advance()
            right_exp = self.parse_expression(pstate, _PREC_PLUS)
            left_exp = pmbl.Sum((left_exp, right_exp))
            did_something = True
        elif pstate.is_next(_minus) and _PREC_PLUS > min_precedence:
            pstate.advance()
            right_exp = self.parse_expression(pstate, _PREC_PLUS)
            right_exp = pmbl.Product((-1, right_exp))
            left_exp = pmbl.Sum((left_exp, right_exp))
            did_something = True
        else:
            return super().parse_postfix(pstate, min_precedence, left_exp)
        return left_exp, did_something

    def parse_terminal(self, pstate):
        if pstate.is_next(self._f_float):
            return self.parse_f_float(pstate.next_str_and_advance())
        if pstate.is_next(self._f_int):
            return self.parse_f_int(pstate.next_str_and_advance())
        if pstate.is_next(self._f_string):
            return self.parse_f_string(pstate.next_str_and_advance())
        if pstate.is_next(self._f_true):
            assert pstate.next_str_and_advance().lower() == ".true."
            return sym.LogicLiteral('.TRUE.')
        if pstate.is_next(self._f_false):
            assert pstate.next_str_and_advance().lower() == ".false."
            return sym.LogicLiteral('.FALSE.')
        return super().parse_terminal(pstate)

    def __call__(self, expr_str, scope=None, evaluate=False, strict=False, context=None):
        """
        Call Loki String Parser to convert expression(s) represented in a string to Loki expression(s)/IR.

        Parameters
        ----------
        expr_str : str
            The expression as a string
        scope : :any:`Scope`
            The scope to which symbol names inside the expression belong
        evaluate : bool, optional
            Whether to evaluate the expression or not (default: `False`)
        strict : bool, optional
            Whether to raise exception for unknown variables/symbols when
            evaluating an expression (default: `False`)
        context : dict, optional
            Symbol context, defining variables/symbols/procedures to help/support
            evaluating an expression

        Returns
        -------
        :any:`Expression`
            The expression tree corresponding to the expression
        """
        result = super().__call__(expr_str)
        context = context or {}
        context = CaseInsensitiveDict(context)
        if evaluate:
            result = LokiEvaluationMapper(context=context, strict=strict)(result)
        ir = PymbolicMapper()(result)
        return AttachScopes().visit(ir, scope=scope or Scope())

    def parse_float(self, s):
        """
        Parse float literals.

        Do not cast to float via 'float()' in order to keep the original
        notation, e.g., do not convert 1E-3 to 0.003.
        """
        return sym.FloatLiteral(value=s.replace("d", "e").replace("D", "e"))

    def parse_f_float(self, s):
        """
        Parse "Fortran-style" float literals.

        E.g., ``3.1415_my_real_kind``.
        """
        stripped = s.split('_', 1)
        if len(stripped) == 2:
            return sym.FloatLiteral(value=self.parse_float(stripped[0]), kind=sym.Variable(name=stripped[1].lower()))
        return self.parse_float(stripped[0])

    def parse_f_int(self, s):
        """
        Parse "Fortran-style" int literals.

        E.g., ``1_my_int_kind``.
        """
        stripped = s.split('_', 1)
        value = int(stripped[0].replace("d", "e").replace("D", "e"))
        return sym.IntLiteral(value=value, kind=sym.Variable(name=stripped[1].lower()))

    def parse_f_string(self, s):
        """
        Parse string literals.
        """
        return sym.StringLiteral(s)


parse_expr = ExpressionParser()
"""
An instance of :any:`ExpressionParser` that allows parsing expression strings into a Loki expression tree.
See :any:`ExpressionParser.__call__` for a description of the available arguments.
"""
