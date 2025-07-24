# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
import numpy as np
from pymbolic.mapper.evaluator import EvaluationMapper
try:
    from fparser.two.Fortran2003 import Intrinsic_Name

    FORTRAN_INTRINSIC_PROCEDURES = Intrinsic_Name.function_names
    """list of intrinsic fortran procedure(s) names"""
except ImportError:
    FORTRAN_INTRINSIC_PROCEDURES = ()

from loki.expression import symbols as sym
from loki.tools.util import CaseInsensitiveDict, as_tuple

__all__ = ['LokiEvaluationMapper', 'eval_expr']

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

    def map_comparison(self, expr):
        import operator # pylint: disable=import-outside-toplevel
        left = self.rec(expr.left)
        right = self.rec(expr.right)
        rel_types = (sym._Literal, float, int)
        if isinstance(left, rel_types) and isinstance(right, rel_types):
            return getattr(operator, expr.operator_to_name[expr.operator])(
                self.rec(expr.left), self.rec(expr.right))
        return sym.Comparison(left=left, operator=expr.operator, right=right)

    def map_logical_and(self, expr):
        children = [self.rec(ch) for ch in expr.children]
        if not all(isinstance(ch, bool) for ch in children):
            new_children = [ch for ch in children if not isinstance(ch, bool) and ch]
            return sym.LogicalAnd(as_tuple(new_children))
        return all(children)

    def map_logic_literal(self, expr):
        return expr.value

    def map_float_literal(self, expr):
        return expr.value
    map_int_literal = map_float_literal

    def map_variable(self, expr):
        _, obj = self._recurse_parent(expr)
        if obj is not None:
            try:
                return self.case_insensitive_getattr(obj, expr.name.split('%')[-1])
            except: # pylint: disable=bare-except
                return expr
        if expr.name.upper() in FORTRAN_INTRINSIC_PROCEDURES:
            return self.map_call(expr)
        if self.strict:
            return super().map_variable(expr)
        if expr.name in self.context:
            return super().map_variable(expr)
        return expr

    def _recurse_parent(self, expr):
        current_expr = expr
        while hasattr(current_expr, 'parent') and current_expr.parent is not None:
            current_expr = current_expr.parent
            obj = self.rec(current_expr)
            return current_expr, obj
        return expr, None

    @staticmethod
    def _evaluate_array(arr, dims):
        """
        Evaluate arrays by converting to numpy array and
        adapting the dimensions corresponding to the different
        starting index.
        """
        return np.array(arr, order='F').item(*[dim-1 for dim in dims])

    def map_array(self, expr):
        new_dims = as_tuple(self.rec(dim) for dim in expr.dimensions)
        return self.map_call(expr.clone(dimensions=new_dims), name=expr.name.lower(), parameters=new_dims)

    def map_call(self, expr, name=None, parameters=None):
        _, obj = self._recurse_parent(expr)
        if obj is not None:
            try:
                _call = self.case_insensitive_getattr(obj, expr.name.split('%')[-1])
                if callable(_call):
                    return _call(*[self.rec(par) for par in expr.dimensions])
                try:
                    return self._evaluate_array(_call,
                            [self.rec(par) for par in expr.dimensions])
                except: # pylint: disable=bare-except
                    pass
                return expr
            except Exception as e:
                if self.strict:
                    raise e
                return expr
        call_name = name or expr.function.name.lower()
        expr_parameters = parameters or expr.parameters
        if call_name == 'min':
            return min(self.rec(par) for par in expr_parameters)
        if call_name == 'max':
            return max(self.rec(par) for par in expr_parameters)
        if call_name == 'modulo':
            args = [self.rec(par) for par in expr_parameters]
            return args[0]%args[1]
        if call_name == 'abs':
            return abs(float([self.rec(par) for par in expr_parameters][0]))
        if call_name == 'int':
            return int(float([self.rec(par) for par in expr_parameters][0]))
        if call_name == 'real':
            return float([self.rec(par) for par in expr_parameters][0])
        if call_name == 'sqrt':
            return math.sqrt(float([self.rec(par) for par in expr_parameters][0]))
        if call_name == 'exp':
            return math.exp(float([self.rec(par) for par in expr_parameters][0]))
        if call_name in self.context:  # and not callable(self.context[call_name]):
            if not callable(self.context[call_name]):
                return self._evaluate_array(self.context[call_name],
                        [self.rec(par) for par in expr_parameters])
            kwargs = CaseInsensitiveDict(expr.kw_parameters) if hasattr(expr, 'kw_parameters') else {}
            return self.rec(self.context[call_name](*[self.rec(par) for par in expr_parameters], **kwargs))
        try:
            return super().map_call(expr)
        except: # pylint: disable=bare-except
            return expr

    def map_inline_call(self, expr):
        _, obj = self._recurse_parent(expr.function)
        if obj is not None:
            try:
                kwargs = {
                    k: self.rec(v)
                    for k, v in expr.kw_parameters.items()}
                return self.case_insensitive_getattr(obj,
                        expr.name.split('%')[-1])(*[self.rec(par) for par in expr.parameters], **kwargs)
            except: # pylint: disable=bare-except
                return expr
        return self.map_call(expr, name=expr.name.lower(),
                parameters=as_tuple(self.rec(param) for param in expr.parameters))

    def map_call_with_kwargs(self, expr):
        args = [self.rec(par) for par in expr.parameters]
        kwargs = {
                k: self.rec(v)
                for k, v in expr.kw_parameters.items()}
        kwargs = CaseInsensitiveDict(kwargs)
        return self.rec(expr.function)(*args, **kwargs)


def eval_expr(expr, context=None, strict=False):
    """
    Call Loki Evaluation Mapper to evaluate expression(s).

    Parameters
    ----------
    expr : :any:`Expression`
        The expression as a string
    strict : bool, optional
        Whether to raise exception for unknown variables/symbols when
        evaluating an expression (default: `False`)
    context : dict, optional
        Symbol context, defining variables/symbols/procedures to help/support
        evaluating an expression

    Returns
    -------
    :any:`Expression`
        The evaluated expression tree corresponding to the expression
    """
    context = context or {}
    context = CaseInsensitiveDict(context)
    mapper = LokiEvaluationMapper(context=context, strict=strict)
    return mapper(expr)
