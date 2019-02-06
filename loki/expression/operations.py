"""
Sub-classes of SymPy's native operations that allow us to express
niche things like mathemtaically irrelevant parenthesis that
nevertheless change code results.
"""

from sympy.core import Add, Mul, Pow
from sympy.core.cache import cacheit
from sympy.logic.boolalg import as_Boolean
from sympy.core.basic import as_Basic
from sympy.core.numbers import One, Zero


class ParenthesisedAdd(Add):
    """
    Specialised version of :class:`Add` that always pretty-prints and
    code-generates with explicit parentheses.
    """
    def _sympyrepr(self, printer=None):
        return '(%s)' % printer._print_Add(self)

    _sympystr = _sympyrepr
    _fcode = _sympyrepr


class ParenthesisedMul(Mul):
    """
    Specialised version of :class:`Mul` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    def _sympyrepr(self, printer=None):
        return '(%s)' % printer._print_Mul(self)

    _sympystr = _sympyrepr
    _fcode = _sympyrepr


class ParenthesisedPow(Pow):
    """
    Specialised version of :class:`Pow` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    def _sympyrepr(self, printer=None):
        return '(%s)' % printer._print_Pow(self)

    _sympystr = _sympyrepr
    _fcode = _sympyrepr
