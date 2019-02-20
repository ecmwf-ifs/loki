"""
Sub-classes of SymPy's native operations that allow us to express
niche things like mathemtaically irrelevant parenthesis that
nevertheless change code results.
"""

from sympy.core import Add, Mul, Pow
from sympy.core.singleton import S


class NonCommutativeAdd(Add):
    """
    Special-casing of :class:`sympy.Add` that honours commutativity flags
    in terms in ``.args``.

    Note, this is required to create a strict defensive mode, where term
    ordering from parser frontends is honoured rigorously to avoid round-off
    errors in "parse-unparse" tests.
    """

    def args_cnc(self, cset=False, warn=True, split_1=True):
        """
        Argument canonicalization that honours commutativity by emulating
        ``sympy.Mul.args_cnc()``.
        """
        args = list(self.args)

        for i, mi in enumerate(args):
            if not mi.is_commutative:
                c = args[:i]
                nc = args[i:]
                break
        else:
            c = args
            nc = []

        if c and split_1 and (
            c[0].is_Number and
            c[0].is_negative and
                c[0] is not S.NegativeOne):
            c[:1] = [S.NegativeOne, -c[0]]

        if cset:
            clen = len(c)
            c = set(c)
            if clen and warn and len(c) != clen:
                raise ValueError('repeated commutative arguments: %s' %
                                 [ci for ci in c if list(self.args).count(ci) > 1])
        return [c, nc]

    def as_ordered_terms(self, order=None):
        """
        Term ordering that honours commutativity by emulating
        ``sympy.Mul.as_ordered_factors()``.
        """
        cpart, ncpart = self.args_cnc()
        cpart.sort(key=lambda expr: expr.sort_key(order=order))
        return cpart + ncpart


class ParenthesisedAdd(NonCommutativeAdd):
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
