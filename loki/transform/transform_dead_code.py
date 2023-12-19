# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utilities to perform Dead Code Elimination.
"""
from loki.visitors import Transformer
from loki.expression.symbolic import simplify
from loki.tools import flatten, as_tuple
from loki.ir import Conditional


__all__ = ['dead_code_elimination', 'DeadCodeEliminationTransformer']


def dead_code_elimination(routine, use_simplify=True):
    """
    Perform Dead Code Elimination on the given :any:`Subroutine` object.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine to which to apply dead code elimination.
    simplify : boolean
        Use :any:`simplify` when evaluating expressions for branch pruning.
    """

    transformer = DeadCodeEliminationTransformer(use_simplify=use_simplify)
    routine.body = transformer.visit(routine.body)


class DeadCodeEliminationTransformer(Transformer):
    """
    :any:`Transformer` class that removes provably unreachable code paths.

    The pirmary modification performed is to prune individual code branches
    under :any:`Conditional` nodes.

    Parameters
    ----------
    simplify : boolean
        Use :any:`simplify` when evaluating expressions for branch pruning.
    """

    def __init__(self, use_simplify=True, **kwargs):
        super().__init__(**kwargs)
        self.use_simplify = use_simplify

    def visit_Conditional(self, o, **kwargs):
        condition = self.visit(o.condition, **kwargs)
        body = as_tuple(flatten(as_tuple(self.visit(o.body, **kwargs))))
        else_body = as_tuple(flatten(as_tuple(self.visit(o.else_body, **kwargs))))

        if self.use_simplify:
            condition = simplify(condition)

        if condition == 'True':
            return body

        if condition == 'False':
            return else_body

        has_elseif = isinstance(else_body, Conditional)
        return self._rebuild(o, tuple((condition,) + (body,) + (else_body,)), has_elseif=has_elseif)
