# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utilities to automatically remove code elements or
section and to perform Dead Code Elimination.
"""

from loki.expression.symbolic import simplify
from loki.tools import flatten, as_tuple
from loki.ir import Conditional, Transformer, Comment
from loki.pragma_utils import is_loki_pragma, pragma_regions_attached


__all__ = [
    'remove_dead_code', 'RemoveDeadCodeTransformer',
    'remove_marked_regions', 'RemoveRegionTransformer',
    'remove_calls', 'RemoveCallsTransformer'
]


def remove_dead_code(routine, use_simplify=True):
    """
    Perform Dead Code Elimination on the given :any:`Subroutine` object.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine to which to apply dead code elimination.
    simplify : boolean
        Use :any:`simplify` when evaluating expressions for branch pruning.
    """

    transformer = RemoveDeadCodeTransformer(use_simplify=use_simplify)
    routine.body = transformer.visit(routine.body)


class RemoveDeadCodeTransformer(Transformer):
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

        has_elseif = o.has_elseif and else_body and isinstance(else_body[0], Conditional)
        return self._rebuild(o, tuple((condition,) + (body,) + (else_body,)), has_elseif=has_elseif)


def remove_marked_regions(routine, mark_with_comment=True):
    """
    Utility routine to remove code regions marked with
    ``!$loki remove`` pragmas from a subroutine's body.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine to which to apply dead code elimination.
    mark_with_comment : boolean
        Flag to trigger the insertion of a marker comment when
        removing a region; default: ``True``.
    """

    transformer = RemoveRegionTransformer(
        mark_with_comment=mark_with_comment
    )

    with pragma_regions_attached(routine):
        routine.body = transformer.visit(routine.body)


class RemoveRegionTransformer(Transformer):
    """
    A :any:`Transformer` that removes code regions marked with
    ``!$loki remove`` pragmas.

    This :any:`Transformer` only removes :any:`PragmaRegion` nodes,
    and thus requires the IR tree to have pragma regions attached, for
    example via :method:`pragma_regions_attached`.

    When removing a marked code region the transformer may leave a
    comment in the source to mark the previous location, or remove the
    code region entirely.

    Parameters
    ----------
    mark_with_comment : boolean
        Flag to trigger the insertion of a marker comment when
        removing a region; default: ``True``.
    """

    def __init__(self, mark_with_comment=True, **kwargs):
        super().__init__(**kwargs)
        self.mark_with_comment = mark_with_comment

    def visit_PragmaRegion(self, o, **kwargs):
        """ Remove :any:`PragmaRegion` nodes with ``!$loki remove`` pragmas """

        if is_loki_pragma(o.pragma, starts_with='remove'):
            # Leave a comment to mark the removed region in source
            if self.mark_with_comment:
                return Comment(text='![Loki] Removed content of pragma-marked region!')

            return None

        # Recurse into the pragama region and rebuild
        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)


def remove_calls(
        routine, call_names=None, intrinsic_names=None,
        import_names=None
):
    """
    Utility routine to remove all :any:`CallStatement` nodes
    to specific named subroutines in a :any:`Subroutine`.

    For more information, see :any:`RemoveCallsTransformer`.

    Parameters
    ----------
    call_names : list of str
        List of subroutine names against which to match
        :any:`CallStatement` nodes.
    intrinsic_names : list of str
        List of module names against which to match :any:`Intrinsic`
        nodes.
    import_names : list of str
        List of module names against which to match :any:`Import`
        nodes.
    """

    transformer = RemoveCallsTransformer(
        call_names=call_names, intrinsic_names=intrinsic_names,
        import_names=import_names
    )
    routine.spec = transformer.visit(routine.spec)
    routine.body = transformer.visit(routine.body)


class RemoveCallsTransformer(Transformer):
    """
    A :any:`Transformer` that removes all :any:`CallStatement` nodes
    to specific named subroutines.

    This :any:`Transformer` will by default also remove the enclosing
    inline-conditional when encountering calls of the form ```if
    (flag) call named_procedure()``.`

    This :any:`Transformer` will also attempt to match and remove
    :any:`Intrinsic` nodes against a given list of name strings.  This
    allows removing intrinsic calls like ``write (*,*) "..."``.

    In addition, this :any:`Transformer` can also attempt to match and
    remove :any:`Import` nodes if given a list of strings to
    match. This can be used to remove the associated imports of the
    removed subroutines.

    Parameters
    ----------
    call_names : list of str
        List of subroutine names against which to match
        :any:`CallStatement` nodes.
    intrinsic_names : list of str
        List of module names against which to match :any:`Intrinsic`
        nodes.
    import_names : list of str
        List of module names against which to match :any:`Import`
        nodes.
    """

    def __init__(
            self, call_names=None, intrinsic_names=None,
            import_names=None, **kwargs
    ):
        super().__init__(**kwargs)

        self.call_names = as_tuple(call_names)
        self.intrinsic_names = as_tuple(intrinsic_names)
        self.import_names = as_tuple(import_names)

    def visit_CallStatement(self, o, **kwargs):
        """ Match and remove :any:`CallStatement` nodes against name patterns """
        if o.name in self.call_names:
            return None

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)

    def visit_Conditional(self, o, **kwargs):
        """ Remove inline-conditionals after recursing into their body """

        # First, recurse into condition and bodies
        cond, body, else_body = tuple(self.visit(i, **kwargs) for i in o.children)

        # Capture and remove newly empty inline conditionals
        if o.inline and len(body) == 0:
            return None

        return self._rebuild(o, (cond, body, else_body))

    def visit_Intrinsic(self, o, **kwargs):
        """ Match and remove :any:`Intrinsic` nodes against name patterns """
        if self.intrinsic_names:
            if any(str(c).lower() in o.text.lower() for c in self.intrinsic_names):
                return None

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)

    def visit_Import(self, o, **kwargs):
        """ Match and remove :any:`Import` nodes against name patterns """
        if self.import_names:
            if any(str(c).lower() in o.module.lower() for c in self.import_names):
                return None

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)
