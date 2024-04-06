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
from loki.transform.transformation import Transformation


__all__ = [
    'RemoveCodeTransformation',
    'do_remove_dead_code', 'RemoveDeadCodeTransformer',
    'do_remove_marked_regions', 'RemoveRegionTransformer',
    'do_remove_calls', 'RemoveCallsTransformer'
]


class RemoveCodeTransformation(Transformation):
    """
    A :any:`Transformation` that provides named call and import
    removal, code removal of pragma-marked regions and Dead Code
    Elimination for batch processing vis the :any:`Scheduler`.

    The transformation will apply the following methods in order:
    * :method:`do_remove_calls`
    * :method:`do_remove_marked_regions`
    * :method:`do_remove_dead_code`

    Parameters
    ----------
    remove_marked_regions : boolean
        Flag to trigger the use of :method:`remove_marked_regions`;
        default: ``True``
    mark_with_comment : boolean
        Flag to trigger the insertion of a marker comment when
        removing a region; default: ``True``.
    remove_dead_code : boolean
        Flag to trigger the use of :method:`remove_dead_code`;
        default: ``False``
    use_simplify : boolean
        Use :any:`simplify` when branch pruning in during
        :method:`remove_dead_code`.
    call_names : list of str
        List of subroutine names against which to match
        :any:`CallStatement` nodes during :method:`remove_calls`.
    import_names : list of str
        List of module names against which to match :any:`Import`
        nodes during :method:`remove_calls`.
    intrinsic_names : list of str
        List of module names against which to match :any:`Intrinsic`
        nodes during :method:`remove_calls`.
    kernel_only : boolean
        Only apply the configured removal to subroutines marked as
        "kernel"; default: ``False``
    """

    # Recurse to subroutines in ``contains`` clause
    recurse_to_internal_procedures = True

    def __init__(
            self, remove_marked_regions=True, mark_with_comment=True,
            remove_dead_code=False, use_simplify=True,
            call_names=None, import_names=None,
            intrinsic_names=None, kernel_only=False
    ):
        self.remove_marked_regions = remove_marked_regions
        self.mark_with_comment = mark_with_comment

        self.remove_dead_code = remove_dead_code
        self.use_simplify = use_simplify

        self.call_names = as_tuple(call_names)
        self.import_names = as_tuple(import_names)
        self.intrinsic_names = as_tuple(intrinsic_names)

        self.kernel_only = kernel_only

    def transform_subroutine(self, routine, **kwargs):

        if self.kernel_only and not kwargs.get('role') == 'kernel':
            return

        # Apply named node removal to strip specific calls
        if self.call_names or self.intrinsic_names:
            do_remove_calls(
                routine, call_names=self.call_names,
                import_names=self.import_names,
                intrinsic_names=self.intrinsic_names
            )

        # Apply marked region removal
        if self.remove_marked_regions:
            do_remove_marked_regions(
                routine, mark_with_comment=self.mark_with_comment
            )

        # Apply Dead Code Elimination
        if self.remove_dead_code:
            do_remove_dead_code(routine, use_simplify=self.use_simplify)


def do_remove_dead_code(routine, use_simplify=True):
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
    use_simplify : boolean
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


def do_remove_marked_regions(routine, mark_with_comment=True):
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


def do_remove_calls(
        routine, call_names=None, import_names=None, intrinsic_names=None,
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
    import_names : list of str
        List of module names against which to match :any:`Import`
        nodes.
    intrinsic_names : list of str
        List of module names against which to match :any:`Intrinsic`
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
    import_names : list of str
        List of module names against which to match :any:`Import`
        nodes.
    intrinsic_names : list of str
        List of module names against which to match :any:`Intrinsic`
        nodes.
    """

    def __init__(
            self, call_names=None, import_names=None,
            intrinsic_names=None, **kwargs
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
