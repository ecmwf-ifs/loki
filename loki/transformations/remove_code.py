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

from loki.batch import Transformation
from loki.expression import simplify, symbols as sym
from loki.tools import flatten, as_tuple
from loki.ir import Conditional, Transformer, Comment, CallStatement, FindNodes, FindVariables
from loki.ir.pragma_utils import is_loki_pragma, pragma_regions_attached
from loki.analyse import dataflow_analysis_attached
from loki.types import BasicType


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
    Elimination for batch processing via the :any:`Scheduler`.

    The transformation will apply the following methods in order:

    * :any:`do_remove_calls`
    * :any:`do_remove_marked_regions`
    * :any:`do_remove_dead_code`

    Parameters
    ----------
    remove_marked_regions : boolean
        Flag to trigger the use of :meth:`remove_marked_regions`;
        default: ``True``
    mark_with_comment : boolean
        Flag to trigger the insertion of a marker comment when
        removing a region; default: ``True``.
    remove_dead_code : boolean
        Flag to trigger the use of :meth:`remove_dead_code`;
        default: ``False``
    use_simplify : boolean
        Use :any:`simplify` when branch pruning in during
        :meth:`remove_dead_code`.
    call_names : list of str
        List of subroutine names against which to match
        :any:`CallStatement` nodes during :meth:`remove_calls`.
    intrinsic_names : list of str
        List of module names against which to match :any:`Intrinsic`
        nodes during :meth:`remove_calls`.
    remove_imports : boolean
        Flag indicating whether to remove symbols from :any:`Import`
        objects during :meth:`remove_calls`; default: ``True``
    kernel_only : boolean
        Only apply the configured removal to subroutines marked as
        "kernel"; default: ``False``
    """

    _key = 'RemoveCodeTransformation'

    # Recurse to subroutines in ``contains`` clause
    recurse_to_internal_procedures = True
    reverse_traversal = True

    def __init__(
            self, remove_marked_regions=True, mark_with_comment=True,
            remove_dead_code=False, use_simplify=True,
            call_names=None, intrinsic_names=None,
            remove_imports=True, kernel_only=False,
            remove_unused_args=False
    ):
        self.remove_marked_regions = remove_marked_regions
        self.mark_with_comment = mark_with_comment

        self.remove_dead_code = remove_dead_code
        self.use_simplify = use_simplify

        self.call_names = as_tuple(call_names)
        self.intrinsic_names = as_tuple(intrinsic_names)
        self.remove_imports = remove_imports

        self.kernel_only = kernel_only
        self.remove_unused_args = remove_unused_args

    def transform_subroutine(self, routine, **kwargs):

        if kwargs.get('role') == 'kernel' or not self.kernel_only:
            # Apply named node removal to strip specific calls
            if self.call_names or self.intrinsic_names:
                do_remove_calls(
                    routine, call_names=self.call_names,
                    intrinsic_names=self.intrinsic_names,
                    remove_imports=self.remove_imports
                )

            # Apply marked region removal
            if self.remove_marked_regions:
                do_remove_marked_regions(
                    routine, mark_with_comment=self.mark_with_comment
                )

            # Apply Dead Code Elimination
            if self.remove_dead_code:
                do_remove_dead_code(routine, use_simplify=self.use_simplify)

        if self.remove_unused_args and (item := kwargs['item']):
            # collect unused args from successors
            successors = kwargs['successors']
            unused_args_map = {successor.ir: successor.trafo_data.get(self._key, {}).get('unused_args', {})
                               for successor in successors}
            do_remove_unused_call_args(routine, unused_args_map)

            if item.config.get('remove_unused_args', True) and kwargs['role'] == 'kernel':
                # find unused args
                unused_args = find_unused_args(routine)
                do_remove_unused_dummy_args(routine, unused_args)
                # store unused args
                item.trafo_data[self._key] = {'unused_args': unused_args}


def do_remove_unused_dummy_args(routine, unused_args):

    routine.variables = [a for a in routine.variables
                         if not a.name.lower() in unused_args]


def do_remove_unused_call_args(routine, unused_args_map):

    for call in FindNodes(CallStatement).visit(routine.body):
        if call.routine is BasicType.DEFERRED or not unused_args_map.get(call.routine, None):
            continue

        unused_args = [call.arguments[c] for c in unused_args_map[call.routine].values() if c < len(call.arguments)]
        unused_kwargs = [(kw, arg) for kw, arg in call.kwarguments if kw.lower() in unused_args_map[call.routine]]

        new_args = [arg for arg in call.arguments if not arg in unused_args]
        new_kwargs = [(kw, arg) for kw, arg in call.kwarguments if not (kw, arg) in unused_kwargs]

        call._update(arguments=as_tuple(new_args), kwarguments=as_tuple(new_kwargs))


def find_unused_args(routine):

    variable_map = routine.symbol_map
    with dataflow_analysis_attached(routine):
        used_or_defined_symbols = routine.body.uses_symbols | routine.body.defines_symbols

        # we search for symbols used to define array sizes
        used_or_defined_array_shapes = [s.shape for s in used_or_defined_symbols if isinstance(s, sym.Array)]
        used_or_defined_symbols |= set(FindVariables().visit(used_or_defined_array_shapes))

        used_or_defined_symbols |= set(variable_map.get(v.name_parts[0], v) for v in used_or_defined_symbols)

        unused_args = {a.clone(dimensions=None): c for c, a in enumerate(routine.arguments)
                       if not a.name.lower() in used_or_defined_symbols}

    return unused_args


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

    The primary modification performed is to prune individual code branches
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
    example via :meth:`pragma_regions_attached`.

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
                return Comment(text='! [Loki] Removed content of pragma-marked region!')

            return None

        # Recurse into the pragama region and rebuild
        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)


def do_remove_calls(
        routine, call_names=None, intrinsic_names=None, remove_imports=True
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
    remove_imports : boolean
        Flag indicating whether to remove the respective procedure
        symbols from :any:`Import` objects; default: ``True``.
    """

    transformer = RemoveCallsTransformer(
        call_names=call_names, intrinsic_names=intrinsic_names,
        remove_imports=remove_imports
    )
    routine.spec = transformer.visit(routine.spec)
    routine.body = transformer.visit(routine.body)


class RemoveCallsTransformer(Transformer):
    """
    A :any:`Transformer` that removes all :any:`CallStatement` nodes
    to specific named subroutines.

    This :any:`Transformer` will by default also remove the enclosing
    inline-conditional when encountering calls of the form ```if
    (flag) call named_procedure()``.

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
    remove_imports : boolean
        Flag indicating whether to remove the respective procedure
        symbols from :any:`Import` objects; default: ``True``.
    """

    def __init__(
            self, call_names=None, intrinsic_names=None,
            remove_imports=True, **kwargs
    ):
        super().__init__(**kwargs)

        self.call_names = as_tuple(call_names)
        self.intrinsic_names = as_tuple(intrinsic_names)
        self.remove_imports = remove_imports

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
        """ Remove the symbol of any named calls from Import nodes """

        symbols_found = False
        if o.c_import and 'intfb' in o.module:
            symbols = as_tuple(o.module.split('.')[0])
            symbols_found = symbols[0] in self.call_names
        else:
            symbols = o.symbols
            symbols_found = any(s in self.call_names for s in symbols)
        if self.remove_imports and symbols_found:
            new_symbols = tuple(s for s in symbols if s not in self.call_names)
            return o.clone(symbols=new_symbols) if new_symbols else None

        rebuilt = tuple(self.visit(i, **kwargs) for i in o.children)
        return self._rebuild(o, rebuilt)
