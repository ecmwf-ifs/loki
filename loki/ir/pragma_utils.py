# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from collections import defaultdict
from contextlib import contextmanager
from codetiming import Timer

from loki.ir.nodes import VariableDeclaration, Pragma, PragmaRegion
from loki.ir.find import FindNodes
from loki.ir.transformer import Transformer
from loki.ir.visitor import Visitor
from loki.tools.util import as_tuple, replace_windowed
from loki.logging import debug, warning


__all__ = [
    'is_loki_pragma', 'get_pragma_parameters', 'process_dimension_pragmas',
    'attach_pragmas', 'detach_pragmas',
    'pragmas_attached', 'attach_pragma_regions', 'detach_pragma_regions',
    'pragma_regions_attached', 'PragmaAttacher', 'PragmaDetacher'
]


def is_loki_pragma(pragma, starts_with=None):
    """
    Checks for a pragma annotation and, if it exists, for the ``loki`` keyword.
    Optionally, the pragma content is tested for a specific start.

    Parameters
    ----------
    pragma : :any:`Pragma` or `list`/`tuple` of :any:`Pragma` or `None`
        the pragma or list of pragmas to check.
    starts_with : str, optional
        the keyword the pragma content must start with.
    """
    pragma = as_tuple(pragma)
    if not pragma:
        return False
    loki_pragmas = [p for p in pragma if p.keyword.lower() == 'loki']
    if not loki_pragmas:
        return False
    if starts_with is not None and not any(p.content and p.content.startswith(starts_with) for p in loki_pragmas):
        return False
    return True


class PragmaParameters:
    """
    Utility class to parse strings for parameters in the form ``<command>[(<arg>)]`` and
    return them as a map ``{<command>: <arg> or None}``.
    """

    _pattern_opening_parenthesis = re.compile(r'\(')
    _pattern_closing_parenthesis = re.compile(r'\)')
    _pattern_quoted_string = re.compile(r'(?:\'.*?\')|(?:".*?")')

    @classmethod
    def find(cls, string):
        """
        Find parameters in the form ``<command>[(<arg>)]`` and
        return them as a map ``{<command>: <arg> or None}``.

        .. note::
            This allows nested parenthesis by matching pairs of
            parentheses starting at the end by pushing and popping
            from a stack.
        """
        string = cls._pattern_quoted_string.sub('', string)
        if not string.strip():
            # Early bail-out on empty strings
            return {}

        p_open = [match.start() for match in cls._pattern_opening_parenthesis.finditer(string)]
        p_close = [match.start() for match in cls._pattern_closing_parenthesis.finditer(string)]
        assert len(p_open) == len(p_close)

        def _match_spans(open_, close_):
            # We match pairs of parentheses starting at the end by pushing and popping from a stack.
            # Whenever the stack runs out, we have fully resolved a set of (nested) parenthesis and
            # record the corresponding span
            if not close_:
                return []
            spans = []
            stack = [close_.pop()]
            while open_:
                if not close_ or open_[-1] > close_[-1]:
                    assert stack
                    start = open_.pop()
                    end = stack.pop()
                    if not stack:
                        spans.append((start, end))
                else:
                    stack.append(close_.pop())
            assert not (stack or open_ or close_)
            return spans

        p_spans = _match_spans(p_open, p_close)
        spans = []
        while p_spans:
            spans.append(p_spans.pop())
        if p_spans:
            spans += p_spans[::-1]

        # Build the list of parameters from the matched spans
        parameters = defaultdict(list)
        for i, span in enumerate(spans):
            keys = string[spans[i-1][1]+1 if i>=1 else 0:span[0]].strip().split(' ')
            for key in keys[:-1]:
                if key:
                    parameters[key].append(None)
            parameters[keys[-1]].append(string[span[0]+1:span[1]])

        # Tail handling (including strings without any matched spans)
        tail_span = spans[-1][1] + 1 if spans else 0
        for key in string[tail_span:].strip().split(' '):
            if key != '':
                parameters[key].append(None)
        parameters = {k: v if len(v) > 1 else v[0] for k, v in parameters.items()}
        return parameters


def get_pragma_parameters(pragma, starts_with=None, only_loki_pragmas=True):
    """
    Parse the pragma content for parameters in the form ``<command>[(<arg>)]`` and
    return them as a map ``{<command>: <arg> or None}``.

    Optionally, look only at the pragma with the given keyword at the beginning.

    Note that if multiple pragma are given as a tuple/list, arguments with the same
    name will overwrite previous definitions.

    Parameters
    ----------
    pragma : :any:`Pragma` or `list`/`tuple` of :any:`Pragma` or `None`
        the pragma or list of pragmas to check.
    starts_with : str, optional
        the keyword the pragma content should start with.
    only_loki_pragmas : bool, optional
        restrict parameter extraction to ``loki`` pragmas only.

    Returns
    -------
    dict :
        Mapping of parameters ``{<command>: <arg> or <None>}`` with the values being a list
        when multiple entries have the same key
    """
    pragma_parameters = PragmaParameters()
    pragma = as_tuple(pragma)
    parameters = defaultdict(list)
    for p in pragma:
        if only_loki_pragmas and p.keyword.lower() != 'loki':
            continue
        content = p.content or ''
        # Remove any line-continuation markers
        content = content.replace('&', '')
        if starts_with is not None:
            if not content.lower().startswith(starts_with.lower()):
                continue
            content = content[len(starts_with):]
        parameter = pragma_parameters.find(content)
        for key in parameter:
            parameters[key].append(parameter[key])
    parameters = {k: v if len(v) > 1 else v[0] for k, v in parameters.items()}
    return parameters


def process_dimension_pragmas(ir, scope=None):
    """
    Process any ``!$loki dimension`` pragmas to override deferred dimensions

    Note that this assumes :any:`attach_pragmas` has been run on :data:`ir` to
    attach any pragmas to the :any:`VariableDeclaration` nodes.

    Parameters
    ----------
    ir : :any:`Node`
        Root node of the (section of the) internal representation to process
    """
    from loki.expression.parser import parse_expr  # pylint: disable=import-outside-toplevel

    # print(f"process_dimension_pragmas ...")
    for decl in FindNodes(VariableDeclaration).visit(ir):
        # print(f"  decl.pragma {decl.pragma} ? {is_loki_pragma(decl.pragma, starts_with='dimension')}")
        if is_loki_pragma(decl.pragma, starts_with='dimension'):
            for v in decl.symbols:
                # Found dimension override for variable
                dims = get_pragma_parameters(decl.pragma)['dimension']
                dims = [d.strip() for d in dims.split(',')]
                # parse each dimension
                shape = tuple(parse_expr(d, scope=scope) for d in dims)
                # print(f"  v: new shape: {shape}")
                # update symbol table
                v.scope.symbol_attrs[v.name] = v.type.clone(shape=shape)
    return ir


class PragmaAttacher(Visitor):
    """
    Utility visitor that finds pragmas preceding (or optionally also
    trailing) nodes of given types and attaches them to these nodes as
    ``pragma`` property.

    Note that this operates by updating (instead of rebuilding) the relevant
    nodes, thus only nodes to which pragmas are attached get modified and
    the tree as a whole is not modified if no pragmas are found. This means
    existing node references should remain valid.

    .. note::
        When using :data:`attach_pragma_post` and two nodes qualifying according to
        :data:`node_type` are separated only by :any:`Pragma` nodes inbetween, it
        is not possible to decide to which node these pragmas belong. In such cases,
        they are attached to the second node as ``pragma`` property takes precedence.
        Such situations can only be resolved by full knowledge about the pragma
        language specification (_way_ out of scope) or modifying the original source,
        e.g. by inserting a comment between the relevant pragmas.

    Parameters
    ----------
    node_type :
        the IR node type (or a list of them) to attach pragmas to.
    attach_pragma_post : bool, optional
        look for pragmas after the node, too, and attach as ``pragma_post`` if applicable.

    """

    def __init__(self, node_type, attach_pragma_post=True):
        super().__init__()
        self.node_type = as_tuple(node_type)
        self.attach_pragma_post = attach_pragma_post

    def visit_tuple(self, o, **kwargs):
        pragmas = []
        updated = []
        for i in o:
            if isinstance(i, Pragma):
                # Collect pragmas, anticipating a possible node to attach to
                pragmas += [i]
            else:
                # Recurse first
                i = self.visit(i, **kwargs)
                if pragmas:
                    if isinstance(i, self.node_type):
                        # Found a node of given type: attach pragmas
                        i._update(pragma=as_tuple(pragmas))
                    elif (
                          self.attach_pragma_post and updated and
                          isinstance(updated[-1], self.node_type) and
                          hasattr(updated[-1], 'pragma_post')
                    ):
                        # Encountered a different node but have some pragmas: attach to last
                        # node as pragma_post if type matches
                        updated[-1]._update(pragma_post=as_tuple(pragmas))
                    else:
                        # Not attaching pragmas anywhere: re-insert into list
                        updated += pragmas
                    pragmas = []
                updated += [i]
        if self.attach_pragma_post and pragmas:
            # Take care of leftover pragmas
            if updated and isinstance(updated[-1], self.node_type):
                updated[-1]._update(pragma_post=as_tuple(pragmas))
                pragmas = []
        return as_tuple(updated + pragmas)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) for i in o.children)
        # Modify the node in-place instead of rebuilding it to leave existing references
        # to IR nodes intact
        o._update(*children)
        return o

    def visit_object(self, o, **kwargs):
        # Any other objects (e.g., expression trees) are to be left untouched
        return o


class PragmaDetacher(Visitor):
    """
    Utility visitor that detaches inlined pragmas from nodes of given types
    and inserts them before/after the nodes into the IR.

    Note that this operates by updating (instead of rebuilding) the relevant
    nodes, thus only nodes to which pragmas are attached get modified and
    the tree as a whole is not modified if no pragmas are found. This means
    existing node references should remain valid.

    Parameters
    ----------
    node_type :
        the IR node type (or a list of them) to detach pragmas from.
    detach_pragma_post : bool, optional
        detach ``pragma_post`` properties, if applicable.
    """

    def __init__(self, node_type, detach_pragma_post=False):
        super().__init__()
        self.node_type = as_tuple(node_type)
        self.detach_pragma_post = detach_pragma_post

    def visit_tuple(self, o, **kwargs):
        updated = ()
        for i in o:
            i = self.visit(i, **kwargs)
            if isinstance(i, self.node_type) and getattr(i, 'pragma', None):
                # Pragmas need to go before the node
                updated += as_tuple(i.pragma)
                # Modify the node in-place to leave existing references intact
                i._update(pragma=None)
            # Insert node into the tuple
            updated += (i,)
            if self.detach_pragma_post and isinstance(i, self.node_type) and getattr(i, 'pragma_post', None):
                # pragma_post need to go after the node
                updated += as_tuple(i.pragma_post)
                # Modify the node in-place to leave existing references intact
                i._update(pragma_post=None)
        return updated

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) for i in o.children)
        # Modify the node in-place instead of rebuilding it to leave existing references
        # to IR nodes intact
        o._update(*children)
        return o

    def visit_object(self, o, **kwargs):
        # Any other objects (e.g., expression trees) are to be left untouched
        return o


def attach_pragmas(ir, node_type, attach_pragma_post=True):
    """
    Find pragmas and merge them onto the given node type(s).

    This can be done for all IR nodes that have a ``pragma`` property
    (:any:`VariableDeclaration`, :any:`Loop`, :any:`WhileLoop`,
    :any:`CallStatement`).
    Optionally, attaching pragmas after nodes as ``pragma_post`` can be
    disabled by setting :data:`attach_pragma_post` to `False`
    (relevant only for :any:`Loop` and :any:`WhileLoop`).

    .. note::
        Pragmas are not discovered by :any:`FindNodes` while attached to IR nodes.

    This is implemented using :any:`PragmaAttacher`. Therefore, the IR
    is not rebuilt but updated and existing references should remain valid.

    Parameters
    ----------
    ir : :any:`Node`
        the root of (a section of the) intermediate representation in which
        pragmas are to be attached.
    node_type : list
        the (list of) :any:`Node` types pragmas should be attached to.
    attach_pragma_post : bool, optional
        process ``pragma_post`` attachments.
    """
    return PragmaAttacher(node_type, attach_pragma_post=attach_pragma_post).visit(ir)


def detach_pragmas(ir, node_type, detach_pragma_post=True):
    """
    Revert the inlining of pragmas, e.g. as done by :any:`attach_pragmas`.

    This can be done for all IR nodes that have a ``pragma`` property
    (:class:``Declaration``, :class:``Loop``, :class:``WhileLoop`,
    :class:``CallStatement``).
    Optionally, detaching of pragmas after nodes (for nodes with a
    ``pragma_post`` property) can be disabled by setting
    :data:`detach_pragma_post` to `False` (relevant only for :any:`Loop`
    and :any:`WhileLoop`).

    This is implemented using :any:`PragmaDetacher`. Therefore, the IR
    is not rebuilt but updated and existing references should remain valid.

    Parameters
    ----------
    ir : :any:`Node`
        the root node of the (section of the) intermediate representation
        in which pragmas are to be detached.
    node_type :
        the (list of) :any:`Node` types that pragmas should be detached from.
    detach_pragma_post: bool, optional
        process ``pragma_post`` attachments.
    """
    return PragmaDetacher(node_type, detach_pragma_post=detach_pragma_post).visit(ir)


@contextmanager
def pragmas_attached(module_or_routine, node_type, attach_pragma_post=True):
    """
    Create a context in which pragmas preceding nodes of given type(s) inside
    the module's or routine's IR are attached to these nodes.

    This can be done for all IR nodes that have a ``pragma`` property
    (:any:`VariableDeclaration`, :any:`ProcedureDeclaration`, :any:`Loop`,
    :any:`WhileLoop`, :any:`CallStatement`). Inside the created context,
    attached pragmas are no longer standalone IR nodes but accessible via the
    corresponding node's ``pragma`` property.

    Pragmas after nodes are attached as ``pragma_post``, which can be disabled
    by setting :data:`attach_pragma_post` to `False` (for :any:`Loop` and
    :any:`WhileLoop`).

    .. note::
        Pragmas are not discovered by :any:`FindNodes` while attached to IR nodes.

    When leaving the context all pragmas for nodes of the given type
    are detached, irrespective of whether they had already been attached or not
    when entering the context.

    .. note::
        Pragma attachment is only done for the object itself (i.e. its spec and
        body), not for any contained subroutines.

    This is implemented using :any:`PragmaAttacher` and
    :any:`PragmaDetacher`, respectively. Therefore, the IR is not rebuilt
    but updated and existing references should remain valid when entering the
    context and stay valid beyond exiting the context.

    Example:

    .. code-block:: python

        loop_of_interest = None
        with pragmas_attached(routine, Loop):
            for loop in FindNodes(Loop).visit(routine.body):
                if is_loki_pragma(loop.pragma, starts_with='foobar'):
                    loop_of_interest = loop
                    break
        # Do something with that loop
        loop_body = loop_of_interest.body
        # Note that loop_body.pragma == None!

    Parameters
    ----------
    module_or_routine : :any:`Module` or :any:`Subroutine`
        the program unit in which pragmas are to be inlined.
    node_type :
        the (list of) :any:`Node` types, that pragmas should be
        attached to.
    attach_pragma_post : bool, optional
        process ``pragma_post`` attachments.
    """
    if hasattr(module_or_routine, 'spec'):
        module_or_routine.spec = attach_pragmas(module_or_routine.spec, node_type,
                                                attach_pragma_post=attach_pragma_post)
    if hasattr(module_or_routine, 'body'):
        module_or_routine.body = attach_pragmas(module_or_routine.body, node_type,
                                                attach_pragma_post=attach_pragma_post)
    try:
        yield module_or_routine
    finally:
        if hasattr(module_or_routine, 'spec'):
            module_or_routine.spec = detach_pragmas(module_or_routine.spec, node_type,
                                                    detach_pragma_post=attach_pragma_post)
        if hasattr(module_or_routine, 'body'):
            module_or_routine.body = detach_pragmas(module_or_routine.body, node_type,
                                                    detach_pragma_post=attach_pragma_post)


def get_matching_region_pragmas(pragmas):
    """
    Given a list of :any:`Pragma` objects return a list of matching pairs
    that define a pragma region.

    Matching pragma pairs are assumed to be of the form
    ``!$<keyword> <marker>`` and ``!$<keyword> end <marker>``.
    """

    def _matches_starting_pragma(start, p):
        """ Definition of which pragmas match """
        stok = start.content.lower().split(' ')
        ptok = p.content.lower().split(' ')
        if 'end' not in ptok:
            return False
        if not start.keyword == p.keyword:
            return False
        idx = ptok.index('end')
        return ptok[idx+1] == stok[idx]

    matches = []
    stack = []
    for i, p in enumerate(pragmas):
        if 'end' not in p.content.lower():
            # If we encounter one that does have a match, stack it
            if any(_matches_starting_pragma(p, p2) for p2 in pragmas[i:]):
                stack.append(p)

        elif 'end' in p.content.lower() and stack:
            # If we and end that matches our last stacked, keep it!
            if _matches_starting_pragma(stack[-1], p):
                p1 = stack.pop()
                matches.append((p1, p))

    return matches


class PragmaRegionAttacher(Transformer):
    """
    Utility transformer that inserts :any:`PragmaRegion` objects to
    mark code section between matching :any:`Pragma` pairs.

    Matching pragma pairs are assumed to be of the form
    ``!$<keyword> <marker>`` and ``!$<keyword> end <marker>``.

    The matching of pragma pairs only happens if the matching pragmas
    are stored within the same tuple, or in other words at the same
    depth of the IR tree. Ending a pragma region in a different
    nesting depth, eg. inside a loop body, will result in a warning
    and no region object being inserted into the IR tree.

    Parameters
    ----------
    pragma_pairs : tuple of tuple of :any:`Pragma`
        Tuple of 2-tuples of matching pragma pairs
    """

    def __init__(self, pragma_pairs=None, **kwargs):
        self.pragma_pairs = pragma_pairs

        super().__init__(**kwargs)

    def visit_tuple(self, o, **kwargs):
        """ Replace pragma-body-end in tuples """
        for start, stop in self.pragma_pairs:
            if start in o:
                # If a pair does not live in the same tuple we have a problem.
                if stop not in o:
                    warning('[Loki::IR] Cannot find matching end for pragma {start} at same IR level!')
                    continue

                # Create the PragmaRegion node and replace in tuple
                idx_start = o.index(start)
                idx_stop = o.index(stop)
                region = PragmaRegion(
                    body=o[idx_start+1:idx_stop], pragma=start, pragma_post=stop
                )
                o = o[:idx_start] + (region,) + o[idx_stop+1:]

        # Then recurse over the new nodes
        visited = tuple(self.visit(i, **kwargs) for i in o)

        # Strip empty sublists/subtuples or None entries
        return tuple(i for i in visited if i is not None and as_tuple(i))

    visit_list = visit_tuple


@Timer(logger=debug, text=lambda s: f'[Loki::IR] Executed attach_pragma_regions in {s:.2f}s')
def attach_pragma_regions(ir):
    """
    Create :any:`PragmaRegion` node objects for all matching pairs of
    region pragmas.

    Matching pragma pairs are assumed to be of the form
    ``!$<keyword> <marker>`` and ``!$<keyword> end <marker>``.

    The defining :any:`Pragma` nodes are accessible via the ``pragma``
    and ``pragma_post`` attributes of the region object. Insertion
    is performed in-place, without rebuilding any IR nodes.
    """
    pragmas = FindNodes(Pragma).visit(ir)
    pragma_pairs = get_matching_region_pragmas(pragmas)

    return PragmaRegionAttacher(pragma_pairs=pragma_pairs, inplace=True).visit(ir)


class PragmaRegionDetacher(Transformer):
    """
    Remove any :any:`PragmaRegion` node objects and insert the tuple
    of ``(r.pragma, r.body, r.pragma_post)`` in the enclosing tuple.
    """

    def visit_tuple(self, o, **kwargs):
        """ Unpack :any:`PragmaRegion` objects and insert in current tuple """

        # We unpack regions here to avoid creating nested tuples, or
        # forcing general tuple-flattening, which can affect other
        # nodes types.
        regions = tuple(n for n in o if isinstance(n, PragmaRegion))
        for r in regions:
            handle = (r.pragma,) + self.visit(r.body, **kwargs) + (r.pragma_post,)
            o = replace_windowed(o, r, subs=handle)

        # First recurse over the new nodes
        visited = tuple(self.visit(i, **kwargs) for i in o)

        # Strip empty sublists/subtuples or None entries
        return tuple(i for i in visited if i is not None and as_tuple(i))

    visit_list = visit_tuple


@Timer(logger=debug, text=lambda s: f'[Loki::IR] Executed detach_pragma_regions in {s:.2f}s')
def detach_pragma_regions(ir):
    """
    Remove any :any:`PragmaRegion` node objects and replace each with a
    tuple of ``(r.pragma, r.body, r.pragma_post)``, where ``r`` is the
    :any:`PragmaRegion` node object.

    All replacements are performed in-place, without rebuilding any IR
    nodes.
    """

    return PragmaRegionDetacher(inplace=True).visit(ir)


@contextmanager
def pragma_regions_attached(module_or_routine):
    """
    Create a context in which :any:`PragmaRegion` node objects are
    inserted into the IR to define code regions marked by matching
    pairs of pragmas.

    Matching pragma pairs are assumed to be of the form
    ``!$<keyword> <marker>`` and ``!$<keyword> end <marker>``.

    In the resulting context ``FindNodes(PragmaRegion).visit(ir)`` can
    be used to select code regions marked by pragma pairs as node
    objects.

    The defining :any:`Pragma` nodes are accessible via the ``pragma``
    and ``pragma_post`` attributes of the region object. Importantly,
    Pragmas are not discovered by :any:`FindNodes` while attached
    to IR nodes.

    When leaving the context all :any:`PragmaRegion` objects are replaced
    with a tuple of ``(r.pragma, r.body, r.pragma_post)``, where ``r``
    is the :any:`PragmaRegion` node object.

    Throughout the setup and teardown of the context IR nodes are only
    updated, never rebuild, meaning node mappings from inside the
    context are valid outside of it.

    Example:

    .. code-block:: python

        with pragma_regions_attached(routine):
            for region in FindNodes(PragmaRegion).visit(routine.body):
                if is_loki_pragma(region.pragma, starts_with='foobar'):
                    <transform code in region.body>

    Parameters
    ----------
    module_or_routine : :any:`Module` or :any:`Subroutine` in
        which :any:`PragmaRegion` objects are to be inserted.
    """
    if hasattr(module_or_routine, 'spec'):
        module_or_routine.spec = attach_pragma_regions(module_or_routine.spec)
    if hasattr(module_or_routine, 'body'):
        module_or_routine.body = attach_pragma_regions(module_or_routine.body)

    try:
        yield module_or_routine
    finally:
        if hasattr(module_or_routine, 'spec'):
            module_or_routine.spec = detach_pragma_regions(module_or_routine.spec)
        if hasattr(module_or_routine, 'body'):
            module_or_routine.body = detach_pragma_regions(module_or_routine.body)
