import re
from contextlib import contextmanager

from loki.expression import symbols as sym
from loki.ir import Declaration, Pragma, PragmaRegion
from loki.tools.util import as_tuple, flatten
from loki.types import BasicType
from loki.visitors import FindNodes, Visitor, Transformer, MaskedTransformer


__all__ = [
    'is_loki_pragma', 'get_pragma_parameters', 'process_dimension_pragmas',
    'attach_pragmas', 'detach_pragmas', 'extract_pragma_region',
    'pragmas_attached', 'attach_pragma_regions', 'detach_pragma_regions',
    'pragma_regions_attached'
]


def is_loki_pragma(pragma, starts_with=None):
    """
    Checks for a pragma annotation and, if it exists, for the `loki` keyword.
    Optionally, the pragma content is tested for a specific start.

    :param pragma: the pragma or list of pragmas to check.
    :type pragma: :class:``ir.Pragma`` or ``list``/``tuple`` of ``ir.Pragma`` or ``None``
    :param str starts_with: the keyword the pragma content should start with.
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


_get_pragma_parameters_re = re.compile(r'(?P<command>[\w-]+)\s*(?:\((?P<arg>.+?)\))?')

def get_pragma_parameters(pragma, starts_with=None, only_loki_pragmas=True):
    """
    Parse the pragma content for parameters in the form `<command>[(<arg>)]` and
    return them as a map {<command>: <arg> or None}`.

    Optionally, look only at the pragma with the given keyword at the beginning.

    Note that if multiple pragma are given as a tuple/list, arguments with the same
    name will overwrite previous definitions.

    :param pragma: the pragma or list of pragmas to check.
    :type pragma: :class:``ir.Pragma`` or ``list``/``tuple`` of ``ir.Pragma`` or ``None``
    :param str starts_with: the keyword the pragma content should start with.
    :param bool only_loki_pragmas: restrict parameter extraction to ``loki`` pragmas only.

    :return: Mapping of parameters ``{<command>: <arg> or <None>}``.
    :rtype: ``dict``
    """
    pragma = as_tuple(pragma)
    parameters = {}
    for p in pragma:
        if only_loki_pragmas and p.keyword.lower() != 'loki':
            continue
        content = p.content or ''
        if starts_with is not None:
            if not content.startswith(starts_with):
                continue
            content = content[len(starts_with):]
        parameters.update({match.group('command'): match.group('arg')
                           for match in re.finditer(_get_pragma_parameters_re, content)})
    return parameters


def process_dimension_pragmas(ir):
    """
    Process any '!$loki dimension' pragmas to override deferred dimensions

    Note that this assumes `inline_pragmas` has been run on :param ir: to
    attach any pragmas to the `Declaration` nodes.
    """
    for decl in FindNodes(Declaration).visit(ir):
        if is_loki_pragma(decl.pragma, starts_with='dimension'):
            for v in decl.variables:
                # Found dimension override for variable
                dims = get_pragma_parameters(decl.pragma)['dimension']
                dims = [d.strip() for d in dims.split(',')]
                shape = []
                for d in dims:
                    if d.isnumeric():
                        shape += [sym.Literal(value=int(d), type=BasicType.INTEGER)]
                    else:
                        shape += [sym.Variable(name=d, scope=v.scope)]
                v.scope.symbols[v.name] = v.type.clone(shape=as_tuple(shape))
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

    :param node_type: the IR node type (or a list of them) to attach pragmas to.
    :param bool attach_pragma_post: to look for pragmas after the node, too,
        and attach as ``pragma_post`` if applicable.

    NB: When using ``attach_pragma_post`` and two nodes qualifying according to
    ``node_type`` are separated only by :class:``Pragma`` nodes inbetween, it
    is not possible to decide to which node these pragmas belong. In such cases,
    attaching to the second node as ``pragma`` takes precedence. Such
    situations can only be resolved in the original source, e.g. by inserting
    a comment between the relevant pragmas.
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

    :param node_type: the IR node type (or a list of them) to detach pragmas from.
    :param bool detach_pragma_post: to detach ``pragma_post`` properties, if applicable.
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
    (:class:``Declaration``, :class:``Loop``, :class:``WhileLoop`,
    :class:``CallStatement``).
    Optionally, attaching pragmas after nodes as ``pragma_post`` can be
    disabled by setting ``attach_pragma_post`` to ``False``
    (relevant only for :class:``Loop`` and :class:``WhileLoop``).

    NB: Pragmas are not discovered by :class:``FindNodes`` while attached
    to IR nodes.

    This is implemented using :class:``PragmaAttacher``. Therefore, the IR
    is not rebuilt but updated and existing references should remain valid.

    :param ir:
        the intermediate representation in which pragmas are to be attached.
    :param node_type:
        the (list of) :class:``ir.Node`` types pragmas should be attached to.
    :param bool attach_pragma_post:
        process ``pragma_post`` attachments.
    """
    return PragmaAttacher(node_type, attach_pragma_post=attach_pragma_post).visit(ir)


def detach_pragmas(ir, node_type, detach_pragma_post=True):
    """
    Revert the inlining of pragmas, e.g. as done by ``inline_pragmas``.

    This can be done for all IR nodes that have a ``pragma`` property
    (:class:``Declaration``, :class:``Loop``, :class:``WhileLoop`,
    :class:``CallStatement``).
    Optionally, detaching of pragmas after nodes (for nodes with a
    ``pragma_post`` property) can be disabled by setting
    ``detach_pragma_post`` to ``False`` (relevant only for :class:``Loop``
    and :class:``WhileLoop``).

    This is implemented using :class:``PragmaDetacher``. Therefore, the IR
    is not rebuilt but updated and existing references should remain valid.

    :param ir:
        the intermediate representation in which pragmas are to be detached.
    :param node_type:
        the (list of) :class:``ir.Node`` types pragmas should be detached from.
    :param bool detach_pragma_post:
        process ``pragma_post`` attachments.
    """
    return PragmaDetacher(node_type, detach_pragma_post=detach_pragma_post).visit(ir)


@contextmanager
def pragmas_attached(module_or_routine, node_type, attach_pragma_post=True):
    """
    Create a context in which pragmas preceding nodes of given type(s) inside
    the module's or routine's IR are attached to these nodes.

    This can be done for all IR nodes that have a ``pragma`` property
    (:class:``Declaration``, :class:``Loop``, :class:``WhileLoop`,
    :class:``CallStatement``). Inside the created context, attached pragmas
    are no longer standalone IR nodes but accessible via the corresponding
    node's ``pragma`` property.

    Pragmas after nodes are attached as ``pragma_post``, which can be disabled
    by setting ``attach_pragma_post`` to ``False`` (for :class:``Loop`` and
    :class:``WhileLoop``).

    NB: Pragmas are not discovered by :class:``FindNodes`` while attached
    to IR nodes.

    NB: When leaving the context all pragmas for nodes of the given type
    are detached, irrespective of whether they had already been attached or not
    when entering the context.

    NB: Pragma attachment is only done for the object itself (i.e. its spec and
    body), not for any contained subroutines.

    This is implemented using :class:``PragmaAttacher`` and
    :class:``PragmaDetacher``, respectively. Therefore, the IR is not rebuilt
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

    :param module_or_routine: the :class:``Module`` or :class:``Subroutine`` in
        which pragmas are to be inlined.
    :param node_type: the (list of) :class:``ir.Node`` types pragmas should be
        attached to.
    :param bool attach_pragma_post: process ``pragma_post`` attachments.
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
    Given a list of ``Pragma`` objects return a list of matching pairs
    that define a pragma region.

    Matching pragma pairs are assumed to be of the form
    ``!$<keyword> <marker>`` and ``!$<keyword> end <marker>``.
    """

    def _matches_starting_pragma(start, p):
        """ Definition of which pragmas match """
        if 'end' not in p.content.lower():
            return False
        if not start.keyword == p.keyword:
            return False
        stok = start.content.lower().split(' ')
        ptok = p.content.lower().split(' ')
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


def extract_pragma_region(ir, start, end):
    """
    Create a ``PragmaRegion`` object defined by two ``Pragma`` node
    objects ``start`` and ``end``.

    The resulting ``PragmaRegion`` object will be inserted into the
    ``ir`` tree without rebuilding any IR nodes via ``Transformer(...,
    inplace=True)``.
    """
    assert isinstance(start, Pragma)
    assert isinstance(end, Pragma)

    # Pick out the marked code block for the PragmaRegion
    block = MaskedTransformer(start=start, stop=end, inplace=True).visit(ir)
    block = as_tuple(flatten(block))[1:]  # Drop the initial pragma node
    region = PragmaRegion(body=block, pragma=start, pragma_post=end)

    # Remove the content of the code region and replace
    # starting pragma with new PragmaRegion node.
    mapper = {}
    for node in block:
        mapper[node] = None
    mapper[start] = region
    mapper[end] = None

    return Transformer(mapper, inplace=True).visit(ir)


def attach_pragma_regions(ir):
    """
    Create ``PragmaRegion`` node objects for all matching pairs of
    region pragmas.

    Matching pragma pairs are assumed to be of the form
    ``!$<keyword> <marker>`` and ``!$<keyword> end <marker>``.

    The defining ``Pragma`` nodes are accessible via the ``pragma``
    and ``pragma_post`` attributes of the region object. Insertion
    is performed in-place, without rebuilding any IR nodes.
    """
    pragmas = FindNodes(Pragma).visit(ir)
    for start, end in get_matching_region_pragmas(pragmas):
        ir = extract_pragma_region(ir, start=start, end=end)
    return ir

def detach_pragma_regions(ir):
    """
    Remove any ``PragmaRegion`` node objects and replace with with a
    tuple of ``(r.pragma, r.body, r.pragma_post)``, where ``r`` is the
    ``PragmaRegion`` node object.

    All replacements are performed in-place, without rebuilding any IR
    nodes.
    """
    mapper = {region: (region.pragma, region.body, region.pragma_post)
              for region in FindNodes(PragmaRegion).visit(ir)}
    return Transformer(mapper, inplace=True).visit(ir)


@contextmanager
def pragma_regions_attached(module_or_routine):
    """
    Create a context in which ``PragmaRegion`` node objects are
    inserted into the IR to define code regions marked by matching
    pairs of pragmas.

    Matching pragma pairs are assumed to be of the form
    ``!$<keyword> <marker>`` and ``!$<keyword> end <marker>``.

    In the resulting context ``FindNodes(PragmaRegion).visit(ir)`` can
    be used to select code regions marked by pragma pairs as node
    objects.

    The defining ``Pragma`` nodes are accessible via the ``pragma``
    and ``pragma_post`` attributes of the region object. Importantly,
    Pragmas are not discovered by :class:``FindNodes`` while attached
    to IR nodes.

    When leaving the context all ``PragmaRegion`` objects are replaced
    with a tuple of ``(r.pragma, r.body, r.pragma_post)``, where ``r``
    is the ``PragmaRegion`` node object.

    Throughout the setup and teardown of the context IR nodes are only
    updated, never rebuild, meaning node mappings from inside the
    context are valid outside of it.

    Example:

    .. code-block:: python

        with pragma_regions_attached(routine):
            for region in FindNodes(PragmaRegion).visit(routine.body):
                if is_loki_pragma(region.pragma, starts_with='foobar'):
                    <transform code in region.body>

    :param module_or_routine: the :class:``Module`` or :class:``Subroutine`` in
        which ``PragmaRegion`` objects are to be inserted.
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
