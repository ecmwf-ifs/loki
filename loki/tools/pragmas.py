import re
from loki.expression import symbols as sym
from loki.ir import CallStatement, Declaration, Loop, WhileLoop, Pragma
from loki.tools.util import as_tuple
from loki.types import BasicType, SymbolType
from loki.visitors import FindNodes, NestedTransformer
from loki.frontend.util import PatternFinder, SequenceFinder


__all__ = [
    'is_loki_pragma', 'get_pragma_parameters', 'process_dimension_pragmas',
    'inline_pragmas', 'detach_pragmas'
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
                        _type = SymbolType(BasicType.INTEGER)
                        shape += [sym.Variable(name=d, scope=v.scope, type=_type)]
                v.type = v.type.clone(shape=as_tuple(shape))
    return ir


def inline_pragmas(ir):
    """
    Find pragmas and merge them onto declarations and subroutine calls

    Note: Pragmas in derived types are already associated with the
    declaration due to way we parse derived types.
    """
    # Look for pragma sequences and make them accessible via the last pragma in sequence
    pragma_groups = {seq[-1]: seq for seq in SequenceFinder(node_type=Pragma).visit(ir)}

    # (Pragma, <target_node>) patterns to look for
    patterns = [(Pragma, Declaration), (Pragma, Loop), (Pragma, WhileLoop)]

    # TODO: Generally pragma inlining does not repsect type restriction
    # (eg. omp do pragas to loops) or "post_pragmas". This needs a deeper
    # rethink, so diabling the problematic corner case for now.
    # patterns += [(Pragma, CallStatement)]

    mapper = {}
    for pattern in patterns:
        for seq in PatternFinder(pattern=pattern).visit(ir):
            # Merge pragmas with IR node and delete
            pragmas = as_tuple(pragma_groups.get(seq[0], seq[0]))
            mapper.update({pragma: None for pragma in pragmas})
            mapper[seq[-1]] = seq[-1]._rebuild(pragma=pragmas)
    return NestedTransformer(mapper, invalidate_source=False).visit(ir)


def detach_pragmas(ir):
    """
    Revert the inlining of pragmas, e.g. as done by ``inline_pragmas``.

    Take any ``<IR node>.pragma`` and ``<IR node>.pragma_post`` properties and insert
    them as stand alone :class:``Pragma`` nodes before/after the respective node.

    Currently, this is done for :class:``Loop``, :class:``WhileLoop``, :class:``Declaration``,
    and :class:``CallStatement`` nodes.
    """
    mapper = {}
    for node in FindNodes((Loop, WhileLoop, Declaration, CallStatement)).visit(ir):
        if hasattr(node, 'pragma_post'):
            if node.pragma or node.pragma_post:
                pragma = as_tuple(node.pragma)
                pragma_post = as_tuple(node.pragma_post) if hasattr(node, 'pragma_post') else ()
                seq = pragma + (node.clone(pragma=None, pragma_post=None),) + pragma_post
                mapper[node] = seq
        elif node.pragma:
            pragma = as_tuple(node.pragma)
            mapper[node] = pragma + (node.clone(pragma=None),)
    return NestedTransformer(mapper, invalidate_source=False).visit(ir)
