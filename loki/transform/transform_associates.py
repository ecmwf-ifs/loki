from loki.expression import FindVariables, SubstituteExpressions
from loki.ir import Associate
from loki.tools import CaseInsensitiveDict
from loki.visitors import Transformer, FindNodes


__all__ = ['resolve_associates']


def resolve_associates(routine):
    """
    Resolve :any`Associate` mappings in the body of a given routine.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine for which to resolve all associate blocks.
    """
    assoc_map = {}
    vmap = {}
    for assoc in FindNodes(Associate).visit(routine.body):
        invert_assoc = CaseInsensitiveDict({v.name: k for k, v in assoc.associations})
        for v in FindVariables(unique=False).visit(assoc.body):
            if v.name in invert_assoc:
                inv = invert_assoc[v.name]
                vmap[v] = v.clone(parent=inv.parent, scope=inv.scope)
        assoc_map[assoc] = assoc.body

    routine.body = Transformer(assoc_map).visit(routine.body)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Ensure that all symbols have the appropriate scope attached
    routine.rescope_symbols()
