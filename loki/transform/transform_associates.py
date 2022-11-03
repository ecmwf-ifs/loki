from loki.expression import FindVariables, SubstituteExpressions
from loki.ir import Associate
from loki.tools import CaseInsensitiveDict
from loki.visitors import Transformer, FindNodes


__all__ = ['resolve_associates', 'ResolveAssociatesTransformer']


def resolve_associates(routine):
    """
    Resolve :any`Associate` mappings in the body of a given routine.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine for which to resolve all associate blocks.
    """
    routine.body = ResolveAssociatesTransformer().visit(routine.body)

    # Ensure that all symbols have the appropriate scope attached.
    # This is needed, as the parent of a symbol might have changed,
    # which affects the symbol's type-defining scope.
    routine.rescope_symbols()


class ResolveAssociatesTransformer(Transformer):
    """
    :any:`Transformer` class that replaces :any:`Associate` node with its body,
    with  have been replaced by their ass
    """

    def visit_Associate(self, o):
        # First head-recurse, so that all associate blocks beneath are resolved
        body = self.visit(o.body)

        # Create an inverse association map to look up replacements
        invert_assoc = CaseInsensitiveDict({v.name: k for k, v in o.associations})

        # Build the expression substitution map
        vmap = {}
        for v in FindVariables(unique=False).visit(body):
            if v.name in invert_assoc:
                # Clone the expression to update its parentage and scoping
                inv = invert_assoc[v.name]
                vmap[v] = v.clone(name=inv.name, parent=inv.parent, scope=inv.scope)

        # Return the body of the associate block with all expressions replaced
        return SubstituteExpressions(vmap).visit(body)
