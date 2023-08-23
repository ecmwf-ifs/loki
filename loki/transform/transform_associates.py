# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import FindVariables, SubstituteExpressions
from loki.tools import CaseInsensitiveDict
from loki.transform.transform_utilities import recursive_expression_map_update
from loki.visitors import NestedTransformer


__all__ = ['resolve_associates', 'ResolveAssociatesTransformer']


def resolve_associates(routine):
    """
    Resolve :any:`Associate` mappings in the body of a given routine.

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


class ResolveAssociatesTransformer(NestedTransformer):
    """
    :any:`Transformer` class to resolve :any:`Associate` nodes in IR trees

    This will replace each :any:`Associate` node with its own body,
    where all `identifier` symbols have been replaced with the
    corresponding `selector` expression defined in ``associations``.
    """

    def visit_Associate(self, o, **kwargs):
        # First head-recurse, so that all associate blocks beneath are resolved
        body = self.visit(o.body, **kwargs)

        # Create an inverse association map to look up replacements
        invert_assoc = CaseInsensitiveDict({v.name: k for k, v in o.associations})

        # Build the expression substitution map
        vmap = {}
        for v in FindVariables().visit(body):
            if v.name in invert_assoc:
                # Clone the expression to update its parentage and scoping
                inv = invert_assoc[v.name]
                if hasattr(v, 'dimensions'):
                    vmap[v] = inv.clone(dimensions=v.dimensions)
                else:
                    vmap[v] = inv

        # Apply the expression substitution map to itself to handle nested expressions
        vmap = recursive_expression_map_update(vmap)

        # Mark the associate block for replacement with its body, with all expressions replaced
        self.mapper[o] = SubstituteExpressions(vmap).visit(body)

        # Return the original object unchanged and let the tuple injection mechanism take care
        # of replacing it by its body - otherwise we would end up with nested tuples
        return o
