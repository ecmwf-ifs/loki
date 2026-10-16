# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.config import config
from loki.expression.mappers import ExpressionRetriever


__all__ = ['loki_make_stringifier', 'StrCompareMixin', 'ForeignMapperFallbackMixin']


def loki_make_stringifier(self, originating_stringifier=None):  # pylint: disable=unused-argument
    """
    Return a :any:`LokiStringifyMapper` instance that can be used to generate a
    human-readable representation of :data:`self`.

    This is used as common abstraction for the :meth:`make_stringifier` method in
    Pymbolic expression nodes.
    """
    from loki.expression.mappers import LokiStringifyMapper  # pylint: disable=import-outside-toplevel
    return LokiStringifyMapper()


class StrCompareMixin:
    """
    Mixin to enable comparing expressions to strings.

    The purpose of the string comparison override is to reliably and flexibly
    identify expression symbols from equivalent strings.
    """

    @staticmethod
    def _canonical(s):
        """ Define canonical string representations (lower-case, no spaces) """
        if config['case-sensitive']:
            return str(s).replace(' ', '')
        return str(s).lower().replace(' ', '')

    def __hash__(self):
        return hash(self._canonical(self))

    def __eq__(self, other):
        if isinstance(other, (str, type(self))):
            # Do comparsion based on canonical string representations
            return self._canonical(self) == self._canonical(other)

        return super().__eq__(other)

    def __contains__(self, other):
        # Assess containment via a retriver with node-wise string comparison
        return len(ExpressionRetriever(lambda x: x == other).retrieve(self)) > 0

    make_stringifier = loki_make_stringifier


class ForeignMapperFallbackMixin:
    """
    Mixin for Loki leaf nodes to make pymbolic's own mappers defer to the node.

    Mappers dispatch an expression node via its :attr:`mapper_method`. If a
    mapper does not provide this method, pymbolic walks the MRO of the node's
    class and uses the first :attr:`mapper_method` that the mapper provides.
    If none is found, it calls :meth:`handle_unsupported_expression`, which
    pymbolic's stringifiers implement by delegating to the node's own
    :meth:`make_stringifier`, and evaluation mappers by raising
    :any:`pymbolic.mapper.UnsupportedExpressionError` (a :any:`ValueError`).

    Since pymbolic 2024.1, :class:`pymbolic.primitives.AlgebraicLeaf` and
    :class:`pymbolic.primitives.Leaf` carry a ``map_algebraic_leaf``
    mapper method that raises :any:`NotImplementedError`. For Loki leaf
    nodes this would be found before the fallback above, so that, e.g.,
    ``str()`` of a pymbolic :class:`pymbolic.primitives.Sum` with Loki
    children or ``float()`` of a Loki symbol fails.

    Placing this mixin before the pymbolic base class in the MRO restores the
    fallback to :meth:`handle_unsupported_expression` for mappers that do not
    know a Loki leaf node. Mappers that provide the node's own mapper method,
    such as all Loki mappers, are not affected. Classes using this mixin must
    define their own :attr:`mapper_method`.
    """

    mapper_method = 'handle_unsupported_expression'
