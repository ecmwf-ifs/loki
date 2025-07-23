# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.config import config
from loki.expression.mappers import ExpressionRetriever


__all__ = ['loki_make_stringifier', 'StrCompareMixin']


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
