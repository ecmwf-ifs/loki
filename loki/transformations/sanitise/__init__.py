# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
Sub-package with assorted utility :any:`Transformation` classes to
harmonize the look-and-feel of input source code.
"""
from loki.batch import Transformation

from loki.transformations.sanitise.associates import * # noqa
from loki.transformations.sanitise.sequence_associations import * # noqa


__all__ = ['SanitiseTransformation']


class SanitiseTransformation(Transformation):
    """
    :any:`Transformation` object to apply several code sanitisation
    steps when batch-processing large source trees via the :any:`Scheduler`.

    Parameters
    ----------
    resolve_associate_mappings : bool
        Resolve ASSOCIATE mappings in body of processed subroutines; default: True.
    resolve_sequence_association : bool
        Replace scalars that are passed to array arguments with array
        ranges; default: False.
    """

    def __init__(
            self, resolve_associate_mappings=True, resolve_sequence_association=False
    ):
        self.resolve_associate_mappings = resolve_associate_mappings
        self.resolve_sequence_association = resolve_sequence_association

    def transform_subroutine(self, routine, **kwargs):

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        if self.resolve_associate_mappings:
            do_resolve_associates(routine)

        # Transform arrays passed with scalar syntax to array syntax
        if self.resolve_sequence_association:
            do_resolve_sequence_association(routine)
