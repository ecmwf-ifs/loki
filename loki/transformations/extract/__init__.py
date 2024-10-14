# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
Transformations sub-package that provides various forms of
source-code extraction into standalone :any:`Subroutine` objects.

The various extractions mechanisms are provided as standalone utility
methods, or via the :any:`ExtractTransformation` class for for batch
processing.

These utilities represent the conceptual inverse operation to
"inlining", as done by the :any:`InlineTransformation`.
"""

from loki.transformations.extract.internal import * # noqa
from loki.transformations.extract.outline import * # noqa

from loki.batch import Transformation


__all__ = ['ExtractTransformation']


class ExtractTransformation(Transformation):
    """
    :any:`Transformation` class to apply several types of source
    extraction when batch-processing large source trees via the
    :any:`Scheduler`.

    Parameters
    ----------
    extract_internals : bool
        Extract internal procedure (see :any:`extract_internal_procedures`);
        default: False.
    outline_regions : bool
        Outline pragma-annotated code regions to :any:`Subroutine` objects.
        (see :any:`outline_pragma_regions`); default: True.
    """
    def __init__(self, extract_internals=False, outline_regions=True):
        self.extract_internals = extract_internals
        self.outline_regions = outline_regions

    def transform_module(self, module, **kwargs):
        """
        Extract internals procedures and marked subroutines and add
        them to the given :any:`Module`.
        """

        # Extract internal (contained) procedures into standalone ones
        if self.extract_internals:
            for routine in module.subroutines:
                new_routines = extract_internal_procedures(routine)
                module.contains.append(new_routines)

        # Extract pragma-marked code regions into standalone subroutines
        if self.outline_regions:
            for routine in module.subroutines:
                new_routines = outline_pragma_regions(routine)
                module.contains.append(new_routines)

    def transform_file(self, sourcefile, **kwargs):
        """
        Extract internals procedures and marked subroutines and add
        them to the given :any:`Sourcefile`.
        """

        # Extract internal (contained) procedures into standalone ones
        if self.extract_internals:
            for routine in sourcefile.subroutines:
                new_routines = extract_internal_procedures(routine)
                sourcefile.ir.append(new_routines)

        # Extract pragma-marked code regions into standalone subroutines
        if self.outline_regions:
            for routine in sourcefile.subroutines:
                new_routines = outline_pragma_regions(routine)
                sourcefile.ir.append(new_routines)
