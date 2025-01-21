# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
Transformations sub-package that provides various forms of
source-level code inlining.

The various inline mechanisms are provided as standalone utility methods,
or via the :any:`InlineTransformation` class for for batch processing.
"""

from loki.transformations.inline.constants import * # noqa
from loki.transformations.inline.functions import * # noqa
from loki.transformations.inline.mapper import * # noqa
from loki.transformations.inline.procedures import * # noqa
from loki.transformations.inline.transformation import * # noqa
