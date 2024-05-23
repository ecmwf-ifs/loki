# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from importlib.metadata import version, PackageNotFoundError

from transformations.derived_types import * # noqa
from transformations.argument_shape import * # noqa
from transformations.data_offload import * # noqa
from transformations.single_column_annotate import * # noqa
from transformations.single_column_base import * # noqa
from transformations.single_column_claw import * # noqa
from transformations.single_column_coalesced import * # noqa
from transformations.single_column_coalesced_vector import * # noqa
from transformations.single_column_hoist import * # noqa
from transformations.drhook import * # noqa
from transformations.single_column_coalesced_extended import * # noqa
# from transformations.utility_routines import * # noqa
from transformations.scc_cuf import * # noqa
from transformations.pool_allocator import * # noqa
from transformations.block_index_transformations import * # noqa
from transformations.scc_low_level import * # noqa

try:
    __version__ = version("transformations")
except PackageNotFoundError:
    # package is not installed
    pass
