# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation


class IdemTransformation(Transformation):
    """
    A custom transformation that does absolutely nothing!

    This can be used to test simple parse-unparse cycles.
    """

    def transform_subroutine(self, routine, **kwargs):
        pass
