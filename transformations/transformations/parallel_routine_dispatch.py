# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.transform import Transformation

__all__ = ['ParallelRoutineDispatchTransformation']


class ParallelRoutineDispatchTransformation(Transformation):

    def __init__(self):
        self.dummy_return_value = []

    def transform_subroutine(self, routine, **kwargs):
        self.dummy_return_value += [routine.name.lower()]
