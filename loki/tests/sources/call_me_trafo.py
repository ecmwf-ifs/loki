# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import Transformation, FindNodes, CallStatement


class CallMeMaybeTrafo(Transformation):
    """ Test transformation for dynamically loading remote transformations. """

    def __init__(self, name='Dave', horizontal=None):
        self.name = name
        self.horizontal = horizontal

    def transform_subroutine(self, routine, **kwargs):
        for call in FindNodes(CallStatement).visit(routine.body):
            call._update(name=self.name)
