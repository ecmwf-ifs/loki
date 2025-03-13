# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.tools import flatten

__all__ = ['Binary']


class Binary:
    """
    A binary build target to generate executables.
    """

    def __init__(self, name, objs=None, libs=None):
        self.name = name
        self.objs = objs or []
        self.libs = libs or []

    def build(self, builder):

        # Trigger build for object dependencies
        for obj in flatten(self.objs):
            obj.build(builder=builder)

        # Trigger build for library dependencies
        for lib in flatten(self.libs):
            lib.build(builder=builder)

        # TODO: Link the final binary
