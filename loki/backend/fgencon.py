# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.backend.fgen import FortranCodegen
from loki.frontend.source import SourceStatus


__all__ = ['FortranCodegenConservative']


class FortranCodegenConservative(FortranCodegen):
    """
    Strictly conservative version of :any:`FortranCodegen` visitor
    that will attempt to use existing :any:`Source` information from
    the frontends where possible.
    """

    def visit(self, o, *args, **kwargs):
        """
        Overwrite standard visit routine to inject original source in conservative mode.
        """
        if hasattr(o, 'source') and o.source and o.source.status == SourceStatus.VALID:
            # Re-use original source associated with node
            return o.source.string
        return super().visit(o, *args, **kwargs)
