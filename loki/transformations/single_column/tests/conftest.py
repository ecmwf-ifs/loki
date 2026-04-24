# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Shared test helpers for vertical loop transformation tests.
"""

from loki.ir import FindNodes, Loop


def _count_jk_loops(routine):
    """Count all DO JK=... loops in the routine."""
    return sum(1 for l in FindNodes(Loop).visit(routine.body)
               if l.variable.name.lower() == 'jk')


def _find_jk_loops(routine):
    """Return all DO JK=... loops in the routine, in source order."""
    return [l for l in FindNodes(Loop).visit(routine.body)
            if l.variable.name.lower() == 'jk']
