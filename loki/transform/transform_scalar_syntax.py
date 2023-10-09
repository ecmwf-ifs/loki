# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import (
    Product, IntLiteral
    )
from loki.ir import CallStatement
from loki.visitors import FindNodes
from loki import Array, RangeIndex


__all__ = [
    'fix_scalar_syntax'
]


def fix_scalar_syntax(routine):
    """
    Housekeeping routine to replace scalar syntax when passing arrays as arguments
    For example, a call like

    call myroutine(a(i,j))

    where myroutine looks like

    subroutine myroutine(a)
        real :: a(5)
    end subroutine myroutine

    should be changed to

    call myroutine(a(i:i+5,j)


    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine where calls will be changed
    """

    #Define minus one for later
    minus_one = Product((-1, IntLiteral(1)))

    calls = FindNodes(CallStatement).visit(routine.body)

    for call in calls:

        arg_map = {}


        for dummy, arg in call.arg_map.items():
            if isinstance(arg, Array) and isinstance(dummy, Array):
                if arg.dimensions:
                    n_dummy_ranges = sum(1 for d in arg.dimensions if isinstance(d, RangeIndex))
                    if n_dummy_ranges == 0:
                        print(call)
                        print(arg, dummy)
                        for s in dummy.shape:
                            print(s, s.__class__)
                        print()







