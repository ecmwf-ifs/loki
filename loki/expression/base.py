# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
Base class of all expression tree nodes.

pymbolic 2024.1 renamed ``Expression`` to ``ExpressionNode`` and keeps
``Expression`` only as a deprecated alias. :data:`ExpressionNode` resolves to
whichever of the two the installed pymbolic version provides.
"""

import pymbolic.primitives as pmbl


__all__ = ['ExpressionNode']


if hasattr(pmbl, 'ExpressionNode'):
    ExpressionNode = pmbl.ExpressionNode
else:
    ExpressionNode = pmbl.Expression
