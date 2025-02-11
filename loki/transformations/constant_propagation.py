# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.analyse.constant_propagation_analysis import ConstantPropagationAnalysis
from loki import Transformer, Subroutine

__all__ = ['ConstantPropagationTransformer']

class ConstantPropagationTransformer(Transformer):

    def __init__(self, fold_floats=True, unroll_loops=True):
        self.fold_floats = fold_floats
        self.unroll_loops = unroll_loops
        super().__init__()

    def visit(self, expr, *args, **kwargs):
        const_prop = ConstantPropagationAnalysis(self.fold_floats, self.unroll_loops, True)
        constants_map = kwargs.get('constants_map', dict())
        try:
            declarations_map = const_prop.generate_declarations_map(expr)
            # If a user specifies their own map, they probably want it to override these
            declarations_map.update(constants_map)
            constants_map = declarations_map
        except AttributeError:
            pass

        is_routine = isinstance(expr, Subroutine)
        target = expr.body if is_routine else expr

        target = const_prop.get_attacher().visit(target, constants_map=constants_map)
        target = const_prop.get_detacher().visit(target)

        if is_routine:
            expr.body = target
            return expr

        return target