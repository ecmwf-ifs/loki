# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import deepcopy

from loki import Subroutine, Transformer
from loki.analyse.constant_propagation_analysis import ConstantPropagationAnalysis

__all__ = ['ConstantPropagationTransformer']


class ConstantPropagationTransformer(Transformer):
    """Apply constant-propagation analysis as a transformation driver."""

    def __init__(self, fold_floats=True, unroll_loops=True):
        self.fold_floats = fold_floats
        self.unroll_loops = unroll_loops
        super().__init__()

    def visit(self, expr, *args, **kwargs):
        constants_map = deepcopy(kwargs.pop('constants_map', {}))
        const_prop = ConstantPropagationAnalysis(
            fold_floats=self.fold_floats,
            unroll_loops=self.unroll_loops,
            apply_transform=True,
        )

        if isinstance(expr, Subroutine):
            declarations_map = const_prop.generate_declarations_map(expr)
            declarations_map.update(constants_map)
            attacher = const_prop.get_attacher()
            detacher = const_prop.get_detacher()
            if expr.spec:
                expr.spec = attacher.visit(expr.spec, *args, constants_map=declarations_map, **kwargs)
                detacher.visit(expr.spec)
            if expr.body:
                expr.body = attacher.visit(expr.body, *args, constants_map=declarations_map, **kwargs)
                detacher.visit(expr.body)
            return expr

        target = const_prop.get_attacher().visit(expr, *args, constants_map=constants_map, **kwargs)
        target = const_prop.get_detacher().visit(target)
        return target
