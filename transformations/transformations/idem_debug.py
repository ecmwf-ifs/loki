# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pdb
import operator as _op
from loki import (
     Transformation, FindNodes, CallStatement, Assignment, Scalar, RangeIndex,
     simplify, Sum, Product, IntLiteral, DeferredTypeSymbol, as_tuple, SubstituteExpressions,
     symbolic_op, warning
)

__all__ = ['IdemDebugTransformation']

class IdemDebugTransformation(Transformation):

    @staticmethod
    def _resolve_range(dim, arg, count):
        if not dim.upper:
            if isinstance(arg.shape[count], RangeIndex):
                upper = arg.shape[count].upper
            else:
                upper = arg.shape[count]
        else:
            upper = dim.upper

        if not dim.lower:
            if isinstance(arg.shape[count], RangeIndex):
                lower = arg.shape[count].lower
            else:
                lower = IntLiteral(1)
        else:
            lower = dim.lower

        return lower, upper

    @staticmethod
    def range_to_sum(lower, upper):
        return Sum((IntLiteral(1), upper, Product((IntLiteral(-1), lower))))

    def get_explicit_arg_size(self, arg, dims):
        if isinstance(arg, Scalar):
            size = as_tuple(IntLiteral(1))
        else:
            size = ()
            for dim in dims:
                if isinstance(dim, RangeIndex):
                    size += as_tuple(simplify(self.range_to_sum(dim.lower, dim.upper)))
                else:
                    size += as_tuple(dim)

        return size

    def get_implicit_arg_size(self, arg, dims):
        size = ()
        for count, dim in enumerate(dims):
            if isinstance(dim, RangeIndex):
                if not dim.upper:
                    if isinstance(arg.shape[count], RangeIndex):
                        upper = arg.shape[count].upper
                    else:
                        upper = arg.shape[count]
                else:
                    upper = dim.upper
                if not dim.lower:
                    if isinstance(arg.shape[count], RangeIndex):
                        lower = arg.shape[count].lower
                    else:
                        lower = IntLiteral(1)
                else:
                    lower = dim.lower

                size += as_tuple(self.range_to_sum(lower, upper))
            else:
                size += as_tuple(dim)

        return size

    def argument_size_mismatch(self, routine, **kwargs):

        assign_map = {a.lhs: a.rhs for a in FindNodes(Assignment).visit(routine.body)}

        targets = as_tuple(kwargs.get('targets', None))
        calls = FindNodes(CallStatement).visit(routine.body)
        calls = [c for c in calls if c.name.name.lower() in targets]

        for call in calls:

            # check if calls are enriched
            if not call.routine:
                continue

            arg_map_f = {carg: rarg for rarg, carg in call.arg_iter()}
            arg_map_r = {rarg: carg for rarg, carg in call.arg_iter()}
            for arg in call.arguments:

                # we can't proceed if dummy arg has assumed shape component
                if isinstance(arg_map_f[arg], Scalar):
                    dummy_arg_size = as_tuple(IntLiteral(1))
                else:
                    if any(None in (dim.lower, dim.upper)
                           for dim in arg_map_f[arg].shape if isinstance(dim, RangeIndex)):
                        continue
                    dummy_arg_size = self.get_explicit_arg_size(arg_map_f[arg], arg_map_f[arg].shape)

                dummy_arg_size = SubstituteExpressions(arg_map_r).visit(dummy_arg_size)
                dummy_arg_size = SubstituteExpressions(assign_map).visit(dummy_arg_size)
                dummy_arg_size = Product(dummy_arg_size)


                arg_size = ()
                alt_arg_size = ()
                # check if argument is scalar
                if isinstance(arg, Scalar):
                    arg_size += as_tuple(IntLiteral(1))
                    alt_arg_size += as_tuple(IntLiteral(1))
                else:
                    # check if arg has assumed size component
                    if any(None in (dim.lower, dim.upper)
                           for dim in arg.shape if isinstance(dim, RangeIndex)):

                        # each dim must have explicit range-index to be sure of arg size
                        if not arg.dimensions:
                            continue
                        if not all(isinstance(dim, RangeIndex) for dim in arg.dimensions):
                            continue
                        if any(None in (dim.lower, dim.upper) for dim in arg.dimensions):
                            continue

                        arg_size = self.get_explicit_arg_size(arg, arg.dimensions)
                        alt_arg_size = arg_size
                    else:
                        # compute dim sizes assuming single element
                        if arg.dimensions:
                            arg_size = self.get_implicit_arg_size(arg, arg.dimensions)
                            arg_size = as_tuple([IntLiteral(1) if not isinstance(a, Sum) else simplify(a)
                                                 for a in arg_size])
                        else:
                            arg_size = self.get_explicit_arg_size(arg, arg.shape)

                        # compute dim sizes assuming element reference
                        alt_arg_size = self.get_implicit_arg_size(arg, arg.dimensions)
                        alt_arg_size = as_tuple([simplify(Sum((Product((IntLiteral(-1), a)),
                                                               a.shape[i], IntLiteral(1))))
                                                 if not isinstance(a, Sum) else simplify(a)
                                                 for i, a in enumerate(alt_arg_size)])
                        alt_arg_size += self.get_explicit_arg_size(arg, arg.shape[len(arg.dimensions):])

                stat = False
                for i in range(len(arg_size)):
                    dims = ()
                    for a in alt_arg_size[:i]:
                        dims += as_tuple(a)
                    for a in arg_size[i:]:
                        dims += as_tuple(a)

                    dims = Product(dims)

                    if symbolic_op(dims, _op.eq, dummy_arg_size):
                        stat = True

                if not stat:
                    warn = f'[Loki::IdemDebug] Size mismatch:: arg: {arg}, dummy_arg: {arg_map_f[arg]} '
                    warn += f'in {call} in {routine}'
                    warning(warn)

    def transform_subroutine(self, routine, **kwargs):

        # check for argument size mismatch across subroutine calls
        self.argument_size_mismatch(routine, **kwargs)
