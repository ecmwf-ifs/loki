# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import operator as _op
from loki import (
     FindNodes, CallStatement, Assignment, Scalar, RangeIndex,
     simplify, Sum, Product, IntLiteral, as_tuple, SubstituteExpressions,
     symbolic_op, StringLiteral, is_constant, LogicLiteral, VariableDeclaration, flatten
)
from loki.lint import GenericRule, RuleType

class ArgSizeMismatchRule(GenericRule):
    """
    Rule to check for argument size mismatch in subroutine/function calls
    """

    type = RuleType.WARN

    @staticmethod
    def range_to_sum(lower, upper):
        """
        Method to convert lower and upper bounds of a :any:`RangeIndex` to a
        :any:`Sum` expression.
        """

        return Sum((IntLiteral(1), upper, Product((IntLiteral(-1), lower))))

    @classmethod
    def get_explicit_arg_size(cls, arg, dims):
        """
        Method to return the size of a subroutine argument whose bounds are
        explicitly declared.
        """

        if isinstance(arg, Scalar):
            size = as_tuple(IntLiteral(1))
        else:
            size = ()
            for dim in dims:
                if isinstance(dim, RangeIndex):
                    size += as_tuple(simplify(cls.range_to_sum(dim.lower, dim.upper)))
                else:
                    size += as_tuple(dim)

        return size

    @classmethod
    def get_implicit_arg_size(cls, arg, dims):
        """
        Method to return the size of a subroutine argument whose bounds are
        potentially implicitly declared.
        """

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

                size += as_tuple(cls.range_to_sum(lower, upper))
            else:
                size += as_tuple(dim)

        return size

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
        """
        Method to check for argument size mismatches across subroutine calls.
        It requires all :any:`CallStatement` nodes to be enriched, and requires
        all subroutine arguments *to not be* of type :any:`DeferredTypeSymbol`.
        Therefore relevant modules should be parsed before parsing the current
        :any:`Subroutine`.
        """

        assign_map = {a.lhs: a.rhs for a in FindNodes(Assignment).visit(subroutine.body)}
        decl_symbols = flatten([decl.symbols for decl in FindNodes(VariableDeclaration).visit(subroutine.spec)])
        decl_symbols = [sym for sym in decl_symbols if sym.type.initial]
        assign_map.update({sym: sym.initial for sym in decl_symbols})

        targets = as_tuple(kwargs.get('targets', None))
        calls = [c for c in FindNodes(CallStatement).visit(subroutine.body) if c.name in targets]

        for call in calls:

            # check if calls are enriched
            if not call.routine:
                continue

            arg_map_f = {carg: rarg for rarg, carg in call.arg_iter()}
            arg_map_r = dict(call.arg_iter())

            # combine args and kwargs into single iterable
            arguments = call.arguments
            arguments += as_tuple([arg for kw, arg in call.kwarguments])

            for arg in arguments:

                if isinstance(arg_map_f[arg], Scalar):
                    dummy_arg_size = as_tuple(IntLiteral(1))
                else:
                    # we can't proceed if dummy arg has assumed shape component
                    if any(None in (dim.lower, dim.upper)
                           for dim in arg_map_f[arg].shape if isinstance(dim, RangeIndex)):
                        continue
                    dummy_arg_size = cls.get_explicit_arg_size(arg_map_f[arg], arg_map_f[arg].shape)

                dummy_arg_size = SubstituteExpressions(arg_map_r).visit(dummy_arg_size)
                dummy_arg_size = SubstituteExpressions(assign_map).visit(dummy_arg_size)
                dummy_arg_size = Product(dummy_arg_size)

                # TODO: skip string literal args
                if isinstance(arg, StringLiteral):
                    continue

                arg_size = ()
                alt_arg_size = ()
                # check if argument is scalar
                if isinstance(arg, (Scalar, LogicLiteral)) or is_constant(arg):
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

                        arg_size = cls.get_explicit_arg_size(arg, arg.dimensions)
                        alt_arg_size = arg_size
                    else:
                        # compute dim sizes assuming single element
                        if arg.dimensions:
                            arg_size = cls.get_implicit_arg_size(arg, arg.dimensions)
                            arg_size = as_tuple([IntLiteral(1) if not isinstance(a, Sum) else simplify(a)
                                                 for a in arg_size])
                        else:
                            arg_size = cls.get_explicit_arg_size(arg, arg.shape)

                        # compute dim sizes assuming array sequence
                        alt_arg_size = cls.get_implicit_arg_size(arg, arg.dimensions)
                        ubounds = [dim.upper if isinstance(dim, RangeIndex) else dim for dim in arg.shape]
                        alt_arg_size = as_tuple([simplify(Sum((Product((IntLiteral(-1), a)),
                                                               ubounds[i], IntLiteral(1))))
                                                 if not isinstance(a, Sum) else simplify(a)
                                                 for i, a in enumerate(alt_arg_size)])
                        alt_arg_size += cls.get_explicit_arg_size(arg, arg.shape[len(arg.dimensions):])

                stat = False
                for i in range(len(arg_size) + 1):
                    dims = tuple(alt_arg_size[:i])
                    dims += tuple(arg_size[i:])

                    dims = Product(dims)

                    if symbolic_op(dims, _op.eq, dummy_arg_size):
                        stat = True
                        break

                if not stat:
                    msg = f'Size mismatch:: arg: {arg}, dummy_arg: {arg_map_f[arg]} '
                    msg += f'in {call} in {subroutine}'
                    rule_report.add(msg, call)

# Create the __all__ property of the module to contain only the rule names
__all__ = tuple(name for name in dir() if name.endswith('Rule') and name != 'GenericRule')
