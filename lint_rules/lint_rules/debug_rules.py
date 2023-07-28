# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import operator as _op
from loki import (
     FindNodes, CallStatement, Assignment, Scalar, RangeIndex, resolve_associates,
     simplify, Sum, Product, IntLiteral, as_tuple, SubstituteExpressions, Array,
     symbolic_op, StringLiteral, is_constant, LogicLiteral, VariableDeclaration, flatten,
     FindInlineCalls, Conditional, FindExpressions, Comparison
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

    @staticmethod
    def compare_sizes(arg_size, alt_arg_size, dummy_arg_size):
        """
        Compare all possible argument size candidates with dummy arg size.
        """
        for i in range(len(arg_size) + 1):
            dims = tuple(alt_arg_size[:i])
            dims += tuple(arg_size[i:])

            dims = Product(dims)

            if symbolic_op(dims, _op.eq, dummy_arg_size):
                return True

        return False

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

        # first resolve associates
        resolve_associates(subroutine)

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

            arg_map = {carg: rarg for rarg, carg in call.arg_iter()}
            for arg in arg_map:

                if isinstance(arg_map[arg], Scalar):
                    dummy_arg_size = as_tuple(IntLiteral(1))
                else:
                    # we can't proceed if dummy arg has assumed shape component
                    if any(None in (dim.lower, dim.upper)
                           for dim in arg_map[arg].shape if isinstance(dim, RangeIndex)):
                        continue
                    dummy_arg_size = cls.get_explicit_arg_size(arg_map[arg], arg_map[arg].shape)
                    dummy_arg_size = SubstituteExpressions(dict(call.arg_iter())).visit(dummy_arg_size)

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

                # first check using unmodified dimension names
                dummy_size = Product(dummy_arg_size)
                stat = cls.compare_sizes(arg_size, alt_arg_size, dummy_size)

                # if necessary, update dimension names and check
                if not stat:
                    dummy_size = Product(SubstituteExpressions(assign_map).visit(dummy_arg_size))
                    stat = cls.compare_sizes(arg_size, alt_arg_size, dummy_size)

                if not stat:
                    msg = f'Size mismatch:: arg: {arg}, dummy_arg: {arg_map[arg]} '
                    msg += f'in {call} in {subroutine}'
                    rule_report.add(msg, call)

class DynamicUboundCheckRule(GenericRule):
    """
    Rule to check for run-time ubound checks for assumed shape dummy arguments
    """

    type = RuleType.WARN
    fixable = True

    @staticmethod
    def is_assumed_shape(arg):
        """
        Method to check if argument is an assumed shape array.
        """

        if all(isinstance(dim, RangeIndex) for dim in arg.shape):
            return all(dim.upper is None and dim.lower is None for dim in arg.shape)
        return False

    @staticmethod
    def get_ubound_checks(subroutine):
        """
        Method to return UBOUND checks nested within a :any:`Conditional`.
        """

        cond_map = {cond: FindInlineCalls(unique=False).visit(cond.condition)
                    for cond in FindNodes(Conditional).visit(subroutine.body)}
        return {call: cond for cond, calls in cond_map.items() for call in calls}

    @classmethod
    def get_assumed_shape_args(cls, subroutine):
        """
        Method to return all assumed-shape dummy arguments in a :any:`Subroutine`.
        """
        args = [arg for arg in subroutine.arguments if isinstance(arg, Array)]
        return [arg for arg in args if cls.is_assumed_shape(arg)]

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
        """
        Method to check for run-time ubound checks for assumed shape dummy arguments
        """

        ubound_checks = cls.get_ubound_checks(subroutine)
        args = cls.get_assumed_shape_args(subroutine)

        for arg in args:
            checks = [c for c in ubound_checks if arg.name in c.arguments]
            params = flatten([p for c in checks for p in c.arguments if not p == arg])
            if all(IntLiteral(d+1) in params for d in range(len(arg.shape))):
                msg = f'Run-time UBOUND checks for assumed-shape arg: {arg}'
                rule_report.add(msg, subroutine)

    @classmethod
    def fix_subroutine(cls, subroutine, rule_report, config):
        """
        Method to fix run-time ubound checks for assumed shape dummy arguments
        """

        ubound_checks = cls.get_ubound_checks(subroutine)
        args = cls.get_assumed_shape_args(subroutine)

        new_vars = ()
        node_map = {}

        for arg in args:
            checks = [c for c in ubound_checks if arg.name in c.arguments]
            params = {p: c for c in checks for p in c.arguments if not p == arg}

            # check if ubounds of all dimensions are tested
            if all(IntLiteral(d+1) in params for d in range(len(arg.shape))):
                new_shape = ()
                for d in range(len(arg.shape)):
                    conditional = ubound_checks[params[IntLiteral(d+1)]]
                    node_map[conditional] = None

                    # extract comparison expressions in case they are nested in a logical operation
                    conditions = [c for c in FindExpressions().visit(conditional.condition)
                                  if isinstance(c, Comparison)]
                    conditions = [c for c in conditions if c.operator in ('<', '>')]

                    cond = [c for c in conditions if arg.name in c and IntLiteral(d+1) in c][0]

                    # build ordered tuple for declaration shape
                    if 'ubound' in FindExpressions().visit(cond.left):
                        new_shape += as_tuple(cond.right)
                    else:
                        new_shape += as_tuple(cond.left)

                vtype = arg.type.clone(shape=new_shape, scope=subroutine)
                new_vars += as_tuple(arg.clone(type=vtype, dimensions=new_shape, scope=subroutine))

        #TODO: add 'VariableDeclaration.symbols' should be of type 'Variable' rather than 'Expression'
        # to enable case-insensitive search here
        new_var_names = [v.name.lower() for v in new_vars]
        subroutine.variables = [var for var in subroutine.variables if not var.name.lower() in new_var_names]
        subroutine.variables += new_vars

        return node_map

# Create the __all__ property of the module to contain only the rule names
__all__ = tuple(name for name in dir() if name.endswith('Rule') and name != 'GenericRule')
