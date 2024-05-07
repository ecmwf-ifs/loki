# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki.transform import resolve_associates
from loki import (
    ir, Transformation, FindNodes, Transformer,
    as_tuple, FindExpressions,
    SymbolAttributes, BasicType, SubstituteExpressions,
    FindVariables
)


__all__ = ['SCCBaseTransformation']


class SCCBaseTransformation(Transformation):
    """
    A basic set of utilities used in the SCC transformation. These utilities
    can either be used as a transformation in their own right, or the contained
    class methods can be called directly.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    directive : string or None
        Directives flavour to use for parallelism annotations; either
        ``'openacc'`` or ``None``.
    """

    def __init__(self, horizontal, directive=None):
        self.horizontal = horizontal

        assert directive in [None, 'openacc']
        self.directive = directive

    # TODO: correct "definition" of a pure/elemental routine (take e.g. loki serial into account ...)
    @staticmethod
    def is_elemental(routine):
        """
        Check whether :any:`Subroutine` ``routine`` is an elemental routine.
        Need for distinguishing elemental and non-elemental function to transform
        those in a different way.
        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to check whether elemental
        """
        for prefix in routine.prefix:
            if prefix.lower() == 'elemental':
                return True
        return False

    @staticmethod
    def check_array_dimensions_in_calls(routine):
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        for call in calls:
            for arg in call.arguments:
                if isinstance(arg, sym.Array):
                    if any(dim == sym.RangeIndex((None, None)) for dim in arg.dimensions):
                        return False
        return True

    @staticmethod
    def remove_dimensions(routine, calls_only=False):
        if calls_only:
            calls = FindNodes(ir.CallStatement).visit(routine.body)
            for call in calls:
                arguments = ()
                for arg in call.arguments:
                    if isinstance(arg, sym.Array):
                        if all(dim == sym.RangeIndex((None, None)) for dim in arg.dimensions):
                            new_dimensions = None
                            arguments += (arg.clone(dimensions=new_dimensions),)
                        else:
                            arguments += (arg,)
                    else:
                        arguments += (arg,)
                call._update(arguments=arguments)

        else:
            arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
            array_map = {}
            for array in arrays:
                if all(dim == sym.RangeIndex((None, None)) for dim in array.dimensions):
                    new_dimensions = None
                    array_map[array] = array.clone(dimensions=new_dimensions)
            routine.body = SubstituteExpressions(array_map).visit(routine.body)

    @staticmethod
    def explicit_dimensions(routine):
        arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
        array_map = {}
        for array in arrays:
            if not array.dimensions:
                new_dimensions = (sym.RangeIndex((None, None)),) * len(array.shape)
                array_map[array] = array.clone(dimensions=new_dimensions)
        routine.body = SubstituteExpressions(array_map).visit(routine.body)

    @classmethod
    def check_routine_pragmas(cls, routine, directive):
        """
        Check if routine is marked as sequential or has already been processed.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to perform checks on.
        directive: string or None
            Directives flavour to use for parallelism annotations; either
            ``'openacc'`` or ``None``.
        """

        pragmas = FindNodes(ir.Pragma).visit(routine.ir)
        routine_pragmas = [p for p in pragmas if p.keyword.lower() in ['loki', 'acc']]
        routine_pragmas = [p for p in routine_pragmas if 'routine' in p.content.lower()]

        seq_pragmas = [r for r in routine_pragmas if 'seq' in r.content.lower()]
        if seq_pragmas:
            loki_seq_pragmas = [r for r in routine_pragmas if 'loki' == r.keyword.lower()]
            if loki_seq_pragmas:
                if directive == 'openacc':
                    # Mark routine as acc seq
                    mapper = {seq_pragmas[0]: None}
                    routine.spec = Transformer(mapper).visit(routine.spec)
                    routine.body = Transformer(mapper).visit(routine.body)

                    # Append the acc pragma to routine.spec, regardless of where the corresponding
                    # loki pragma is found
                    routine.spec.append(ir.Pragma(keyword='acc', content='routine seq'))
            return True

        vec_pragmas = [r for r in routine_pragmas if 'vector' in r.content.lower()]
        if vec_pragmas:
            if directive == 'openacc':
                return True

        return False

    @classmethod
    def check_horizontal_var(cls, routine, horizontal):
        """
        Check for horizontal loop bounds in a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to perform checks on.
        horizontal : :any:`Dimension`
            :any:`Dimension` object describing the variable conventions used in code
            to define the horizontal data dimension and iteration space.
        """

        if horizontal.bounds[0] not in routine.variable_map:
            raise RuntimeError(f'No horizontal start variable found in {routine.name}')
        if horizontal.bounds[1] not in routine.variable_map:
            raise RuntimeError(f'No horizontal end variable found in {routine.name}')

    @classmethod
    def get_horizontal_loop_bounds(cls, routine, horizontal):
        """
        Check for horizontal loop bounds in a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to perform checks on.
        horizontal : :any:`Dimension`
            :any:`Dimension` object describing the variable conventions used in code
            to define the horizontal data dimension and iteration space.
        """

        bounds = ()
        variables = routine.variables
        for name, _bounds in zip(['start', 'end'], horizontal.bounds_expressions):
            for bound in _bounds:
                if bound.split('%', maxsplit=1)[0] in variables:
                    bounds += (bound,)
                    break
            else:
                raise RuntimeError(f'No horizontol {name} variable matching {_bounds[0]} found in {routine.name}')

        return bounds

    @classmethod
    def get_integer_variable(cls, routine, name):
        """
        Find a local variable in the routine, or create an integer-typed one.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in which to find the variable
        name : string
            Name of the variable to find the in the routine.
        """
        if not (v_index := routine.symbol_map.get(name, None)):
            dtype = SymbolAttributes(BasicType.INTEGER)
            v_index = sym.Variable(name=name, type=dtype, scope=routine)
        return v_index

    @classmethod
    def resolve_masked_stmts(cls, routine, loop_variable):
        """
        Resolve :any:`MaskedStatement` (WHERE statement) objects to an
        explicit combination of :any:`Loop` and :any:`Conditional` combination.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in which to resolve masked statements
        loop_variable : :any:`Scalar`
            The induction variable for the created loops.
        """
        mapper = {}
        for masked in FindNodes(ir.MaskedStatement).visit(routine.body):
            # TODO: Currently limited to simple, single-clause WHERE stmts
            assert len(masked.conditions) == 1 and len(masked.bodies) == 1
            ranges = [e for e in FindExpressions().visit(masked.conditions[0]) if isinstance(e, sym.RangeIndex)]
            exprmap = {r: loop_variable for r in ranges}
            assert len(ranges) > 0
            assert all(r == ranges[0] for r in ranges)
            bounds = sym.LoopRange((ranges[0].start, ranges[0].stop, ranges[0].step))
            cond = ir.Conditional(condition=masked.conditions[0], body=masked.bodies[0], else_body=masked.default)
            loop = ir.Loop(variable=loop_variable, bounds=bounds, body=(cond,))
            # Substitute the loop ranges with the loop index and add to mapper
            mapper[masked] = SubstituteExpressions(exprmap).visit(loop)

        routine.body = Transformer(mapper).visit(routine.body)

        # if loops have been inserted, check if loop variable is declared
        if mapper and loop_variable not in routine.variables:
            routine.variables += as_tuple(loop_variable)

    @classmethod
    def resolve_vector_dimension(cls, routine, loop_variable, bounds):
        """
        Resolve vector notation for a given dimension only. The dimension
        is defined by a loop variable and the bounds of the given range.

        TODO: Consolidate this with the internal
        `loki.transform.transform_array_indexing.resolve_vector_notation`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in which to resolve vector notation usage.
        loop_variable : :any:`Scalar`
            The induction variable for the created loops.
        bounds : tuple of :any:`Scalar`
            Tuple defining the iteration space of the inserted loops.
        """
        bounds_str = f'{bounds[0]}:{bounds[1]}'

        bounds_v = (sym.Variable(name=bounds[0]), sym.Variable(name=bounds[1]))

        mapper = {}
        for stmt in FindNodes(ir.Assignment).visit(routine.body):
            ranges = [e for e in FindExpressions().visit(stmt)
                      if isinstance(e, sym.RangeIndex) and e == bounds_str]
            if ranges:
                exprmap = {r: loop_variable for r in ranges}
                loop = ir.Loop(
                    variable=loop_variable, bounds=sym.LoopRange(bounds_v),
                    body=as_tuple(SubstituteExpressions(exprmap).visit(stmt))
                )
                mapper[stmt] = loop

        routine.body = Transformer(mapper).visit(routine.body)

        # if loops have been inserted, check if loop variable is declared
        if mapper and loop_variable not in routine.variables:
            routine.variables += as_tuple(loop_variable)

    @staticmethod
    def is_driver_loop(loop, targets):
        """
        Test/check whether a given loop is a *driver loop*.

        Parameters
        ----------
        loop : :any: `Loop`
            The loop to test if it is a *driver loop*.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """
        if loop.pragma:
            for pragma in loop.pragma:
                if pragma.keyword.lower() == "loki" and pragma.content.lower() == "driver-loop":
                    return True
        for call in FindNodes(ir.CallStatement).visit(loop.body):
            if call.name in targets:
                return True
        return False

    @classmethod
    def find_driver_loops(cls, routine, targets):
        """
        Find and return all driver loops of a given `routine`.

        A *driver loop* is specified either by a call to a routine within
        `targets` or by the pragma `!$loki driver-loop`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in which to find the driver loops.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """

        driver_loops = []
        nested_driver_loops = []
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if loop in nested_driver_loops:
                continue

            if not cls.is_driver_loop(loop, targets):
                continue

            driver_loops.append(loop)
            loops = FindNodes(ir.Loop).visit(loop.body)
            nested_driver_loops.extend(loops)
        return driver_loops

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCBase utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']

        if role == 'kernel':
            self.process_kernel(routine)
        if role == 'driver':
            self.process_driver(routine)

    def process_kernel(self, routine):
        """
        Applies the SCCBase utilities to a "kernel". This consists simply
        of resolving associations, masked statements and vector notation.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Bail if routine is marked as sequential or routine has already been processed
        if self.check_routine_pragmas(routine, self.directive):
            return

        # check for horizontal loop bounds in subroutine symbol table
        self.check_horizontal_var(routine, self.horizontal)

        # Find the iteration index variable for the specified horizontal
        v_index = self.get_integer_variable(routine, name=self.horizontal.index)

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        resolve_associates(routine)

        # Resolve WHERE clauses
        self.resolve_masked_stmts(routine, loop_variable=v_index)

        # Resolve vector notation, eg. VARIABLE(KIDIA:KFDIA)
        self.resolve_vector_dimension(routine, loop_variable=v_index, bounds=self.horizontal.bounds)

    def process_driver(self, routine):
        """
        Applies the SCCBase utilities to a "driver". This consists simply
        of resolving associations.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Resolve associates, since the PGI compiler cannot deal with
        # implicit derived type component offload by calling device
        # routines.
        resolve_associates(routine)
