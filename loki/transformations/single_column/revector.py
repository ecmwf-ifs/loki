# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

from loki.batch import Transformation
from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, Transformer, is_loki_pragma,
    pragmas_attached, pragma_regions_attached
)
from loki.tools import as_tuple

from loki.transformations.utilities import (
    get_integer_variable, get_loop_bounds, find_driver_loops,
    check_routine_sequential, single_variable_declaration
)


__all__ = [
    'SCCRevectorTransformation', 'SCCVecRevectorTransformation',
    'SCCSeqRevectorTransformation', 'wrap_vector_section',
    'RevectorSectionTransformer'
]


def wrap_vector_section(section, routine, bounds, index, insert_pragma=True):
    """
    Wrap a section of nodes in a vector-level loop across the horizontal.

    Parameters
    ----------
    section : tuple of :any:`Node`
        A section of nodes to be wrapped in a vector-level loop
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    insert_pragma: bool, optional
        Adds a ``!$loki vector`` pragma around the created loop
    """
    # Create a single loop around the horizontal from a given body
    index = get_integer_variable(routine, index)
    bounds = sym.LoopRange(bounds)

    # Ensure we clone all body nodes, to avoid recursion issues
    body = Transformer().visit(section)

    # Add a marker pragma for later annotations
    pragma = (ir.Pragma('loki', content='loop vector'),) if insert_pragma else None
    vector_loop = ir.Loop(variable=index, bounds=bounds, body=body, pragma=pragma)

    # Add a comment before and after the pragma-annotated loop to ensure
    # we do not overlap with neighbouring pragmas
    return (ir.Comment(''), vector_loop, ir.Comment(''))


class RevectorSectionTransformer(Transformer):
    """
    :any:`Transformer` that replaces :any:`Section` objects labelled
    with ``"vector_section"`` with vector-level loops across the
    horizontal.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in the vector loops should be removed.
    horizontal: :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    insert_pragma: bool, optional
        Adds a ``!$loki vector`` pragma around the created loop
    """
    # pylint: disable=unused-argument

    def __init__(self, routine, horizontal, *args, insert_pragma=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.routine = routine
        self.horizontal = horizontal

        self.insert_pragma = insert_pragma

    def visit_Section(self, s, **kwargs):
        if s.label == 'vector_section':
            # Derive the loop bounds wrap section in loop
            bounds = get_loop_bounds(self.routine, dimension=self.horizontal)
            return wrap_vector_section(
                s.body, self.routine, bounds=bounds, index=self.horizontal.index,
                insert_pragma=self.insert_pragma
            )

        # Rebuild loop after recursing to children
        return self._rebuild(s, self.visit(s.children))


class BaseRevectorTransformation(Transformation):
    """
    A base/parent class for transformation to wrap thread-parallel IR sections within a horizontal loop.
    This transformation relies on markers placed by :any:`SCCDevectorTransformation`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    _reduction_match_pattern = r'reduction\([\+\*\.\w \t]+:[\w\, \t]+\)'

    def __init__(self, horizontal):
        self.horizontal = horizontal

    def mark_vector_reductions(self, routine, section):
        """
        Mark vector-reduction loops in marked vector-reduction
        regions.

        If a region explicitly marked with
        ``!$loki vector-reduction(<reduction clause>)``/
        ``!$loki end vector-reduction`` is encountered, we replace
        existing ``!$loki loop vector`` loop pragmas and add the
        reduction keyword and clause. These will be turned into
        OpenACC equivalents by :any:`SCCAnnotate`.
        """
        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(section):
                if is_loki_pragma(region.pragma, starts_with='vector-reduction'):
                    if (reduction_clause := re.search(self._reduction_match_pattern, region.pragma.content)):

                        loops = FindNodes(ir.Loop).visit(region)
                        assert len(loops) == 1
                        pragma = ir.Pragma(keyword='loki', content=f'loop vector {reduction_clause[0]}')
                        # Update loop and region in place to remove marker pragmas
                        loops[0]._update(pragma=(pragma,))
                        region._update(pragma=None, pragma_post=None)

    def mark_seq_loops(self, section):
        """
        Mark interior sequential loops in a thread-parallel section
        with ``!$loki loop seq`` for later annotation.

        This utility requires loop-pragmas to be attached via
        :any:`pragmas_attached`. It also updates loops in-place.

        Parameters
        ----------
        section : tuple of :any:`Node`
            Code section in which to mark "seq loops".
        """
        for loop in FindNodes(ir.Loop).visit(section):

            # Skip loops explicitly marked with `!$loki/claw nodep`
            if loop.pragma and any('nodep' in p.content.lower() for p in as_tuple(loop.pragma)):
                continue

            # Mark loop as sequential with `!$loki loop seq`
            if loop.variable != self.horizontal.index:
                loop._update(pragma=(ir.Pragma(keyword='loki', content='loop seq'),))

    def mark_driver_loop(self, routine, loop):
        """
        Add ``!$loki loop driver`` pragmas to outer block loops and
        add ``vector-length(size)`` clause for later annotations.

        This method assumes that pragmas have been attached via
        :any:`pragmas_attached`.
        """

        # Skip loops with existing parallel annotations
        if loop.pragma:
            if any(pragma.keyword.lower() in ['omp', 'acc'] and 'parallel' in pragma.content.lower()
                   for pragma in loop.pragma):
                return

        # Find a horizontal size variable to mark vector_length
        symbol_map = routine.symbol_map
        sizes = tuple(
            routine.resolve_typebound_var(size, symbol_map) for size in self.horizontal.size_expressions
            if size.split('%')[0] in symbol_map
        )
        vector_length = f' vector_length({sizes[0]})' if sizes else ''

        # Replace existing `!$loki loop driver markers, but leave all others
        pragma = ir.Pragma(keyword='loki', content=f'loop driver{vector_length}')
        loop_pragmas = tuple(
            p for p in as_tuple(loop.pragma) if not is_loki_pragma(p, starts_with='driver-loop')
        )
        loop._update(pragma=loop_pragmas + (pragma,))


class SCCVecRevectorTransformation(BaseRevectorTransformation):
    """
    A transformation to wrap thread-parallel IR sections within a horizontal loop.
    This transformation relies on markers placed by :any:`SCCDevectorTransformation`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def transform_subroutine(self, routine, **kwargs):
        """
        Wrap vector-parallel sections in vector :any:`Loop` objects.

        This wraps all thread-parallel sections within "kernel"
        routines or within the parallel loops in "driver" routines.

        The markers placed by :any:`SCCDevectorTransformation` are removed

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : str
            Must be either ``"kernel"`` or ``"driver"``
        targets : tuple or str
            Tuple of target routine names for determining "driver" loops
        """
        role = kwargs['role']
        targets = kwargs.get('targets', ())

        if role == 'kernel':
            # Skip if kernel is marked as `!$loki routine seq`
            if check_routine_sequential(routine):
                return

            # Revector all marked vector sections within the kernel body
            routine.body = RevectorSectionTransformer(routine, self.horizontal).visit(routine.body)

            with pragmas_attached(routine, ir.Loop):
                # Check for explicitly labelled vector-reduction regions
                self.mark_vector_reductions(routine, routine.body)

                # Mark sequential loops inside vector sections
                self.mark_seq_loops(routine.body)

            # Mark subroutine as vector parallel for later annotation
            routine.spec.append(ir.Pragma(keyword='loki', content='routine vector'))

        if role == 'driver':
            with pragmas_attached(routine, ir.Loop):
                driver_loops = find_driver_loops(section=routine.body, targets=targets)

                for loop in driver_loops:
                    # Revector all marked sections within the driver loop body
                    loop._update(body=RevectorSectionTransformer(routine, self.horizontal).visit(loop.body))

                    # Check for explicitly labelled vector-reduction regions
                    self.mark_vector_reductions(routine, loop.body)

                    # Mark sequential loops inside vector sections
                    self.mark_seq_loops(loop.body)

                    # Mark outer driver loops
                    self.mark_driver_loop(routine, loop)

# alias for backwards compability
SCCRevectorTransformation = SCCVecRevectorTransformation

class SCCSeqRevectorTransformation(BaseRevectorTransformation):
    """
    A transformation to wrap thread-parallel IR sections within a horizontal loop
    in a way that the horizontal loop is hoisted/moved to the driver level while
    the horizontal/loop index is passed as an argument.
    This transformation relies on markers placed by :any:`SCCDevectorTransformation`.
    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    process_ignored_items = True

    def remove_vector_sections(self, section):
        """
        Remove all thread-parallel :any:`Section` objects within a given
        code section
        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        section : tuple of :any:`Node`
            Code section in which to replace vector-parallel
            :any:`Section` objects.
        """
        # Wrap all thread-parallel sections into horizontal thread loops
        mapper = {
            s: s.body
            for s in FindNodes(ir.Section).visit(section)
            if s.label == 'vector_section'
        }
        return Transformer(mapper).visit(section)

    def mark_vector_reductions(self, routine, section):
        """
        Vector reductions are not applicable to sequential routines
        so we raise an axception here.
        """

        with pragma_regions_attached(routine):
            for region in FindNodes(ir.PragmaRegion).visit(section):
                if is_loki_pragma(region.pragma, starts_with='vector-reduction'):
                    if re.search(self._reduction_match_pattern, region.pragma.content):
                        raise RuntimeError(f'[Loki::SCCSeq] Vector reduction invalid in seq routine {routine.name}')

    @staticmethod
    def _get_loop_bound(bound, call_arg_map):
        if isinstance(bound, tuple):
            for alias in bound:
                alias_arg = alias
                elem = None
                if '%' in alias:
                    elem = alias_arg.split('%')[1]
                    alias_arg = alias_arg.split('%')[0]
                if alias_arg in call_arg_map:
                    return (call_arg_map[alias_arg.lower()], elem)
        return (call_arg_map[bound.lower()], None)

    def transform_subroutine(self, routine, **kwargs):
        """
        Wrap vector-parallel sections in vector :any:`Loop` objects.
        This wraps all thread-parallel sections within "kernel"
        routines or within the parallel loops in "driver" routines.
        The markers placed by :any:`SCCDevectorTransformation` are removed
        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : str
            Must be either ``"kernel"`` or ``"driver"``
        targets : tuple or str
            Tuple of target routine names for determining "driver" loops
        """
        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))
        # ignore = kwargs.get('ignore', ())
        item = kwargs.get('item', None)
        ignore = item.ignore if item else () + tuple(str(t).lower() for t in as_tuple(kwargs.get('ignore', None)))

        if role == 'kernel':
            # Skip if kernel is marked as `!$loki routine seq`
            if check_routine_sequential(routine):
                return

            if self.horizontal.index not in routine.variables:
                jl = get_integer_variable(routine, self.horizontal.index)
                routine.arguments += (jl.clone(type=jl.type.clone(intent='in')),)
            else:
                single_variable_declaration(routine, variables=(self.horizontal.index,))
                routine.symbol_attrs.update({self.horizontal.index:\
                    routine.variable_map[self.horizontal.index].type.clone(intent='in')})
                if self.horizontal.index not in routine.arguments:
                    routine.arguments += (get_integer_variable(routine, self.horizontal.index),)

            # add horizontal.index as argument for calls/routines being in targets
            call_map = {}
            for call in FindNodes(ir.CallStatement).visit(routine.body):
                if str(call.name).lower() in targets or call.routine.name.lower() in ignore:
                    if check_routine_sequential(call.routine):
                        continue
                    if self.horizontal.index not in call.arg_map:
                        new_kwarg = (self.horizontal.index, get_integer_variable(routine, self.horizontal.index))
                        updated_call = call.clone(kwarguments=call.kwarguments + (new_kwarg,))
                        call_map[call] = updated_call
                    if call.routine.name.lower() in ignore:
                        if self.horizontal.index not in call.routine.variables:
                            jl = get_integer_variable(call.routine, self.horizontal.index)
                            call.routine.arguments += (jl.clone(type=jl.type.clone(intent='in')),)
                        else:
                            single_variable_declaration(call.routine, variables=(self.horizontal.index,))
                            call.routine.symbol_attrs.update({self.horizontal.index:\
                                call.routine.variable_map[self.horizontal.index].type.clone(intent='in')})
                            if self.horizontal.index not in call.routine.arguments:
                                call.routine.arguments += (get_integer_variable(call.routine, self.horizontal.index),)
            routine.body = Transformer(call_map).visit(routine.body)

            # Revector all marked vector sections within the kernel body
            routine.body = self.remove_vector_sections(routine.body)

            with pragmas_attached(routine, ir.Loop):
                # Check for explicitly labelled vector-reduction regions
                self.mark_vector_reductions(routine, routine.body)

                # Mark sequential loops inside vector sections
                self.mark_seq_loops(routine.body)

            # Mark subroutine as seq for later annotation
            routine.spec.append(ir.Pragma(keyword='loki', content='routine seq'))

        if role == 'driver':

            # add horizontal.index, e.g., 'jl'
            index = get_integer_variable(routine, self.horizontal.index)
            routine.variables += (index,)

            with pragmas_attached(routine, ir.Loop):
                driver_loops = find_driver_loops(section=routine.body, targets=targets)

                for loop in driver_loops:

                    # Wrap calls being in targets in a horizontal loop and add horizontal.index as argument
                    call_map = {}
                    for call in FindNodes(ir.CallStatement).visit(loop.body):
                        if str(call.name).lower() in targets:
                            if self.horizontal.index not in call.arg_map:
                                new_kwarg = (self.horizontal.index,
                                        get_integer_variable(routine, self.horizontal.index))
                                updated_call = call.clone(kwarguments=call.kwarguments + (new_kwarg,))
                                call_arg_map = {k.name.lower(): v for (k, v) in call.arg_map.items()}
                                # loop bound(s) could be derived types ...
                                ltmp = self._get_loop_bound(self.horizontal.lower, call_arg_map)
                                utmp = self._get_loop_bound(self.horizontal.upper, call_arg_map)
                                lower = ltmp[0] if ltmp[1] is None else sym.Variable(name=ltmp[1], parent=ltmp[0])
                                upper = utmp[0] if utmp[1] is None else sym.Variable(name=utmp[1], parent=utmp[0])
                                # wrap call with horizontal loop
                                loop_bounds = (lower, upper)
                                call_map[call] = wrap_vector_section((updated_call,), routine, bounds=loop_bounds,
                                        insert_pragma=True, index=self.horizontal.index)

                    loop._update(body=Transformer(call_map).visit(loop.body))

                    # Revector all marked sections within the driver loop body
                    loop._update(body=RevectorSectionTransformer(routine, self.horizontal).visit(loop.body))

                    # Check for explicitly labelled vector-reduction regions
                    super().mark_vector_reductions(routine, loop.body)

                    # Mark sequential loops inside vector sections
                    self.mark_seq_loops(loop.body)

                    # Mark outer driver loops
                    self.mark_driver_loop(routine, loop)
