# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import Array, is_dimension_constant
from loki.expression import symbols as sym
from loki.frontend import HAVE_FP
from loki.ir import nodes as ir, FindNodes, FindInlineCalls, FindVariables, Transformer
from loki.tools import as_tuple, CaseInsensitiveDict, OrderedSet

from loki.transformations.array_indexing import demote_variables
from loki.transformations.utilities import get_integer_variable

if HAVE_FP:
    from fparser.two import Fortran2003


__all__ = ['SCCDemoteTransformation']


class SCCDemoteTransformation(Transformation):
    """
    A set of utilities to determine which arrays can be safely demoted in a
    :any:`Subroutine` as part of a transformation pass.

    Unless the option `demote_local_arrays` is set to `False`, this transformation will demote
    arrays that are used in at most one vector section. Specific arrays in individual
    routines can also be marked for preservation by assigning them to the `preserve_arrays` list
    in the :any:`SchedulerConfig`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    _key = 'SCCDemoteTransformation'
    reverse_traversal = True

    def __init__(self, horizontal, demote_local_arrays=True):
        self.horizontal = horizontal

        self.demote_local_arrays = demote_local_arrays

    @classmethod
    def get_variables_to_demote(cls, routine, sections, horizontal):
        """
        Collect all array variables used in vector sections and return
        those that are safe to demote.

        Demotion is considered safe if the variable:
        * Is used in at most one vector section
        * Has the ``horizontal`` as the innermost dimension
        * Has only constant dimensions beyond the innermost

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine to scan for variables.
        sections : list of :any:`Section`
            The vector sections to scan for array variables.
        horizontal : :any:`Dimension`
            Dimension object describing the horizontal data dimension.

        Returns
        -------
        list of str
            Lower-case names of variables safe to demote.
        """
        # Collect array variable names used in each section
        vars_per_section = {
            s: OrderedSet(
                v.name.lower() for v in FindVariables(unique=False).visit(s)
                if isinstance(v, Array)
            ) for s in sections
        }

        # Collect all unique variable names across sections
        all_vars = OrderedSet()
        for names in vars_per_section.values():
            all_vars |= names

        # Keep only those appearing in at most one section
        single_section_vars = [
            name for name in all_vars
            if sum(1 for s_vars in vars_per_section.values() if name in s_vars) <= 1
        ]

        # Build a lookup from declared variables to check shape properties
        variable_map = CaseInsensitiveDict(
            (v.name, v) for v in routine.variables if isinstance(v, Array)
        )

        # Only demote arrays with the horizontal as fast dimension
        # and whose remaining dimensions are known constants
        to_demote = []
        for name in single_section_vars:
            v = variable_map.get(name)
            if v is None or not v.shape:
                continue
            if v.shape[0] not in horizontal.sizes:
                continue
            if not all(is_dimension_constant(d) for d in v.shape[1:]):
                continue
            to_demote.append(name)

        # Exclude arrays used in reduction intrinsics (e.g. SUM, MAXVAL, ...)
        # Search the entire routine body because reduction calls may sit
        # outside vector sections (they act as separators during devectorization).
        reduction_arrays = cls._get_reduction_arrays(routine.body, variable_map)
        to_demote = [name for name in to_demote if name not in reduction_arrays]

        return to_demote

    @staticmethod
    def _get_reduction_arrays(body, variable_map):
        """
        Collect lower-case names of arrays passed to reduction intrinsics
        (e.g. ``SUM``, ``MAXVAL``, ...) within *body*.

        The entire routine body is searched rather than just the vector
        sections, because reduction calls act as separators during
        devectorization and therefore sit outside any vector section.

        After devectorization, array notation may have been resolved so
        that a reference like ``ZCOUNT(JROF)`` appears as a ``Scalar``.
        We therefore also match parameters whose name corresponds to a
        declared array in *variable_map*.
        """
        if HAVE_FP:
            reduction_names = {
                name.lower()
                for name in Fortran2003.Intrinsic_Name.array_reduction_names
            }
        else:
            reduction_names = {'sum', 'product', 'maxval', 'minval',
                               'maxloc', 'minloc', 'any', 'all', 'count'}

        reduction_arrays = OrderedSet()
        for call in FindInlineCalls().visit(body):
            if call.name.lower() in reduction_names:
                for param in call.parameters:
                    name = getattr(param, 'name', None)
                    if name and name.lower() in variable_map:
                        reduction_arrays.add(name.lower())
        return reduction_arrays

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCDemote utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
            or ``"driver"``
        """
        role = kwargs['role']
        item = kwargs.get('item', None)

        sub_sgraph = kwargs.get('sub_sgraph', None)
        successors = sub_sgraph.successors(item=item) if sub_sgraph is not None and item else ()

        if role == 'kernel':
            demote_locals = self.demote_local_arrays
            preserve_arrays = []
            if item:
                demote_locals = item.config.get('demote_locals', self.demote_local_arrays)
                preserve_arrays = item.config.get('preserve_arrays', [])
            self.process_kernel(routine, item=item, successors=successors,
                                demote_locals=demote_locals, preserve_arrays=preserve_arrays)

        if role == 'driver':
            self.process_driver(routine, successors=successors)

    def process_kernel(self, routine, item=None, successors=(), demote_locals=True, preserve_arrays=None):
        """
        Applies the SCCDemote utilities to a "kernel" and demotes all suitable arrays.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        item : :any:`Item`, optional
            Scheduler work item for this routine.
        successors : tuple of :any:`Item`
            Successor items whose trafo_data may contain demoted variable info.
        demote_locals : bool, optional
            Flag to trigger demotion of local arrays; default: True.
        preserve_arrays : list, optional
            List of array names to preserve from demotion.
        """

        # Find vector sections marked in the SCCDevectorTransformation
        sections = [
            s for s in FindNodes(ir.Section).visit(routine.body)
            if s.label == 'vector_section'
        ]

        # Determine which variables appear in at most one section
        to_demote = self.get_variables_to_demote(routine, sections, self.horizontal)

        # Filter out arrays marked explicitly for preservation
        if preserve_arrays:
            to_demote = [v for v in to_demote if v not in preserve_arrays]

        # Build successor map: callee Subroutine -> set of demoted variable names
        successor_map = self._build_successor_map(successors)

        # Cannot demote a variable if a child doesn't demote the corresponding dummy arg
        to_demote = self._filter_by_children(routine, to_demote, successor_map)

        # Demote all arrays that do not buffer values between sections
        if demote_locals and to_demote:
            demote_variables(
                routine, variable_names=to_demote,
                dimensions=self.horizontal.sizes
            )

        # Update call args: add horizontal.index where child demoted but parent didn't
        self._update_call_args(routine, successor_map, to_demote)

        # Store demoted variable names in trafo_data for downstream use
        if item:
            item.trafo_data[self._key] = {'demoted_variables': to_demote}

    def process_driver(self, routine, successors=()):
        """
        Update the driver routine's call interfaces to match demoted
        variables in callee kernels.

        For each call to a successor whose variables were demoted, the
        corresponding caller-side arguments are updated by inserting
        ``horizontal.index`` into the horizontal dimension.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The driver subroutine to update.
        successors : tuple of :any:`Item`
            Successor items whose trafo_data may contain demoted variable info.
        """
        successor_map = self._build_successor_map(successors)
        self._update_call_args(routine, successor_map, to_demote=[])

    def _build_successor_map(self, successors):
        """
        Build a map from callee :any:`Subroutine` to the set of
        demoted variable names stored in its trafo_data.
        """
        return {
            successor.ir: set(
                successor.trafo_data.get(self._key, {}).get('demoted_variables', [])
            )
            for successor in successors
        }

    @staticmethod
    def _filter_by_children(routine, to_demote, successor_map):
        """
        Remove from *to_demote* any variable that is passed to a child
        call whose corresponding dummy argument was **not** demoted.
        """
        to_demote_set = OrderedSet(to_demote)
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if call.routine not in successor_map:
                continue
            demoted_in_child = successor_map[call.routine]
            for dummy, arg in call.arg_map.items():
                if isinstance(arg, Array) and arg.name.lower() in to_demote_set:
                    if dummy.name.lower() not in demoted_in_child:
                        to_demote_set.discard(arg.name.lower())
        return list(to_demote_set)

    def _update_call_args(self, routine, successor_map, to_demote):
        """
        For each call to a successor, update arguments where the callee
        demoted a dummy but the caller did not demote the corresponding
        variable: insert ``horizontal.index`` into the horizontal
        dimension of the call argument.
        """
        to_demote_set = set(to_demote)
        idx = get_integer_variable(routine, self.horizontal.index)

        call_map = {}
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if call.routine not in successor_map:
                continue
            demoted_in_child = successor_map[call.routine]
            if not demoted_in_child:
                continue

            # Rebuild positional arguments
            new_arguments = list(call.arguments)
            for i, (dummy, arg) in enumerate(zip(call.routine.arguments, call.arguments)):
                if dummy.name.lower() not in demoted_in_child:
                    continue
                if not isinstance(arg, Array):
                    continue
                if arg.name.lower() in to_demote_set:
                    # Both sides demoted — call arg already consistent
                    continue
                new_arguments[i] = self._add_horizontal_index(arg, idx)

            # Rebuild keyword arguments
            r_args = CaseInsensitiveDict(
                (a.name, a) for a in call.routine.arguments
            )
            new_kwarguments = []
            for kw, arg in as_tuple(call.kwarguments):
                dummy = r_args.get(kw)
                if (dummy and dummy.name.lower() in demoted_in_child
                        and isinstance(arg, Array)
                        and arg.name.lower() not in to_demote_set):
                    new_kwarguments.append((kw, self._add_horizontal_index(arg, idx)))
                else:
                    new_kwarguments.append((kw, arg))

            call_map[call] = call.clone(
                arguments=as_tuple(new_arguments),
                kwarguments=as_tuple(new_kwarguments)
            )

        if call_map:
            routine.body = Transformer(call_map).visit(routine.body)

    @staticmethod
    def _add_horizontal_index(arg, idx):
        """
        Return a clone of *arg* with its first (horizontal) dimension
        replaced by *idx*.
        """
        if arg.dimensions:
            new_dims = (idx,) + arg.dimensions[1:]
        else:
            new_dims = (idx,) + tuple(
                sym.RangeIndex((None, None)) for _ in arg.shape[1:]
            )
        return arg.clone(dimensions=new_dims)
