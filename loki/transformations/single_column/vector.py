# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from more_itertools import split_at

from loki.analyse import dataflow_analysis_attached
from loki.batch import Transformation
from loki.expression import (
    symbols as sym, is_dimension_constant, FindVariables
)
from loki.ir import (
    nodes as ir, FindNodes, FindScopes, Transformer,
    NestedTransformer, is_loki_pragma, pragmas_attached
)
from loki.tools import as_tuple, flatten
from loki.types import BasicType

from loki.transformations.array_indexing import demote_variables
from loki.transformations.utilities import (
    get_integer_variable, get_loop_bounds, find_driver_loops,
    get_local_arrays, check_routine_pragmas
)


__all__ = [
    'SCCDevectorTransformation', 'SCCRevectorTransformation',
    'SCCDemoteTransformation', 'wrap_vector_section'
]


class SCCDevectorTransformation(Transformation):
    """
    A set of utilities that can be used to strip vector loops from a :any:`Subroutine`
    and determine the regions of the IR to be placed within thread-parallel loop directives.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    trim_vector_sections : bool
        Flag to trigger trimming of extracted vector sections to remove
        nodes that are not assignments involving vector parallel arrays.
    """

    def __init__(self, horizontal, trim_vector_sections=False):
        self.horizontal = horizontal
        self.trim_vector_sections = trim_vector_sections

    @classmethod
    def kernel_remove_vector_loops(cls, routine, horizontal):
        """
        Remove all vector loops over the specified dimension.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in the vector loops should be removed.
        horizontal : :any:`Dimension`
            The dimension specifying the horizontal vector dimension
        """
        loop_map = {}
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if loop.variable == horizontal.index:
                loop_map[loop] = loop.body
        routine.body = Transformer(loop_map).visit(routine.body)

    @classmethod
    def extract_vector_sections(cls, section, horizontal):
        """
        Extract a contiguous sections of nodes that contains vector-level
        computations and are not interrupted by recursive subroutine calls
        or nested control-flow structures.

        Parameters
        ----------
        section : tuple of :any:`Node`
            A section of nodes from which to extract vector-level sub-sections
        horizontal: :any:`Dimension`
            The dimension specifying the horizontal vector dimension
        """

        _scope_node_types = (ir.Loop, ir.Conditional, ir.MultiConditional)

        # Identify outer "scopes" (loops/conditionals) constrained by recursive routine calls
        calls = FindNodes(ir.CallStatement).visit(section)
        pragmas = [pragma for pragma in FindNodes(ir.Pragma).visit(section) if pragma.keyword.lower() == "loki" and
                   pragma.content.lower() == "separator"]
        separator_nodes = pragmas

        for call in calls:

            # check if calls have been enriched
            if not call.routine is BasicType.DEFERRED:
                # check if called routine is marked as sequential
                if check_routine_pragmas(routine=call.routine, directive=None):
                    continue

            if call in section:
                # If the call is at the current section's level, it's a separator
                separator_nodes.append(call)

            else:
                # If the call is deeper in the IR tree, it's highest ancestor is used
                ancestors = flatten(FindScopes(call).visit(section))
                ancestor_scopes = [a for a in ancestors if isinstance(a, _scope_node_types)]
                if len(ancestor_scopes) > 0 and ancestor_scopes[0] not in separator_nodes:
                    separator_nodes.append(ancestor_scopes[0])

        for pragma in FindNodes(ir.Pragma).visit(section):
            # Reductions over thread-parallel regions should be marked as a separator node
            if (is_loki_pragma(pragma, starts_with='vector-reduction') or
                is_loki_pragma(pragma, starts_with='end vector-reduction')):
                separator_nodes.append(pragma)

        # Extract contiguous node sections between separator nodes
        assert all(n in section for n in separator_nodes)
        subsections = [as_tuple(s) for s in split_at(section, lambda n: n in separator_nodes)]

        # Filter sub-sections that do not use the horizontal loop index variable
        subsections = [s for s in subsections if horizontal.index in list(FindVariables().visit(s))]

        # Recurse on all separator nodes that might contain further vector sections
        for separator in separator_nodes:

            if isinstance(separator, ir.Loop):
                subsec_body = cls.extract_vector_sections(separator.body, horizontal)
                if subsec_body:
                    subsections += subsec_body

            if isinstance(separator, ir.Conditional):
                subsec_body = cls.extract_vector_sections(separator.body, horizontal)
                if subsec_body:
                    subsections += subsec_body
                subsec_else = cls.extract_vector_sections(separator.else_body, horizontal)
                if subsec_else:
                    subsections += subsec_else

            if isinstance(separator, ir.MultiConditional):
                for body in separator.bodies:
                    subsec_body = cls.extract_vector_sections(body, horizontal)
                    if subsec_body:
                        subsections += subsec_body
                subsec_else = cls.extract_vector_sections(separator.else_body, horizontal)
                if subsec_else:
                    subsections += subsec_else

        return subsections

    @classmethod
    def get_trimmed_sections(cls, routine, horizontal, sections):
        """
        Trim extracted vector sections to remove nodes that are not assignments
        involving vector parallel arrays.
        """

        trimmed_sections = ()
        with dataflow_analysis_attached(routine):
            for sec in sections:
                vec_nodes = [node for node in sec if horizontal.index.lower() in node.uses_symbols]
                start = sec.index(vec_nodes[0])
                end = sec.index(vec_nodes[-1])

                trimmed_sections += (sec[start:end+1],)

        return trimmed_sections

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCDevector utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']
        targets = kwargs.get('targets', ())

        if role == 'kernel':
            self.process_kernel(routine)
        if role == "driver":
            self.process_driver(routine, targets=targets)

    def process_kernel(self, routine):
        """
        Applies the SCCDevector utilities to a "kernel". This consists simply
        of stripping vector loops and determing which sections of the IR can be
        placed within thread-parallel loops.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Remove all vector loops over the specified dimension
        self.kernel_remove_vector_loops(routine, self.horizontal)

        # Extract vector-level compute sections from the kernel
        sections = self.extract_vector_sections(routine.body.body, self.horizontal)

        if self.trim_vector_sections:
            sections = self.get_trimmed_sections(routine, self.horizontal, sections)

        # Replace sections with marked Section node
        section_mapper = {s: ir.Section(body=s, label='vector_section') for s in sections}
        routine.body = NestedTransformer(section_mapper).visit(routine.body)

    def process_driver(self, routine, targets=()):
        """
        Applies the SCCDevector utilities to a "driver". This consists simply
        of stripping vector loops and determining which sections of the IR can be
        placed within thread-parallel loops.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        targets : list or string
            List of subroutines that are to be considered as part of
            the transformation call tree.
        """

        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            driver_loops = find_driver_loops(routine=routine, targets=targets)

        # remove vector loops
        driver_loop_map = {}
        for loop in driver_loops:
            loop_map = {}
            for l in FindNodes(ir.Loop).visit(loop.body):
                if l.variable == self.horizontal.index:
                    loop_map[l] = l.body
            new_driver_loop = Transformer(loop_map).visit(loop.body)
            new_driver_loop = loop.clone(body=new_driver_loop)
            sections = self.extract_vector_sections(new_driver_loop.body, self.horizontal)
            if self.trim_vector_sections:
                sections = self.get_trimmed_sections(new_driver_loop, self.horizontal, sections)
            section_mapper = {s: ir.Section(body=s, label='vector_section') for s in sections}
            new_driver_loop = NestedTransformer(section_mapper).visit(new_driver_loop)
            driver_loop_map[loop] = new_driver_loop
        routine.body = Transformer(driver_loop_map).visit(routine.body)


def wrap_vector_section(section, routine, horizontal, insert_pragma=True):
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
    bounds = get_loop_bounds(routine, dimension=horizontal)

    # Create a single loop around the horizontal from a given body
    index = get_integer_variable(routine, horizontal.index)
    bounds = sym.LoopRange(bounds)

    # Ensure we clone all body nodes, to avoid recursion issues
    body = Transformer().visit(section)

    # Add a marker pragma for later annotations
    pragma = (ir.Pragma('loki', content='loop vector'),) if insert_pragma else None
    vector_loop = ir.Loop(variable=index, bounds=bounds, body=body, pragma=pragma)

    # Add a comment before and after the pragma-annotated loop to ensure
    # we do not overlap with neighbouring pragmas
    return (ir.Comment(''), vector_loop, ir.Comment(''))


class SCCRevectorTransformation(Transformation):
    """
    A transformation to wrap thread-parallel IR sections within a horizontal loop.
    This transformation relies on markers placed by :any:`SCCDevectorTransformation`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def __init__(self, horizontal, remove_vector_section=False):
        self.horizontal = horizontal
        self.remove_vector_section = remove_vector_section

    def revector_section(self, routine, section):
        """
        Wrap all thread-parallel :any:`Section` objects within a given
        code section in a horizontal loop and mark interior loops as
        ``!$loki loop seq``.

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
            s: wrap_vector_section(s.body, routine, self.horizontal)
            for s in FindNodes(ir.Section).visit(section)
            if s.label == 'vector_section'
        }
        return Transformer(mapper).visit(section)

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
            # Revector all marked vector sections within the kernel body
            routine.body = self.revector_section(routine, routine.body)

            # Mark sequential loops inside vector sections
            with pragmas_attached(routine, ir.Loop):
                self.mark_seq_loops(routine.body)

        if role == 'driver':
            with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
                driver_loops = find_driver_loops(routine=routine, targets=targets)

                for loop in driver_loops:
                    # Revector all marked sections within the driver loop body
                    loop._update(body=self.revector_section(routine, loop.body))

                    # Mark sequential loops inside vector sections
                    self.mark_seq_loops(loop.body)

        if self.remove_vector_section:
            # Remove the vector section wrappers
            # These have been inserted by SCCDevectorTransformation
            section_mapper = {s: s.body for s in FindNodes(ir.Section).visit(routine.body)
                    if s.label == 'vector_section'}
            if section_mapper:
                routine.body = Transformer(section_mapper).visit(routine.body)


class SCCDemoteTransformation(Transformation):
    """
    A set of utilities to determine which local arrays can be safely demoted in a
    :any:`Subroutine` as part of a transformation pass.

    Unless the option `demote_local_arrays` is set to `False`, this transformation will demote
    local arrays that do not buffer values between vector loops. Specific arrays in individual
    routines can also be marked for preservation by assigning them to the `preserve_arrays` list
    in the :any:`SchedulerConfig`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def __init__(self, horizontal, demote_local_arrays=True):
        self.horizontal = horizontal

        self.demote_local_arrays = demote_local_arrays

    @classmethod
    def get_locals_to_demote(cls, routine, sections, horizontal):
        """
        Create a list of local temporary arrays after checking that
        demotion is safe.

        Demotion is considered safe if the temporary is only used
        within one coherent vector-section (see
        :any:`extract_vector_sections`).

        Local temporaries get demoted if they have:
        * Only one dimension, which is the ``horizontal``
        * Have the ``horizontal`` as the innermost dimension, with all
          other dimensions being declared constant parameters.

        """
        # Create a list of local temporary arrays to filter down
        candidates = get_local_arrays(routine, routine.spec)

        # Only demote local arrays with the horizontal as fast dimension
        candidates = [
            v for v in candidates if v.shape and
            v.shape[0] in [horizontal.size, *horizontal._aliases]
        ]
        # Also demote arrays whose remaning dimensions are known constants
        candidates = [
            v for v in candidates
            if all(is_dimension_constant(d) for d in v.shape[1:])
        ]

        # Create an index into all variable uses per vector-level section
        vars_per_section = {
            s: set(
                v.name.lower() for v in get_local_arrays(routine, s, unique=False)
            ) for s in sections
        }

        # Count in how many sections each temporary is used
        counts = {}
        for arr in candidates:
            counts[arr] = sum(
                1 if arr.name.lower() in v else 0
                for v in vars_per_section.values()
            )

        # Demote temporaries that are only used in one section or not at all
        to_demote = [k for k, v in counts.items() if v <= 1]

        # Filter out variables that we will pass down the call tree
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        call_args = flatten(call.arguments for call in calls)
        call_args += flatten(list(dict(call.kwarguments).values()) for call in calls)
        to_demote = [v for v in to_demote if v.name not in call_args]

        return set(to_demote)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCDemote utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']
        item = kwargs.get('item', None)

        if role == 'kernel':
            demote_locals = self.demote_local_arrays
            preserve_arrays = []
            if item:
                demote_locals = item.config.get('demote_locals', self.demote_local_arrays)
                preserve_arrays = item.config.get('preserve_arrays', [])
            self.process_kernel(routine, demote_locals=demote_locals, preserve_arrays=preserve_arrays)

    def process_kernel(self, routine, demote_locals=True, preserve_arrays=None):
        """
        Applies the SCCDemote utilities to a "kernel" and demotes all suitable local arrays.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Find vector sections marked in the SCCDevectorTransformation
        sections = [
            s for s in FindNodes(ir.Section).visit(routine.body)
            if s.label == 'vector_section'
        ]

        # Extract the local variables to demote after we wrap the sections in vector loops.
        # We do this, because need the section blocks to determine which local arrays
        # may carry buffered values between them, so that we may not demote those!
        to_demote = self.get_locals_to_demote(routine, sections, self.horizontal)

        # Filter out arrays marked explicitly for preservation
        if preserve_arrays:
            to_demote = [v for v in to_demote if not v.name in preserve_arrays]

        # Demote all private local variables that do not buffer values between sections
        if demote_locals:
            variables = tuple(v.name for v in to_demote)
            if variables:
                demote_variables(
                    routine, variable_names=variables,
                    dimensions=[self.horizontal.size, *self.horizontal._aliases]
                )
