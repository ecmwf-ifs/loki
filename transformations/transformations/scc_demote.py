# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import symbols as sym
from loki import Transformation, FindNodes, ir, demote_variables, FindVariables, flatten

__all__ = ['SCCDemoteTransformation']

class SCCDemoteTransformation(Transformation):
    """
    A set of utilities to determine which local arrays can be safely demoted in a
    :any:`Subroutine` as part of a transformation pass.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def __init__(self, horizontal, demote_local_arrays=True):
        self.horizontal = horizontal

        self.demote_local_arrays = demote_local_arrays
        self._processed = {}

    @classmethod
    def kernel_get_locals_to_demote(cls, routine, sections, horizontal):

        argument_names = [v.name for v in routine.arguments]

        def _is_constant(d):
            """Establish if a given dimensions symbol is a compile-time constant"""
            if isinstance(d, sym.IntLiteral):
                return True

            if isinstance(d, sym.RangeIndex):
                if d.lower:
                    return _is_constant(d.lower) and _is_constant(d.upper)
                return _is_constant(d.upper)

            if isinstance(d, sym.Scalar) and isinstance(d.initial , sym.IntLiteral):
                return True

            return False

        def _get_local_arrays(section):
            """
            Filters out local argument arrays that solely buffer the
            horizontal vector dimension
            """
            arrays = FindVariables(unique=False).visit(section)
            # Only demote local arrays with the horizontal as fast dimension
            arrays = [v for v in arrays if isinstance(v, sym.Array)]
            arrays = [v for v in arrays if v.name not in argument_names]
            arrays = [v for v in arrays if v.shape and v.shape[0] == horizontal.size]

            # Also demote arrays whose remaning dimensions are known constants
            arrays = [v for v in arrays if all(_is_constant(d) for d in v.shape[1:])]
            return arrays

        # Create a list of all local horizontal temporary arrays
        candidates = _get_local_arrays(routine.body)

        # Create an index into all variable uses per vector-level section
        vars_per_section = {s: set(v.name.lower() for v in _get_local_arrays(s)) for s in sections}

        # Count in how many sections each temporary is used
        counts = {}
        for arr in candidates:
            counts[arr] = sum(1 if arr.name.lower() in v else 0 for v in vars_per_section.values())

        # Mark temporaries that are only used in one section for demotion
        to_demote = [k for k, v in counts.items() if v == 1]

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

        # TODO: we only need this here until the scheduler can combine multiple transformations into single pass
        # Bail if routine has already been processed
        if self._processed.get(routine, None):
            return

        role = kwargs['role']
        item = kwargs.get('item', None)

        if role == 'kernel':
            demote_locals = self.demote_local_arrays
            if item:
                demote_locals = item.config.get('demote_locals', self.demote_local_arrays)
            self.process_kernel(routine, demote_locals=demote_locals)

        # Mark routine as processed
        self._processed[routine] = True

    def process_kernel(self, routine, demote_locals=True):
        """
        Applies the SCCDemote utilities to a "kernel" and demotes all suitable local arrays.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Find vector sections marked in the SCCDevectorTransformation
        sections = [s for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section']

        # Extract the local variables to demote after we wrap the sections in vector loops.
        # We do this, because need the section blocks to determine which local arrays
        # may carry buffered values between them, so that we may not demote those!
        to_demote = self.kernel_get_locals_to_demote(routine, sections, self.horizontal)

        # Demote all private local variables that do not buffer values between sections
        if demote_locals:
            variables = tuple(v.name for v in to_demote)
            if variables:
                demote_variables(routine, variable_names=variables, dimensions=self.horizontal.size)
