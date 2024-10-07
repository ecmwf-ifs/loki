# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import (
    symbols as sym, FindVariables,
)
from loki.ir import (
    nodes as ir, FindNodes, Transformer,
    is_loki_pragma, pragmas_attached,
    get_pragma_parameters
)
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.transformations.transform_loop import loop_fusion, loop_interchange
from loki.transformations.array_indexing import demote_variables
from loki.transformations.utilities import get_local_arrays
from loki.logging import info

__all__ = ['SCCFuseVerticalLoops']

class SCCFuseVerticalLoops(Transformation):
    """
    A transformation to fuse vertical loops and demote temporaries in the vertical
    dimension if possible.

    .. note::
        This transfomation currently relies on pragmas being inserted in the input
        source files. Relevant pragmas are `!$loki loop-interchange` to expose the
        vertical loops (in case vertical loops are nested) and `!$loki loop-fusion`
        possibly grouped via `group(<group name>)`. Further, if there are loops
        that initialize multilevel arrays (`jk +/- 1`) it is possible to mark those
        loops as `!$loki loop-fusion group(<group-name>-init)`. This allows to split
        the relevant node and moves the initialization of those arrays to the top of
        the group.

    Parameters
    ----------
    vertical : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the vertical data dimension and iteration space.
    """

    def __init__(self, vertical=None):
        self.vertical = vertical

    def transform_subroutine(self, routine, **kwargs):
        """
        Fuse vertical loops and demote temporaries in the vertical dimension
        if possible.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in the vertical loops should be fused and
            temporaries be demoted.
        """
        if self.vertical is None:
            info('[SCCFuseVerticalLoops] is not applied as the vertical dimension is not defined!')
            return
        role = kwargs['role']
        if role == 'kernel':
            self.process_kernel(routine)

    def process_kernel(self, routine):
        """
        Current logic (simplified):

        1. loop interchange to expose vertical loops
        2. fuse vertical loops (possibly into multiple groups)
        3. find local arrays to be demoted and apply heuristics to check whether this is safe
        4. demote those arrays which are safe to be demoted
        """
        # find local arrays with a vertical dimension
        relevant_local_arrays = self.find_relevant_local_arrays(routine)
        # find "multilevel" thus "jk +/- 1" arrays
        multilevel_relevant_local_arrays = self.identify_multilevel_arrays(relevant_local_arrays)
        # loop interchange to expose vertical loops as outermost loops
        loop_interchange(routine)
        # handle initialization of arrays "jk +/- 1" arrays
        multilevel_relevant_local_arrays_names = set(arr.name.lower() for arr in multilevel_relevant_local_arrays)
        self.correct_init_of_multilevel_arrays(routine, multilevel_relevant_local_arrays_names)
        # fuse vertical loops
        loop_fusion(routine)
        # demote in vertical dimension if possible
        relevant_local_arrays_names = set(arr.name.lower() for arr in relevant_local_arrays)
        demote_candidates = relevant_local_arrays_names - multilevel_relevant_local_arrays_names
        # check which variables are safe to demote in the vertical
        safe_to_demote = self.check_safe_to_demote(routine, demote_candidates)
        # demote locals in vertical dimension
        dimensions_to_demote = self.vertical.size_expressions + (f"{self.vertical.size}+1",)
        demote_variables(routine, safe_to_demote, dimensions_to_demote)

    def check_safe_to_demote(self, routine, demote_candidates):
        """
        Check whether variables that are candidates to be demoted in the vertical dimension are really
        safe to be demoted.

        Current heuristic: If the candidate is used in more than one vertical loop, assume it is NOT safe
        to demote!
        """
        loop_var_map = CaseInsensitiveDict()
        with pragmas_attached(routine, ir.Loop):
            for loop in FindNodes(ir.Loop).visit(routine.body):
                if loop.variable == self.vertical.index:
                    if is_loki_pragma(loop.pragma, starts_with='fused-loop'):
                        parameters = get_pragma_parameters(loop.pragma, starts_with='fused-loop')
                        group = parameters.get('group', 'default')
                        if group == 'ignore':
                            continue
                        for var in FindVariables().visit(loop.body):
                            if isinstance(var, sym.Array):
                                loop_var_map.setdefault(var.name, set()).add(group)

        safe_to_demote = ()
        for var in demote_candidates:
            if var in loop_var_map and len(loop_var_map[var]) <= 1:
                safe_to_demote += (var,)

        return safe_to_demote

    def find_relevant_local_arrays(self, routine):
        """
        Find local arrays/temporaries that do have the vertical dimension.
        """
        # local/temporary arrays
        local_arrays = get_local_arrays(routine, routine.body)
        # only those with the vertical size within shape
        relevant_local_arrays = [arr for arr in local_arrays if self.vertical.size.lower()
                in FindVariables().visit(arr.shape)]
        # filter arrays to be ignored (for whatever reason)
        ignore_names = self.find_local_arrays_to_be_ignored(routine)
        if ignore_names:
            relevant_local_arrays = [arr for arr in relevant_local_arrays if arr.name.lower() not in ignore_names]
        return relevant_local_arrays

    def find_local_arrays_to_be_ignored(self, routine):
        """
        Identify variables to be ignore regarding demotion for whatever reason.

        Reasons are:

        * explicitly marked to be ignored via pragmas within the input source file, e.g.,
          'loki k-caching ignore(var1, var2, ...)'
        """
        ignore = ()
        pragmas = FindNodes(ir.Pragma).visit(routine.body)
        # look for 'loki k-caching ignore(var1, var2, ...)' pragmas within routine and ignore those vars
        for pragma in pragmas:
            if is_loki_pragma(pragma, starts_with='k-caching'):
                if pragma_ignore := get_pragma_parameters(pragma, starts_with='k-caching').get('ignore', None):
                    ignore += as_tuple(v.strip() for v in pragma_ignore.split(','))
        ignore_names = set(var.lower() for var in ignore)
        return ignore_names

    def identify_multilevel_arrays(self, local_arrays):
        """
        Identify local arrays/temporaries that have an access in the vertical dimension
        that is different to '<vertical.index>', e.g., '<vertical.index> +/- 1'
        """
        multilevel_local_arrays = []
        for arr in local_arrays:
            for dim in arr.dimensions:
                if self.vertical.index in FindVariables().visit(dim):
                    # dim is not equal to vertical.index e.g., vertical.index +/- 1
                    if dim != self.vertical.index:
                        multilevel_local_arrays.append(arr)
        return multilevel_local_arrays

    def correct_init_of_multilevel_arrays(self, routine, multilevel_local_arrays):
        """
        Possibly handle initialization of those multilevel local arrays via
        splitting relevant loops or rather creating a new node with the relevant
        nodes moved to the newly created loop.

        .. note::
            This relies on pragmas being inserted in the input source code!
        """
        loop_map = {}
        # find/identify loops with pragma 'loop-fusion group(<group-name>-init)'
        with pragmas_attached(routine, ir.Loop):
            loop_map = {}
            for loop in FindNodes(ir.Loop).visit(routine.body):
                if is_loki_pragma(loop.pragma, starts_with='loop-fusion'):
                    parameters = get_pragma_parameters(loop.pragma, starts_with='loop-fusion')
                    group = parameters.get('group', 'default')
                    if group.endswith('-init'):
                        nodes_to_be_moved = ()
                        nodes = FindNodes(ir.Assignment).visit(loop.body)
                        node_map = {}
                        node_map_init = {}
                        # find nodes that have multilevel arrays
                        for node in nodes:
                            node_vars = FindVariables().visit(node)
                            if any(node_var.name.lower() in multilevel_local_arrays for node_var in node_vars):
                                nodes_to_be_moved += (node,)
                                node_map[node] = None
                            else:
                                node_map_init[node] = None
                        # split the loop/create a new node to move those nodes with
                        # multilevel arrays to the new node
                        if nodes_to_be_moved:
                            pragmas = loop.pragma
                            new_pragmas = [pragma.clone(content=pragma.content.replace('-init', '')) if '-init'
                                    in pragma.content else pragma for pragma in pragmas]
                            # init part
                            transf_init = Transformer(node_map_init).visit(loop.clone(\
                                    pragma=as_tuple(ir.Pragma(keyword='loki',
                                        content='fused-loop group(ignore)'))))
                            # rest of the original node/loop
                            transf_orig = Transformer(node_map).visit(loop.clone(pragma=as_tuple(new_pragmas)))
                            loop_map[loop] = (ir.Comment('! Loki generated loop for init ...'),
                                    transf_init, transf_orig)
            if loop_map:
                routine.body = Transformer(loop_map).visit(routine.body)
