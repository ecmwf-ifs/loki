# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Utilities for demoting the rank of array variables. """

from loki.ir import FindVariables, SubstituteExpressions
from loki.logging import info
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.transformations.utilities import update_variable_declarations


__all__ = ['demote_variables']


def demote_variables(routine, variable_names, dimensions):
    """
    Demote a list of array variables by removing any occurence of a
    provided set of dimension symbols.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which the variables should be promoted.
    variable_names : list of str
        The names of variables to be promoted. Matching of variables against
        names is case-insensitive.
    dimensions : :py:class:`pymbolic.Expression` or tuple
        Symbol name or tuple of symbol names representing the dimension
        to remove from all occurances of the named variables.
    """
    dimensions = as_tuple(dimensions)

    # Compare lower-case only, since we're not comparing symbols
    vnames = tuple(name.lower() for name in variable_names)

    variables = FindVariables(unique=False).visit(routine.ir)
    variables = tuple(v for v in variables if v.name.lower() in vnames)
    variables = tuple(v for v in variables if hasattr(v, 'shape'))

    if not variables:
        return

    # Record original array shapes
    shape_map = CaseInsensitiveDict({v.name: v.shape for v in variables})

    # Remove shape and dimension entries from each variable in the list
    vmap = {}
    for v in variables:
        old_shape = shape_map[v.name]
        new_shape = tuple(s for s in old_shape if s not in dimensions)
        new_dims = tuple(d for d, s in zip(v.dimensions, old_shape) if s in new_shape)

        new_type = v.type.clone(shape=new_shape or None)
        vmap[v] = v.clone(dimensions=new_dims or None, type=new_type)

    # Propagate the new dimensions to declarations and routine bodys
    routine.body = SubstituteExpressions(vmap).visit(routine.body)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)

    # Ensure all declarations with `DIMENSION` keywords are modified too!
    routine.spec = update_variable_declarations(routine.spec, vmap.values())

    info(f'[Loki::Transform] Demoted variables in {routine.name}: {", ".join(variable_names)}')
