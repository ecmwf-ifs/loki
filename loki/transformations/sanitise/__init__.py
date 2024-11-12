# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
Sub-package with assorted utility :any:`Transformation` classes to
harmonize the look-and-feel of input source code.
"""
from functools import partial

from loki.batch import Transformation, Pipeline

from loki.transformations.sanitise.associates import * # noqa
from loki.transformations.sanitise.sequence_associations import * # noqa
from loki.transformations.sanitise.substitute import * # noqa


"""
:any:`Pipeline` class that provides combined access to the features
provided by the following :any:`Transformation` classes, in sequence:
1. :any:`SubstituteExpressionTransformation` - String-based generic
expression substitution.
2. :any:`AssociatesTransformation` - Full or partial resolution of
nested :any:`Associate` nodes, including optional merging of
independent association pairs.
3. :any:`SequenceAssociationTransformation` - Resolves sequence
association patterns in the call signature of :any:`CallStatement`
nodes.

Parameters
----------
substitute_expressions : bool
    Flag to trigger or suppress expression substitution
expression_map : dict of str to str
    A string-to-string map detailing the substitutions to apply.
substitute_spec : bool
    Flag to trigger or suppress expression substitution in specs.
substitute_body : bool
    Flag to trigger or suppress expression substitution in bodies.
resolve_associates : bool, default: True
    Enable full or partial resolution of only :any:`Associate`
    scopes.
merge_associates : bool, default: False
    Enable merging :any:`Associate` to the outermost possible
    scope in nested associate blocks.
start_depth : int, optional
    Starting depth for partial resolution of :any:`Associate`
    after merging.
max_parents : int, optional
    Maximum number of parent symbols for valid selector to have
    when merging :any:`Associate` nodes.
resolve_sequence_associations : bool
    Flag to trigger or suppress resolution of sequence associations
"""
SanitisePipeline = partial(
    Pipeline, classes=(
        SubstituteExpressionTransformation,
        AssociatesTransformation,
        SequenceAssociationTransformation,
    )
)


class SanitiseTransformation(Transformation):
    """
    :any:`Transformation` object to apply several code sanitisation
    steps when batch-processing large source trees via the :any:`Scheduler`.

    Parameters
    ----------
    resolve_associate_mappings : bool
        Resolve ASSOCIATE mappings in body of processed subroutines; default: True.
    resolve_sequence_association : bool
        Replace scalars that are passed to array arguments with array
        ranges; default: False.
    """

    def __init__(
            self, resolve_associate_mappings=True, resolve_sequence_association=False
    ):
        self.resolve_associate_mappings = resolve_associate_mappings
        self.resolve_sequence_association = resolve_sequence_association

    def transform_subroutine(self, routine, **kwargs):

        # Associates at the highest level, so they don't interfere
        # with the sections we need to do for detecting subroutine calls
        if self.resolve_associate_mappings:
            do_resolve_associates(routine)

        # Transform arrays passed with scalar syntax to array syntax
        if self.resolve_sequence_association:
            do_resolve_sequence_association(routine)
