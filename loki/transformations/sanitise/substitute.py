# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.ir import SubstituteStringExpressions


__all__ = ['SubstituteExpressionTransformation']


class SubstituteExpressionTransformation(Transformation):
    """
    A :any:`Transformation` that allows individual expressions to be
    substituted in :any:`Subroutine` objects.

    The expressions should be provided as a dictionary map of strings,
    which will be parsed in the local :any:`Subroutine` scope to
    determine the respective symbols.

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
    """

    def __init__(
            self, substitute_expressions=True, expression_map=None,
            substitute_body=True, substitute_spec=True
    ):
        self.substitute_expressions = substitute_expressions
        self.expression_map = expression_map or {}
        self.substitute_spec = substitute_spec
        self.substitute_body = substitute_body

    def transform_subroutine(self, routine, **kwargs):

        if self.substitute_expressions:
            substitute = SubstituteStringExpressions(
                self.expression_map, scope=routine
            )

            if self.substitute_spec:
                routine.spec = substitute.visit(routine.spec)

            if self.substitute_body:
                routine.body = substitute.visit(routine.body)
