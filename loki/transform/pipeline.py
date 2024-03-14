# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from inspect import signature


class Pipeline:
    """
    A transformation pipeline that combines multiple :any:`Transformation`
    passes and allows to apply them in unison.

    The associated :any:`Transformation` objects are constructed from keyword
    arguments in the constructor, so shared keywords get same initial value.

    Attributes
    ----------
    transformations : list of :any:`Transformation`
        The list of transformations applied to a source in this pipeline

    Parameters
    ----------
    classes : tuple of types
        A tuple of types from which to instantiate :any:`Transformation` objects.
    *args : optional
        Positional arguments that are passed on to the constructors of
        all transformations
    **kwargs : optional
        Keyword arguments that are matched to the constructor
        signature of the transformations.
    """

    def __init__(self, *args, classes=None, **kwargs):
        self.transformations = []
        for cls in classes:
            # Get signature of the trnasformation constructor
            sig = signature(cls)

            # Filter kwargs for this transformation class specifically
            t_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            # Then bind and infer the appropriate defaults
            bound = sig.bind_partial(*args, **t_kwargs)
            bound.apply_defaults()

            self.transformations.append(cls(**bound.arguments))

    def apply(self, source, **kwargs):
        """
        Apply each associated :any:`Transformation` to :data:`source`

        It dispatches to the respective :meth:`apply` of each
        :any:`Transformation` in the order specified in the constructor.

        Parameters
        ----------
        source : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine`
            The source item to transform.
        **kwargs : optional
            Keyword arguments that are passed on to the methods defining the
            actual transformation.
        """
        for trafo in self.transformations:
            trafo.apply(source, **kwargs)
