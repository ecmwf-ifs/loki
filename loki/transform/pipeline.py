# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from inspect import signature, Parameter


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

            # Get all relevant constructor parameters from teh MRO,
            # but exclude catch-all kwyward args, like ``**kwargs``
            t_parameters = {
                k: v for c in cls.__mro__ for k, v in signature(c).parameters.items()
                if not v.kind == Parameter.VAR_KEYWORD
            }
            # Filter kwargs for this transformation class specifically
            t_kwargs = {k: v for k, v in kwargs.items() if k in t_parameters}

            # We need to apply our own default, if we are to honour inheritance
            t_kwargs.update({
                k: param.default for k, param in t_parameters.items()
                if k not in t_kwargs and param.default is not None
            })

            # Then instantiate with the default *args and the derived **t_kwargs
            self.transformations.append(cls(*args, **t_kwargs))

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
