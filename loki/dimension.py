# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.tools import as_tuple

__all__ = ['Dimension']


class Dimension:
    """
    Dimension object that defines a one-dimensional data and iteration space.

    Parameters
    ----------
    name : string
        Name of the dimension to identify in configurations
    index : string or tuple of str
        String representation of the predominant loop index variable
        associated with this dimension; can be one or several.
    size : string or tuple of str
        String representation of the predominant size variable used
        to declare array shapes; can be one or several.
    lower : string or tuple of str
        String representation of the lower bound variable used to
        declare iteration spaces; can be one or several.
    lower : string or tuple of str
        String representation of the upper bound variable used to
        declare iteration spaces; can be one or several.
    bounds : tuple of strings
        String representation of the variables usually used to denote
        the iteration bounds of this dimension.

        **WARNING:** This argument is deprecated, instead ``lower``
        and ``upper`` should be used.
    aliases : list or tuple of strings
        String representations of alternative size variables that are
        used to define arrays shapes of this dimension (eg. alternative
        names used in "driver" subroutines).

        **WARNING:** This argument is deprecated, instead a tuple of
        variables names should be provided for ``size``.
    bounds_aliases : list or tuple of strings
        String representations of alternative bounds variables that are
        used to define loop ranges.

        **WARNING:** This argument is deprecated, instead a tuple of
        variables names should be provided for ``lower`` and
        ``upper``.
    index_aliases : list or tuple of strings
        String representations of alternative loop index variables associated
        with this dimension.

        **WARNING:** This argument is deprecated, instead a tuple of
        variables names should be provided for ``index``.
    """

    def __init__(
            self, name=None, index=None, size=None, lower=None,
            upper=None, step=None, aliases=None, bounds=None,
            bounds_aliases=None, index_aliases=None
    ):
        self.name = name

        if bounds:
            # Backward compat for ``bounds`` contructor argument
            assert not lower and not upper and len(bounds) == 2
            lower = (bounds[0],)
            upper = (bounds[1],)

        # Store one or more strings for dimension variables
        self._index = as_tuple(index) or None
        self._size = as_tuple(size) or None
        self._lower = as_tuple(lower) or None
        self._upper = as_tuple(upper) or None
        self._step = as_tuple(step) or None

        # Keep backward-compatibility for constructor arguments
        if aliases:
            self._size += as_tuple(aliases)
        if index_aliases:
            self._index += as_tuple(index_aliases)
        if bounds_aliases:
            self._lower = as_tuple(self._lower) + (bounds_aliases[0],)
            self._upper = as_tuple(self._upper) + (bounds_aliases[1],)

    def __repr__(self):
        """ Pretty-print dimension details """
        name = f'<{self.name}>' if self.name else ''
        index = str(self.index) or ''
        size = str(self.size) or ''
        bounds = ','.join(str(b) for b in self.bounds) if self.bounds else ''
        return f'Dimension{name}[{index},{size},({bounds})]'

    @property
    def variables(self):
        return (self.index, self.size) + self.bounds

    @property
    def sizes(self):
        """
        Tuple of strings that match the primary size and all secondary size expressions.

        .. note::
            For derived expressions, like ``end - start + 1`` or
            ``1:size``, please use :any:`size_expressions`.
        """
        return self._size

    @property
    def size(self):
        """
        String that matches the primary size expression of a data space (variable allocation).
        """
        return self.sizes[0] if self.sizes else None

    @property
    def indices(self):
        """
        Tuple of strings that matche the primary index and all secondary index expressions.
        """
        return self._index

    @property
    def index(self):
        """
        String that matches the primary index expression of an iteration space (loop).
        """
        return self.indices[0] if self.indices else None

    @property
    def lower(self):
        """
        String or tuple of strings that matches the lower bound of the iteration space.
        """
        return self._lower[0] if self._lower and len(self._lower) == 1 else self._lower

    @property
    def upper(self):
        """
        String or tuple of strings that matches the upper bound of the iteration space.
        """
        return self._upper[0] if self._upper and len(self._upper) == 1 else self._upper

    @property
    def step(self):
        """
        String or tuple of strings that matches the step size of the iteration space.
        """
        return self._step[0] if self._step and len(self._step) == 1 else self._step

    @property
    def bounds(self):
        """
        Tuple of expression string that represent the bounds of an iteration space.

        .. note::
            If mutiple lower or upper bound string have been provided,
            only the first pair will be used.
        """
        return (
            self.lower[0] if isinstance(self.lower, tuple) else self.lower,
            self.upper[0] if isinstance(self.upper, tuple) else self.upper
        )

    @property
    def range(self):
        """
        String that matches the range expression of an iteration space (loop).

        .. note::
            If mutiple lower or upper bound string have been provided,
            only the first pair will be used.
        """
        return f'{self.bounds[0]}:{self.bounds[1]}'

    @property
    def size_expressions(self):
        """
        A list of all expression strings representing the size of a data space.

        This includes generic aliases, like ``end - start + 1`` or ``1:size`` ranges.
        """
        exprs = self.sizes
        exprs += (f'1:{self.size}', )
        if self.bounds:
            exprs += (f'{self.bounds[1]} - {self.bounds[0]} + 1', )
            exprs += (f'{self.bounds[0]}:{self.bounds[1]}', )
        return exprs

