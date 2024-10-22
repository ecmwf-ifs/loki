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
    index : string
        String representation of the predominant loop index variable
        associated with this dimension.
    size : string
        String representation of the predominant size variable used
        to declare array shapes using this dimension.
    bounds : tuple of strings
        String representation of the variables usually used to denote
        the iteration bounds of this dimension.
    aliases : list or tuple of strings
        String representations of alternative size variables that are
        used to define arrays shapes of this dimension (eg. alternative
        names used in "driver" subroutines).
    bounds_aliases : list or tuple of strings
        String representations of alternative bounds variables that are
        used to define loop ranges.
    index_aliases : list or tuple of strings
        String representations of alternative loop index variables associated
        with this dimension.
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
            lower, upper = bounds

        # Store one or more strings for dimension variables
        self._index = as_tuple(index)
        self._size = as_tuple(size)
        self._lower = as_tuple(lower)
        self._upper = as_tuple(upper)
        self._step = as_tuple(step)

        self._aliases = as_tuple(aliases)
        self._index_aliases = as_tuple(index_aliases)

        self._bounds_aliases = as_tuple(bounds_aliases)

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
    def size(self):
        """
        String that matches the size expression of a data space (variable allocation).
        """
        return self._size[0] if len(self._size) == 1 else self._size

    @property
    def index(self):
        """
        String that matches the primary index expression of an iteration space (loop).
        """
        return self._index[0] if len(self._index) == 1 else self._index

    @property
    def lower(self):
        """
        String or tuple of strings that matches the lower bound of the iteration space.
        """
        return self._lower[0] if len(self._lower) == 1 else self._lower

    @property
    def upper(self):
        """
        String or tuple of strings that matches the upper bound of the iteration space.
        """
        return self._upper[0] if len(self._upper) == 1 else self._upper

    @property
    def step(self):
        """
        String or tuple of strings that matches the step size of the iteration space.
        """
        return self._step[0] if len(self._step) == 1 else self._step

    @property
    def bounds(self):
        """
        Tuple of expression string that represent the bounds of an iteration space.
        """
        return (self.lower, self.upper)

    @property
    def range(self):
        """
        String that matches the range expression of an iteration space (loop).
        """
        return f'{self.bounds[0]}:{self.bounds[1]}'

    @property
    def size_expressions(self):
        """
        A list of all expression strings representing the size of a data space.

        This includes generic aliases, like ``end - start + 1`` or ``1:size`` ranges.
        """
        exprs = as_tuple(self.size)
        if self._aliases:
            exprs += as_tuple(self._aliases)
        exprs += (f'1:{self.size}', )
        if self.bounds:
            exprs += (f'{self.bounds[1]} - {self.bounds[0]} + 1', )
        return exprs
