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
    """

    def __init__(self, name=None, index=None, bounds=None, size=None, aliases=None,
                 bounds_aliases=None):
        self.name = name
        self._index = index
        self._bounds = as_tuple(bounds)
        self._size = size
        self._aliases = as_tuple(aliases)

        if bounds_aliases:
            if len(bounds_aliases) != 2:
                raise RuntimeError(f'Start and end both needed for horizontal bounds aliases in {self.name}')
            if bounds_aliases[0].split('%')[0] != bounds_aliases[1].split('%')[0]:
                raise RuntimeError(f'Inconsistent root name for horizontal bounds aliases in {self.name}')

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
        return self._size

    @property
    def index(self):
        """
        String that matches the primary index expression of an iteration space (loop).
        """
        return self._index

    @property
    def bounds(self):
        """
        Tuple of expression string that represent the bounds of an iteration space.
        """
        return self._bounds

    @property
    def range(self):
        """
        String that matches the range expression of an iteration space (loop).
        """
        return f'{self._bounds[0]}:{self._bounds[1]}'

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
        if self._bounds:
            exprs += (f'{self._bounds[1]} - {self._bounds[0]} + 1', )
        return exprs

    @property
    def bounds_expressions(self):
        """
        A list of all expression strings representing the bounds of a data space.
        """

        exprs = [(b,) for b in self.bounds]
        if self._bounds_aliases:
            exprs = [expr + (b,) for expr, b in zip(exprs, self._bounds_aliases)]

        return as_tuple(exprs)
