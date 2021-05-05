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
    """

    def __init__(self, name=None, index=None, bounds=None, size=None, aliases=None):
        self.name = name
        self._index = index
        self._bounds = as_tuple(bounds)
        self._size = size
        self._aliases = as_tuple(aliases)

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
        return '{}:{}'.format(self._bounds[0], self._bounds[1])

    @property
    def size_expressions(self):
        """
        A list of all expression strings representing the size of a data space.

        This includes generic aliases, like ``end - start + 1`` or ``1:size`` ranges.
        """
        exprs = as_tuple(self.size)
        exprs += as_tuple(self._aliases)
        exprs += ('1:{}'.format(self.size), )
        exprs += ('{} - {} + 1'.format(self._bounds[1], self._bounds[0]), )
        return exprs
