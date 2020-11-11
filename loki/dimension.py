"""
Utility class to manage variables pertaining to a conceptual array dimension.
"""


__all__ = ['Dimension']


class Dimension:
    """
    Dimension that defines a one-dimensional iteration space.

    :param name: Name of the dimension, as used in data array declarations
    :param variable: Name of the iteration variable used in loops in this
                     dimension.
    :param iteration: Tuple defining the start/end variable names or values for
                      loops in this dimension.
    """

    def __init__(self, name=None, aliases=None, variable=None, iteration=None):
        self.name = name
        self.aliases = aliases
        self.variable = variable
        self.iteration = iteration

    @property
    def variables(self):
        return (self.name, self.variable) + self.iteration

    @property
    def size_expressions(self):
        """
        Return a list of expression strings all signifying "dimension size".
        """
        iteration = ['%s - %s + 1' % (self.iteration[1], self.iteration[0])]
        # Add ``1:x`` size expression for OMNI (it will insert an explicit lower bound)
        iteration += ['1:%s - %s + 1' % (self.iteration[1], self.iteration[0])]
        iteration += ['1:%s' % self.name]
        iteration += ['1:%s' % alias for alias in self.aliases]
        return [self.name] + self.aliases + iteration

    @property
    def index_expressions(self):
        """
        Return a list of expression strings all signifying potential
        dimension indices, including range accesses like `START:END`.
        """
        i_range = ['%s:%s' % (self.iteration[0], self.iteration[1])]
        # A somewhat strange expression used in VMASS bracnhes
        i_range += ['%s-%s+1' % (self.variable, self.iteration[0])]
        return [self.variable] + i_range
