"""
Module containing a set of classes to represent and manipuate a
Fortran source code file.
"""

from collections import Iterable

__all__ =['FortranSourceFile']


def flatten(l):
    """Flatten a hierarchy of nested lists into a plain list."""
    newlist = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist


class FortranSourceFile(object):
    """
    Class to handle and manipulate Fortran source files.

    :param filename: Name of the input source file
    """

    def __init__(self, filename):
        self.name = filename

        # Import and store the raw file content
        with open(filename) as f:
            self._raw_string = f.read()

    @property
    def lines(self):
        """
        Sanitizes content into long lines with continuous statements.

        Note: This does not change the content of the file
        """
        return self._raw_string.split('\n')

    @property
    def longlines(self):
        from ecir.helpers import assemble_continued_statement_from_iterator
        srciter = iter(self.lines)
        return [assemble_continued_statement_from_iterator(line, srciter)[0] for line in srciter]

    def write(self, filename=None):
        """
        Write content to file

        :param filename: Optional filename. If not provided, `self.name` is used
        """
        filename = filename or self.name
        with open(filename, 'w') as f:
            f.write(self._raw_string)

    def replace(self, mapping):
        """
        Performs a string-replacement from a given mapping

        Note: The replacement is performed on each raw line. Might
        need to improve this later to unpick linebreaks in the search
        keys.
        """
        rawlines = self.lines
        for k, v in mapping.items():
            rawlines = [line.replace(k, v) for line in rawlines]
        self._raw_string = '\n'.join(rawlines)
