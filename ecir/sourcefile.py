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

        # Store a list of (longline, rawline) tuples
        self._long_raw = self._sanitize(self._raw_string)

    def _sanitize(self, raw_string):
        """
        Sanitize raw lines into continuous statements
        """
        from ecir.helpers import assemble_continued_statement_from_iterator
        raw_lines = raw_string.split('\n')
        srciter = iter(raw_lines)
        return [assemble_continued_statement_from_iterator(line, srciter) for line in srciter]

    @property
    def longlines(self):
        return [long for long, _ in self._long_raw]

    @property
    def rawlines(self):
        return flatten(raw for _, raw in self._long_raw)

    def write(self, filename=None):
        """
        Write content to file

        :param filename: Optional filename. If not provided, `self.name` is used
        """
        filename = filename or self.name
        with open(filename, 'w') as f:
            f.write('\n'.join(self.rawlines))

    def strip_long(self, striplines):
        """
        Strip/delete a set of lines from the file by providing the longline string.

        Note: This modifies the the "raw lines" as well as the "long lines"
        """
        # Ensure longlines are iterable
        striplines = [striplines] if isinstance(striplines, str) else striplines
        self._long_raw = [(longline, rawline) for longline, rawline in self._long_raw
                          if longline not in striplines]

    def replace(self, mapping):
        """
        Performs a string-replacement from a given mapping

        Note: The replacement is performed on the `rawlines` and the
        `longlines` are rebuilt from the `rawlines`. Might need to
        improve this later to unpick linebreaks in the search keys.
        """
        rawlines = self.rawlines
        for k, v in mapping.items():
            rawlines = [line.replace(k, v) for line in rawlines]
        self._raw_string = '\n'.join(rawlines)
        self._long_raw = self._sanitize(self._raw_string)
