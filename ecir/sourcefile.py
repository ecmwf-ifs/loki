"""
Module containing a set of classes to represent and manipuate a
Fortran source code file.
"""
from open_fortran_parser import parse
import re
import time
from collections import Iterable

from ecir.subroutine import Section, Subroutine

__all__ =['FortranSourceFile']


class FortranSourceFile(object):
    """
    Class to handle and manipulate Fortran source files.

    :param filename: Name of the input source file
    """

    def __init__(self, filename):
        self.name = filename

        # Import and store the raw file content
        with open(filename) as f:
            self._raw_source = f.read()

        # Parse the file content into a Fortran AST
        print("Parsing %s..." % filename)
        t0 = time.time()
        self._ast = parse(filename)
        t1 = time.time() - t0
        print("Parsing done! (time: %.2fs)" % t1)

        # Extract subroutines and pre/post sections from file
        ast_routines = self._ast.findall('file/subroutine')

        # Extract pre/post sections
        r_start = int(ast_routines[0].attrib['line_begin'])
        r_end = int(ast_routines[-1].attrib['line_end'])
        self._pre = Section(name='pre', source=''.join(self.lines[:r_start]))
        self._post = Section(name='post', source=''.join(self.lines[r_end:]))

        self.routines = []
        for r in ast_routines:
            lstart = int(r.attrib['line_begin'])
            lend = int(r.attrib['line_end'])
            source = ''.join(self.lines[lstart:lend])
            self.routines.append(Subroutine(name=r.attrib['name'], ast=r,
                                            source=source, raw_source=self.lines))

    @property
    def source(self):
        content = [self._pre] + self.routines + [self._post]
        return ''.join(s.source for s in content)

    def write(self, filename=None):
        """
        Write content to file

        :param filename: Optional filename. If not provided, `self.name` is used
        """
        filename = filename or self.name
        with open(filename, 'w') as f:
            f.write(self.source)

    @property
    def lines(self):
        """
        Sanitizes source content into long lines with continuous statements.

        Note: This does not change the content of the file
        """
        return self._raw_source.splitlines(keepends=True)

    @property
    def longlines(self):
        return self.body.longlines

    def replace(self, mapping):
        """
        Performs a line-by-line string-replacement from a given mapping

        Note: The replacement is performed on each raw line. Might
        need to improve this later to unpick linebreaks in the search
        keys.
        """
        for section in self.sections:
            section.replace(mapping)
