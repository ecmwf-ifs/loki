"""
Module containing a set of classes to represent and manipuate a
Fortran source code file.
"""
import re
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


class SourceSection(object):
    """
    Class to handle and manipulate a source code section.

    :param name: Name of the source section.
    :param source: String with the contained source code.
    """

    def __init__(self, name, source):
        self.name = name

        self._source = source

    @property
    def source(self):
        """
        The raw source code contained in this section.
        """
        return self._source

    @property
    def lines(self):
        """
        Sanitizes source content into long lines with continuous statements.

        Note: This does not change the content of the file
        """
        return self._source.split('\n')

    @property
    def longlines(self):
        from ecir.helpers import assemble_continued_statement_from_iterator
        srciter = iter(self.lines)
        return [assemble_continued_statement_from_iterator(line, srciter)[0] for line in srciter]

    def replace(self, mapping):
        """
        Performs a line-by-line string-replacement from a given mapping

        Note: The replacement is performed on each raw line. Might
        need to improve this later to unpick linebreaks in the search
        keys.
        """
        rawlines = self.lines
        for k, v in mapping.items():
            rawlines = [line.replace(k, v) for line in rawlines]
        self._source = '\n'.join(rawlines)


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

        # Split source into sections
        self.body = SourceSection(name='body', source=self._raw_source)

        # First break source into: <pre> <SUBROUTINE ... END SUBROUTINE> <post>
        re_subroutine = re.compile('(?P<pre>.*)'
                                   '(?P<routine>SUBROUTINE.*?END SUBROUTINE)'
                                   '(?P<post>.*)', re.DOTALL)
        source = re_subroutine.search(self._raw_source).groupdict()
        self._pre = SourceSection(name='pre', source=source['pre'])
        self._post = SourceSection(name='post', source=source['post'])

        # Split subroutine body into:
        # <definition> <modules> <IMPLICIT NONE declarations> <ASSOCIATE associate> <body>
        re_body = re.compile('(?P<definition>\W*SUBROUTINE\W+[A-Z]+.*?\(.*?\)\W*?\n)'
                             '(?P<modules>.*)'
                             '(?P<declarations>IMPLICIT NONE.*)'
                             '(?P<associate>ASSOCIATE\(.*?\)\W*?\n)'
                             '(?P<body>.*)', re.DOTALL)
        routine = re_body.search(source['routine']).groupdict()
        self.definition = SourceSection(name='definition', source=routine['definition'])
        self.modules = SourceSection(name='modules', source=routine['modules'])
        self.declarations = SourceSection(name='declarations', source=routine['declarations'])
        self.associate = SourceSection(name='associate', source=routine['associate'])
        self.body = SourceSection(name='body', source=routine['body'])

    @property
    def sections(self):
        return tuple([self._pre, self.definition, self.modules,
                      self.declarations, self.associate,
                      self.body, self._post])

    @property
    def source(self):
        return ''.join(s.source for s in self.sections)

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
        return self.body.lines

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
