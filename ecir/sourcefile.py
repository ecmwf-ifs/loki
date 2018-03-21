"""
Module containing a set of classes to represent and manipuate a
Fortran source code file.
"""
import re
import time
import pickle
import open_fortran_parser
from pathlib import Path
from collections import Iterable, defaultdict

from ecir.subroutine import Section, Subroutine, Module
from ecir.tools import disk_cached, timeit
from ecir.logging import info, INFO


__all__ =['FortranSourceFile']


class FortranSourceFile(object):
    """
    Class to handle and manipulate Fortran source files.

    :param filename: Name of the input source file
    """

    # Default custom KIND identifiers to use for pre-processing
    _kinds = ['JPIM', 'JPRB']

    def __init__(self, filename, preprocess=True, typedefs=None):
        self.path = Path(filename)
        info_path = self.path.with_suffix('.pp.info')
        file_path = self.path

        # Unfortunately we need a pre-processing step to sanitize
        # the input to the OFP, as it will otherwise drop certain
        # terms due to advanced bugged-ness! :(
        if preprocess:
            pp_path = self.path.with_suffix('.pp.F90')
            self.preprocess(pp_path, info_path)
            file_path = pp_path

        # Import and store the raw file content
        with file_path.open() as f:
            self._raw_source = f.read()

        # Parse the file content into a Fortran AST
        self._ast = self.parse_ast(filename=str(file_path))

        # Extract subroutines and pre/post sections from file
        pp_info = None
        if info_path.exists():
            with info_path.open('rb') as f:
                pp_info = pickle.load(f)

        self.subroutines = [Subroutine(ast=r, raw_source=self._raw_source,
                                       typedefs=typedefs, pp_info=pp_info)
                            for r in self._ast.findall('file/subroutine')]
        self.modules = [Module(ast=m, raw_source=self._raw_source)
                        for m in self._ast.findall('file/module')]

    def preprocess(self, pp_path, info_path, kinds=None):
        """
        A dedicated pre-processing step to ensure smooth source parsing.

        Note: The OFP drops jumbles up valid expression nodes if it
        encounters _KIND type casts (issue #48). To avoid this, we
        remove these here and create a record of the literals and
        their _KINDs, indexed by line. This allows us to the re-insert
        this information after the AST parse when creating `Subroutine`s.
        """
        if pp_path.exists():
            if pp_path.stat().st_mtime > self.path.stat().st_mtime:
                # Already pre-processed this one, skip!
                return
        info("Pre-processing %s => %s" % (self.path, pp_path))

        def repl_number_kind(match):
            m = match.groupdict()
            return m['number'] if m['kind'] in self._kinds else m['all']

        ll_kind_map = defaultdict(list)
        re_number = re.compile('(?P<all>(?P<number>[0-9.]+[eE]?[0-9\-]*)_(?P<kind>[a-zA-Z]+))')
        source = ''
        with self.path.open() as f:
            for ll, line in enumerate(f):
                ll += 1  # Correct for Fortran counting
                matches = re_number.findall(line)
                for m in matches:
                    if m[2] in self._kinds:
                        line = line.replace(m[0], m[1])
                        ll_kind_map[ll] += [(m[1], m[2])]
                source += line

        with pp_path.open('w') as f:
            f.write(source)

        with info_path.open('wb') as f:
            pickle.dump(ll_kind_map, f)

    @timeit()
    @disk_cached(argname='filename')
    def parse_ast(self, filename):
        """
        Read and parse a source file usign the Open Fortran Parser.

        Note: The parsing is cached on disk in ``<filename>.cache``.
        """
        info("Parsing %s" % filename)
        return open_fortran_parser.parse(filename)

    @property
    def source(self):
        content = self.modules + self.subroutines
        return '\n\n'.join(s.source for s in content)

    def write(self, source=None, filename=None):
        """
        Write content to file

        :param source: Optional source string; if not provided `self.source` is used
        :param filename: Optional filename; if not provided, `self.name` is used
        """
        filename = filename or '%s.F90' % self.basename
        source = self.source if source is None else source
        with open(filename, 'w') as f:
            f.write(source)

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
