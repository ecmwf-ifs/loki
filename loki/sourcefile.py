"""
Module containing a set of classes to represent and manipuate a
Fortran source code file.
"""
import pickle
from pathlib import Path
from collections import OrderedDict
from fparser.two import Fortran2003

from loki.subroutine import Subroutine
from loki.module import Module
from loki.tools import flatten, as_tuple
from loki.logging import info
from loki.frontend import OMNI, OFP, FP, blacklist, read_file, Source, extract_source
from loki.frontend.omni import preprocess_omni, parse_omni_file, parse_omni_source
from loki.frontend.ofp import parse_ofp_file, parse_ofp_source
from loki.frontend.fparser import parse_fparser_file, parse_fparser_source, extract_fparser_source
from loki.types import TypeTable
from loki.backend.fgen import fgen


__all__ = ['SourceFile']


class SourceFile:
    """
    Class to handle and manipulate source files.

    :param str path: the name of the source file
    :param tuple routines: the subroutines (functions) contained in this source.
    :param tuple modules: the Fortran modules contained in this source.
    :param ast: optional parser-AST of the original source file
    :param TypeTable symbols: existing type information for all symbols defined within this
            file's scope.
    :param TypeTable types: existing type information for all (derived) types defined within
            this file's scope.
    :param Source source: string and line information about the original source file.
    """

    def __init__(self, path, routines=None, modules=None, ast=None, symbols=None, types=None,
                 source=None):
        self.path = Path(path) if path is not None else path
        self._routines = routines
        self._modules = modules
        self._ast = ast
        self.symbols = symbols if symbols is not None else TypeTable(None)
        self.types = types if types is not None else TypeTable(None)
        self._source = source

    @classmethod
    def from_file(cls, filename, preprocess=False, typedefs=None,
                  xmods=None, includes=None, builddir=None, frontend=OFP):
        if frontend == OMNI:
            return cls.from_omni(filename, typedefs=typedefs, xmods=xmods,
                                 includes=includes, builddir=builddir)
        if frontend == OFP:
            return cls.from_ofp(filename, preprocess=preprocess, typedefs=typedefs)
        if frontend == FP:
            return cls.from_fparser(filename, preprocess=preprocess, typedefs=typedefs)
        raise NotImplementedError('Unknown frontend: %s' % frontend)

    @classmethod
    def from_omni(cls, filename, preprocess=False, typedefs=None, xmods=None,
                  includes=None, builddir=None):
        """
        Use the OMNI compiler frontend to generate internal subroutine
        and module IRs.
        """
        filepath = Path(filename)
        pppath = Path(filename).with_suffix('.omni%s' % filepath.suffix)
        if builddir is not None:
            pppath = Path(builddir)/pppath.name

        preprocess_omni(filename, pppath, includes=includes)

        with filepath.open() as f:
            raw_source = f.read()

        # Parse the file content into an OMNI Fortran AST
        ast = parse_omni_file(filename=str(pppath), xmods=xmods)
        typetable = ast.find('typeTable')
        return cls._from_omni_ast(ast=ast, path=filename, raw_source=raw_source,
                                  typedefs=typedefs, typetable=typetable,
                                  xmods=xmods, includes=includes)

    @classmethod
    def _from_omni_ast(cls, ast, path=None, raw_source=None, typedefs=None, typetable=None,
                       xmods=None, includes=None):
        """
        Generate the full set of `Subroutine` and `Module` members of the `SourceFile`.
        """
        obj = cls(path=path, ast=ast)

        ast_r = ast.findall('./globalDeclarations/FfunctionDefinition')
        routines = [Subroutine.from_omni(ast=routine, typedefs=typedefs, raw_source=raw_source,
                                         typetable=typetable, parent=obj) for routine in ast_r]

        ast_m = ast.findall('./globalDeclarations/FmoduleDefinition')
        modules = [Module.from_omni(ast=module, typedefs=typedefs, raw_source=raw_source,
                                    typetable=typetable, parent=obj) for module in ast_m]

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        obj.__init__(path=path, routines=routines, modules=modules, ast=ast,
                     symbols=obj.symbols, types=obj.types, source=source)
        return obj

    @classmethod
    def from_ofp(cls, filename, preprocess=False, typedefs=None):
        """
        Parse a given source file with the OFP frontend to instantiate
        a `SourceFile` object.
        """
        file_path = Path(filename)
        info_path = file_path.with_suffix('.ofp.info')

        # Unfortunately we need a pre-processing step to sanitize
        # the input to the OFP, as it will otherwise drop certain
        # terms due to advanced bugged-ness! :(
        if preprocess:
            pp_path = file_path.with_suffix('.ofp%s' % file_path.suffix)
            cls.preprocess(OFP, file_path, pp_path, info_path)
            file_path = pp_path

        # Import and store the raw file content
        with file_path.open() as f:
            raw_source = f.read()

        # Parse the file content into a Fortran AST
        ast = parse_ofp_file(filename=str(file_path))

        # Extract subroutines and pre/post sections from file
        pp_info = None
        if info_path.exists():
            with info_path.open('rb') as f:
                pp_info = pickle.load(f)

        return cls._from_ofp_ast(path=filename, ast=ast, typedefs=typedefs,
                                 pp_info=pp_info, raw_source=raw_source)

    @classmethod
    def _from_ofp_ast(cls, ast, path=None, raw_source=None, typedefs=None, pp_info=None):
        """
        Generate the full set of `Subroutine` and `Module` members of the `SourceFile`.
        """
        obj = cls(path=path, ast=ast)

        routines = [Subroutine.from_ofp(ast=routine, raw_source=raw_source, typedefs=typedefs,
                                        parent=obj, pp_info=pp_info)
                    for routine in list(ast.find('file'))
                    if routine.tag in ('subroutine', 'function')]

        modules = [Module.from_ofp(ast=module, typedefs=typedefs, parent=obj, raw_source=raw_source,
                                   pp_info=pp_info) for module in ast.findall('file/module')]

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        obj.__init__(path=path, routines=routines, modules=modules,
                     ast=ast, symbols=obj.symbols, types=obj.types, source=source)
        return obj

    @classmethod
    def from_fparser(cls, filename, preprocess=False, typedefs=None):
        file_path = Path(filename)
        info_path = file_path.with_suffix('.fp.info')

        # Unfortunately we need a pre-processing step to sanitize
        # the input to the FP, as it will otherwise drop certain
        # terms due to missing features in FP
        if preprocess:
            pp_path = file_path.with_suffix('.fp%s' % file_path.suffix)
            cls.preprocess(FP, file_path, pp_path, info_path)
            file_path = pp_path

        # Import and store the raw file content
        with file_path.open() as f:
            raw_source = f.read()

        # Parse the file content into a Fortran AST
        ast = parse_fparser_file(filename=str(file_path))

        # Extract preprocessing replacements from file
        pp_info = None
        if info_path.exists():
            with info_path.open('rb') as f:
                pp_info = pickle.load(f)
        return cls._from_fparser_ast(path=filename, ast=ast, typedefs=typedefs,
                                     pp_info=pp_info, raw_source=raw_source)

    @classmethod
    def _from_fparser_ast(cls, ast, path=None, raw_source=None, typedefs=None, pp_info=None):
        """
        Generate the full set of `Subroutine` and `Module` members of the `SourceFile`.
        """
        obj = cls(path=path, ast=ast)

        routine_types = (Fortran2003.Subroutine_Subprogram, Fortran2003.Function_Subprogram)
        routines = [Subroutine.from_fparser(ast=routine, typedefs=typedefs, parent=obj,
                                            pp_info=pp_info, raw_source=raw_source)
                    for routine in ast.content if isinstance(routine, routine_types)]
        modules = [Module.from_fparser(ast=module, typedefs=typedefs, parent=obj,
                                       pp_info=pp_info, raw_source=raw_source)
                   for module in ast.content if isinstance(module, Fortran2003.Module)]

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        obj.__init__(path=path, routines=routines, modules=modules, ast=ast, symbols=obj.symbols,
                     types=obj.types, source=source)
        return obj

    @classmethod
    def from_source(cls, source, xmods=None, includes=None, typedefs=None, frontend=OFP):

        if frontend == OMNI:
            ast = parse_omni_source(source, xmods=xmods)
            typetable = ast.find('typeTable')
            return cls._from_omni_ast(path=None, ast=ast, raw_source=source, typedefs=typedefs,
                                      typetable=typetable, xmods=xmods, includes=includes)

        if frontend == OFP:
            ast = parse_ofp_source(source)
            return cls._from_ofp_ast(path=None, ast=ast, raw_source=source, typedefs=typedefs)

        if frontend == FP:
            ast = parse_fparser_source(source)
            return cls._from_fparser_ast(path=None, ast=ast, raw_source=source, typedefs=typedefs)

        raise NotImplementedError('Unknown frontend: %s' % frontend)

    @classmethod
    def preprocess(cls, frontend, file_path, pp_path, info_path):
        """
        A dedicated pre-processing step to ensure smooth source parsing.
        """
        # Check for previous preprocessing of this file
        if pp_path.exists() and info_path.exists():
            # Make sure the existing PP data belongs to this file
            if pp_path.stat().st_mtime > file_path.stat().st_mtime:
                with info_path.open('rb') as f:
                    pp_info = pickle.load(f)
                    if pp_info.get('original_file_path') == str(file_path):
                        # Already pre-processed this one, skip!
                        return

        info("Pre-processing %s => %s" % (file_path, pp_path))
        source = read_file(file_path)

        # Apply preprocessing rules and store meta-information
        pp_info = OrderedDict()
        pp_info['original_file_path'] = str(file_path)
        for name, rule in blacklist[frontend].items():
            # Apply rule filter over source file
            rule.reset()
            new_source = ''
            for ll, line in enumerate(source.splitlines(keepends=True)):
                ll += 1  # Correct for Fortran counting
                new_source += rule.filter(line, lineno=ll)

            # Store met-information from rule
            pp_info[name] = rule.info
            source = new_source

        with pp_path.open('w') as f:
            f.write(source)

        with info_path.open('wb') as f:
            pickle.dump(pp_info, f)

    @property
    def source(self):
        return self._source

    def to_fortran(self, conservative=False):
        return fgen(self, conservative=conservative)

    @property
    def modules(self):
        return as_tuple(self._modules)

    @property
    def subroutines(self):
        return as_tuple(self._routines)

    @property
    def all_subroutines(self):
        routines = as_tuple(self._routines)
        routines += as_tuple(flatten(m.subroutines for m in self.modules))
        return routines

    def __getitem__(self, name):
        module_map = {m.name.lower(): m for m in self.modules}
        if name.lower() in module_map:
            return module_map[name.lower()]

        subroutine_map = {s.name.lower(): s for s in self.all_subroutines}
        if name.lower() in subroutine_map:
            return subroutine_map[name.lower()]

        return None

    def apply(self, op, **kwargs):
        """
        Apply a given transformation to the source file object.

        Note that the dispatch routine `op.apply(source)` will ensure
        that all entities of this `SourceFile` are correctly traversed.
        """
        # TODO: Should type-check for an `Operation` object here
        op.apply(self, **kwargs)

    def write(self, path=None, source=None, conservative=False):
        """
        Write content to file

        :param str path: Optional filepath; if not provided, `self.path` is used
        :param str source: Optional source string; if not provided `self.to_fortran()` is used
        :param bool conservative: Enable conservative output of the backend.
        """
        path = self.path if path is None else Path(path)
        source = self.to_fortran(conservative) if source is None else source
        self.to_file(source=source, path=path)

    @classmethod
    def to_file(cls, source, path):
        """
        Same as ``write(source, filename)``, but can be called from a
        static context.
        """
        info("Writing %s" % path)
        with path.open('w') as f:
            f.write(source)
