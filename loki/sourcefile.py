"""
Module containing a set of classes to represent and manipuate a
Fortran source code file.
"""
import pickle
from pathlib import Path
from fparser.two import Fortran2003

from loki.subroutine import Subroutine
from loki.module import Module
from loki.tools import flatten, as_tuple
from loki.logging import info
from loki.frontend import OMNI, OFP, FP, preprocess_internal, Source, read_file
from loki.frontend.omni import preprocess_omni, parse_omni_file, parse_omni_source
from loki.frontend.ofp import parse_ofp_file, parse_ofp_source
from loki.frontend.fparser import parse_fparser_file, parse_fparser_source
from loki.types import TypeTable
from loki.backend.fgen import fgen


__all__ = ['Sourcefile']


class Sourcefile:
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

    def __init__(self, path, routines=None, modules=None, ast=None, source=None):
        self.path = Path(path) if path is not None else path
        self._routines = routines
        self._modules = modules
        self._ast = ast
        self._source = source

    @classmethod
    def from_file(cls, filename, preprocess=False, definitions=None,
                  xmods=None, includes=None, builddir=None, frontend=OFP):
        if frontend == OMNI:
            return cls.from_omni(filename, definitions=definitions, xmods=xmods,
                                 includes=includes, builddir=builddir)
        if frontend == OFP:
            return cls.from_ofp(filename, definitions=definitions)
        if frontend == FP:
            return cls.from_fparser(filename, definitions=definitions)
        raise NotImplementedError('Unknown frontend: %s' % frontend)

    @classmethod
    def from_omni(cls, filename, definitions=None, xmods=None, includes=None, builddir=None):
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
                                  definitions=definitions, typetable=typetable)

    @classmethod
    def _from_omni_ast(cls, ast, path=None, raw_source=None, definitions=None, typetable=None):
        """
        Generate the full set of `Subroutine` and `Module` members of the `Sourcefile`.
        """
        ast_r = ast.findall('./globalDeclarations/FfunctionDefinition')
        routines = [Subroutine.from_omni(ast=routine, definitions=definitions, raw_source=raw_source,
                                         typetable=typetable) for routine in ast_r]

        ast_m = ast.findall('./globalDeclarations/FmoduleDefinition')
        modules = [Module.from_omni(ast=module, definitions=definitions, raw_source=raw_source,
                                    typetable=typetable) for module in ast_m]

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        return cls(path=path, routines=routines, modules=modules, ast=ast, source=source)

    @classmethod
    def from_ofp(cls, filename, definitions=None):
        """
        Parse a given source file with the OFP frontend to instantiate
        a `Sourcefile` object.
        """
        filepath = Path(filename)

        # Preprocess using internal frontend-specific PP rules
        # to sanitize input and work around known frontend problems.
        source, pp_info = preprocess_internal(OFP, filepath=filepath)

        # Parse the file content into a Fortran AST
        ast = parse_ofp_source(source)

        return cls._from_ofp_ast(path=filename, ast=ast, definitions=definitions,
                                 pp_info=pp_info, raw_source=source)

    @classmethod
    def _from_ofp_ast(cls, ast, path=None, raw_source=None, definitions=None, pp_info=None):
        """
        Generate the full set of `Subroutine` and `Module` members of the `Sourcefile`.
        """
        routines = [Subroutine.from_ofp(ast=routine, raw_source=raw_source,
                                        definitions=definitions, pp_info=pp_info)
                    for routine in list(ast.find('file'))
                    if routine.tag in ('subroutine', 'function')]

        modules = [Module.from_ofp(ast=module, definitions=definitions, raw_source=raw_source,
                                   pp_info=pp_info) for module in ast.findall('file/module')]

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        return cls(path=path, routines=routines, modules=modules, ast=ast, source=source)

    @classmethod
    def from_fparser(cls, filename, definitions=None):
        filepath = Path(filename)

        # Preprocess using internal frontend-specific PP rules
        # to sanitize input and work around known frontend problems.
        source, pp_info = preprocess_internal(FP, filepath=filepath)

        # Parse the file content into a Fortran AST
        ast = parse_fparser_source(source)

        return cls._from_fparser_ast(path=filename, ast=ast, definitions=definitions,
                                     pp_info=pp_info, raw_source=source)

    @classmethod
    def _from_fparser_ast(cls, ast, path=None, raw_source=None, definitions=None, pp_info=None):
        """
        Generate the full set of `Subroutine` and `Module` members of the `Sourcefile`.
        """
        routine_types = (Fortran2003.Subroutine_Subprogram, Fortran2003.Function_Subprogram)
        routines = [Subroutine.from_fparser(ast=routine, definitions=definitions,
                                            pp_info=pp_info, raw_source=raw_source)
                    for routine in ast.content if isinstance(routine, routine_types)]
        modules = [Module.from_fparser(ast=module, definitions=definitions,
                                       pp_info=pp_info, raw_source=raw_source)
                   for module in ast.content if isinstance(module, Fortran2003.Module)]

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        return cls(path=path, routines=routines, modules=modules, ast=ast, source=source)

    @classmethod
    def from_source(cls, source, xmods=None, definitions=None, frontend=OFP):

        if frontend == OMNI:
            ast = parse_omni_source(source, xmods=xmods)
            typetable = ast.find('typeTable')
            return cls._from_omni_ast(path=None, ast=ast, raw_source=source,
                                      definitions=definitions, typetable=typetable)

        if frontend == OFP:
            ast = parse_ofp_source(source)
            return cls._from_ofp_ast(path=None, ast=ast, raw_source=source, definitions=definitions)

        if frontend == FP:
            ast = parse_fparser_source(source)
            return cls._from_fparser_ast(path=None, ast=ast, raw_source=source, definitions=definitions)

        raise NotImplementedError('Unknown frontend: %s' % frontend)

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
        that all entities of this `Sourcefile` are correctly traversed.
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
