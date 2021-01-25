"""
Module containing a set of classes to represent and manipuate a
Fortran source code file.
"""
from pathlib import Path
from fparser.two import Fortran2003

from loki.subroutine import Subroutine
from loki.module import Module
from loki.tools import flatten, as_tuple
from loki.logging import info
from loki.frontend import OMNI, OFP, FP, sanitize_input, Source, read_file, preprocess_cpp
from loki.frontend.omni import parse_omni_source
from loki.frontend.ofp import parse_ofp_source
from loki.frontend.fparser import parse_fparser_source
from loki.backend.fgen import fgen


__all__ = ['Sourcefile']


class Sourcefile:
    """
    Class to handle and manipulate source files.

    :param str path: the name of the source file
    :param tuple routines: the subroutines (functions) contained in this source.
    :param tuple modules: the Fortran modules contained in this source.
    :param ast: optional parser-AST of the original source file
    :param Source source: string and line information about the original source file.
    """

    def __init__(self, path, routines=None, modules=None, ast=None, source=None):
        self.path = Path(path) if path is not None else path
        self._routines = routines
        self._modules = modules
        self._ast = ast
        self._source = source

    @classmethod
    def from_file(cls, filename, definitions=None, preprocess=False,
                  includes=None, defines=None, omni_includes=None,
                  xmods=None, frontend=FP):
        """
        Constructor from raw source files that can apply a
        C-preprocessor before invoking frontend parsers.

        Parameters:
        ===========
        * ``filename``: Name of the file to parse into a ``Sourcefile`` object.
        * ``definitions``: (List of) ``Module`` object(s) that may supply external
                           type of procedure definitions.
        * ``preprocess``: Flag to trigger CPP preprocessing (by default ``False``)
        * ``includes``: (List of) include paths to pass to the C-preprocessor.
        * ``defines``: (List of) symbol definitions to pass to the C-preprocessor.
        * ``xmods``: (Optional) path to directory to find and store ``.xmod`` files
                     when using the OMNI frontend.
        * ``omni_includes``: (List of) additional include paths to pass to the
                             preprocessor run as part of the OMNI frontend parse.
                             If set, this replaces(!) ``includes``, if not ``includes``
                             will be used instead.
        * ``frontend``: Frontend to use for AST parsing (default FP).

        Please note that, when using the OMNI frontend, C-preprocessing will always
        be applied, so ``includes`` and ``defines`` may need to be defined eve with
        ``preprocess=False``.
        """
        filepath = Path(filename)
        raw_source = read_file(filepath)

        if preprocess:
            # Trigger CPP-preprocessing explicitly, as includes and
            # defines can also be used by our OMNI frontend
            source = preprocess_cpp(source=raw_source, filepath=filepath,
                                    includes=includes, defines=defines)
        else:
            source = raw_source

        if frontend == OMNI:
            return cls.from_omni(source, filepath, definitions=definitions,
                                 includes=includes, defines=defines,
                                 xmods=xmods, omni_includes=omni_includes)

        if frontend == OFP:
            return cls.from_ofp(source, filepath, definitions=definitions)

        if frontend == FP:
            return cls.from_fparser(source, filepath, definitions=definitions)

        raise NotImplementedError('Unknown frontend: %s' % frontend)

    @classmethod
    def from_omni(cls, raw_source, filepath, definitions=None, includes=None,
                  defines=None, xmods=None, omni_includes=None):
        """
        Use the OMNI compiler frontend to generate internal subroutine
        and module IRs.
        """

        # Always CPP-preprocess source files for OMNI, but optionally
        # use a different set of include paths if specified that way.
        # (It's a hack, I know, but OMNI sucks, so what can I do...?)
        if omni_includes is not None and len(omni_includes) > 0:
            includes = omni_includes
        source = preprocess_cpp(raw_source, filepath=filepath,
                                includes=includes, defines=defines)

        # Parse the file content into an OMNI Fortran AST
        ast = parse_omni_source(source=source, filepath=filepath, xmods=xmods)
        typetable = ast.find('typeTable')
        return cls._from_omni_ast(ast=ast, path=filepath, raw_source=raw_source,
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
    def from_ofp(cls, raw_source, filepath, definitions=None):
        """
        Parse a given source file with the OFP frontend to instantiate
        a `Sourcefile` object.
        """

        # Preprocess using internal frontend-specific PP rules
        # to sanitize input and work around known frontend problems.
        source, pp_info = sanitize_input(source=raw_source, frontend=OFP, filepath=filepath)

        # Parse the file content into a Fortran AST
        ast = parse_ofp_source(source, filepath=filepath)

        return cls._from_ofp_ast(path=filepath, ast=ast, definitions=definitions,
                                 pp_info=pp_info, raw_source=raw_source)

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
    def from_fparser(cls, raw_source, filepath, definitions=None):

        # Preprocess using internal frontend-specific PP rules
        # to sanitize input and work around known frontend problems.
        source, pp_info = sanitize_input(source=raw_source, frontend=FP, filepath=filepath)

        # Parse the file content into a Fortran AST
        ast = parse_fparser_source(source)

        return cls._from_fparser_ast(path=filepath, ast=ast, definitions=definitions,
                                     pp_info=pp_info, raw_source=raw_source)

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

    def __iter__(self):
        raise TypeError('Sourcefiles alone cannot be traversed! Try traversing "Sourcefile.subroutines".')

    def __bool__(self):
        """
        Ensure existing objects register as True in boolean checks, despite
        raising exceptions in `__iter__`.
        """
        return True

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
