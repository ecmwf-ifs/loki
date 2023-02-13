# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Contains the declaration of :any:`Sourcefile` that is used to represent and
manipulate (Fortran) source code files.
"""
from pathlib import Path
from codetiming import Timer

from loki.backend.fgen import fgen
from loki.backend.cufgen import cufgen
from loki.frontend import (
    OMNI, OFP, FP, REGEX, sanitize_input, Source, read_file, preprocess_cpp,
    parse_omni_source, parse_ofp_source, parse_fparser_source,
    parse_omni_ast, parse_ofp_ast, parse_fparser_ast, parse_regex_source

)
from loki.ir import Section, RawSource, Comment, PreprocessorDirective
from loki.logging import info, perf
from loki.module import Module
from loki.program_unit import ProgramUnit
from loki.subroutine import Subroutine
from loki.tools import flatten, as_tuple


__all__ = ['Sourcefile']


class Sourcefile:
    """
    Class to handle and manipulate source files, storing :any:`Module` and
    :any:`Subroutine` objects.

    Reading existing source code from file or string can be done via
    :meth:`from_file` or :meth:`from_source`.

    Parameters
    ----------
    path : str
        The name of the source file.
    ir : :any:`Section`, optional
        The IR of the file content (including :any:`Subroutine`, :any:`Module`,
        :any:`Comment` etc.)
    ast : optional
        Parser-AST of the original source file.
    source : :any:`Source`, optional
        Raw source string and line information about the original source file.
    incomplete : bool, optional
        Mark the object as incomplete, i.e. only partially parsed. This is
        typically the case when it was instantiated using the :any:`Frontend.REGEX`
        frontend and a full parse using one of the other frontends is pending.
    """

    def __init__(self, path, ir=None, ast=None, source=None, incomplete=False):
        self.path = Path(path) if path is not None else path
        if ir is not None and not isinstance(ir, Section):
            ir = Section(body=ir)
        self.ir = ir
        self._ast = ast
        self._source = source
        self._incomplete = incomplete

    @classmethod
    def from_file(cls, filename, definitions=None, preprocess=False,
                  includes=None, defines=None, omni_includes=None,
                  xmods=None, frontend=FP, parser_classes=None):
        """
        Constructor from raw source files that can apply a
        C-preprocessor before invoking frontend parsers.

        Parameters
        ----------
        filename : str
            Name of the file to parse into a :any:`Sourcefile` object.
        definitions : list of :any:`Module`, optional
            :any:`Module` object(s) that may supply external type or procedure
            definitions.
        preprocess : bool, optional
            Flag to trigger CPP preprocessing (by default `False`).

            .. attention::
                Please note that, when using the OMNI frontend, C-preprocessing
                will always be applied, so :data:`includes` and :data:`defines`
                may have to be defined even when disabling :data:`preprocess`.

        includes : list of str, optional
            Include paths to pass to the C-preprocessor.
        defines : list of str, optional
            Symbol definitions to pass to the C-preprocessor.
        xmods : str, optional
            Path to directory to find and store ``.xmod`` files when using the
            OMNI frontend.
        omni_includes: list of str, optional
            Additional include paths to pass to the preprocessor run as part of
            the OMNI frontend parse. If set, this **replaces** (!)
            :data:`includes`, otherwise :data:`omni_includes` defaults to the
            value of :data:`includes`.
        frontend : :any:`Frontend`, optional
            Frontend to use for producing the AST (default :any:`FP`).
        """

        # Log full parses at INFO and regex scans at PERF level
        log = f'[Loki::Sourcefile] Constructed from {filename}' + ' in {:.2f}s'
        with Timer(logger=perf if frontend is REGEX else info, text=log):

            filepath = Path(filename)
            raw_source = read_file(filepath)

            if preprocess:
                # Trigger CPP-preprocessing explicitly, as includes and
                # defines can also be used by our OMNI frontend
                source = preprocess_cpp(source=raw_source, filepath=filepath,
                                        includes=includes, defines=defines)
            else:
                source = raw_source

            if frontend == REGEX:
                return cls.from_regex(source, filepath, parser_classes=parser_classes)

            if frontend == OMNI:
                return cls.from_omni(source, filepath, definitions=definitions,
                                     includes=includes, defines=defines,
                                     xmods=xmods, omni_includes=omni_includes)

            if frontend == OFP:
                return cls.from_ofp(source, filepath, definitions=definitions)

            if frontend == FP:
                return cls.from_fparser(source, filepath, definitions=definitions)

            raise NotImplementedError(f'Unknown frontend: {frontend}')

    @classmethod
    def from_omni(cls, raw_source, filepath, definitions=None, includes=None,
                  defines=None, xmods=None, omni_includes=None):
        """
        Parse a given source string using the OMNI frontend

        Parameters
        ----------
        raw_source : str
            Fortran source string
        filepath : str or :any:`pathlib.Path`
            The filepath of this source file
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        includes : list of str, optional
            Include paths to pass to the C-preprocessor.
        defines : list of str, optional
            Symbol definitions to pass to the C-preprocessor.
        xmods : str, optional
            Path to directory to find and store ``.xmod`` files when using the
            OMNI frontend.
        omni_includes: list of str, optional
            Additional include paths to pass to the preprocessor run as part of
            the OMNI frontend parse. If set, this **replaces** (!)
            :data:`includes`, otherwise :data:`omni_includes` defaults to the
            value of :data:`includes`.
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
        type_map = {t.attrib['type']: t for t in typetable}
        if ast.find('symbols') is not None:
            symbol_map = {s.attrib['type']: s for s in ast.find('symbols')}
        else:
            symbol_map = None

        ir = parse_omni_ast(
            ast=ast, definitions=definitions, raw_source=raw_source,
            type_map=type_map, symbol_map=symbol_map
        )

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        return cls(path=path, ir=ir, ast=ast, source=source)

    @classmethod
    def from_ofp(cls, raw_source, filepath, definitions=None):
        """
        Parse a given source string using the Open Fortran Parser (OFP) frontend

        Parameters
        ----------
        raw_source : str
            Fortran source string
        filepath : str or :any:`pathlib.Path`
            The filepath of this source file
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        """
        # Preprocess using internal frontend-specific PP rules
        # to sanitize input and work around known frontend problems.
        source, pp_info = sanitize_input(source=raw_source, frontend=OFP)

        # Parse the file content into a Fortran AST
        ast = parse_ofp_source(source, filepath=filepath)

        return cls._from_ofp_ast(path=filepath, ast=ast, definitions=definitions,
                                 pp_info=pp_info, raw_source=raw_source)

    @classmethod
    def _from_ofp_ast(cls, ast, path=None, raw_source=None, definitions=None, pp_info=None):
        """
        Generate the full set of :any:`Subroutine` and :any:`Module` members
        in the :any:`Sourcefile`.
        """
        ir = parse_ofp_ast(ast.find('file'), pp_info=pp_info, definitions=definitions, raw_source=raw_source)

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        return cls(path=path, ir=ir, ast=ast, source=source)

    @classmethod
    def from_fparser(cls, raw_source, filepath, definitions=None):
        """
        Parse a given source string using the fparser frontend

        Parameters
        ----------
        raw_source : str
            Fortran source string
        filepath : str or :any:`pathlib.Path`
            The filepath of this source file
        definitions : list
            List of external :any:`Module` to provide derived-type and procedure declarations
        """
        # Preprocess using internal frontend-specific PP rules
        # to sanitize input and work around known frontend problems.
        source, pp_info = sanitize_input(source=raw_source, frontend=FP)

        # Parse the file content into a Fortran AST
        ast = parse_fparser_source(source)

        return cls._from_fparser_ast(path=filepath, ast=ast, definitions=definitions,
                                     pp_info=pp_info, raw_source=raw_source)

    @classmethod
    def _from_fparser_ast(cls, ast, path=None, raw_source=None, definitions=None, pp_info=None):
        """
        Generate the full set of :any:`Subroutine` and :any:`Module` members
        in the :any:`Sourcefile`.
        """
        ir = parse_fparser_ast(ast, pp_info=pp_info, definitions=definitions, raw_source=raw_source)

        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=path)
        return cls(path=path, ir=ir, ast=ast, source=source)

    @classmethod
    def from_regex(cls, raw_source, filepath, parser_classes=None):
        """
        Parse a given source string using the fparser frontend
        """
        ir = parse_regex_source(raw_source, parser_classes=parser_classes)
        lines = (1, raw_source.count('\n') + 1)
        source = Source(lines, string=raw_source, file=filepath)
        return cls(path=filepath, ir=ir, source=source, incomplete=True)

    @classmethod
    def from_source(cls, source, xmods=None, definitions=None, parser_classes=None, frontend=FP):
        """
        Constructor from raw source string that invokes specified frontend parser

        Parameters
        ----------
        source : str
            Fortran source string
        xmods : str, optional
            Path to directory to find and store ``.xmod`` files when using the
            OMNI frontend.
        definitions : list of :any:`Module`, optional
            :any:`Module` object(s) that may supply external type or procedure
            definitions.
        frontend : :any:`Frontend`, optional
            Frontend to use for producing the AST (default :any:`FP`).
        """
        if frontend == REGEX:
            return cls.from_regex(source, filepath=None, parser_classes=parser_classes)

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

        raise NotImplementedError(f'Unknown frontend: {frontend}')

    def make_complete(self, **frontend_args):
        """
        Trigger a re-parse of the source file if incomplete to produce a full Loki IR

        If the source file is marked to be incomplete, i.e. when using the `lazy` constructor
        option, this triggers a new parsing of all :any:`ProgramUnit` objects and any
        :any:`RawSource` nodes in the :attr:`Sourcefile.ir`.

        Existing :any:`Module` and :any:`Subroutine` objects continue to exist and references
        to them stay valid, as they will only be updated instead of replaced.
        """
        if not self._incomplete:
            return

        log = f'[Loki::Sourcefile] Finished constructing from {self.path}' + ' in {:.2f}s'
        with Timer(logger=info, text=log):

            # Sanitize frontend_args
            frontend = frontend_args.pop('frontend', FP)
            if frontend == REGEX:
                frontend_argnames = ['parser_classes']
            elif frontend == OMNI:
                frontend_argnames = ['definitions', 'type_map', 'symbol_map', 'scope']
                xmods = frontend_args.get('xmods')
            elif frontend in (OFP, FP):
                frontend_argnames = ['definitions', 'scope']
            else:
                raise NotImplementedError(f'Unknown frontend: {frontend}')
            sanitized_frontend_args = {k: frontend_args.get(k) for k in frontend_argnames}

            body = []
            for node in self.ir.body:
                if isinstance(node, ProgramUnit):
                    node.make_complete(frontend=frontend, **frontend_args)
                    body += [node]
                elif isinstance(node, RawSource):
                    # Sanitize the input code to ensure non-supported features
                    # do not break frontend parsing ourside of program units
                    raw_source = node.source.string
                    source, pp_info = sanitize_input(source=raw_source, frontend=frontend)

                    # Typically, this should only be comments, PP statements etc., therefore
                    # we are not bothering with type tables, definitions or similar to parse them
                    if frontend == REGEX:
                        ir_ = parse_regex_source(source, **sanitized_frontend_args)
                    elif frontend == OMNI:
                        ast = parse_omni_source(source=source, xmods=xmods)
                        ir_ = parse_omni_ast(ast=ast, raw_source=raw_source, **sanitized_frontend_args)
                    elif frontend == OFP:
                        ast = parse_ofp_source(source=source)
                        ir_ = parse_ofp_ast(ast, raw_source=raw_source, pp_info=pp_info,
                                            **sanitized_frontend_args)
                    elif frontend == FP:
                        # Fparser is unable to parse comment-only source files/strings,
                        # so we see if this is only comments and convert them ourselves
                        # (https://github.com/stfc/fparser/issues/375)
                        # This can be removed once fparser 0.0.17 is released
                        lines = [l.lstrip() for l in source.splitlines()]
                        if all(not l or l[0] in '!#' for l in lines):
                            ir_ = [
                                PreprocessorDirective(text=line.string, source=line)
                                if line.string.lstrip().startswith('#')
                                else Comment(text=line.string, source=line)
                                for line in node.source.clone_lines()
                            ]
                        else:
                            ast = parse_fparser_source(source)
                            ir_ = parse_fparser_ast(ast, raw_source=raw_source, pp_info=pp_info,
                                                    **sanitized_frontend_args)
                    else:
                        raise NotImplementedError(f'Unknown frontend: {frontend}')
                    if isinstance(ir_, Section):
                        ir_ = ir_.body
                    body += flatten([ir_])
                else:
                    body += [node]

            self.ir._update(body=as_tuple(body))
            self._incomplete = frontend == REGEX

    @property
    def source(self):
        return self._source

    def to_fortran(self, conservative=False, cuf=False):
        if cuf:
            return cufgen(self, conservative=conservative)
        return fgen(self, conservative=conservative)

    @property
    def modules(self):
        """
        List of :class:`Module` objects that are members of this :class:`Sourcefile`.
        """
        if self.ir is None:
            return ()
        return as_tuple(
            module for module in self.ir.body if isinstance(module, Module)
        )

    @property
    def routines(self):
        """
        List of :class:`Subroutine` objects that are members of this :class:`Sourcefile`.
        """
        if self.ir is None:
            return ()
        return as_tuple(
            routine for routine in self.ir.body if isinstance(routine, Subroutine)
        )

    subroutines = routines

    @property
    def typedefs(self):
        """
        List of :class:`TypeDef` objects that are declared in the :any:`Module` in this :class:`Sourcefile`.
        """
        if self.ir is None:
            return ()
        return as_tuple(
            tdef for module in self.modules for tdef in module.typedefs.values()
        )

    @property
    def all_subroutines(self):
        routines = self.subroutines
        routines += as_tuple(flatten(m.subroutines for m in self.modules))
        return routines

    @property
    def definitions(self):
        """
        List of all definitions made in this sourcefile, i.e. modules, subroutines and types
        """
        return self.modules + self.subroutines + self.typedefs

    def __getitem__(self, name):
        module_map = {m.name.lower(): m for m in self.modules}
        if name.lower() in module_map:
            return module_map[name.lower()]

        subroutine_map = {s.name.lower(): s for s in self.all_subroutines}  # pylint: disable=no-member
        if name.lower() in subroutine_map:
            return subroutine_map[name.lower()]

        for module in module_map.values():
            typedef_map = module.typedefs
            if name in typedef_map:
                return typedef_map[name]

        return None

    def __iter__(self):
        raise TypeError('Sourcefiles alone cannot be traversed! Try traversing "Sourcefile.ir".')

    def __bool__(self):
        """
        Ensure existing objects register as True in boolean checks, despite
        raising exceptions in `__iter__`.
        """
        return True

    @property
    def _canonical(self):
        """
        Base definition for comparing :any:`Subroutine` objects.
        """
        return (self.path, self.ir, self.source, )

    def __eq__(self, other):
        if isinstance(other, Sourcefile):
            return self._canonical == other._canonical
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._canonical)

    def __getstate__(self):
        # Do not pickle the AST, as it is not pickle-safe for certain frontends
        _ignore = ('_ast',)
        return dict((k, v) for k, v in self.__dict__.items() if k not in _ignore)

    def apply(self, op, **kwargs):
        """
        Apply a given transformation to the source file object.

        Note that the dispatch routine `op.apply(source)` will ensure
        that all entities of this `Sourcefile` are correctly traversed.
        """
        # TODO: Should type-check for an `Operation` object here
        op.apply(self, **kwargs)

    def write(self, path=None, source=None, conservative=False, cuf=False):
        """
        Write content as Fortran source code to file

        Parameters
        ----------
        path : str, optional
            Filepath of target file; if not provided, :attr:`Sourcefile.path` is used
        source : str, optional
            Write the provided string instead of generating via :any:`Sourcefile.to_fortran`
        conservative : bool, optional
            Enable conservative output in the backend, aiming to be as much string-identical
            as possible (default: False)
        cuf: bool, optional
            To use either Cuda Fortran or Fortran backend
        """
        path = self.path if path is None else Path(path)
        source = self.to_fortran(conservative, cuf) if source is None else source
        self.to_file(source=source, path=path)

    @classmethod
    def to_file(cls, source, path):
        """
        Same as :meth:`write` but can be called from a static context.
        """
        info(f'[Loki::Sourcefile] Writing to {path}')
        with path.open('w') as f:
            f.write(source)
            if source[-1] != '\n':
                f.write('\n')
