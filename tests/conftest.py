# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import contextlib
import os
import io
import pytest

from loki import (
    Sourcefile, Module, Subroutine, fgen, OFP, compile_and_load, FindNodes, CallStatement,
    as_tuple, Frontend, Section, REGEX
)
from loki.build import Builder, Lib, Obj
from loki.tools import gettempdir, filehash
import loki.frontend


__all__ = [
    'generate_identity', 'jit_compile', 'jit_compile_lib', 'clean_test',
    'stdchannel_redirected', 'stdchannel_is_captured', 'available_frontends'
]

_f90wrap_kind_map = Path(__file__).parent/'kind_map'


def generate_identity(refpath, routinename, modulename=None, frontend=OFP):
    """
    Generate the "identity" of a single subroutine with a frontend-specific suffix.
    """
    testname = refpath.parent/(f'{refpath.stem}_{routinename}_{frontend}.f90')
    source = Sourcefile.from_file(refpath, frontend=frontend)

    if modulename:
        module = [m for m in source.modules if m.name == modulename][0]
        module.name += f'_{routinename}_{frontend}'
        for routine in source.all_subroutines:
            routine.name += f'_{frontend}'
            for call in FindNodes(CallStatement).visit(routine.body):  # pylint: disable=no-member
                call.name += f'_{frontend}'
        source.write(path=testname, source=fgen(module))
    else:
        routine = [r for r in source.subroutines if r.name == routinename][0]
        routine.name += f'_{frontend}'
        source.write(path=testname, source=fgen(routine))

    pymod = compile_and_load(testname, cwd=str(refpath.parent), use_f90wrap=True, f90wrap_kind_map=_f90wrap_kind_map)

    if modulename:
        # modname = '_'.join(s.capitalize() for s in refpath.stem.split('_'))
        return getattr(pymod, testname.stem)
    return pymod


def jit_compile(source, filepath=None, objname=None):
    """
    Generate, Just-in-Time compile and load a given item
    for interactive execution.

    Parameters
    ----------
    source : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine`
        The item to compile and load
    filepath : str or :any:`Path`, optional
        Path of the source file to write (default: hashed name in :any:`gettempdir()`)
    objname : str, optional
        Return a specific object (module or subroutine) in :attr:`source`
    """
    if isinstance(source, Sourcefile):
        filepath = source.filepath if filepath is None else Path(filepath)
        source.write(path=filepath)
    else:
        source = fgen(source)
        if filepath is None:
            filepath = gettempdir()/filehash(source, prefix='', suffix='.f90')
        else:
            filepath = Path(filepath)
        Sourcefile(filepath).write(source=source)

    pymod = compile_and_load(filepath, cwd=str(filepath.parent), use_f90wrap=True, f90wrap_kind_map=_f90wrap_kind_map)

    if objname:
        return getattr(pymod, objname)
    return pymod


def jit_compile_lib(sources, path, name, wrap=None, builder=None):
    """
    Generate, just-in-time compile and load a set of items into a
    library and import dynamically into the Python runtime.

    Parameters
    ----------
    source : list
        Source items or filepaths to compile and add to lib
    path : str or :any:`Path`
        Basepath for on-the-fly creation of source files
    name : str
        Name of created lib
    wrap : list, optional
        File names to pass to ``f90wrap``. Defaults to list of source files.
    builder : :any:`Builder`, optional
        Builder object to use for lib compilation and linking
    """
    builder = builder or Builder(source_dirs=path, build_dir=path)
    sourcefiles = []

    for source in sources:
        if isinstance(source, (str, Path)):
            sourcefiles.append(source)

        if isinstance(source, Sourcefile):
            filepath = source.path or path/f'{source.name}.f90'
            source.write(path=filepath)
            sourcefiles.append(source.path)

        elif isinstance(source, (Module, Subroutine)):
            filepath = path/f'{source.name}.f90'
            source = Sourcefile(filepath, ir=Section(body=as_tuple(source)))
            source.write(path=filepath)
            sourcefiles.append(source.path)

    objects = [Obj(source_path=s) for s in sourcefiles]
    lib = Lib(name=name, objs=objects, shared=False)
    lib.build(builder=builder)
    wrap = wrap or sourcefiles
    return lib.wrap(modname=name, sources=wrap, builder=builder, kind_map=_f90wrap_kind_map)


def clean_test(filepath):
    """
    Clean test directory based on JIT'ed source file.
    """
    file_list = [filepath.with_suffix('.f90'), filepath.with_suffix('.o'),
                 filepath.with_suffix('.py'), filepath.parent/'f90wrap_toplevel.f90',
                 filepath.with_suffix('.mod'), filepath.with_suffix('.xmod')]
    for f in file_list:
        if f.exists():
            f.unlink()
    for sofile in filepath.parent.glob(f'_{filepath.stem}.*.so'):
        sofile.unlink()
    f90wrap_path = filepath.parent/f'f90wrap_{filepath.name}'
    if f90wrap_path.exists():
        f90wrap_path.unlink()


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:

    ```
    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    ```

    Source: https://stackoverflow.com/a/17753573

    Note, that this only works when pytest is invoked with '--show-capture' (or '-s').
    This can be checked using `stdchannel_is_captured(capsys)`.
    Additionally, capturing of sys.stdout/sys.stderr needs to be disabled explicitly,
    i.e., use the fixture `capsys` and wrap the above:

    ```
    with capsys.disabled():
        with stdchannel_redirected(sys.stdout, 'stdout.log'):
            function()
    ```
    """

    def try_dup(fd):
        try:
            oldfd = os.dup(fd.fileno())
        except io.UnsupportedOperation:
            oldfd = None
        return oldfd

    def try_dup2(fd, fd2, fd_fileno=True):
        try:
            if fd_fileno:
                os.dup2(fd.fileno(), fd2.fileno())
            else:
                os.dup2(fd, fd2.fileno())
        except io.UnsupportedOperation:
            pass

    oldstdchannel, dest_file = None, None
    try:
        oldstdchannel = try_dup(stdchannel)
        dest_file = open(dest_filename, 'w')
        try_dup2(dest_file, stdchannel)

        yield
    finally:
        if oldstdchannel is not None:
            try_dup2(oldstdchannel, stdchannel, fd_fileno=False)
        if dest_file is not None:
            dest_file.close()


def stdchannel_is_captured(capsys):
    """
    Utility function to verify if pytest captures stdout/stderr.

    This hinders redirecting stdout/stderr for f2py/f90wrap functions.

    Parameters
    ----------
    capsys :
        The capsys fixture of the test.

    Returns
    -------
    bool
        `True` if pytest captures output, otherwise `False`.
    """

    capturemanager = capsys.request.config.pluginmanager.getplugin("capturemanager")
    return capturemanager._global_capturing.out is not None


def available_frontends(xfail=None, skip=None, include_regex=False):
    """
    Provide list of available frontends to parametrize tests with

    To run tests for every frontend, an argument :attr:`frontend` can be added to
    a test with the return value of this function as parameter.

    For any unavailable frontends where ``HAVE_<frontend>`` is `False` (e.g.
    because required dependencies are not installed), :attr:`test` is marked as
    skipped.

    Use as

    ..code-block::
        @pytest.mark.parametrize('frontend', available_frontends(xfail=[OMNI, (OFP, 'Because...')]))
        def my_test(frontend):
            source = Sourcefile.from_file('some.F90', frontend=frontend)
            # ...

    Parameters
    ----------
    xfail : list, optional
        Provide frontends that are expected to fail, optionally as tuple with reason
        provided as string. By default `None`
    skip : list, optional
        Provide frontends that are always skipped, optionally as tuple with reason
        provided as string. By default `None`
    include_regex : bool, optional
        Include the :any:`REGEX` frontend in the list. By default `false`.
    """
    if xfail:
        xfail = dict((as_tuple(f) + (None,))[:2] for f in xfail)
    else:
        xfail = {}

    if skip:
        skip = dict((as_tuple(f) + (None,))[:2] for f in skip)
    else:
        skip = {}

    # Unavailable frontends
    unavailable_frontends = {
        f: f'{f} is not available' for f in Frontend
        if not getattr(loki.frontend, f'HAVE_{str(f).upper()}')
    }
    skip.update(unavailable_frontends)

    # Build the list of parameters
    params = []
    for f in Frontend:
        if f in skip:
            params += [pytest.param(f, marks=pytest.mark.skip(reason=skip[f]))]
        elif f in xfail:
            params += [pytest.param(f, marks=pytest.mark.xfail(reason=xfail[f]))]
        elif f != REGEX or include_regex:
            params += [f]

    return params


def graphviz_present():
    """
    Test if graphviz is present and works
    The import will work as long as the graphviz python wrapper is available,
    but the underlying binaries may be missing.
    """
    try:
        import graphviz as gviz # pylint: disable=import-outside-toplevel
    except ImportError:
        return False

    try:
        gviz.Graph().pipe()
    except gviz.ExecutableNotFound:
        return False

    return True
