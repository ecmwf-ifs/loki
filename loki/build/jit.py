# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
Utilities to facilitate Just-in-Time compilation for testing purposes.
"""
from pathlib import Path

from loki.backend import fgen
from loki.build.builder import Builder
from loki.build.compiler import compile_and_load
from loki.build.lib import Lib
from loki.build.obj import Obj
from loki.frontend import OFP
from loki.ir import CallStatement, FindNodes, Section
from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine
from loki.tools import as_tuple, gettempdir, filehash


__all__ = ['jit_compile', 'jit_compile_lib', 'clean_test']


_f90wrap_kind_map = Path(__file__).parent.parent/'tests/kind_map'


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
        filepath = source.path if filepath is None else Path(filepath)
        if filepath is None:
            filepath = Path(gettempdir()/filehash(source, prefix='', suffix='.f90'))
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
    file_list = [
        filepath.with_suffix('.f90'), filepath.with_suffix('.o'),
        filepath.with_suffix('.py'), filepath.parent/'f90wrap_toplevel.f90',
        filepath.with_suffix('.mod'), filepath.with_suffix('.xmod')
    ]
    for f in file_list:
        if f.exists():
            f.unlink()
    for sofile in filepath.parent.glob(f'_{filepath.stem}.*.so'):
        sofile.unlink()
    f90wrap_path = filepath.parent/f'f90wrap_{filepath.name}'
    if f90wrap_path.exists():
        f90wrap_path.unlink()
