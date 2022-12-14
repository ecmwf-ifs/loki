# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from os import environ
from pathlib import Path

from loki.logging import info
from loki.tools import execute, as_tuple, delete, flatten
from loki.build.compiler import clean


__all__ = ['clean_max', 'compile_all', 'compile_c', 'compile_maxj', 'generate_max',
           'get_max_includes', 'get_max_libdirs', 'get_max_libs', 'link_obj']


def get_classpath():
    """
    Extract the CLASSPATH for Maxeler tools from the environment.
    """
    cp_base = Path(environ['MAXCOMPILERDIR']) / 'lib'
    classpath = [cp_base / j for j in ('Max5Platform.jar', 'Max4Platform.jar', 'MaxCompiler.jar')]
    return ':'.join([str(p) for p in classpath])


def get_max_includes():
    """
    Build the includes for Maxeler tools from the environment.
    """
    incdirs = [Path(environ['MAXCOMPILERDIR']) / 'include']
    incdirs += [Path(environ['MAXCOMPILERDIR']) / 'include' / 'slic']
    incdirs += [Path(environ['MAXELEROSDIR']) / 'include']
    return incdirs


def get_max_libs():
    """
    Build the libraries to be linked for Maxeler tools from the environment.
    """
    return ['slic', 'maxeleros', 'm', 'pthread', 'curl']


def get_max_libdirs():
    """
    Build the library include dirs for Maxeler tools from the environment.
    """
    return [Path(environ['MAXCOMPILERDIR'])/'lib', Path(environ['MAXELEROSDIR'])/'lib']


def get_max_output_dir(build_dir, max_filename):
    """
    Generates the (absolute) directory name in which output from maxJavaRun is stored.
    """
    return (Path(build_dir)/f'{max_filename}_MAX5C_DFE_SIM'/'results').resolve()


def clean_max(build_dir, package=None):
    """
    Cleans output from a previous build.
    """
    filepath = Path(build_dir)
    dirs = ['*_MAX5C_DFE_SIM'] + ([package] or [])
    for p in as_tuple(dirs):
        for f in filepath.glob(p):
            delete(f, force=True)


def compile_maxj(src, build_dir):
    """
    Compiles maxj-files to java binaries.

    :param src: Path to maxj source files.
    :param target: Root build directory for class files.
    """
    info(f'Compiling .maxj: {src}')
    src = Path(src).resolve()

    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)

    build = ['maxjc', '-classpath', get_classpath(), '-1.8']
    build += ['-d', '.', str(src)]
    execute(build, cwd=build_dir)


def generate_max(manager, maxj_src, max_filename, build_dir, package=None):
    """
    Generate max-file (and matching header) using java binaries.

    :param manager: Name of manager to be run.
    :param max_filename: Target filename.
    :param build_dir: Root build directory for max file.
    """
    output_file = get_max_output_dir(build_dir, max_filename)/f'{max_filename}.max'
    info(f'Generating: {output_file}')

    maxj_src = Path(maxj_src).resolve()
    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)

    build = ['maxJavaRun', '-m', '8192']  # TODO: Make memsize configurable
    build += [str(manager) if package is None else f'{package}.{manager}']
    build += ['DFEModel=GALAVA', 'target=DFE_SIM']
    build += [f'maxFileName={max_filename}']

    env = environ.copy()
    env.update({'MAXSOURCEDIRS': str(maxj_src), 'MAXAPPJCP': '.', 'CLASSPATH': get_classpath()})
    execute(build, cwd=build_dir, env=env)

    return output_file


def compile_max(max_filename, obj_filename, build_dir):
    """
    Generate .o object file from given max file.

    :param max_filename: max file to use.
    :param target: Output filename for object file.
    :param build_dir: Build dir where target is stored.
    """
    info(f'Compiling: {max_filename}')
    max_filename = Path(max_filename).resolve()
    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)
    build = ['sliccompile', str(max_filename), obj_filename]
    execute(build, cwd=build_dir)

    return (build_dir / obj_filename).resolve()


def compile_c(src, build_dir, include_dirs=None):
    """
    Compile .c to .o files.

    Parameters
    ----------
    src : str or :any:`pathlib.Path`
        Filename of C file or directory (will compile ``src/*.c``).
    build_dir : str or :any:`pathlib.Path`
        Build dir where .o files are stored.
    include_dirs : list, optional
        Optional list of header include paths.
    """
    info(f'Compiling .c: {src}')
    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)
    src = Path(src).resolve()
    obj_filename = '*.o' if src.is_dir() else f'{src.stem}.o'
    src = src.glob('*.c') if src.is_dir() else [src]

    build = ['gcc', '-c', '-fPIC', '-std=c99', '-O0', '-Wall', '-Wextra']
    include_dirs = (include_dirs or []) + get_max_includes()
    build += flatten([('-I', str(inc)) for inc in include_dirs])
    build += [str(s) for s in src]
    execute(build, cwd=build_dir)

    return list(build_dir.resolve().glob(obj_filename))


def link_obj(objs, target, build_dir):
    """
    Links object files to build an executable or shared library.

    :param objs: List of object files.
    :param target: Output filename of executable or shared library.
    """
    info(f'Linking: {target}')
    objs = set(str(o) for o in objs)  # Convert to set of str to eliminate doubles
    build = ['gcc']
    if Path(target).suffix == '.so':
        build += ['-shared']
    libs = flatten([('-l', str(l)) for l in get_max_libs()])
    libdirs = flatten([('-L', str(l)) for l in get_max_libdirs()])
    build += ['-o', str(target)] + list(objs) + libdirs + libs
    execute(build, cwd=build_dir)


def compile_all(maxj_src, c_src, build_dir, target, manager, package=None):
    """
    Compiles given MaxJ kernel and manager, generates the max object file from it and compiles
    the corresponding SLiC interface.
    """
    clean(build_dir)
    clean_max(build_dir, package)

    compile_maxj(maxj_src, build_dir=build_dir)
    max_filename = generate_max(manager, maxj_src, package or Path(target).stem,
                                build_dir=build_dir, package=package)
    mobj_filename = compile_max(max_filename, f'{max_filename.stem}_max.o', build_dir=build_dir)

    obj_filename = compile_c(c_src, build_dir, include_dirs=[Path(max_filename).parent])
    link_obj(obj_filename + [mobj_filename], target, build_dir)
