from os import environ
from pathlib import Path

from loki.logging import info
from loki.build.compiler import clean
from loki.build.tools import as_tuple, delete, execute, flatten


__all__ = ['clean_max', 'compile', 'compile_c',
           'compile_maxj', 'generate_max', 'link_obj']


def get_classpath():
    """
    Extract the CLASSPATH for Maxeler tools from the environment.
    """
    cp_base = Path(environ['MAXCOMPILERDIR']) / 'lib'
    classpath = [cp_base / j for j in ('Max5Platform.jar', 'Max4Platform.jar', 'MaxCompiler.jar')]
    return ':'.join([str(p) for p in classpath])


def get_includes():
    """
    Build the includes for Maxeler tools from the environment.
    """
    incdirs = [Path(environ['MAXCOMPILERDIR']) / 'include']
    incdirs += [Path(environ['MAXCOMPILERDIR']) / 'include' / 'slic']
    incdirs += [Path(environ['MAXELEROSDIR']) / 'include']
    return flatten([('-I', str(inc)) for inc in incdirs])


def get_libs():
    """
    Build the library includes and links for Maxeler tools from the environment.
    """
    libs = [Path(environ['MAXCOMPILERDIR']) / 'lib']
    libs += [Path(environ['MAXELEROSDIR']) / 'lib']
    libs = flatten([('-L', str(lib)) for lib in libs])
    return libs + ['-lmaxeleros', '-lslic', '-lm', '-lpthread', '-lcurl']


def get_max_output_dir(build_dir, max_filename):
    """
    Generates the (absolute) directory name in which output from maxJavaRun is stored.
    """
    return (Path(build_dir) / ('%s_MAX5C_DFE_SIM' % max_filename) / 'results').resolve()


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
    info('Compiling .maxj: %s' % src)
    src = Path(src).resolve()

    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)

    build = ['maxjc', '-classpath', get_classpath(), '-1.6']
    build += ['-d', '.', str(src)]
    execute(build, cwd=build_dir)


def generate_max(manager, maxj_src, max_filename, build_dir, package=None):
    """
    Generate max-file (and matching header) using java binaries.

    :param manager: Name of manager to be run.
    :param max_filename: Target filename.
    :param build_dir: Root build directory for max file.
    """
    output_file = get_max_output_dir(build_dir, max_filename) / ('%s.max' % max_filename)
    info('Generating: %s' % output_file)

    maxj_src = Path(maxj_src).resolve()
    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)

    build = ['maxJavaRun', '-m', '8192']  # TODO: Make memsize configurable
    build += [str(manager) if package is None else '%s.%s' % (package, manager)]
    build += ['DFEModel=GALAVA', 'target=DFE_SIM']
    build += ['maxFileName=%s' % max_filename]

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
    info('Compiling: %s' % max_filename)
    max_filename = Path(max_filename).resolve()
    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)
    build = ['sliccompile', str(max_filename), obj_filename]
    execute(build, cwd=build_dir)

    return (build_dir / obj_filename).resolve()


def compile_c(src, build_dir, include_dirs=None):
    """
    Compile .c to .o files.

    :param src: Filename of C file or directory (will compile 'src/*.c').
    :param build_dir: Build dir where .o files are stored.
    :param include_dirs: Optional list of header include paths.
    """
    info('Compiling .c: %s' % src)
    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)
    src = Path(src).resolve()
    obj_filename = '%s.o' % '*' if src.is_dir() else src.stem
    src = src.glob('*.c') if src.is_dir() else [src]

    build = ['gcc', '-c', '-fPIC', '-std=c99', '-O0', '-Wall', '-Wextra']
    build += get_includes()
    if include_dirs is not None:
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
    info('Linking: %s' % target)
    objs = set(str(o) for o in objs)  # Convert to set of str to eliminate doubles
    build = ['gcc']
    if Path(target).suffix == '.so':
        build += ['-shared'] 
    build += ['-o', str(target)] + list(objs) + get_libs()
    execute(build, cwd=build_dir)


def compile(maxj_src, c_src, build_dir, target, manager, package=None):
    clean(build_dir)
    clean_max(build_dir, package)

    compile_maxj(maxj_src, build_dir=build_dir)
    max_filename = generate_max(manager, maxj_src, package or Path(target).stem,
                                build_dir=build_dir, package=package)
    mobj_filename = compile_max(max_filename, '%s_max.o' % max_filename.stem, build_dir=build_dir)

    obj_filename = compile_c(c_src, build_dir, include_dirs=[Path(max_filename).parent])
    link_obj(obj_filename + [mobj_filename], target, build_dir)

