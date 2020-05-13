import pytest  # pylint: disable=unused-import
from pathlib import Path

from loki import clean, SourceFile, fgen, OFP, compile_and_load, FindNodes, CallStatement
from loki.tools import gettempdir, filehash


__all__ = ['ogenerate_identity', 'jit_compile', 'clean_test']


def generate_identity(refpath, routinename, modulename=None, frontend=OFP):
    """
    Generate the "identity" of a single subroutine with a frontend-specific suffix.
    """
    testname = refpath.parent/('%s_%s_%s.f90' % (refpath.stem, routinename, frontend))
    source = SourceFile.from_file(refpath, frontend=frontend)

    if modulename:
        module = [m for m in source.modules if m.name == modulename][0]
        module.name += '_%s_%s' % (routinename, frontend)
        for routine in source.all_subroutines:
            routine.name += '_%s' % frontend
            for call in FindNodes(CallStatement).visit(routine.body):
                call.name += '_%s' % frontend
        source.write(source=fgen(module), filename=testname)
    else:
        routine = [r for r in source.subroutines if r.name == routinename][0]
        routine.name += '_%s' % frontend
        source.write(source=fgen(routine), filename=testname)

    pymod = compile_and_load(testname, cwd=str(refpath.parent), use_f90wrap=True)

    if modulename:
        # modname = '_'.join(s.capitalize() for s in refpath.stem.split('_'))
        return getattr(pymod, testname.stem)
    return pymod


def jit_compile(source, filepath=None, objname=None):
    """
    Generate, Just-in-Time compile and load a given item (`Module` or
    `Subroutine`) for interactive execution.
    """
    if isinstance(source, SourceFile):
        filepath = source.filepath if filepath is None else Path(filepath)
        source.write(filename=filepath)
    else:
        source = fgen(source)
        if filepath is None:
            filepath = gettempdir()/filehash(source, prefix='', suffix='.f90')
        else:
            filepath = Path(filepath)
        SourceFile(filepath).write(source)

    pymod = compile_and_load(filepath, cwd=str(filepath.parent), use_f90wrap=True)

    if objname:
        return getattr(pymod, objname)
    return pymod


def clean_test(filepath):
    """
    Clean test directory based on JIT'ed source file.
    """
    filepath.with_suffix('.f90').unlink()
    filepath.with_suffix('.o').unlink()
    filepath.with_suffix('.py').unlink()
    f90wrap_toplevel = filepath.parent/'f90wrap_toplevel.f90'
    if f90wrap_toplevel.exists():
        f90wrap_toplevel.unlink()
    for sofile in filepath.parent.glob('_%s.*.so' % filepath.stem):
        sofile.unlink()
