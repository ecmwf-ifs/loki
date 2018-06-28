import pytest

from loki import SourceFile, fgen, OFP, compile_and_load, clean


def generate_identity(refpath, routinename, modulename=None, frontend=OFP):
    """
    Generate the "identity" of a single subroutine with a frontend-specific suffix.
    """
    testname = refpath.parent/('%s_%s_%s.f90' % (refpath.stem, routinename, frontend))
    source = SourceFile.from_file(refpath, frontend=frontend)

    if modulename:
        module = [m for m in source.modules if m.name == modulename][0]
        module.name += '_%s_%s' % (routinename, frontend)
        for routine in source.subroutines:
            routine.name += '_%s' % frontend
        source.write(source=fgen(module), filename=testname)
    else:
        routine = [r for r in source.subroutines if r.name == routinename][0]
        routine.name += '_%s' % frontend
        source.write(source=fgen(routine), filename=testname)

    return compile_and_load(testname, cwd=str(refpath.parent),
                            use_f90wrap=modulename is not None)
