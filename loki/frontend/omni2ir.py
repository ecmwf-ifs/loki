from subprocess import check_call, CalledProcessError
from pathlib import Path
import xml.etree.ElementTree as ET

from loki.logging import info, error, DEBUG
from loki.frontend import OMNI
from loki.tools import timeit, disk_cached


__all__ = ['parse_omni']


def preprocess_omni(filename, outname, includes=None):
    """
    Call C-preprocessor to sanitize input for OMNI frontend.
    """
    filepath = Path(filename)
    outpath = Path(outname)
    includes = [Path(incl) for incl in includes or []]

    # TODO Make CPP driveable via flags/config
    cmd = ['gfortran', '-E', '-cpp']
    for incl in includes:
        cmd += ['-I', '%s' % Path(incl)]
    cmd += ['-o', '%s' % outpath]
    cmd += ['%s' % filepath]

    try:
        check_call(cmd)
    except CalledProcessError as e:
        error('[%s] Preprocessing failed: %s' % (OMNI, ' '.join(cmd)))
        raise e


@timeit(log_level=DEBUG)
def parse_omni(filename, xmods=None):
    """
    Deploy the OMNI compiler's frontend (F_Front) to generate the OMNI AST.
    """
    filepath = Path(filename)
    info("[Frontend.OMNI] Parsing %s" % filepath.name)

    xml_path = filepath.with_suffix('.omni.F90')
    xmods = xmods or []

    cmd = ['F_Front']
    for m in xmods:
        cmd += ['-M', '%s' % Path(m)]
    cmd += ['-o', '%s' % xml_path]
    cmd += ['%s' % filepath]

    try:
        check_call(cmd)
    except CalledProcessError as e:
        error('[%s] Parsing failed: %s' % (OMNI, ' '.join(cmd)))
        raise e

    return ET.parse(xml_path).getroot()
