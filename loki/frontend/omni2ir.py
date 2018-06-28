from subprocess import check_call, CalledProcessError
from pathlib import Path
import xml.etree.ElementTree as ET

from loki.logging import info, error, DEBUG
from loki.frontend import OMNI
from loki.tools import timeit, disk_cached


__all__ = ['parse_omni']


@timeit(log_level=DEBUG)
@disk_cached(argname='filename', suffix='omniast')
def parse_omni(filename, xmods=None):
    """
    Deploy the OMNI compiler's frontend (F_Front) to generate the OMNI AST.
    """
    filepath = Path(filename)
    info("[Frontend.OFP] Parsing %s" % filepath.name)

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
