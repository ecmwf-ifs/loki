from pkg_resources import get_distribution, DistributionNotFound

# Import the global configuration map
from loki.config import *  # noqa

from loki.frontend import *  # noqa
from loki.sourcefile import *  # noqa
from loki.subroutine import *  # noqa
from loki.module import *  # noqa
from loki.ir import *  # noqa
from loki.expression import *  # noqa
from loki.types import *  # noqa
from loki.visitors import *  # noqa
from loki.tools import *  # noqa
from loki.logging import *  # noqa
from loki.backend import *  # noqa
from loki.transform import * # noqa
from loki.build import *  # noqa  # pylint: disable=redefined-builtin
from loki.debug import *  # noqa
from loki.scheduler import *  # noqa
from loki.lint import *  # noqa
from loki.pragma_utils import *  # noqa
from loki.analyse import *  # noqa
from loki.dimension import *  # noqa

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


# Add flag to trigger an initial print out of the global config
config.register('print-config', False, env_variable='LOKI_PRINT_CONFIG',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Define Loki's global config options
config.register('log-level', 'INFO', env_variable='LOKI_LOGGING',
                callback=set_log_level, preprocess=lambda i: log_levels[i])

# Define Loki's temporary directory for generating intermediate files
config.register('tmp-dir', None, env_variable='LOKI_TMP_DIR')

# Causes external frontend preprocessor to dump intermediate soruce files
config.register('cpp-dump-files', False, env_variable='LOKI_CPP_DUMP_FILES',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Causes OMNI frontend to dump intermediate XML files to LOKI_TMP_DIR
config.register('omni-dump-xml', False, env_variable='LOKI_OMNI_DUMP_XML',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Use internal frontend preprocessor caching (see loki/frontend/preprocessing.py)
config.register('frontend-pp-cache', False, env_variable='LOKI_FRONTEND_PP_CACHE',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Disk-caching, which causes OFP ASTs to be cached on disk for
# fast re-parsing of unchanged source files
config.register('disk-cache', False, env_variable='LOKI_DISK_CACHE',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Trigger configuration initialisation, including
# a scan of the current environment variables
config.initialize()

if config['print-config']:
    config.print_state()
