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

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


# Define Loki's global config options
config.register('log-level', 'INFO', env_variable='LOKI_LOGGING',
                callback=set_log_level, preprocess=lambda i: log_levels[i])

# Disk-caching, which causes OFP ASTs to be cached on disk for
# fast re-parsing of unchanged source files
config.register('disk-cache', False, env_variable='LOKI_DISK_CACHE',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Trigger configuration initialisation, including
# a scan of the current environment variables
config.initialize()
