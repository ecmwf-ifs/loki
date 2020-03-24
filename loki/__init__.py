from pkg_resources import get_distribution, DistributionNotFound

from loki.frontend import * # noqa
from loki.sourcefile import * # noqa
from loki.subroutine import * # noqa
from loki.module import * # noqa
from loki.ir import *  # noqa
from loki.expression import *  # noqa
from loki.types import *  # noqa
from loki.visitors import *  # noqa
from loki.tools import * # noqa
from loki.logging import * # noqa
from loki.backend import * # noqa
from loki.transform import * # noqa
from loki.build import * # noqa  # pylint: disable=redefined-builtin
from loki.debug import * # noqa
from loki.scheduler import * # noqa
from loki.lint import * # noqa

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
