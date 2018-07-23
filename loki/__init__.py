from loki.frontend import * # noqa
from loki.sourcefile import * # noqa
from loki.subroutine import * # noqa
from loki.ir import *  # noqa
from loki.expression import *  # noqa
from loki.types import *  # noqa
from loki.visitors import *  # noqa
from loki.tools import * # noqa
from loki.logging import * # noqa
from loki.codegen import * # noqa
from loki.transformation import * # noqa
from loki.build import * # noqa

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
