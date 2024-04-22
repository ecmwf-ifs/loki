# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from importlib.metadata import version, PackageNotFoundError

# Import the global configuration map
from loki.config import *  # noqa

from loki.frontend import *  # noqa
from loki.sourcefile import *  # noqa
from loki.subroutine import *  # noqa
from loki.program_unit import * # noqa
from loki.module import *  # noqa
from loki.ir import *  # noqa
from loki.expression import *  # noqa
from loki.types import *  # noqa
from loki.scope import *  # noqa
from loki.tools import *  # noqa
from loki.logging import *  # noqa
from loki.backend import *  # noqa
from loki.build import *  # noqa  # pylint: disable=redefined-builtin
from loki.batch import *  # noqa
from loki.lint import *  # noqa
from loki.analyse import *  # noqa
from loki.dimension import *  # noqa
from loki.transformations import *  # noqa


try:
    __version__ = version("loki")
except PackageNotFoundError:
    # package is not installed
    pass


# Add flag to trigger an initial print out of the global config
config.register('print-config', False, env_variable='LOKI_PRINT_CONFIG',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Define Loki's global config options
config.register('log-level', 'INFO', env_variable='LOKI_LOGGING',
                callback=set_log_level, preprocess=lambda i: log_levels[i])

config.register('debug', None, env_variable='LOKI_DEBUG',
                callback=set_excepthook, preprocess=lambda i: auto_post_mortem_debugger if i else None)

# Define Loki's temporary directory for generating intermediate files
config.register('tmp-dir', None, env_variable='LOKI_TMP_DIR')

# Causes external frontend preprocessor to dump intermediate soruce files
config.register('cpp-dump-files', False, env_variable='LOKI_CPP_DUMP_FILES',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Causes OMNI frontend to dump intermediate XML files to LOKI_TMP_DIR
config.register('omni-dump-xml', False, env_variable='LOKI_OMNI_DUMP_XML',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Enable strict frontend behaviour (fail on unknown/unsupported language features)
config.register('frontend-strict-mode', False, env_variable='LOKI_FRONTEND_STRICT_MODE',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Disk-caching, which causes OFP ASTs to be cached on disk for
# fast re-parsing of unchanged source files
config.register('disk-cache', False, env_variable='LOKI_DISK_CACHE',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Force symbol comparison and object equality to be case sensitive
config.register('case-sensitive', False, env_variable='LOKI_CASE_SENSITIVE',
                preprocess=lambda i: bool(i) if isinstance(i, int) else i)

# Specify a timeout for the REGEX frontend to catch catastrophic backtracking
config.register('regex-frontend-timeout', 30, env_variable='LOKI_REGEX_FRONTEND_TIMEOUT', preprocess=int)

# Trigger configuration initialisation, including
# a scan of the current environment variables
config.initialize()

if config['print-config']:
    config.print_state()
