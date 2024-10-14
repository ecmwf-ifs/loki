# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
Loki's logger classes and logging utilities.
"""

import logging
import sys


__all__ = ['logger', 'log_levels', 'set_log_level', 'FileLogger',
           'debug', 'detail', 'perf', 'info', 'warning', 'error', 'log']


def FileLogger(name, filename, level=None, file_level=None, fmt=None,
               mode='a'):
    """
    Logger that emits to a single logfile, as well as stdout/stderr.
    """
    level = level or INFO
    file_level = file_level or level

    _logger = logging.getLogger(name)
    _logger.setLevel(level if level <= file_level else file_level)

    fmt = fmt or '%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'
    fh = logging.FileHandler(str(filename), mode=mode)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(file_level)
    _logger.addHandler(fh)

    # Install the colored logging handlers
    try:
        import coloredlogs  # pylint: disable=import-outside-toplevel
        coloredlogs.install(level=level, logger=_logger)
    except ImportError:
        pass

    # TODO: For concurrent file writes, initialize queue and
    # main logging thread.

    return _logger


# Initialize base logger
logger = logging.getLogger('Loki')
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

# This one is primarily used by loki.build
default_logger = logger

# Note, this a remnant from loki.build.logging, which not only adds
# colour, but also adds hostname and timestamps, etc. to the log line
# We might want to re-eanble this under some specific logging options

# coloredlogs.install(level=default_level, logger=logger)


# Define available log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
PERF = 15
DETAIL = 12

# Internally accepted log levels
log_levels = {
    'DEBUG': DEBUG,
    'DETAIL': DETAIL,
    'PERF': PERF,
    'INFO': INFO,
    'WARNING': WARNING,
    'ERROR': ERROR,
    # Lower case keywords for env variables
    'debug': DEBUG,
    'detail': DETAIL,
    'perf': PERF,
    'info': INFO,
    'warning': WARNING,
    'error': ERROR,
    # Enum keys for idempotence
    DEBUG: DEBUG,
    DETAIL: DETAIL,
    PERF: PERF,
    INFO: INFO,
    WARNING: WARNING,
    ERROR: ERROR,
}

# Internally used log colours (in simple mode)
NOCOLOR = '%s'
RED = '\033[1;37;31m%s\033[0m'
BLUE = '\033[1;37;34m%s\033[0m'
GREEN = '\033[1;37;32m%s\033[0m'
colors = {
    DEBUG: NOCOLOR,
    DETAIL: GREEN,
    PERF: GREEN,
    INFO: GREEN,
    WARNING: BLUE,
    ERROR: RED,
}

def set_log_level(level):
    """
    Set the log level for the Loki logger.
    """
    if level not in log_levels.values():
        raise ValueError(f'Illegal logging level {level}')

    logger.setLevel(level)


def log(msg, level, *args, **kwargs):
    """
    Wrapper of the main Python's logging function. Print 'msg % args' with
    the severity 'level'.

    :param msg: the message to be printed.
    """
    color = colors[level] if sys.stdout.isatty() and sys.stderr.isatty() else '%s'
    logger.log(level, color % msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """
    Logger method for most verbose level of output

    Parameters
    ----------
    msg : str
        Message to log at :any:`DEBUG` level.
    """
    log(msg, DEBUG, *args, **kwargs)

def detail(msg, *args, **kwargs):
    """
    Logger method for detailed, per-file information.

    This level should be used for timing and detailed information at a
    per-file level, which can get verbose.

    Parameters
    ----------
    msg : str
        Message to log at :any:`DETAIL` level.
    """
    log(msg, DETAIL, *args, **kwargs)

def perf(msg, *args, **kwargs):
    """
    Logger method for performance information.

    This level should be used for timing individual processes at a
    global granularity during batch-processing.

    Parameters
    ----------
    msg : str
        Message to log at :any:`DETAIL` level.
    """
    log(msg, PERF, *args, **kwargs)

def info(msg, *args, **kwargs):
    """
    Logger method for high-level progress information.

    This is the default output logging and should only be used at a
    global granularity during batch-processing.

    Parameters
    ----------
    msg : str
        Message to log at :any:`INFO` level.
    """
    log(msg, INFO, *args, **kwargs)

def warning(msg, *args, **kwargs):
    """
    Logger method for high-level progress information.

    This level should be used for potentially dangerous, but not fatal
    information.

    Parameters
    ----------
    msg : str
        Message to log at :any:`WARN` level.
    """
    log(msg, WARNING, *args, **kwargs)

def error(msg, *args, **kwargs):
    """
    Logger method for high-level progress information.

    This level should be used to provide additional information in
    case of failures.

    Parameters
    ----------
    msg : str
        Message to log at :any:`ERROR` level.
    """
    log(msg, ERROR, *args, **kwargs)
