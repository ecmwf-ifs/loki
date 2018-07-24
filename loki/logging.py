import logging
import sys


__all__ = ['logger', 'debug', 'info', 'warning', 'error', 'log',
           'DEBUG', 'INFO', 'WARNING', 'ERROR']


DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR

logger = logging.getLogger('loki')
_ch = logging.StreamHandler()
logger.addHandler(_ch)
logger.setLevel(INFO)

NOCOLOR = '%s'
RED = '\033[1;37;31m%s\033[0m'
BLUE = '\033[1;37;34m%s\033[0m'
GREEN = '\033[1;37;32m%s\033[0m'
COLORS = {
    DEBUG: NOCOLOR,
    INFO: GREEN,
    WARNING: BLUE,
    ERROR: RED,
}


def log(msg, level=INFO, *args, **kwargs):
    """
    Wrapper of the main Python's logging function. Print 'msg % args' with
    the severity 'level'.
    :param msg: the message to be printed.
    :param level: accepted values are: DEBUG, INFO, AUTOTUNER, DSE, DSE_WARN,
                  DLE, DLE_WARN, WARNING, ERROR, CRITICAL
    """
    assert level in [DEBUG, INFO, WARNING, ERROR]

    color = COLORS[level] if sys.stdout.isatty() and sys.stderr.isatty() else '%s'
    logger.log(level, color % msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    log(msg, DEBUG, *args, **kwargs)


def info(msg, *args, **kwargs):
    log(msg, INFO, *args, **kwargs)


def warning(msg, *args, **kwargs):
    log(msg, WARNING, *args, **kwargs)


def error(msg, *args, **kwargs):
    log(msg, ERROR, *args, **kwargs)
