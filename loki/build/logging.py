import logging
import coloredlogs
import sys


__all__ = ['logger', 'FileLogger', '_default_logger',
           'debug', 'info', 'warning', 'error', 'log',
           'DEBUG', 'INFO', 'WARNING', 'ERROR']

def FileLogger(name, filename, logger=None, level=None, fmt=None, mode='w'):
    level = level or INFO
    fmt = fmt or '%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'

    logger = logger or logging.getLogger(name)
    logger.setLevel(level)

    fh = logging.FileHandler(str(filename), mode=mode)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    return logger

# Wrap the usual log level flags
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR

# TODO: Make configurable from options dict
default_level = INFO

# Create the deault logger with color and timings
logger = logging.getLogger('Loki')
coloredlogs.install(level=default_level, logger=logger)
_default_logger = logger

# Wrap the common invocation methods
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
log = logger.log
