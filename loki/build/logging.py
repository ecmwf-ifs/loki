import logging
import coloredlogs
import sys

from loki.build import multiprocessing_logging


__all__ = ['logger', 'FileLogger', 'default_logger',
           'debug', 'info', 'warning', 'error', 'log',
           'DEBUG', 'INFO', 'WARNING', 'ERROR']


def FileLogger(name, filename, level=None, file_level=None, fmt=None,
               mode='a'):
    """
    Logger that emits to a single logfile, as well as stdout/stderr.
    """
    level = level or INFO
    file_level = file_level or level

    logger = logging.getLogger(name)
    logger.setLevel(level if level <= file_level else file_level)

    fmt = fmt or '%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'
    fh = logging.FileHandler(str(filename), mode=mode)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(file_level)
    logger.addHandler(fh)

    # Install the colored logging handlers
    coloredlogs.install(level=level, logger=logger)

    # TODO: For concurrent file writes, initialize queue and
    # main logging thread.

    return logger

# Wrap the usual log level flags
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR



# TODO: Make configurable from options dict
default_level = INFO

# # Create the deault logger with color and timings
# logger = logging.getLogger('Loki')
# coloredlogs.install(level=default_level, logger=logger)

# TODO: This is a hack to enable file-logging from parallel
# workers. Since logger objects themselves are not pickled and
# shared, we need to re-run the parallel file init on all procs,
# ie. in this global __init__ space. Ideally we would initialize
# a logging thread and a queue listener that the worker threads
# log into, but I'm out of patience now...
logger = FileLogger(name='Loki', level=INFO, file_level=DEBUG,
                    filename='build/build.log', mode='a')

default_logger = logger


# Wrap the common invocation methods
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
log = logger.log
