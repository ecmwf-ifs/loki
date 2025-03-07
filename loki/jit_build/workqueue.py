# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from contextlib import contextmanager
from multiprocessing import Manager
from logging.handlers import QueueListener, QueueHandler
from concurrent.futures import ProcessPoolExecutor

from loki.logging import default_logger
from loki.tools import execute


__all__ = ['workqueue', 'wait_and_check', 'MEMORY_URL', 'DEFAULT_TIMEOUT']


MEMORY_URL = 'memory://'

# TODO: REALLY NEED TO MAKE THIS USER CONFIGURABLE!
DEFAULT_TIMEOUT = 60


class DummyQueue:
    """
    Dummy queue object to funnel workqueue requests to the current
    main process.
    """

    @staticmethod
    def execute(*args, **kwargs):
        execute(*args, **kwargs)

    @staticmethod
    def call(fn, *args, **kwargs):
        return fn(*args, **kwargs)


"""
A global flag to make worker initialization happen once only.
This is hacky, but it requires a stable Python3.7 to fix.
"""
_initialized = False


def init_worker(log_queue=None):
    """
    Process-local initialization of the worker process. This sets up
    the queue-based logging, etc.
    """
    if log_queue is not None:
        from loki import config  # pylint: disable=import-outside-toplevel
        log_level = config['log-level']

        # Set up logger to funnel logs back to master via ``log_queue``
        qh = QueueHandler(log_queue)
        qh.setLevel(log_level)

        # Wipe all local handlers, since we dispatch to the master.
        # We also drop the logging level, so that the master may
        # decide what to do.
        for handler in default_logger.handlers:
            default_logger.removeHandler(handler)
        default_logger.addHandler(qh)
        default_logger.setLevel(log_level)


def init_call(fn, *args, **kwargs):
    """
    Hack alert: This small wrapper function ensure that an initialization
    function is called once and only once per worker from the within the
    work scheduler. This is done to work around the fact that a global worker
    initialization mechanism is only added to :class:`ProcessPoolExecutor`
    in Python3.7, which (at the time of writing) is not out or mature yet.
    """
    global _initialized  # pylint: disable=global-statement
    log_queue = kwargs.pop('log_queue', None)
    if not _initialized:
        init_worker(log_queue=log_queue)
        _initialized = True

    return fn(*args, **kwargs)


def wait_and_check(task, timeout=DEFAULT_TIMEOUT, logger=None):
    """
    Wait for :param:`task` to complete and check for possible exceptions.
    """
    logger = logger or default_logger

    if task is not None:
        try:
            # Get result from the worker task and sanity check
            task.result(timeout=timeout)
            error = task.exception(timeout=timeout)

            if error is not None:
                logger.error('Failed compilation task: %s', task)
                raise error

        except TimeoutError as e:
            logger.error('Compilation task timed out: %s', task)
            raise e


class ParallelQueue:
    """
    Dummy queue object to funnel workqueue requests to the current
    main process.
    """

    def __init__(self, executor, logger=None, manager=None):
        self.executor = executor

        self.manager = None
        self.listener = None
        self.log_queue = None

        if logger is not None:
            # Initialize a listener for the logging queue that dispatches
            # to our pre-configured handlers on the master process
            self.manager = manager or Manager()
            self.log_queue = self.manager.Queue()
            self.listener = QueueListener(self.log_queue, *(logger.handlers),
                                          respect_handler_level=True)

    def execute(self, *args, **kwargs):
        """
        Wrapper around the ``tools.execute(cmd)`` function presented by the
        :class:`ParallelQueue` object to its users.
        """
        return self.executor.submit(init_call, execute, *args, **kwargs)

    def call(self, fn, *args, **kwargs):
        """
        Arbitrary interface to submit function calls to the
        :class:`ParallelQueue` object.
        """
        return self.executor.submit(init_call, fn, *args, **kwargs)


@contextmanager
def workqueue(workers=None, logger=None, manager=None):
    """
    Parallel work queue manager that creates a worker pool and exposes
    the ``q.execute(cmd)`` utility to invoke shell commands in parallel.
    """
    if workers is None:
        yield DummyQueue()
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        q = ParallelQueue(executor, logger=logger, manager=manager)

        # We have to manually start and stop the queue listener
        # for our funneled logging setup.
        if q.listener:
            q.listener.start()

        yield q

        if q.listener:
            q.listener.stop()
