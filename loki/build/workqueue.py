from worq import get_broker, get_queue
from worq.pool.process import WorkerPool
from contextlib import contextmanager

from loki.build.tools import execute


__all__ = ['workqueue', 'MEMORY_URL', 'DEFAULT_TIMEOUT']


MEMORY_URL = 'memory://'

# TODO: REALLY NEED TO MAKE THIS USER CONFIGURABLE!
DEFAULT_TIMEOUT = 60


def worker_init(url=None):
    url = url or MEM_URL
    broker = get_broker(url)
    broker.expose(execute)
    return broker


class DummyQueue(object):
    """
    Dummy queue object to funnel workqueue requests to the current
    main process.
    """

    def execute(self, *args, **kwargs):
        execute(*args, **kwargs)


@contextmanager
def workqueue(workers=None, timeout=1, url=None):
    """
    Parallel work queue manager that creates a worker pool and exposes
    the ``q.execute(cmd)`` utility to invoke shell commands in parallel.
    """
    if workers is None:
        yield DummyQueue()
        return

    url = url or MEMORY_URL
    broker = get_broker(url)

    # TODO: We could cache the worker pool if necessary
    pool = WorkerPool(broker, worker_init, workers=workers)
    pool.start(timeout=DEFAULT_TIMEOUT, handle_sigterm=False)

    # Get the queu object and hand it to caller
    q = get_queue(MEMORY_URL)
    try:
        yield q
    finally:
        # Cleanup the worker queue
        pool.stop(join=False)
        pool.join()
        broker.discard_pending_tasks()
