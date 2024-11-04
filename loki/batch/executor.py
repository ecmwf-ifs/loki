# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A dummy "executor" utility that allows parallel non-threaded execution
under the same API as thread or ProcessPoolExecutors.
"""

from concurrent.futures import Executor, Future

__all__ = ['SerialExecutor']


class SerialExecutor(Executor):
    """
    A dummy "executor" utility that allows parallel non-threaded
    execution with the same API as a ``ProcessPoolExecutors``.
    """

    def submit(self, fn, *args, **kwargs):  # pylint: disable=arguments-differ
        """
        Executes the callable, *fn* as ``fn(*args, **kwargs)``
        and wraps the return value in a *Future* object.
        """
        f = Future()
        try:
            # Execute function on given args
            result = fn(*args, **kwargs)
        except BaseException as e:
            f.set_exception(e)
        else:
            f.set_result(result)

        return f

    def map(self, fn, *args, **kwargs):
        """
        Maps the callable *fn* via ``map(fn, *args, **kwargs)``.
        """
        return map(fn, *args, **kwargs)
