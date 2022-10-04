import pytest

from loki import as_tuple, Frontend
import loki.frontend


def available_frontends(xfail=None, skip=None):
    """
    Provide list of available frontends to parametrize tests with

    To run tests for every frontend, an argument :attr:`frontend` can be added to
    a test with the return value of this function as parameter.

    For any unavailable frontends where ``HAVE_<frontend>`` is `False` (e.g.
    because required dependencies are not installed), :attr:`test` is marked as
    skipped.

    Use as

    ..code-block::
        @pytest.mark.parametrize('frontend', available_frontends(xfail=[OMNI, (OFP, 'Because...')]))
        def my_test(frontend):
            source = Sourcefile.from_file('some.F90', frontend=frontend)
            # ...

    Parameters
    ----------
    xfail : list, optional
        Provide frontends that are expected to fail, optionally as tuple with reason
        provided as string. By default `None`
    skip : list, optional
        Provide frontends that are always skipped, optionally as tuple with reason
        provided as string. By default `None`
    """
    if xfail:
        xfail = dict((as_tuple(f) + (None,))[:2] for f in xfail)
    else:
        xfail = {}

    if skip:
        skip = dict((as_tuple(f) + (None,))[:2] for f in skip)
    else:
        skip = {}

    # Unavailable frontends
    unavailable_frontends = {
        f: f'{f} is not available' for f in Frontend
        if not getattr(loki.frontend, f'HAVE_{str(f).upper()}')
    }
    skip.update(unavailable_frontends)

    # Build the list of parameters
    params = []
    for f in Frontend:
        if f in skip:
            params += [pytest.param(f, marks=pytest.mark.skip(reason=skip[f]))]
        elif f in xfail:
            params += [pytest.param(f, marks=pytest.mark.xfail(reason=xfail[f]))]
        elif f != Frontend.REGEX:
            params += [f]

    return params
