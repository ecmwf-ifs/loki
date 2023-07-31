# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import shutil
import pytest
import yaml

from loki import as_tuple, Frontend
import loki.frontend

__all__ = ['available_frontends', 'local_loki_setup', 'local_loki_cleanup']

def available_frontends(xfail=None, skip=None):
    """
    Provide list of available frontends to parametrize tests with

    To run tests for every frontend, an argument :attr:`frontend` can be added to
    a test with the return value of this function as parameter.

    For any unavailable frontends where ``HAVE_<frontend>`` is `False` (e.g.
    because required dependencies are not installed), :attr:`test` is marked as
    skipped.

    Use as

    .. code-block::

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


def write_env_launch_script(here, binary, args):
    # Write a script to source env.sh and launch the binary
    script = Path(here/f'build/run_{binary}.sh')
    script.write_text(f"""
#!/bin/bash

source env.sh >&2
bin/{binary} {' '.join(args)}
exit $?
    """.strip())
    script.chmod(0o750)

    return script


def local_loki_setup(here):
    lokidir = Path(__file__).parent.parent.parent
    target = here/'source/loki'
    backup = here/'source/loki.bak'
    bundlefile = here/'bundle.yml'

    # Do not overwrite any existing Loki copy
    if target.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(target, backup)

    # Change bundle to symlink for Loki
    bundle = yaml.safe_load(bundlefile.read_text())
    loki_index = [i for i, p in enumerate(bundle['projects']) if 'loki' in p]
    assert len(loki_index) == 1

    return str(lokidir.resolve()), target, backup


def local_loki_cleanup(target, backup):
    if target.is_symlink():
        target.unlink()
    if not target.exists() and backup.exists():
        shutil.move(backup, target)
