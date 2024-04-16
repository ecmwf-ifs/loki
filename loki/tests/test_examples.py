# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Automatically test the provided examples
"""
from pathlib import Path

import pytest
import nbformat
from jupyter_client.kernelspec import find_kernel_specs
from nbconvert.preprocessors import ExecutePreprocessor

example_path = Path(__file__).parent.parent/'example'

def is_ipython_available():
    """
    Check if Jupyter Notebook kernel is available
    """
    is_available = False
    try:
        import IPython  # pylint: disable=import-outside-toplevel,unused-import
        is_available = True
    except ImportError:
        pass
    return is_available

# Skip tests in this module if Jupyter Kernel not available
pytestmark = pytest.mark.skipif(
    not is_ipython_available() or 'python3' not in find_kernel_specs(),
    reason='IPython or Jupyter kernel are not available'
)

@pytest.mark.parametrize("notebook", example_path.glob('*.ipynb'))
def test_notebooks(notebook, monkeypatch):
    """
    Convert all example Jupyter notebooks to scripts and run them, making sure
    they run through without any problems
    """
    monkeypatch.chdir(example_path)

    with notebook.open() as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
        assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
