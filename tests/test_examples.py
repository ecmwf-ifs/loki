"""
Automatically test the provided examples
"""
from pathlib import Path

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

example_path = Path(__file__).parent.parent/'example'

@pytest.mark.parametrize("notebook", example_path.glob('*.ipynb'))
def test_notebooks(notebook, monkeypatch):
    """
    Convert all example Jupyter notebooks to scripts and run them, making sure
    they run through without any problems
    """
    monkeypatch.chdir(example_path)

    with notebook.open() as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600,kernel_name='python3')
        assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
