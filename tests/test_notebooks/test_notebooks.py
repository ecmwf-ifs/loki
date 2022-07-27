import fnmatch,os
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebookpath = '../../example/'
def get_filelist():
    return fnmatch.filter(os.listdir(notebookpath),'*.ipynb')

@pytest.fixture(scope='module', name='nbpath')
def fixture_here():
    return notebookpath

@pytest.mark.parametrize("notebook",get_filelist())
def test_notebooks(notebook,monkeypatch,nbpath):
 
    monkeypatch.chdir(nbpath)
    
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600,kernel_name='python3')
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            assert False, f"Failed executing {notebook}"
