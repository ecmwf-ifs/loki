import pytest

from loki import config


@pytest.fixture(name='reset_frontend_mode')
def fixture_reset_frontend_mode():
    original_frontend_mode = config['frontend-strict-mode']
    yield
    config['frontend-strict-mode'] = original_frontend_mode
