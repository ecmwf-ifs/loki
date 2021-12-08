import sys
from pathlib import Path

# Bootstrap the tests directory for custom test utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Bootstrap the local transformations directory for custom transformations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.conftest import *  # pylint: disable=wildcard-import,unused-wildcard-import,wrong-import-position
