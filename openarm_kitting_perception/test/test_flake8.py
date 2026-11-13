import os

from ament_flake8.main import main_with_errors
import pytest

CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "setup.cfg")      # the repository's flake8 settings


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    rc, errors = main_with_errors(argv=["--config", CONFIG])
    assert rc == 0, f"Found {len(errors)} code style errors / warnings:\n" + "\n".join(errors)
