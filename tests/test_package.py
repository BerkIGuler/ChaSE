import tomllib
from pathlib import Path

import chase

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject():
    with open(PYPROJECT, "rb") as f:
        assert chase.__version__ == tomllib.load(f)["project"]["version"]
