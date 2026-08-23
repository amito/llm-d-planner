"""Resolve paths to bundled data files.

Works from both source checkouts and installed wheels.
"""

from importlib.resources import files
from pathlib import Path

_DEFAULT_DATA_ROOT = Path(str(files("planner.data")))


def data_path(relative: str, data_dir: Path | None = None) -> Path:
    """Resolve a data file path relative to planner.data.

    Uses bundled package data unless data_dir is provided.
    """
    root = data_dir or _DEFAULT_DATA_ROOT
    return root / relative
