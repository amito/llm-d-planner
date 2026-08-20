"""Resolve paths to bundled quality scoring data files.

Works from both source checkouts and installed wheels.
"""

from importlib.resources import files
from pathlib import Path

_DEFAULT_DATA_ROOT = Path(str(files("quality_scoring.data")))


def quality_data_path(relative: str = "", data_dir: Path | None = None) -> Path:
    """Resolve a quality data file path.

    Args:
        relative: Path relative to the data directory,
                  e.g. "arena_models.json". Empty string returns the directory.
        data_dir: Optional override directory.

    Returns:
        Absolute Path to the data file or directory.
    """
    root = data_dir or _DEFAULT_DATA_ROOT
    if relative:
        return root / relative
    return root
