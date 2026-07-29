"""Resolve paths to bundled data files.

Works from both source checkouts and installed wheels (unpacked).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def data_path(relative: str) -> Path:
    """Resolve a data file path relative to planner.data.

    Args:
        relative: Path relative to the data directory,
                  e.g. "configuration/model_catalog.json"

    Returns:
        Absolute Path to the data file.
    """
    ref = files("planner.data").joinpath(relative)
    return Path(str(ref))
