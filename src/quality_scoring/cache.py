"""Shared cache utilities for quality scoring clients."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path


def get_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the cache directory for quality scoring data.

    Resolution order:
    1. Explicit ``cache_dir`` argument (highest priority)
    2. ``LLM_QUALITY_CACHE_DIR`` environment variable
    3. Default ``.model_cache/`` relative to the repo root
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    env = os.environ.get("LLM_QUALITY_CACHE_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(__file__).parent.parent.parent / ".model_cache"


def is_cache_stale(fetched_at: str | None, max_age_hours: int = 24) -> bool:
    """Return True if *fetched_at* is None or older than *max_age_hours*."""
    if fetched_at is None:
        return True
    try:
        fetched = datetime.fromisoformat(fetched_at)
    except ValueError:
        return True
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=UTC)
    age = datetime.now(UTC) - fetched
    return age.total_seconds() > max_age_hours * 3600
