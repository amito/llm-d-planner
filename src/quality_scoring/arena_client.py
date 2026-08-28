"""Arena leaderboard cache management."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from quality_scoring.cache import get_cache_dir

logger = logging.getLogger(__name__)

# Only these fields are used by the scoring engine
SCORING_FIELDS = {"model_name", "category", "rating", "rating_lower", "rating_upper"}


def get_cache_path(cache_dir: Path | None = None) -> Path:
    return get_cache_dir(cache_dir) / "arena_models.json"


def fetch_from_hf() -> list[dict]:
    """Fetch the Arena leaderboard dataset from HuggingFace."""
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("lmarena-ai/leaderboard-dataset", "text_style_control", split="latest")
    df = pd.DataFrame(ds)
    return df.to_dict(orient="records")  # type: ignore[no-any-return]


def save_cache(rows: list[dict], cache_dir: Path | None = None) -> Path:
    """Write rows to cache file using atomic write."""
    cache_path = get_cache_path(cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "row_count": len(rows),
        "rows": rows,
    }

    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(envelope, f)
        os.replace(tmp_path, cache_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return cache_path


def load_cache(cache_dir: Path | None = None) -> tuple[list[dict], str | None]:
    """Load rows from cache. Returns (row_dicts, fetched_at) or ([], None) if no cache."""
    cache_path = get_cache_path(cache_dir)
    if not cache_path.exists():
        return [], None
    try:
        with open(cache_path) as f:
            envelope = json.load(f)
        return envelope.get("rows", []), envelope.get("fetched_at")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Corrupt cache file %s: %s", cache_path, e)
        return [], None


def get_dist_cache_path(cache_dir: Path | None = None) -> Path:
    return get_cache_dir(cache_dir) / "arena_dist.json"


def compute_distribution(rows: list[dict]) -> dict:
    """Compute distribution stats from overall-category ratings."""
    import statistics

    overall = [r["rating"] for r in rows if r.get("category") == "overall"]
    if not overall:
        raise ValueError("No overall-category rows found in Arena data")

    overall_sorted = sorted(overall)
    n = len(overall_sorted)
    p25_idx = int(n * 0.25)
    p75_idx = int(n * 0.75)

    return {
        "stats": {
            "count": n,
            "min": min(overall),
            "max": max(overall),
            "median": statistics.median(overall),
            "mean": statistics.mean(overall),
            "stdev": statistics.stdev(overall) if n > 1 else 0.0,
            "p25": overall_sorted[p25_idx],
            "p75": overall_sorted[p75_idx],
        },
        "scores": overall_sorted,
    }


def save_dist_cache(dist: dict, cache_dir: Path | None = None) -> Path:
    """Write distribution stats to cache file."""
    cache_path = get_dist_cache_path(cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(dist, f)
        os.replace(tmp_path, cache_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return cache_path


def load_dist_cache(cache_dir: Path | None = None) -> dict | None:
    """Load cached distribution stats. Computes from model cache if missing."""
    cache_path = get_dist_cache_path(cache_dir)
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt dist cache file %s: %s", cache_path, e)

    rows, _ = load_cache(cache_dir)
    if not rows:
        return None
    try:
        dist = compute_distribution(rows)
        save_dist_cache(dist, cache_dir)
        return dist
    except ValueError:
        return None


def sync(cache_dir: Path | None = None) -> tuple[int, Path]:
    """Fetch from HuggingFace, sort rows, save cache and distribution stats. Returns (row_count, cache_path)."""
    logger.info("Fetching Arena leaderboard from HuggingFace...")
    rows = fetch_from_hf()
    logger.info("Received %d rows", len(rows))
    # Sort key must match scripts/format_quality_data.py for consistent ordering.
    rows.sort(key=lambda r: (r.get("model_name") or "", r.get("category") or ""))
    # Strip to scoring-relevant fields only
    rows = [{k: v for k, v in r.items() if k in SCORING_FIELDS} for r in rows]
    cache_path = save_cache(rows, cache_dir)

    dist = compute_distribution(rows)
    dist_path = save_dist_cache(dist, cache_dir)
    logger.info(
        "Distribution stats cached to %s (%d overall models)", dist_path, dist["stats"]["count"]
    )

    return len(rows), cache_path
