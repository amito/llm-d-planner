"""Artificial Analysis API client and local cache management."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from quality_scoring.cache import get_cache_dir as _shared_get_cache_dir
from quality_scoring.cache import is_cache_stale  # noqa: F401 — re-export for backward compat

logger = logging.getLogger(__name__)

API_BASE = "https://artificialanalysis.ai/api/v2/language/models/free"


def get_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return cache directory, delegating to shared implementation."""
    return _shared_get_cache_dir(cache_dir)


def get_cache_path(cache_dir: Path | None = None) -> Path:
    return get_cache_dir(cache_dir) / "aa_models.json"


_MAX_PAGES = 100


class AAModelRecord(TypedDict):
    name: str
    slug: str
    intelligence_index: int | None
    coding_index: int | None
    agentic_index: int | None


def fetch_from_api(api_key: str) -> list[dict[str, Any]]:
    """Fetch all models from the AA free API, handling pagination."""
    import httpx

    all_models: list[dict[str, Any]] = []
    page = 1

    with httpx.Client(timeout=30) as client:
        while page <= _MAX_PAGES:
            resp = client.get(
                API_BASE,
                headers={"x-api-key": api_key},
                params={"page": page},
            )
            if resp.status_code == 401:
                raise RuntimeError("AA API authentication failed (401). Check your API key.")
            if resp.status_code == 429:
                raise RuntimeError("AA API rate limit exceeded (429). Try again later.")
            resp.raise_for_status()

            body = resp.json()
            if not isinstance(body, dict) or "data" not in body:
                raise RuntimeError(f"Unexpected API response structure: {type(body)}")
            if not isinstance(body["data"], list):
                raise RuntimeError(f"Expected list for 'data' field, got {type(body['data'])}")

            all_models.extend(body["data"])

            pagination = body.get("pagination", {})
            if not pagination.get("has_more", False):
                break
            page += 1

    if page > _MAX_PAGES:
        logger.warning(
            "Pagination capped at %d pages (%d models fetched). "
            "Some models may be missing from quality data.",
            _MAX_PAGES,
            len(all_models),
        )

    return all_models


def _safe_int(val: object) -> int | None:
    if val is None:
        return None
    try:
        f = float(str(val))
        return int(f)
    except (TypeError, ValueError):
        return None


def _map_api_model(api_obj: dict) -> AAModelRecord:
    """Map a single API model object to a normalized model dict.

    Only fields consumed by the scoring engine are saved.
    """
    evals = api_obj.get("evaluations") or {}

    return {
        "name": api_obj.get("name", ""),
        "slug": api_obj.get("slug", ""),
        "intelligence_index": _safe_int(evals.get("artificial_analysis_intelligence_index")),
        "coding_index": _safe_int(evals.get("artificial_analysis_coding_index")),
        "agentic_index": _safe_int(evals.get("artificial_analysis_agentic_index")),
    }


def save_cache(models: list[dict[str, Any]], cache_dir: Path | None = None) -> Path:
    """Write models to cache file using atomic write."""
    cache_path = get_cache_path(cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "model_count": len(models),
        "models": models,
    }

    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(envelope, f, indent=2)
        os.replace(tmp_path, cache_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return cache_path


def load_cache(cache_dir: Path | None = None) -> tuple[list[dict], str | None]:
    """Load models from cache. Returns (model_dicts, fetched_at) or ([], None) if no cache."""
    cache_path = get_cache_path(cache_dir)
    if not cache_path.exists():
        return [], None
    try:
        with open(cache_path) as f:
            envelope = json.load(f)
        return envelope.get("models", []), envelope.get("fetched_at")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Corrupt cache file %s: %s", cache_path, e)
        return [], None


def cache_age_display(fetched_at: str) -> str:
    """Human-readable cache age like '2 hours ago' or '3 days ago'."""
    try:
        fetched = datetime.fromisoformat(fetched_at)
    except ValueError:
        return "unknown age"

    now = datetime.now(UTC)
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=UTC)

    delta = now - fetched
    total_seconds = int(delta.total_seconds())

    if total_seconds < 60:
        return "just now"
    if total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if total_seconds < 86400:
        hours = total_seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = total_seconds // 86400
    return f"{days} day{'s' if days != 1 else ''} ago"


def get_dist_cache_path(cache_dir: Path | None = None) -> Path:
    return get_cache_dir(cache_dir) / "aa_dist.json"


def compute_distribution(models: list[dict[str, Any]]) -> dict:
    """Compute distribution stats from intelligence_index values."""
    import statistics

    scores = [m["intelligence_index"] for m in models if m.get("intelligence_index") is not None]
    if not scores:
        raise ValueError("No models with intelligence_index found in AA data")

    scores_sorted = sorted(scores)
    n = len(scores_sorted)
    p25_idx = int(n * 0.25)
    p75_idx = int(n * 0.75)

    return {
        "stats": {
            "count": n,
            "min": min(scores),
            "max": max(scores),
            "median": statistics.median(scores),
            "mean": statistics.mean(scores),
            "stdev": statistics.stdev(scores) if n > 1 else 0.0,
            "p25": scores_sorted[p25_idx],
            "p75": scores_sorted[p75_idx],
        },
        "scores": scores_sorted,
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

    models, _ = load_cache(cache_dir)
    if not models:
        return None
    try:
        dist = compute_distribution(models)
        save_dist_cache(dist, cache_dir)
        return dist
    except ValueError:
        return None


def sync(api_key: str, cache_dir: Path | None = None) -> tuple[int, Path]:
    """Fetch from API, map and sort models, save cache and distribution stats. Returns (model_count, cache_path)."""
    logger.info("Fetching models from AA API...")
    raw_models = fetch_from_api(api_key)
    logger.info("Received %d models from API", len(raw_models))

    mapped = [_map_api_model(m) for m in raw_models]
    with_index = [m for m in mapped if m.get("intelligence_index") is not None]
    logger.info(
        "%d models have intelligence_index (filtered from %d total)",
        len(with_index),
        len(mapped),
    )

    # Sort key must match scripts/format_quality_data.py for consistent ordering.
    mapped.sort(key=lambda m: m.get("slug") or "")

    cache_path = save_cache(mapped, cache_dir)  # type: ignore[arg-type]

    dist = compute_distribution(mapped)  # type: ignore[arg-type]
    dist_path = save_dist_cache(dist, cache_dir)
    logger.info("Distribution stats cached to %s (%d models)", dist_path, dist["stats"]["count"])

    return len(mapped), cache_path
