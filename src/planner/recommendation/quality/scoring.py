"""Quality scoring helper — applies use-case category weights to ScoringEngine scorecards."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from quality_scoring import aa_client, arena_client
from quality_scoring.cache import is_cache_stale
from quality_scoring.categories import CATEGORY_MAP
from quality_scoring.engine import ScoringEngine
from quality_scoring.models import ModelScorecard

logger = logging.getLogger(__name__)


def compute_quality_score(
    scorecard: ModelScorecard,
    category_weights: dict[str, int],
) -> float:
    """Compute a weighted quality score from a scorecard and category weights.

    category_weights maps category names to integer weights (e.g.,
    {"coding": 5, "math": 3}).  Weights are normalized internally.

    For categories where the model has no data, the model's overall
    composite percentile is used as a fill-in.  This is a known
    simplification — see docs/quality-scoring-guide.md for future
    alternatives (minimum-by-size estimation, pre-computed fill-ins).
    """
    overall_pct = scorecard.overall.percentile if scorecard.overall else 0.0

    total_weight = 0
    weighted_sum = 0.0
    for cat, weight in category_weights.items():
        cs = scorecard.categories.get(cat)
        if cs is not None:
            weighted_sum += weight * cs.percentile
        else:
            weighted_sum += weight * overall_pct
        total_weight += weight

    if total_weight > 0:
        return round(weighted_sum / total_weight, 2)

    return overall_pct


def load_quality_weights(path: Path) -> dict:
    """Load use-case category weights from a JSON file. Returns {} if file is missing."""
    if not path.is_file():
        logger.warning("Quality weights file not found: %s", path)
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def validate_quality_weights(weights: dict) -> None:
    """Log warnings for any category keys not in CATEGORY_MAP."""
    valid_keys = set(CATEGORY_MAP.keys())
    for use_case, config in weights.items():
        cats = config.get("categories", {})
        for cat_key in cats:
            if cat_key not in valid_keys:
                logger.warning(
                    "Unknown category %r in quality weights for use case %r. Valid categories: %s",
                    cat_key,
                    use_case,
                    ", ".join(sorted(valid_keys)),
                )


def build_scoring_engine(
    cache_dir: Path | None = None,
    auto_update: bool | None = None,
    aa_api_key: str | None = None,
) -> tuple[ScoringEngine, dict[str, Any]]:
    """Build a ScoringEngine, handling cache loading and optional auto-update.

    Args:
        cache_dir: Explicit cache directory for runtime cache files.
            When provided the clients read/write there instead of
            consulting ``LLM_QUALITY_CACHE_DIR``.  The environment
            variable is **never** mutated.
        auto_update: Override for the QUALITY_AUTO_UPDATE env var.
            When provided, this value is used instead of the env var.
        aa_api_key: Artificial Analysis API key. Falls back to
            AA_API_KEY env var.

    Returns:
        A ``(engine, metadata)`` tuple.  *metadata* contains counts
        and ``fetched_at`` timestamps for each data source so callers
        can cache them without re-reading the full files.
    """
    from quality_scoring.data._resolver import quality_data_path

    checked_in_dir = quality_data_path()
    runtime_cache_dir = cache_dir or Path(os.environ.get("LLM_QUALITY_CACHE_DIR", ".quality_cache"))

    if auto_update is None:
        auto_update = os.environ.get("QUALITY_AUTO_UPDATE", "false").lower() in (
            "true",
            "1",
            "yes",
        )
    if aa_api_key is None:
        aa_api_key = os.environ.get("AA_API_KEY")

    arena_rows: list = []
    aa_models: list = []
    arena_fetched_at: str | None = None
    aa_fetched_at: str | None = None

    if auto_update:
        runtime_cache_dir.mkdir(parents=True, exist_ok=True)

        arena_rows_cached, arena_fetched = arena_client.load_cache(cache_dir=runtime_cache_dir)
        if not arena_rows_cached or is_cache_stale(arena_fetched):
            logger.info("Auto-updating Arena data...")
            try:
                arena_client.sync(cache_dir=runtime_cache_dir)
                arena_rows, arena_fetched_at = arena_client.load_cache(cache_dir=runtime_cache_dir)
            except Exception:
                arena_rows = arena_rows_cached
                arena_fetched_at = arena_fetched
                logger.warning("Arena sync failed, falling back to cached data")
        else:
            arena_rows = arena_rows_cached
            arena_fetched_at = arena_fetched

        aa_cached, aa_fetched = aa_client.load_cache(cache_dir=runtime_cache_dir)
        if aa_api_key and (not aa_cached or is_cache_stale(aa_fetched)):
            logger.info("Auto-updating AA data...")
            try:
                aa_client.sync(api_key=aa_api_key, cache_dir=runtime_cache_dir)
                aa_models, aa_fetched_at = aa_client.load_cache(cache_dir=runtime_cache_dir)
            except Exception:
                aa_models = aa_cached
                aa_fetched_at = aa_fetched
                logger.warning("AA sync failed, falling back to cached data")
        elif aa_cached:
            aa_models = aa_cached
            aa_fetched_at = aa_fetched
        else:
            if not aa_api_key:
                logger.warning("AA_API_KEY not set — using checked-in AA data")

    # Fall back to checked-in data when no runtime data loaded,
    # or when checked-in snapshots are newer than runtime cache.
    # Note: fetched_at comparison is lexicographic (ISO 8601 with consistent tz format).
    arena_path = checked_in_dir / "arena_models.json"
    if arena_path.is_file():
        with open(arena_path) as f:
            checked_in = json.load(f)
        checked_in_rows = checked_in.get("rows", [])
        checked_in_fetched = checked_in.get("fetched_at")
        if not arena_rows or (
            checked_in_fetched and arena_fetched_at and checked_in_fetched > arena_fetched_at
        ):
            arena_rows = checked_in_rows
            arena_fetched_at = checked_in_fetched

    aa_path = checked_in_dir / "aa_models.json"
    if aa_path.is_file():
        with open(aa_path) as f:
            checked_in = json.load(f)
        checked_in_models = checked_in.get("models", [])
        checked_in_fetched = checked_in.get("fetched_at")
        if not aa_models or (
            checked_in_fetched and aa_fetched_at and checked_in_fetched > aa_fetched_at
        ):
            aa_models = checked_in_models
            aa_fetched_at = checked_in_fetched

    logger.info(
        "Building ScoringEngine: %d arena rows, %d AA models", len(arena_rows), len(aa_models)
    )

    metadata: dict[str, Any] = {
        "arena_count": len(arena_rows),
        "arena_fetched": arena_fetched_at,
        "aa_count": len(aa_models),
        "aa_fetched": aa_fetched_at,
    }

    return ScoringEngine(arena_rows=arena_rows, aa_models=aa_models), metadata
