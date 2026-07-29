"""Quality data management endpoints."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Request
from pydantic import BaseModel

from quality_scoring import aa_client, arena_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["quality"])


class AutoUpdateStatus(BaseModel):
    enabled: bool
    arena_last_updated: str | None = None
    aa_last_updated: str | None = None
    arena_model_count: int = 0
    aa_model_count: int = 0


class AutoUpdateToggle(BaseModel):
    enabled: bool


class RefreshResult(BaseModel):
    arena_rows: int = 0
    aa_models: int = 0
    arena_last_updated: str | None = None
    aa_last_updated: str | None = None
    errors: list[str] = []


def _is_auto_update_enabled(request: Request) -> bool:
    """Check auto-update state from app.state, falling back to env var on first call."""
    enabled = getattr(request.app.state, "quality_auto_update", None)
    if enabled is None:
        enabled = os.environ.get("QUALITY_AUTO_UPDATE", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        request.app.state.quality_auto_update = enabled
    return enabled


@router.get("/quality/auto-update")
async def get_auto_update_status(request: Request) -> AutoUpdateStatus:
    """Get current auto-update status and data freshness info."""
    metadata = getattr(request.app.state, "quality_metadata", None) or {}

    return AutoUpdateStatus(
        enabled=_is_auto_update_enabled(request),
        arena_last_updated=metadata.get("arena_fetched"),
        aa_last_updated=metadata.get("aa_fetched"),
        arena_model_count=metadata.get("arena_count", 0),
        aa_model_count=metadata.get("aa_count", 0),
    )


@router.put("/quality/auto-update")
async def set_auto_update(request: Request, body: AutoUpdateToggle) -> AutoUpdateStatus:
    """Enable or disable auto-update of quality benchmark data."""
    request.app.state.quality_auto_update = body.enabled
    logger.info("Quality auto-update %s", "enabled" if body.enabled else "disabled")
    return await get_auto_update_status(request)


@router.post("/quality/refresh")
async def refresh_quality_data(request: Request) -> RefreshResult:
    """Trigger immediate sync of quality benchmark data and rebuild engine."""
    from pathlib import Path

    from planner.recommendation.quality.scoring import (
        build_scoring_engine,
        load_quality_weights,
        validate_quality_weights,
    )

    aa_api_key = os.environ.get("AA_API_KEY")
    arena_count = 0
    aa_count = 0
    arena_fetched = None
    aa_fetched = None
    errors: list[str] = []

    # Determine runtime cache dir (same logic as build_scoring_engine default)
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    runtime_cache_dir = repo_root / ".quality_cache"
    runtime_cache_dir.mkdir(parents=True, exist_ok=True)

    # Sync Arena (no key needed)
    try:
        arena_count, _ = arena_client.sync(cache_dir=runtime_cache_dir)
        _, arena_fetched = arena_client.load_cache(cache_dir=runtime_cache_dir)
        logger.info("Arena sync complete: %d rows", arena_count)
    except Exception as e:
        logger.exception("Arena sync failed")
        errors.append(f"Arena sync failed: {e}")

    # Sync AA (needs key)
    if aa_api_key:
        try:
            aa_count, _ = aa_client.sync(api_key=aa_api_key, cache_dir=runtime_cache_dir)
            _, aa_fetched = aa_client.load_cache(cache_dir=runtime_cache_dir)
            logger.info("AA sync complete: %d models", aa_count)
        except Exception as e:
            logger.exception("AA sync failed")
            errors.append(f"AA sync failed: {e}")
    else:
        logger.warning("AA_API_KEY not set — skipping AA sync")

    # Rebuild engine and swap into app state
    new_engine, new_metadata = build_scoring_engine(
        cache_dir=runtime_cache_dir,
        auto_update=_is_auto_update_enabled(request),
    )
    request.app.state.scoring_engine = new_engine
    request.app.state.quality_metadata = new_metadata

    # Reload quality weights and update config_finder via public API
    weights_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "data"
        / "configuration"
        / "quality_weights.json"
    )
    new_weights = load_quality_weights(weights_path)
    validate_quality_weights(new_weights)
    if hasattr(request.app.state, "workflow") and hasattr(
        request.app.state.workflow, "config_finder"
    ):
        request.app.state.workflow.config_finder.update_engine(new_engine, new_weights)
        logger.info("Reloaded quality_weights.json")

    return RefreshResult(
        arena_rows=arena_count,
        aa_models=aa_count,
        arena_last_updated=arena_fetched,
        aa_last_updated=aa_fetched,
        errors=errors,
    )
