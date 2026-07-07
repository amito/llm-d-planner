"""Database management API routes.

Provides endpoints for uploading benchmark data, checking DB status,
and resetting the benchmark database. These endpoints enable remote
management of Planner deployments (e.g., on Kubernetes) without
needing shell access.
"""

import hmac
import json
import logging
import os

import psycopg2
from fastapi import APIRouter, File, Header, HTTPException, UploadFile, status

from planner.api.dependencies import _get_benchmark_source_type
from planner.knowledge_base.loader import (
    extract_metadata,
    get_db_stats,
    insert_benchmarks,
    reset_benchmarks,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["database"])

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:planner@localhost:5432/planner",
)


_DB_ADMIN_PASSWORD = os.getenv("DB_ADMIN_PASSWORD")


def _check_admin_password(password: str | None) -> None:
    """Verify admin password if one is configured."""
    if not _DB_ADMIN_PASSWORD:
        return
    if not password or not hmac.compare_digest(password, _DB_ADMIN_PASSWORD):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin password",
        )


def _get_connection():
    """Get a database connection for DB management operations."""
    return psycopg2.connect(_DATABASE_URL)


@router.get("/db/admin-required")
async def admin_required():
    """Check whether DB admin operations require a password."""
    return {"required": _DB_ADMIN_PASSWORD is not None}


@router.post("/db/verify-admin")
async def verify_admin(x_admin_password: str | None = Header(None)):
    """Verify the admin password without performing any action."""
    _check_admin_password(x_admin_password)
    return {"verified": True}


def _build_benchmark_source_info() -> dict:
    """Build the benchmark_source section for status responses."""
    source_type = _get_benchmark_source_type()
    info: dict = {"type": source_type}
    if source_type == "model_catalog":
        info["model_catalog_source_id"] = (
            os.getenv("MODEL_CATALOG_SOURCE_ID", "").strip() or "redhat_ai_validated_models"
        )
    return info


@router.get("/db/status")
async def db_status():
    """Get current benchmark database statistics."""
    try:
        conn = _get_connection()
        try:
            stats = get_db_stats(conn)
            return {
                "success": True,
                **stats,
                "benchmark_source": _build_benchmark_source_info(),
            }
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to get DB status: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database not accessible: {e}"
        ) from e


@router.post("/db/upload-benchmarks")
async def upload_benchmarks(
    file: UploadFile = File(...),
    x_admin_password: str | None = Header(None),
):
    """Upload a benchmark JSON file and load it into the database.

    The JSON file should have a top-level "benchmarks" array containing
    benchmark records. Duplicates (same model/hardware/traffic config)
    are silently skipped.

    Usage:
        curl -X POST -F 'file=@benchmarks.json' http://host/api/v1/db/upload-benchmarks
    """
    _check_admin_password(x_admin_password)

    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a .json file"
        )

    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid JSON: {e}"
        ) from e

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='No benchmarks found. JSON must have a top-level "benchmarks" array.',
        )

    meta = extract_metadata(data)
    source = meta["source"] or "local"
    confidence_level = meta["confidence_level"] or "estimated"

    try:
        conn = _get_connection()
        try:
            stats = insert_benchmarks(
                conn,
                benchmarks,
                source=source,
                confidence_level=confidence_level,
            )
            logger.info(
                f"Uploaded {len(benchmarks)} benchmarks from {file.filename}, "
                f"DB now has {stats['total_benchmarks']} total"
            )
            return {
                "success": True,
                "filename": file.filename,
                "records_in_file": len(benchmarks),
                **stats,
            }
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load benchmarks: {e}",
        ) from e


@router.post("/db/reset")
async def reset_database(x_admin_password: str | None = Header(None)):
    """Reset the benchmark database by removing all benchmark data.

    This truncates the exported_summaries table (cascading to related tables).
    The schema is preserved — only data is removed.

    Usage:
        curl -X POST http://host/api/v1/db/reset
    """
    _check_admin_password(x_admin_password)

    try:
        conn = _get_connection()
        try:
            reset_benchmarks(conn)
            stats = get_db_stats(conn)
            logger.info("Benchmark database reset via API")
            return {
                "success": True,
                "message": "Benchmark database has been reset",
                **stats,
            }
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset database: {e}",
        ) from e
