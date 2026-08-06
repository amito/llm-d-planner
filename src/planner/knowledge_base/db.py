"""Database connection utilities."""

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def get_db_path() -> str:
    """Resolve the database file path from env or default.

    The default path is relative to the project root (not cwd) so the
    same database is used regardless of which directory commands run from.
    """
    env_path = os.environ.get("PLANNER_DB_PATH")
    if env_path:
        return env_path
    return str(find_project_root() / "data" / "planner.db")


def create_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Create a new database connection with WAL mode and schema initialization.

    Each call returns a new connection — callers own the lifecycle.
    Uses check_same_thread=False for FastAPI's async thread pool.
    """
    path = db_path or get_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    from planner.knowledge_base.loader import ensure_schema

    ensure_schema(conn)
    logger.info("Database connection established: %s", path)
    return conn
