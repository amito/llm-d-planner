"""Pytest configuration and shared fixtures.

Creates a temporary SQLite database for database tests, loading
a small static fixture dataset. No external infrastructure needed.
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from planner.knowledge_base.benchmarks import BenchmarkRepository

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_db_path():
    """Create and populate a temporary SQLite database for the test session.

    Yields the database file path, then cleans up.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    fixture_path = FIXTURES_DIR / "test_benchmarks.json"
    BenchmarkRepository.from_files(fixture_path, db_path=db_path)

    yield db_path

    Path(db_path).unlink(missing_ok=True)
    Path(db_path + "-wal").unlink(missing_ok=True)
    Path(db_path + "-shm").unlink(missing_ok=True)


@pytest.fixture(scope="session")
def test_quality_data():
    """Load test quality score fixture data for Arena and AA.

    Returns a dict with 'arena_rows' and 'aa_models' keys.
    """
    fixture_path = FIXTURES_DIR / "test_quality_scores.json"
    with open(fixture_path) as f:
        data = json.load(f)

    return {
        "arena_rows": data["arena_rows"],
        "aa_models": data["aa_models"],
    }
