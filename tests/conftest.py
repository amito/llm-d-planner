"""Pytest configuration and shared fixtures.

Creates a temporary SQLite database for database tests, loading
a small static fixture dataset. No external infrastructure needed.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest

from planner.knowledge_base.benchmarks import BenchmarkRepository

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _disable_cluster_gpu_detection(request, monkeypatch):
    """Prevent tests from contacting a real Kubernetes cluster.

    Skipped for test_gpu_detector which tests detection itself with mocked K8s.
    """
    if "test_gpu_detector" in request.fspath.basename:
        return
    monkeypatch.setenv("PLANNER_DETECT_CLUSTER_GPUS", "false")


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


CANNED_INTENT_DICT = {
    "use_case": "chatbot_conversational",
    "user_count": 1000,
    "quality_priority": "high",
    "cost_priority": "low",
    "latency_priority": "medium",
    "preferred_gpu_types": [],  # Empty to allow all GPUs
    "preferred_models": [],  # Empty to allow all models
    "domain_specialization": ["general"],
}


class MockLLMClient:
    """Mock LLM client that returns canned intent extraction responses."""

    def chat(
        self,
        messages: list[dict[str, str]],
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        return {"message": {"role": "assistant", "content": json.dumps(CANNED_INTENT_DICT)}}

    def extract_structured_data(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        return CANNED_INTENT_DICT.copy()

    def is_available(self) -> bool:
        return True


@pytest.fixture
def mock_llm_client():
    """A mock LLM client that returns canned intent responses."""
    return MockLLMClient()


@pytest.fixture
def canned_intent():
    """The canned intent dict that MockLLMClient returns."""
    return CANNED_INTENT_DICT.copy()


def _build_mock_scoring_engine(quality_data):
    """Build a ScoringEngine from test fixture data. Shared helper for mock_scoring_engine fixture."""
    from quality_scoring.engine import ScoringEngine

    engine = ScoringEngine(
        arena_rows=quality_data["arena_rows"],
        aa_models=quality_data["aa_models"],
    )
    metadata = {
        "arena_count": len(quality_data["arena_rows"]),
        "arena_fetched": "2026-08-14T00:00:00Z",
        "aa_count": len(quality_data["aa_models"]),
        "aa_fetched": "2026-08-14T00:00:00Z",
    }
    return engine, metadata


@pytest.fixture
def mock_scoring_engine(test_quality_data):
    """A mock build_scoring_engine that uses canned quality data.

    Usage: patch build_scoring_engine with side_effect=mock_scoring_engine.
    Returns a callable with the same signature as build_scoring_engine.
    """

    def _mock(cache_dir=None, auto_update=None, aa_api_key=None):
        return _build_mock_scoring_engine(test_quality_data)

    return _mock
