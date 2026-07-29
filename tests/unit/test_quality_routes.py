"""Unit tests for quality data management API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from planner.api.routes.quality import router

_app = FastAPI()
_app.include_router(router)


@pytest.fixture
def client():
    return TestClient(_app)


@pytest.fixture
def mock_request():
    """Create a mock request with app.state."""
    mock_req = MagicMock()
    mock_req.app.state.scoring_engine = MagicMock()
    mock_req.app.state.workflow = MagicMock()
    mock_req.app.state.workflow.config_finder = MagicMock()
    return mock_req


# --- GET /api/v1/quality/auto-update ---


@pytest.mark.unit
class TestGetAutoUpdateStatus:
    def test_returns_disabled_when_not_set(self, client):
        _app.state.quality_metadata = {}
        _app.state.quality_auto_update = None  # Reset so it re-reads env
        with patch.dict("os.environ", {}, clear=False):
            resp = client.get("/api/v1/quality/auto-update")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["arena_last_updated"] is None
        assert body["aa_last_updated"] is None
        assert body["arena_model_count"] == 0
        assert body["aa_model_count"] == 0

    def test_returns_enabled_with_metadata(self, client):
        _app.state.quality_metadata = {
            "arena_count": 1,
            "arena_fetched": "2026-07-01T12:00:00Z",
            "aa_count": 1,
            "aa_fetched": "2026-07-01T13:00:00Z",
        }
        _app.state.quality_auto_update = None  # Reset so it re-reads env
        with patch.dict("os.environ", {"QUALITY_AUTO_UPDATE": "true"}, clear=False):
            resp = client.get("/api/v1/quality/auto-update")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert body["arena_last_updated"] == "2026-07-01T12:00:00Z"
        assert body["aa_last_updated"] == "2026-07-01T13:00:00Z"
        assert body["arena_model_count"] == 1
        assert body["aa_model_count"] == 1

    def test_handles_various_true_values(self, client):
        _app.state.quality_metadata = {}
        for val in ["true", "True", "TRUE", "1", "yes", "YES"]:
            _app.state.quality_auto_update = None  # Reset so it re-reads env
            with patch.dict("os.environ", {"QUALITY_AUTO_UPDATE": val}, clear=False):
                resp = client.get("/api/v1/quality/auto-update")
            assert resp.status_code == 200
            assert resp.json()["enabled"] is True


# --- PUT /api/v1/quality/auto-update ---


@pytest.mark.unit
class TestSetAutoUpdate:
    def test_enables_auto_update(self, client):
        _app.state.quality_metadata = {}
        _app.state.quality_auto_update = False
        resp = client.put("/api/v1/quality/auto-update", json={"enabled": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert _app.state.quality_auto_update is True

    def test_disables_auto_update(self, client):
        _app.state.quality_metadata = {}
        _app.state.quality_auto_update = True
        resp = client.put("/api/v1/quality/auto-update", json={"enabled": False})
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert _app.state.quality_auto_update is False

    def test_logs_state_change(self, client):
        _app.state.quality_metadata = {}
        _app.state.quality_auto_update = False
        with patch("planner.api.routes.quality.logger") as mock_logger:
            client.put("/api/v1/quality/auto-update", json={"enabled": True})
        mock_logger.info.assert_called_once()
        # Check format string and args
        call_args = mock_logger.info.call_args
        assert "auto-update" in call_args[0][0].lower()
        assert call_args[0][1] == "enabled"


# --- POST /api/v1/quality/refresh ---


@pytest.mark.unit
class TestRefreshQualityData:
    def test_successful_sync_both_sources(self, client):
        mock_engine = MagicMock()
        mock_metadata = {
            "arena_count": 100,
            "aa_count": 50,
            "arena_fetched": None,
            "aa_fetched": None,
        }
        with (
            patch.dict("os.environ", {"AA_API_KEY": "test-key"}, clear=False),
            patch(
                "planner.api.routes.quality.arena_client.sync",
                return_value=(100, MagicMock()),
            ),
            patch(
                "planner.api.routes.quality.arena_client.load_cache",
                return_value=([], "2026-07-12T10:00:00Z"),
            ),
            patch(
                "planner.api.routes.quality.aa_client.sync",
                return_value=(50, MagicMock()),
            ),
            patch(
                "planner.api.routes.quality.aa_client.load_cache",
                return_value=([], "2026-07-12T10:30:00Z"),
            ),
            patch(
                "planner.recommendation.quality.scoring.build_scoring_engine",
                return_value=(mock_engine, mock_metadata),
            ),
        ):
            # Add mock app state
            _app.state.scoring_engine = MagicMock()
            _app.state.quality_metadata = {}
            _app.state.workflow = MagicMock()
            _app.state.workflow.config_finder = MagicMock()

            resp = client.post("/api/v1/quality/refresh")

        assert resp.status_code == 200
        body = resp.json()
        assert body["arena_rows"] == 100
        assert body["aa_models"] == 50
        assert body["arena_last_updated"] == "2026-07-12T10:00:00Z"
        assert body["aa_last_updated"] == "2026-07-12T10:30:00Z"
        assert body["errors"] == []

    def test_skips_aa_when_no_key(self, client):
        import os

        env = os.environ.copy()
        env.pop("AA_API_KEY", None)  # Remove AA_API_KEY if it exists

        mock_engine = MagicMock()
        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "planner.api.routes.quality.arena_client.sync",
                return_value=(100, MagicMock()),
            ),
            patch(
                "planner.api.routes.quality.arena_client.load_cache",
                return_value=([], "2026-07-12T10:00:00Z"),
            ),
            patch("planner.api.routes.quality.aa_client.sync") as mock_aa_sync,
            patch(
                "planner.recommendation.quality.scoring.build_scoring_engine",
                return_value=(mock_engine, {}),
            ),
        ):
            # Add mock app state
            _app.state.scoring_engine = MagicMock()
            _app.state.quality_metadata = {}
            _app.state.workflow = MagicMock()
            _app.state.workflow.config_finder = MagicMock()

            resp = client.post("/api/v1/quality/refresh")

        assert resp.status_code == 200
        body = resp.json()
        assert body["arena_rows"] == 100
        assert body["aa_models"] == 0
        # Verify AA sync was never called when no API key
        mock_aa_sync.assert_not_called()

    def test_handles_arena_sync_failure(self, client):
        mock_engine = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=False),
            patch(
                "planner.api.routes.quality.arena_client.sync",
                side_effect=Exception("Connection failed"),
            ),
            patch(
                "planner.recommendation.quality.scoring.build_scoring_engine",
                return_value=(mock_engine, {}),
            ),
            patch("planner.api.routes.quality.logger") as mock_logger,
        ):
            # Add mock app state
            _app.state.scoring_engine = MagicMock()
            _app.state.quality_metadata = {}
            _app.state.workflow = MagicMock()
            _app.state.workflow.config_finder = MagicMock()

            resp = client.post("/api/v1/quality/refresh")

        assert resp.status_code == 200
        body = resp.json()
        assert body["arena_rows"] == 0
        assert any("Arena sync failed" in e for e in body["errors"])
        mock_logger.exception.assert_called()

    def test_handles_aa_sync_failure(self, client):
        mock_engine = MagicMock()
        with (
            patch.dict("os.environ", {"AA_API_KEY": "test-key"}, clear=False),
            patch(
                "planner.api.routes.quality.arena_client.sync",
                return_value=(100, MagicMock()),
            ),
            patch(
                "planner.api.routes.quality.arena_client.load_cache",
                return_value=([], "2026-07-12T10:00:00Z"),
            ),
            patch(
                "planner.api.routes.quality.aa_client.sync",
                side_effect=Exception("API error"),
            ),
            patch(
                "planner.recommendation.quality.scoring.build_scoring_engine",
                return_value=(mock_engine, {}),
            ),
            patch("planner.api.routes.quality.logger") as mock_logger,
        ):
            # Add mock app state
            _app.state.scoring_engine = MagicMock()
            _app.state.quality_metadata = {}
            _app.state.workflow = MagicMock()
            _app.state.workflow.config_finder = MagicMock()

            resp = client.post("/api/v1/quality/refresh")

        assert resp.status_code == 200
        body = resp.json()
        assert body["arena_rows"] == 100
        assert body["aa_models"] == 0
        assert any("AA sync failed" in e for e in body["errors"])
        mock_logger.exception.assert_called()

    def test_rebuilds_and_swaps_engine(self, client):
        mock_new_engine = MagicMock()
        mock_metadata = {
            "arena_count": 100,
            "aa_count": 0,
            "arena_fetched": None,
            "aa_fetched": None,
        }
        with (
            patch.dict("os.environ", {}, clear=False),
            patch(
                "planner.api.routes.quality.arena_client.sync",
                return_value=(100, MagicMock()),
            ),
            patch(
                "planner.api.routes.quality.arena_client.load_cache",
                return_value=([], "2026-07-12T10:00:00Z"),
            ),
            patch(
                "planner.recommendation.quality.scoring.build_scoring_engine",
                return_value=(mock_new_engine, mock_metadata),
            ),
        ):
            # Add mock app state
            _app.state.scoring_engine = MagicMock()
            _app.state.quality_metadata = {}
            _app.state.workflow = MagicMock()
            _app.state.workflow.config_finder = MagicMock()

            resp = client.post("/api/v1/quality/refresh")

        assert resp.status_code == 200
        assert _app.state.scoring_engine == mock_new_engine
        assert _app.state.quality_metadata == mock_metadata
        # Verify update_engine was called instead of direct _engine mutation
        _app.state.workflow.config_finder.update_engine.assert_called()
