"""Unit tests for database management API routes."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from planner.api.routes.database import router

_app = FastAPI()
_app.include_router(router)


@pytest.fixture
def client():
    return TestClient(_app)


def _sample_stats():
    return {
        "num_models": 3,
        "num_hardware_types": 2,
        "num_traffic_profiles": 4,
        "total_benchmarks": 100,
        "traffic_distribution": [],
    }


# --- GET /api/v1/db/admin-required ---


@pytest.mark.unit
class TestAdminRequired:
    def test_returns_false_when_no_password(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None):
            resp = client.get("/api/v1/db/admin-required")
        assert resp.status_code == 200
        assert resp.json() == {"required": False}

    def test_returns_true_when_password_set(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", "secret"):
            resp = client.get("/api/v1/db/admin-required")
        assert resp.status_code == 200
        assert resp.json() == {"required": True}


# --- POST /api/v1/db/verify-admin ---


@pytest.mark.unit
class TestVerifyAdmin:
    def test_no_password_configured_always_passes(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None):
            resp = client.post("/api/v1/db/verify-admin")
        assert resp.status_code == 200
        assert resp.json() == {"verified": True}

    def test_correct_password(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", "secret"):
            resp = client.post("/api/v1/db/verify-admin", headers={"x-admin-password": "secret"})
        assert resp.status_code == 200

    def test_wrong_password(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", "secret"):
            resp = client.post("/api/v1/db/verify-admin", headers={"x-admin-password": "wrong"})
        assert resp.status_code == 401

    def test_missing_password_when_required(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", "secret"):
            resp = client.post("/api/v1/db/verify-admin")
        assert resp.status_code == 401


# --- GET /api/v1/db/status ---


@pytest.mark.unit
class TestDbStatus:
    def test_returns_stats(self, client):
        mock_conn = MagicMock()
        stats = _sample_stats()
        with (
            patch("planner.api.routes.database._get_connection", return_value=mock_conn),
            patch("planner.api.routes.database.get_db_stats", return_value=stats),
            patch("planner.api.routes.database._get_benchmark_source_type", return_value="postgresql"),
        ):
            resp = client.get("/api/v1/db/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total_benchmarks"] == 100
        assert body["benchmark_source"] == {"type": "postgresql"}
        mock_conn.close.assert_called_once()

    def test_includes_benchmark_source_model_catalog(self, client):
        mock_conn = MagicMock()
        stats = _sample_stats()
        with (
            patch("planner.api.routes.database._get_connection", return_value=mock_conn),
            patch("planner.api.routes.database.get_db_stats", return_value=stats),
            patch("planner.api.routes.database._get_benchmark_source_type", return_value="model_catalog"),
            patch.dict("os.environ", {"MODEL_CATALOG_SOURCE_ID": "my_catalog"}),
        ):
            resp = client.get("/api/v1/db/status")
        assert resp.status_code == 200
        source = resp.json()["benchmark_source"]
        assert source == {"type": "model_catalog", "model_catalog_source_id": "my_catalog"}

    def test_returns_503_on_db_error(self, client):
        with patch(
            "planner.api.routes.database._get_connection",
            side_effect=Exception("connection refused"),
        ):
            resp = client.get("/api/v1/db/status")
        assert resp.status_code == 503


# --- POST /api/v1/db/upload-benchmarks ---


@pytest.mark.unit
class TestUploadBenchmarks:
    def _upload(self, client, filename, content, headers=None):
        data = json.dumps(content).encode()
        return client.post(
            "/api/v1/db/upload-benchmarks",
            files={"file": (filename, io.BytesIO(data), "application/json")},
            headers=headers or {},
        )

    def test_successful_upload(self, client):
        mock_conn = MagicMock()
        stats = _sample_stats()
        payload = {
            "_metadata": {"source": "blis", "confidence_level": "benchmarked"},
            "benchmarks": [{"model_hf_repo": "m", "hardware": "H100"}],
        }
        with (
            patch("planner.api.routes.database._get_connection", return_value=mock_conn),
            patch("planner.api.routes.database.insert_benchmarks", return_value=stats),
            patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None),
        ):
            resp = self._upload(client, "bench.json", payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["records_in_file"] == 1
        assert body["filename"] == "bench.json"
        mock_conn.close.assert_called_once()

    def test_rejects_non_json_file(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None):
            resp = client.post(
                "/api/v1/db/upload-benchmarks",
                files={"file": ("data.csv", io.BytesIO(b"a,b"), "text/csv")},
            )
        assert resp.status_code == 400
        assert "json" in resp.json()["detail"].lower()

    def test_rejects_invalid_json(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None):
            resp = client.post(
                "/api/v1/db/upload-benchmarks",
                files={"file": ("data.json", io.BytesIO(b"not json{"), "application/json")},
            )
        assert resp.status_code == 400
        assert "Invalid JSON" in resp.json()["detail"]

    def test_rejects_empty_benchmarks(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None):
            resp = self._upload(client, "empty.json", {"benchmarks": []})
        assert resp.status_code == 400
        assert "No benchmarks" in resp.json()["detail"]

    def test_rejects_missing_benchmarks_key(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None):
            resp = self._upload(client, "no_key.json", {"data": [1, 2]})
        assert resp.status_code == 400

    def test_requires_admin_password(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", "secret"):
            resp = self._upload(client, "bench.json", {"benchmarks": [{"a": 1}]})
        assert resp.status_code == 401

    def test_accepts_with_correct_password(self, client):
        mock_conn = MagicMock()
        stats = _sample_stats()
        payload = {"benchmarks": [{"model_hf_repo": "m", "hardware": "H100"}]}
        with (
            patch("planner.api.routes.database._get_connection", return_value=mock_conn),
            patch("planner.api.routes.database.insert_benchmarks", return_value=stats),
            patch("planner.api.routes.database._DB_ADMIN_PASSWORD", "secret"),
        ):
            resp = self._upload(client, "b.json", payload, headers={"x-admin-password": "secret"})
        assert resp.status_code == 200


# --- POST /api/v1/db/reset ---


@pytest.mark.unit
class TestResetDatabase:
    def test_successful_reset(self, client):
        mock_conn = MagicMock()
        stats = _sample_stats()
        stats["total_benchmarks"] = 0
        with (
            patch("planner.api.routes.database._get_connection", return_value=mock_conn),
            patch("planner.api.routes.database.reset_benchmarks"),
            patch("planner.api.routes.database.get_db_stats", return_value=stats),
            patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None),
        ):
            resp = client.post("/api/v1/db/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total_benchmarks"] == 0
        mock_conn.close.assert_called_once()

    def test_requires_admin_password(self, client):
        with patch("planner.api.routes.database._DB_ADMIN_PASSWORD", "secret"):
            resp = client.post("/api/v1/db/reset")
        assert resp.status_code == 401

    def test_returns_500_on_db_error(self, client):
        with (
            patch(
                "planner.api.routes.database._get_connection",
                side_effect=Exception("db down"),
            ),
            patch("planner.api.routes.database._DB_ADMIN_PASSWORD", None),
        ):
            resp = client.post("/api/v1/db/reset")
        assert resp.status_code == 500
