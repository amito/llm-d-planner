"""Tests that removed endpoints are gone."""

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.mark.unit
class TestRemovedEndpoints:
    def test_recommend_removed(self, client):
        """Verify POST /api/v1/recommend endpoint is removed."""
        response = client.post("/api/v1/recommend", json={"message": "test"})
        assert response.status_code in (404, 405)

    def test_test_endpoint_removed(self, client):
        """Verify POST /api/v1/test endpoint is removed."""
        response = client.post("/api/v1/test")
        assert response.status_code in (404, 405)

    def test_mock_status_removed(self, client):
        """Verify GET /api/v1/deployments/{id}/status endpoint is removed."""
        response = client.get("/api/v1/deployments/test-id/status")
        assert response.status_code in (404, 405)
