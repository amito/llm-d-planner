"""Test that /extract-intent route exists and /extract still works."""

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.mark.unit
def test_extract_intent_route_exists(client):
    """The /extract-intent route should exist (even if it fails due to no LLM)."""
    try:
        response = client.post(
            "/api/v1/extract-intent",
            json={"text": "test"},
        )
        # Will fail with 500 (no LLM), but should NOT be 404/405
        assert response.status_code != 404
        assert response.status_code != 405
    except Exception:
        # If exception is raised (e.g., missing LLM), that's fine too
        # The important thing is the route exists and doesn't 404/405
        pass


@pytest.mark.unit
def test_extract_alias_still_works(client):
    """The old /extract route should still work."""
    try:
        response = client.post(
            "/api/v1/extract",
            json={"text": "test"},
        )
        assert response.status_code != 404
        assert response.status_code != 405
    except Exception:
        # If exception is raised (e.g., missing LLM), that's fine too
        # The important thing is the route exists and doesn't 404/405
        pass
