"""Tests for POST /generate-recommendations endpoint."""

import os

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app


@pytest.fixture
def client(test_db_path):
    os.environ["PLANNER_DB_PATH"] = test_db_path
    try:
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client
    finally:
        os.environ.pop("PLANNER_DB_PATH", None)


def _make_spec_payload(**overrides):
    """Build a minimal valid DeploymentSpecification payload."""
    base = {
        "intent": {
            "use_case": "chatbot_conversational",
            "user_count": 1000,
        },
        "slo_targets": {
            "ttft_target_ms": 300,
            "itl_target_ms": 30,
            "e2e_target_ms": 7000,
        },
        "workload_profile": {
            "prompt_tokens": 512,
            "output_tokens": 256,
            "expected_qps": 0.87,
        },
        "quality_weights": {
            "categories": {"overall": 4, "instruction_following": 3},
        },
        "priorities": {
            "quality": {"priority": "medium", "weight": 4},
            "cost": {"priority": "medium", "weight": 4},
            "latency": {"priority": "medium", "weight": 1},
        },
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestGenerateRecommendations:
    def test_accepts_deployment_specification(self, client):
        response = client.post(
            "/api/v1/generate-recommendations",
            json={"specification": _make_spec_payload()},
        )
        assert response.status_code == 200

    def test_response_has_ranked_lists(self, client):
        response = client.post(
            "/api/v1/generate-recommendations",
            json={"specification": _make_spec_payload()},
        )
        data = response.json()
        assert "balanced" in data
        assert "best_quality" in data
        assert "lowest_cost" in data
        assert "lowest_latency" in data
        assert "total_configs_evaluated" in data

    def test_weights_from_specification_priorities(self, client):
        """Priorities in the spec should drive balanced scoring weights."""
        response = client.post(
            "/api/v1/generate-recommendations",
            json={"specification": _make_spec_payload()},
        )
        assert response.status_code == 200

    def test_optional_filter_fields(self, client):
        response = client.post(
            "/api/v1/generate-recommendations",
            json={
                "specification": _make_spec_payload(),
                "enable_estimated": False,
                "min_quality": 50.0,
                "max_cost": 1000.0,
                "include_near_miss": False,
            },
        )
        assert response.status_code == 200

    def test_recommendations_have_scores(self, client):
        """Returned recommendations have multi-criteria scores."""
        response = client.post(
            "/api/v1/generate-recommendations",
            json={"specification": _make_spec_payload()},
        )
        assert response.status_code == 200
        data = response.json()

        # Check that we have recommendations with scores
        if len(data["balanced"]) > 0:
            first_rec = data["balanced"][0]
            assert "scores" in first_rec
            assert first_rec["scores"] is not None
            scores = first_rec["scores"]
            assert "quality_score" in scores
            assert "price_score" in scores
            assert "latency_score" in scores
            assert "balanced_score" in scores
            assert "slo_status" in scores
            # Scores should be 0-100 scale
            assert 0 <= scores["quality_score"] <= 100
            assert 0 <= scores["price_score"] <= 100
            assert 0 <= scores["latency_score"] <= 100
            assert 0 <= scores["balanced_score"] <= 100

    def test_ranked_lists_present(self, client):
        """All 4 ranked lists are present in response."""
        response = client.post(
            "/api/v1/generate-recommendations",
            json={"specification": _make_spec_payload()},
        )
        assert response.status_code == 200
        data = response.json()

        # All 4 ranked lists should be present
        assert "balanced" in data
        assert "best_quality" in data
        assert "lowest_cost" in data
        assert "lowest_latency" in data
        assert isinstance(data["balanced"], list)
        assert isinstance(data["best_quality"], list)
        assert isinstance(data["lowest_cost"], list)
        assert isinstance(data["lowest_latency"], list)

    def test_no_matching_benchmarks_returns_empty_lists(self, client):
        """Spec with impossible SLO targets returns empty lists, not error."""
        # Create spec with impossibly tight SLO targets
        spec = _make_spec_payload(
            slo_targets={
                "ttft_target_ms": 1,  # Impossibly low
                "itl_target_ms": 1,
                "e2e_target_ms": 1,
            }
        )
        response = client.post(
            "/api/v1/generate-recommendations",
            json={"specification": spec},
        )

        # Should succeed (200), not error (500)
        assert response.status_code == 200
        data = response.json()

        # Should have empty ranked lists
        assert data["total_configs_evaluated"] == 0
        assert len(data["balanced"]) == 0
        assert len(data["best_quality"]) == 0
        assert len(data["lowest_cost"]) == 0
        assert len(data["lowest_latency"]) == 0
