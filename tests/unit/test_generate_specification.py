"""Tests for POST /generate-specification endpoint."""

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.unit
class TestGenerateSpecification:
    def test_basic_chatbot(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "chatbot_conversational", "user_count": 1000},
        )
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data
        assert "slo_targets" in data
        assert "workload_profile" in data
        assert "quality_weights" in data
        assert "priorities" in data

    def test_intent_echoed_with_defaults(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "chatbot_conversational", "user_count": 1000},
        )
        data = response.json()
        intent = data["intent"]
        assert intent["use_case"] == "chatbot_conversational"
        assert intent["user_count"] == 1000
        assert intent["quality_priority"] == "medium"
        assert intent["cost_priority"] == "medium"
        assert intent["latency_priority"] == "medium"
        assert intent["domain_specialization"] == ["general"]

    def test_slo_defaults_match_latency_priority(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={
                "use_case": "chatbot_conversational",
                "user_count": 1000,
                "latency_priority": "high",
            },
        )
        data = response.json()
        slo = data["slo_targets"]
        # High priority = 25th percentile of range
        # TTFT range 100-500: 25th = 200
        assert slo["ttft_target_ms"] == 200
        assert slo["ttft_range"]["min"] == 100
        assert slo["ttft_range"]["max"] == 500

    def test_workload_profile_from_template(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "chatbot_conversational", "user_count": 1000},
        )
        data = response.json()
        wp = data["workload_profile"]
        assert wp["prompt_tokens"] == 512
        assert wp["output_tokens"] == 256
        assert wp["expected_qps"] > 0

    def test_quality_weights_from_config(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "chatbot_conversational", "user_count": 1000},
        )
        data = response.json()
        qw = data["quality_weights"]["categories"]
        assert "overall" in qw
        assert qw["overall"] == 4

    def test_priorities_from_intent(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={
                "use_case": "chatbot_conversational",
                "user_count": 1000,
                "quality_priority": "high",
                "cost_priority": "low",
            },
        )
        data = response.json()
        p = data["priorities"]
        assert p["quality"]["priority"] == "high"
        assert p["quality"]["weight"] == 8
        assert p["cost"]["priority"] == "low"
        assert p["cost"]["weight"] == 2

    def test_unknown_use_case_returns_error(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "nonexistent", "user_count": 1000},
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "chatbot_conversational"},
        )
        assert response.status_code == 422

    def test_gpu_preferences_pass_through(self, client):
        response = client.post(
            "/api/v1/generate-specification",
            json={
                "use_case": "chatbot_conversational",
                "user_count": 1000,
                "preferred_gpu_types": ["H100", {"gpu_type": "L4", "max_count": 2}],
            },
        )
        data = response.json()
        gpus = data["intent"]["preferred_gpu_types"]
        assert len(gpus) == 2

    def test_output_is_valid_deployment_specification(self, client):
        """Output can be deserialized back into DeploymentSpecification."""
        from planner.shared.schemas.specification import DeploymentSpecification

        response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "code_completion", "user_count": 500},
        )
        data = response.json()
        spec = DeploymentSpecification(**data)
        assert spec.intent.use_case == "code_completion"
