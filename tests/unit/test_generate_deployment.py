"""Tests for POST /generate-deployment endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from planner.api.routes import configuration_router
from planner.configuration import DeploymentGenerator, YAMLValidator
from planner.configuration.llmd_generator import LlmdDeploymentGenerator
from planner.shared.schemas import DeploymentConfiguration, DeploymentIntent, SLOTargets
from planner.shared.schemas.recommendation import DeploymentRecommendation, GPUConfig
from planner.shared.schemas.specification import TrafficProfile


@pytest.fixture
def client(tmp_path):
    """Create a test client with mocked app state (no DB or disk side-effects)."""
    app = FastAPI()

    with patch("planner.configuration.generator.ModelCatalog"):
        app.state.deployment_generator = DeploymentGenerator(
            output_dir=str(tmp_path / "vllm"), simulator_mode=False
        )
    app.state.llmd_deployment_generator = LlmdDeploymentGenerator(output_dir=str(tmp_path / "llmd"))
    app.state.yaml_validator = YAMLValidator()
    app.state.cluster_managers = {}
    app.state.cluster_manager_lock = MagicMock()

    app.include_router(configuration_router)

    return TestClient(app)


def _make_recommendation_payload():
    rec = DeploymentRecommendation(
        intent=DeploymentIntent(use_case="chatbot_conversational", user_count=1000),
        traffic_profile=TrafficProfile(prompt_tokens=512, output_tokens=256, expected_qps=0.87),
        slo_targets=SLOTargets(ttft_target_ms=200, itl_target_ms=25, e2e_target_ms=7000),
        model_id="meta-llama/llama-3.1-8b-instruct",
        model_name="Llama 3.1 8B Instruct",
        model_uri=None,
        gpu_config=GPUConfig(gpu_type="H100", gpu_count=1, tensor_parallel=1, replicas=1),
        meets_slo=True,
        reasoning="Test recommendation",
    )
    return rec.model_dump()


def _make_configuration_payload():
    config = DeploymentConfiguration(
        model_id="meta-llama/llama-3.1-8b-instruct",
        model_name="Llama 3.1 8B Instruct",
        model_uri=None,
        gpu_config=GPUConfig(gpu_type="H100", gpu_count=1, tensor_parallel=1, replicas=1),
        use_case="chatbot_conversational",
        expected_qps=0.87,
        prompt_tokens=512,
        output_tokens=256,
        e2e_target_ms=7000,
    )
    return config.model_dump()


@pytest.mark.unit
class TestGenerateDeployment:
    def test_returns_deployment_bundle(self, client):
        response = client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": _make_configuration_payload(),
                "namespace": "default",
                "stack": "vllm",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "deployment_id" in data
        assert "namespace" in data
        assert "stack" in data
        assert "files" in data
        assert "configuration" in data

    def test_deploy_route_removed(self, client):
        """Old /deploy route should be removed."""
        response = client.post(
            "/api/v1/deploy",
            json={
                "recommendation": _make_recommendation_payload(),
                "namespace": "default",
                "stack": "vllm",
            },
        )
        assert response.status_code in (404, 405)

    def test_llm_d_stack_produces_bundle(self, client):
        """Generate deployment with stack='llm-d' produces a bundle."""
        response = client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": _make_configuration_payload(),
                "namespace": "default",
                "stack": "llm-d",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stack"] == "llm-d"
        assert "deployment_id" in data
        assert "namespace" in data
        assert "files" in data
        assert "configuration" in data
        # llm-d stack should produce different files than vllm
        assert isinstance(data["files"], dict)
        assert len(data["files"]) > 0
