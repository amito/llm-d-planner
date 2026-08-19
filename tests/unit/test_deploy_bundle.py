"""Tests for POST /deploy-bundle-to-cluster endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from planner.api.routes import configuration_router
from planner.configuration import YAMLValidator
from planner.shared.schemas import DeploymentConfiguration, DeploymentIntent, SLOTargets
from planner.shared.schemas.recommendation import (
    DeploymentBundle,
    DeploymentRecommendation,
    GPUConfig,
)
from planner.shared.schemas.specification import TrafficProfile


@pytest.fixture
def mock_cluster_manager():
    """Mock cluster manager that simulates successful deployment."""
    manager = MagicMock()
    manager.namespace = "test-namespace"
    manager.create_namespace_if_not_exists = MagicMock(return_value=None)
    manager.apply_yaml_content = MagicMock(return_value={"success": True})
    return manager


@pytest.fixture
def client(tmp_path, mock_cluster_manager):
    """Create a test client with mocked dependencies."""
    app = FastAPI()

    # Mock app state
    app.state.yaml_validator = YAMLValidator()
    app.state.cluster_managers = {}
    app.state.cluster_manager_lock = MagicMock()

    # Patch get_cluster_manager_or_raise to return our mock
    async def _mock_get_cluster_manager(request, namespace="default"):
        return mock_cluster_manager

    with patch(
        "planner.api.routes.configuration.get_cluster_manager_or_raise",
        side_effect=_mock_get_cluster_manager,
    ):
        app.include_router(configuration_router)
        with TestClient(app) as test_client:
            yield test_client, mock_cluster_manager


def _make_deployment_bundle():
    """Build a minimal valid DeploymentBundle payload."""
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
    bundle = DeploymentBundle(
        deployment_id="test-deployment-123",
        namespace="test-namespace",
        stack="vllm",
        configuration=config,
        files={
            "inferenceservice": """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: test-deployment-123
spec:
  predictor:
    containers:
    - name: kserve-container
      image: vllm/vllm-openai:v0.6.2
""",
            "autoscaling": """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: test-deployment-123
spec:
  scaleTargetRef:
    name: test-deployment-123
""",
        },
    )
    return bundle


@pytest.mark.unit
class TestDeployBundleToCluster:
    def test_accepts_deployment_bundle_and_returns_success(self, client):
        """Endpoint accepts a DeploymentBundle and returns success with files_applied list."""
        test_client, mock_manager = client
        bundle = _make_deployment_bundle()

        response = test_client.post(
            "/api/v1/deploy-bundle-to-cluster",
            json={"bundle": bundle.model_dump()},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deployment_id"] == "test-deployment-123"
        assert data["namespace"] == "test-namespace"
        assert "files_applied" in data
        assert isinstance(data["files_applied"], list)
        assert len(data["files_applied"]) == 2
        assert "inferenceservice" in data["files_applied"]
        assert "autoscaling" in data["files_applied"]

    def test_calls_create_namespace_if_not_exists(self, client):
        """Endpoint calls create_namespace_if_not_exists on the cluster manager."""
        test_client, mock_manager = client
        bundle = _make_deployment_bundle()

        response = test_client.post(
            "/api/v1/deploy-bundle-to-cluster",
            json={"bundle": bundle.model_dump()},
        )

        assert response.status_code == 200
        mock_manager.create_namespace_if_not_exists.assert_called_once()

    def test_validates_yaml_before_applying(self, client):
        """Endpoint rejects invalid YAML with 400 before applying to cluster."""
        test_client, mock_manager = client
        bundle = _make_deployment_bundle()
        bundle.files["inferenceservice"] = "apiVersion: v1\nmetadata: {name: test"

        response = test_client.post(
            "/api/v1/deploy-bundle-to-cluster",
            json={"bundle": bundle.model_dump()},
        )

        assert response.status_code == 400
        mock_manager.apply_yaml_content.assert_not_called()

    def test_returns_error_when_apply_fails(self, client):
        """Endpoint returns error when cluster apply fails."""
        test_client, mock_manager = client
        bundle = _make_deployment_bundle()

        # Mock apply_yaml_content to fail
        mock_manager.apply_yaml_content = MagicMock(
            return_value={"success": False, "error": "kubectl apply failed"}
        )

        response = test_client.post(
            "/api/v1/deploy-bundle-to-cluster",
            json={"bundle": bundle.model_dump()},
        )

        assert response.status_code == 500
        assert "Failed to apply" in response.json()["detail"]
