"""E2E pipeline test via REST API: extract_intent → deploy_bundle_to_cluster.

Uses mock LLM and canned fixture data for deterministic results.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app
from planner.cluster import KubernetesClusterManager
from planner.shared.schemas import DeploymentSpecification
from planner.shared.schemas.recommendation import DeploymentBundle


@pytest.fixture
def e2e_client(test_db_path, mock_scoring_engine, mock_llm_client, monkeypatch):
    """Create a test client with mocked LLM, canned benchmarks, and canned quality data."""
    monkeypatch.setenv("PLANNER_DB_PATH", test_db_path)
    monkeypatch.setenv("PLANNER_DETECT_CLUSTER_GPUS", "false")

    with (
        patch(
            "planner.recommendation.quality.scoring.build_scoring_engine",
            side_effect=mock_scoring_engine,
        ),
        patch("planner.api.routes.intent.create_llm_client", return_value=mock_llm_client),
    ):
        app = create_app()
        with TestClient(app) as client:
            yield client


@pytest.mark.unit
class TestE2EPipelineAPI:
    """Full pipeline via REST API endpoints."""

    def test_full_pipeline_extract_to_deploy(self, e2e_client, canned_intent):
        """Test complete pipeline: extract_intent → generate_spec → recommendations → deployment → deploy."""

        # Stage 1: Extract intent (uses MockLLMClient)
        intent_response = e2e_client.post(
            "/api/v1/extract-intent",
            json={"text": "I need a chatbot for 1000 users, high quality is critical"},
        )
        assert intent_response.status_code == 200
        intent_data = intent_response.json()
        assert intent_data["use_case"] == "chatbot_conversational"
        assert intent_data["user_count"] == 1000

        # Stage 2: Generate specification
        spec_response = e2e_client.post(
            "/api/v1/generate-specification",
            json=intent_data,
        )
        assert spec_response.status_code == 200
        spec_data = spec_response.json()
        spec = DeploymentSpecification(**spec_data)
        assert spec.slo_targets.ttft_target_ms > 0
        assert spec.workload_profile.expected_qps > 0
        assert spec.quality_weights is not None

        # Stage 3: Generate recommendations
        rec_response = e2e_client.post(
            "/api/v1/generate-recommendations",
            json={"specification": spec_data, "enable_estimated": True},
        )
        assert rec_response.status_code == 200
        rec_data = rec_response.json()
        assert rec_data["total_configs_evaluated"] > 0
        assert len(rec_data["balanced"]) > 0

        top_rec = rec_data["balanced"][0]
        configuration = top_rec["configuration"]

        # Stage 4: Generate deployment bundle
        deploy_response = e2e_client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": configuration,
                "namespace": "test-ns",
                "stack": "vllm",
            },
        )
        assert deploy_response.status_code == 200
        bundle_data = deploy_response.json()
        bundle = DeploymentBundle(**bundle_data)
        assert "inferenceservice" in bundle.files
        assert "autoscaling" in bundle.files

        # Stage 5: Deploy to cluster (mocked K8s)
        mock_manager = MagicMock(spec=KubernetesClusterManager)
        mock_manager.create_namespace_if_not_exists.return_value = None
        mock_manager.apply_yaml_content.return_value = {"success": True}

        with patch(
            "planner.api.routes.configuration.get_cluster_manager_or_raise",
            return_value=mock_manager,
        ):
            cluster_response = e2e_client.post(
                "/api/v1/deploy-bundle-to-cluster",
                json={"bundle": bundle_data},
            )
            assert cluster_response.status_code == 200
            result = cluster_response.json()
            assert result["success"] is True
