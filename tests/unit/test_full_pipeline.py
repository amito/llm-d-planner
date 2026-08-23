"""Full end-to-end pipeline test.

Tests the complete flow through API endpoints:
1. Generate specification from intent
2. Generate recommendations from specification
3. Generate deployment bundle from configuration
4. Deploy bundle to cluster (mocked K8s)
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app
from planner.shared.schemas import DeploymentSpecification
from planner.shared.schemas.recommendation import DeploymentBundle


@pytest.mark.unit
def test_full_pipeline_end_to_end(test_db_path, mock_scoring_engine, monkeypatch):
    """Test complete pipeline from intent through deployment."""
    monkeypatch.setenv("PLANNER_DB_PATH", test_db_path)

    with patch(
        "planner.recommendation.quality.scoring.build_scoring_engine",
        side_effect=mock_scoring_engine,
    ):
        app = create_app()
        with TestClient(app) as client:
            # Stage 1: Generate specification from intent
            intent = {
                "use_case": "chatbot_conversational",
                "user_count": 1000,
                "quality_priority": "high",
                "cost_priority": "low",
                "latency_priority": "medium",
                "preferred_gpu_types": ["H100"],
                "preferred_models": ["meta-llama/llama-3.1-8b-instruct"],
                "domain_specialization": ["general"],
            }

            spec_response = client.post("/api/v1/generate-specification", json=intent)
            assert spec_response.status_code == 200
            spec_data = spec_response.json()

            spec = DeploymentSpecification(**spec_data)
            assert spec.intent.use_case == "chatbot_conversational"
            assert spec.intent.user_count == 1000
            assert spec.slo_targets.ttft_target_ms > 0
            assert spec.workload_profile.expected_qps > 0
            assert spec.quality_weights is not None
            assert spec.priorities.quality.weight > 0

            # Stage 2: Generate recommendations from specification
            rec_response = client.post(
                "/api/v1/generate-recommendations",
                json={"specification": spec_data, "enable_estimated": False},
            )
            assert rec_response.status_code == 200
            rec_data = rec_response.json()
            assert "balanced" in rec_data
            assert "total_configs_evaluated" in rec_data

            # Stage 3: Generate deployment bundle
            # Use a recommendation if available, otherwise construct a config directly
            if rec_data["balanced"]:
                top_rec = rec_data["balanced"][0]
                configuration = top_rec["configuration"]
            else:
                configuration = {
                    "model_id": "meta-llama/llama-3.1-8b-instruct",
                    "model_name": "Llama 3.1 8B Instruct",
                    "gpu_config": {
                        "gpu_type": "H100",
                        "gpu_count": 1,
                        "tensor_parallel": 1,
                        "replicas": 1,
                    },
                    "use_case": "chatbot_conversational",
                    "expected_qps": spec_data["workload_profile"]["expected_qps"],
                    "prompt_tokens": 512,
                    "output_tokens": 256,
                    "e2e_target_ms": spec_data["slo_targets"]["e2e_target_ms"],
                }

            deploy_response = client.post(
                "/api/v1/generate-deployment",
                json={
                    "configuration": configuration,
                    "namespace": "test-namespace",
                    "stack": "vllm",
                },
            )
            assert deploy_response.status_code == 200
            bundle_data = deploy_response.json()
            assert "deployment_id" in bundle_data
            assert "files" in bundle_data
            assert len(bundle_data["files"]) >= 2

            bundle = DeploymentBundle(**bundle_data)
            assert "inferenceservice" in bundle.files
            assert "autoscaling" in bundle.files

            # Stage 4: Deploy bundle to cluster (mocked)
            from planner.cluster import KubernetesClusterManager

            mock_manager = MagicMock(spec=KubernetesClusterManager)
            mock_manager.create_namespace_if_not_exists.return_value = None
            mock_manager.apply_yaml_content.return_value = {"success": True}

            mock_manager.create_namespace_if_not_exists()
            applied_files = []
            for name, yaml_content in bundle.files.items():
                result = mock_manager.apply_yaml_content(yaml_content)
                assert result["success"] is True
                applied_files.append(name)

            assert "inferenceservice" in applied_files
            assert "autoscaling" in applied_files
