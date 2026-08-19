"""Pipeline integration tests: output of each stage feeds to the next."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app
from planner.configuration import DeploymentGenerator, LlmdDeploymentGenerator, YAMLValidator
from planner.shared.schemas.recommendation import DeploymentBundle
from planner.shared.schemas.specification import DeploymentSpecification


@pytest.fixture
def client(test_db_path, test_quality_data):
    """Create a test client with test database and quality data."""
    os.environ["PLANNER_DB_PATH"] = test_db_path

    # Patch build_scoring_engine to use test quality data instead of production files
    def _mock_build_scoring_engine(cache_dir=None, auto_update=None):
        from quality_scoring.engine import ScoringEngine

        engine = ScoringEngine(
            arena_rows=test_quality_data["arena_rows"],
            aa_models=test_quality_data["aa_models"],
        )
        metadata = {
            "arena_count": len(test_quality_data["arena_rows"]),
            "arena_fetched": "2026-08-14T00:00:00Z",
            "aa_count": len(test_quality_data["aa_models"]),
            "aa_fetched": "2026-08-14T00:00:00Z",
        }
        return engine, metadata

    with patch(
        "planner.recommendation.quality.scoring.build_scoring_engine",
        side_effect=_mock_build_scoring_engine,
    ):
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client

    os.environ.pop("PLANNER_DB_PATH", None)


@pytest.mark.unit
class TestPipelineIntegration:
    def test_generate_specification_output_feeds_to_recommendations(self, client):
        """generate-specification output is valid input for generate-recommendations."""
        # Step 1: Generate specification
        spec_response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "chatbot_conversational", "user_count": 1000},
        )
        assert spec_response.status_code == 200
        spec_data = spec_response.json()

        # Verify it's a valid DeploymentSpecification
        spec = DeploymentSpecification(**spec_data)
        assert spec.intent.use_case == "chatbot_conversational"

        # Step 2: Feed directly to generate-recommendations
        rec_response = client.post(
            "/api/v1/generate-recommendations",
            json={"specification": spec_data},
        )
        assert rec_response.status_code == 200
        rec_data = rec_response.json()
        assert "balanced" in rec_data
        assert "total_configs_evaluated" in rec_data

    def test_all_use_cases_produce_valid_specifications(self, client):
        """Every known use case produces a valid specification."""
        use_cases = [
            "chatbot_conversational",
            "code_completion",
            "code_generation_detailed",
            "translation",
            "content_generation",
            "summarization_short",
            "document_analysis_rag",
            "long_document_summarization",
            "research_legal_analysis",
        ]
        for use_case in use_cases:
            response = client.post(
                "/api/v1/generate-specification",
                json={"use_case": use_case, "user_count": 100},
            )
            assert response.status_code == 200, f"Failed for {use_case}"
            spec = DeploymentSpecification(**response.json())
            assert spec.intent.use_case == use_case
            assert spec.slo_targets.ttft_target_ms > 0
            assert spec.workload_profile.prompt_tokens > 0
            assert spec.quality_weights is not None
            assert len(spec.quality_weights.categories) > 0
            assert spec.priorities.quality.weight > 0

    def test_recommendations_to_deployment_pipeline(self, client):
        """Chain recommendations → deployment: pick a recommendation, generate bundle."""
        # Step 1: Generate specification
        spec_response = client.post(
            "/api/v1/generate-specification",
            json={"use_case": "chatbot_conversational", "user_count": 1000},
        )
        assert spec_response.status_code == 200
        spec_data = spec_response.json()

        # Step 2: Get recommendations
        rec_response = client.post(
            "/api/v1/generate-recommendations",
            json={"specification": spec_data},
        )
        assert rec_response.status_code == 200
        rec_data = rec_response.json()

        # If we have recommendations, test the deployment generation
        if len(rec_data["balanced"]) > 0:
            # Step 3: Pick the top balanced recommendation
            top_recommendation = rec_data["balanced"][0]

            # Step 4: Generate deployment bundle
            # Patch deployment generators to avoid disk writes
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                mock_vllm_gen = DeploymentGenerator(
                    output_dir=str(tmp_path / "vllm"), simulator_mode=False
                )
                mock_llmd_gen = LlmdDeploymentGenerator(output_dir=str(tmp_path / "llmd"))

                with (
                    patch(
                        "planner.api.routes.configuration.get_deployment_generator",
                        return_value=mock_vllm_gen,
                    ),
                    patch(
                        "planner.api.routes.configuration.get_llmd_deployment_generator",
                        return_value=mock_llmd_gen,
                    ),
                    patch(
                        "planner.api.routes.configuration.get_yaml_validator",
                        return_value=YAMLValidator(),
                    ),
                    patch("planner.configuration.generator.ModelCatalog"),
                ):
                    deploy_response = client.post(
                        "/api/v1/generate-deployment",
                        json={
                            "configuration": top_recommendation["configuration"],
                            "namespace": "default",
                            "stack": "vllm",
                        },
                    )

                    assert deploy_response.status_code == 200
                    bundle_data = deploy_response.json()

                    # Verify it's a valid DeploymentBundle
                    bundle = DeploymentBundle(**bundle_data)
                    assert bundle.deployment_id is not None
                    assert bundle.namespace == "default"
                    assert bundle.stack == "vllm"
                    assert len(bundle.files) > 0
                    # Verify the configuration was preserved
                    assert bundle.configuration.model_name == top_recommendation["model_name"]
