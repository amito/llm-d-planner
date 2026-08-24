"""E2E pipeline test via Python Library: extract_intent → deploy_bundle_to_cluster.

Uses mock LLM and canned fixture data for deterministic results.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from planner import Planner
from planner.cluster.manager import KubernetesClusterManager
from planner.shared.schemas import DeploymentIntent

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def e2e_planner(mock_scoring_engine, mock_llm_client):
    """Create a Planner instance with canned data and mocked LLM."""
    with patch(
        "planner.planner.build_scoring_engine",
        side_effect=mock_scoring_engine,
    ):
        p = Planner(llm_provider="ollama")

    # Load canned benchmark JSON (not the SQLite test_db_path)
    p.load_benchmarks(str(FIXTURES_DIR / "test_benchmarks.json"))

    # Inject mock LLM client after initialization
    from planner.intent_extraction import IntentExtractor

    p._llm_client = mock_llm_client
    p._extractor = IntentExtractor(mock_llm_client)

    yield p


@pytest.mark.unit
class TestE2EPipelineLibrary:
    """Full pipeline via Planner class methods."""

    def test_full_pipeline_extract_to_deploy(self, e2e_planner):
        """Test complete pipeline: extract_intent → generate_spec → recommendations → deployment → deploy."""
        p = e2e_planner

        # Stage 1: Extract intent (uses MockLLMClient)
        intent = p.extract_intent("I need a chatbot for 1000 users, high quality is critical")
        assert isinstance(intent, DeploymentIntent)
        assert intent.use_case == "chatbot_conversational"
        assert intent.user_count == 1000

        # Stage 2: Generate specification
        spec = p.generate_specification(intent)
        assert spec.slo_targets.ttft_target_ms > 0
        assert spec.workload_profile.expected_qps > 0
        assert spec.quality_weights is not None

        # Stage 3: Generate recommendations
        recs = p.generate_recommendations(spec, enable_estimated=True)
        assert recs.total_configs_evaluated > 0
        assert len(recs.balanced) > 0

        top_rec = recs.balanced[0]

        # Stage 4: Generate deployment bundle
        bundle = p.generate_deployment(
            config=top_rec.configuration,
            namespace="test-ns",
            stack="vllm",
        )
        assert "inferenceservice" in bundle.files
        assert "autoscaling" in bundle.files

        # Stage 5: Deploy to cluster (mocked)
        mock_manager = MagicMock(spec=KubernetesClusterManager)
        mock_manager.create_namespace_if_not_exists.return_value = None
        mock_manager.apply_yaml_content.return_value = {"success": True}

        with patch(
            "planner.cluster.manager.KubernetesClusterManager",
            return_value=mock_manager,
        ):
            result = p.deploy_bundle_to_cluster(bundle)
            assert result["success"] is True

    def test_pipeline_with_direct_intent(self, e2e_planner):
        """Test pipeline starting from a manually constructed intent (no LLM)."""
        p = e2e_planner

        # Skip stage 1 — construct intent directly
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=500,
            quality_priority="medium",
            cost_priority="medium",
            latency_priority="medium",
        )

        # Stages 2-4
        spec = p.generate_specification(intent)
        recs = p.generate_recommendations(spec, enable_estimated=True)
        assert recs.total_configs_evaluated > 0

        assert recs.balanced, "expected at least one ranked config"

        bundle = p.generate_deployment(
            config=recs.balanced[0].configuration,
            namespace="test-ns",
            stack="vllm",
        )
        assert len(bundle.files) >= 2
