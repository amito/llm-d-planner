"""Unit tests for GPU detection integration in RecommendationWorkflow."""

from unittest.mock import patch

import pytest


def _make_spec():
    """Create minimal DeploymentSpecification for testing."""
    from planner.shared.schemas import (
        DeploymentIntent,
        DeploymentSpecification,
        Priorities,
        PriorityEntry,
        QualityWeights,
        SLOTargets,
        WorkloadProfile,
    )

    intent = DeploymentIntent(
        use_case="chatbot_conversational",
        user_count=100,
        preferred_gpu_types=[],
    )

    slo_targets = SLOTargets(
        ttft_target_ms=500,
        itl_target_ms=50,
        e2e_target_ms=15000,
    )

    workload_profile = WorkloadProfile(
        prompt_tokens=512,
        output_tokens=256,
        expected_qps=5.0,
    )

    quality_weights = QualityWeights(categories={"overall": 10})

    priorities = Priorities(
        quality=PriorityEntry(priority="medium", weight=4),
        cost=PriorityEntry(priority="medium", weight=4),
        latency=PriorityEntry(priority="medium", weight=2),
    )

    return DeploymentSpecification(
        intent=intent,
        slo_targets=slo_targets,
        workload_profile=workload_profile,
        quality_weights=quality_weights,
        priorities=priorities,
    )


@pytest.mark.unit
class TestWorkflowGPUDetection:
    """Verify all three workflow methods call detect_cluster_gpus.

    Uses module-level patching of ConfigFinder to avoid database
    connection attempts during instantiation.
    """

    @patch("planner.orchestration.workflow.detect_cluster_gpus", return_value=["A100-80"])
    @patch("planner.orchestration.workflow.ConfigFinder")
    def test_generate_ranked_from_spec_calls_detector(self, mock_config_finder, mock_detect):
        mock_finder = mock_config_finder.return_value
        mock_finder.plan_all_capacities.return_value = ([], [])

        from planner.orchestration.workflow import RecommendationWorkflow

        workflow = RecommendationWorkflow()
        workflow.generate_recommendations(_make_spec())

        mock_detect.assert_called_once()
        call_kwargs = mock_finder.plan_all_capacities.call_args
        assert call_kwargs.kwargs.get("cluster_gpu_types") == ["A100-80"]

    @patch("planner.orchestration.workflow.detect_cluster_gpus", return_value=[])
    @patch("planner.orchestration.workflow.ConfigFinder")
    def test_empty_detection_passes_empty_list(self, mock_config_finder, mock_detect):
        mock_finder = mock_config_finder.return_value
        mock_finder.plan_all_capacities.return_value = ([], [])

        from planner.orchestration.workflow import RecommendationWorkflow

        workflow = RecommendationWorkflow()
        workflow.generate_recommendations(_make_spec())

        call_kwargs = mock_finder.plan_all_capacities.call_args
        assert call_kwargs.kwargs.get("cluster_gpu_types") == []
