"""Unit tests for GPU detection integration in RecommendationWorkflow."""

from unittest.mock import patch

import pytest


def _make_specs():
    """Create minimal specifications dict for testing."""
    return {
        "intent": {
            "use_case": "chatbot_conversational",
            "user_count": 100,
            "preferred_gpu_types": [],
        },
        "traffic_profile": {
            "prompt_tokens": 512,
            "output_tokens": 256,
            "expected_qps": 5.0,
        },
        "slo_targets": {
            "ttft_target_ms": 500,
            "itl_target_ms": 50,
            "e2e_target_ms": 15000,
        },
    }


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
        workflow.generate_recommendations(_make_specs())

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
        workflow.generate_recommendations(_make_specs())

        call_kwargs = mock_finder.plan_all_capacities.call_args
        assert call_kwargs.kwargs.get("cluster_gpu_types") == []
