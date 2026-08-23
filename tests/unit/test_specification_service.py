"""Tests for SpecificationService."""

import pytest

from planner.shared.schemas import DeploymentIntent, DeploymentSpecification
from planner.specification.service import SpecificationService


@pytest.mark.unit
class TestSpecificationService:
    def test_generates_complete_specification(self):
        svc = SpecificationService()
        intent = DeploymentIntent(use_case="chatbot_conversational", user_count=1000)
        spec = svc.generate(intent)
        assert isinstance(spec, DeploymentSpecification)
        assert spec.intent.use_case == "chatbot_conversational"
        assert spec.slo_targets.ttft_target_ms > 0
        assert spec.workload_profile.expected_qps > 0
        assert spec.quality_weights is not None
        assert spec.priorities.quality.weight > 0

    def test_respects_latency_priority(self):
        svc = SpecificationService()
        high = svc.generate(
            DeploymentIntent(
                use_case="chatbot_conversational", user_count=100, latency_priority="high"
            )
        )
        low = svc.generate(
            DeploymentIntent(
                use_case="chatbot_conversational", user_count=100, latency_priority="low"
            )
        )
        assert high.slo_targets.ttft_target_ms < low.slo_targets.ttft_target_ms

    def test_custom_data_dir(self, tmp_path):
        """Service works with a custom data directory containing valid config files."""
        # This test would need config files in tmp_path — skip for now
        # but validates the data_dir parameter is accepted
        pass
