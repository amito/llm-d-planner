"""Tests for range-percentile SLO default generation."""

from typing import Literal

import pytest

from planner.shared.schemas import DeploymentIntent, SLOTargets
from planner.specification.traffic_profile import TrafficProfileGenerator


def _make_intent(
    latency_priority: Literal["low", "medium", "high"] = "medium",
) -> DeploymentIntent:
    return DeploymentIntent(
        use_case="chatbot_conversational",
        user_count=1000,
        latency_priority=latency_priority,
    )


@pytest.mark.unit
class TestSLORangePercentile:
    def test_medium_uses_50th_percentile(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("medium"))
        # chatbot_conversational TTFT range: 100-500
        # 50th percentile: 100 + (500-100)*0.5 = 300, rounded to nearest 5 = 300
        assert slo.ttft_target_ms == 300

    def test_high_uses_25th_percentile(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("high"))
        # 25th percentile: 100 + (500-100)*0.25 = 200
        assert slo.ttft_target_ms == 200

    def test_low_uses_75th_percentile(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("low"))
        # 75th percentile: 100 + (500-100)*0.75 = 400
        assert slo.ttft_target_ms == 400

    def test_ranges_populated(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("medium"))
        assert slo.ttft_range is not None
        assert slo.ttft_range.min == 100
        assert slo.ttft_range.max == 500
        assert slo.itl_range is not None
        assert slo.e2e_range is not None

    def test_all_metrics_use_same_percentile(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("high"))
        # ITL range for chatbot: 15-50
        # 25th: 15 + (50-15)*0.25 = 23.75 -> round to 25
        assert slo.itl_target_ms == 25
        # E2E range for chatbot: 3940-13300
        # 25th: 3940 + (13300-3940)*0.25 = 6280
        assert slo.e2e_target_ms == 6280

    def test_code_completion_high_priority(self):
        intent = DeploymentIntent(
            use_case="code_completion", user_count=500, latency_priority="high"
        )
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(intent)
        # TTFT range: 50-200, 25th: 50 + 150*0.25 = 87.5 -> 90
        assert slo.ttft_target_ms == 90
