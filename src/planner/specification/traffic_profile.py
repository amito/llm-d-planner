"""Traffic profile generation from deployment intent."""

import json
import logging
from pathlib import Path

from planner.knowledge_base.slo_templates import SLOTemplateRepository
from planner.shared.schemas import DeploymentIntent, SLORange, SLOTargets, TrafficProfile

logger = logging.getLogger(__name__)

# Percentile values for each latency priority level
LATENCY_PRIORITY_PERCENTILE = {
    "high": 0.25,
    "medium": 0.50,
    "low": 0.75,
}


def _round_to_nearest(value: float, nearest: int = 5) -> int:
    """Round a value to the nearest multiple."""
    return int(round(value / nearest) * nearest)


class TrafficProfileGenerator:
    """Generate traffic profiles and SLO targets from deployment intent."""

    def __init__(
        self,
        slo_repo: SLOTemplateRepository | None = None,
        usecase_data_path: Path | None = None,
    ):
        """
        Initialize traffic profile generator.

        Args:
            slo_repo: SLO template repository (creates default if not provided)
            usecase_data_path: Path to usecase_slo_workload.json
        """
        self.slo_repo = slo_repo or SLOTemplateRepository()
        if usecase_data_path is None:
            usecase_data_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "configuration"
                / "usecase_slo_workload.json"
            )
        self.usecase_data_path = usecase_data_path
        self._usecase_data: dict | None = None

    def _load_usecase_data(self) -> dict:
        """Load use case data from usecase_slo_workload.json (cached)."""
        if self._usecase_data is None:
            with open(self.usecase_data_path) as f:
                data = json.load(f)
                self._usecase_data = data["use_case_slo_workload"]
        return self._usecase_data

    def generate_profile(self, intent: DeploymentIntent) -> TrafficProfile:
        """
        Generate traffic profile from deployment intent.

        Uses traffic profile from SLO templates aligned with GuideLLM configurations.

        Args:
            intent: Deployment intent

        Returns:
            TrafficProfile with exact GuideLLM traffic profile
        """
        template = self.slo_repo.get_template(intent.use_case)
        if not template:
            raise ValueError(f"Unknown use_case: {intent.use_case}")

        expected_qps = self._estimate_qps(
            user_count=intent.user_count,
            use_case=intent.use_case,
        )

        return TrafficProfile(
            prompt_tokens=template.prompt_tokens,
            output_tokens=template.output_tokens,
            expected_qps=expected_qps,
        )

    def generate_slo_targets(self, intent: DeploymentIntent) -> SLOTargets:
        """
        Generate SLO targets from deployment intent using range-percentile defaults.

        Reads SLO ranges from usecase_slo_workload.json and calculates defaults
        based on latency priority (high=25th, medium=50th, low=75th percentile).

        Args:
            intent: Deployment intent

        Returns:
            SLOTargets with p95 target latencies and ranges populated
        """
        usecase_data = self._load_usecase_data()
        use_case_config = usecase_data.get(intent.use_case)
        if not use_case_config:
            raise ValueError(f"Unknown use_case: {intent.use_case}")

        # Get SLO ranges from the data
        slo_targets_data = use_case_config.get("slo_targets", {})
        ttft_range_data = slo_targets_data.get("ttft_ms", {})
        itl_range_data = slo_targets_data.get("itl_ms", {})
        e2e_range_data = slo_targets_data.get("e2e_ms", {})

        # Get percentile for latency priority
        percentile = LATENCY_PRIORITY_PERCENTILE.get(intent.latency_priority, 0.5)

        # Calculate defaults using min + (max - min) * percentile, rounded to nearest 5
        ttft_min = ttft_range_data.get("min", 100)
        ttft_max = ttft_range_data.get("max", 500)
        ttft_target = _round_to_nearest(ttft_min + (ttft_max - ttft_min) * percentile)

        itl_min = itl_range_data.get("min", 15)
        itl_max = itl_range_data.get("max", 50)
        itl_target = _round_to_nearest(itl_min + (itl_max - itl_min) * percentile)

        e2e_min = e2e_range_data.get("min", 5000)
        e2e_max = e2e_range_data.get("max", 25000)
        e2e_target = _round_to_nearest(e2e_min + (e2e_max - e2e_min) * percentile)

        return SLOTargets(
            ttft_target_ms=ttft_target,
            itl_target_ms=itl_target,
            e2e_target_ms=e2e_target,
            ttft_range=SLORange(min=ttft_min, max=ttft_max),
            itl_range=SLORange(min=itl_min, max=itl_max),
            e2e_range=SLORange(min=e2e_min, max=e2e_max),
        )

    def _estimate_qps(self, user_count: int, use_case: str) -> float:
        """
        Estimate peak QPS based on user count and per-use-case workload parameters.

        Args:
            user_count: Number of users
            use_case: Use case identifier

        Returns:
            Estimated peak QPS
        """
        # Load use case data
        usecase_data = self._load_usecase_data()
        use_case_config = usecase_data.get(use_case)

        if not use_case_config:
            raise ValueError(f"Unknown use_case: {use_case}")

        workload = use_case_config.get("workload", {})
        active_fraction = workload.get("active_fraction", 0.2)
        requests_per_active_user_per_min = workload.get("requests_per_active_user_per_min", 0.4)
        peak_multiplier = workload.get("peak_multiplier", 2.0)

        # Formula: expected_rps = (user_count * active_fraction * requests_per_min) / 60
        expected_concurrent = int(user_count * active_fraction)
        expected_rps = (expected_concurrent * requests_per_active_user_per_min) / 60

        # Apply peak multiplier for capacity buffer
        peak_rps = expected_rps * peak_multiplier

        # Ensure minimum QPS of 0.1 for small workloads
        peak_rps = max(0.1, peak_rps)

        return float(round(peak_rps, 2))
