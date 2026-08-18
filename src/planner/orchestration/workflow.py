"""Workflow orchestration for end-to-end recommendation flow."""

import logging

from planner.recommendation.analyzer import Analyzer
from planner.recommendation.config_finder import ConfigFinder
from planner.shared.schemas import DeploymentSpecification, RankedRecommendations
from planner.specification.service import SpecificationService

logger = logging.getLogger(__name__)

# Optional cluster GPU detection (requires kubernetes package)
try:
    from planner.cluster.gpu_detector import detect_cluster_gpus

    _HAS_GPU_DETECTION = True
except ImportError:
    _HAS_GPU_DETECTION = False

    def detect_cluster_gpus() -> list[str]:
        """Fallback when kubernetes package is not installed."""
        return []


class RecommendationWorkflow:
    """Orchestrate the recommendation workflow."""

    def __init__(
        self,
        config_finder: ConfigFinder | None = None,
        spec_service: SpecificationService | None = None,
    ):
        self.config_finder = config_finder or ConfigFinder()
        self.spec_service = spec_service or SpecificationService()

    def generate_recommendations(
        self,
        spec: DeploymentSpecification,
        min_quality: float | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
        weights: dict[str, int] | None = None,
        enable_estimated: bool = True,
    ) -> RankedRecommendations:
        """
        Generate ranked recommendation lists from a DeploymentSpecification.

        This bypasses intent extraction and uses the provided spec directly.
        Used when UI has already extracted and potentially edited the spec.

        Args:
            spec: DeploymentSpecification with intent, slo_targets, workload_profile
            min_quality: Minimum quality score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            include_near_miss: Whether to include near-SLO configurations
            weights: Optional custom weights for balanced score (0-10 scale)
                     Keys: quality, price, latency
            enable_estimated: Whether to include estimated performance data

        Returns:
            RankedRecommendations with 4 ranked lists
        """
        logger.info("Starting ranked recommendation workflow from specification")

        intent = spec.intent
        slo_targets = spec.slo_targets
        workload_profile = spec.workload_profile

        logger.info(
            f"Specs: {intent.use_case}, {intent.user_count} users, "
            f"{workload_profile.expected_qps} QPS, "
            f"TTFT target={slo_targets.ttft_target_ms}ms (p95)"
        )

        # Get ALL configurations with scores
        # Detect cluster GPUs for filtering (optional - requires kubernetes package)
        cluster_gpu_types = detect_cluster_gpus()
        logger.info("Planning capacity for all model/GPU combinations")
        logger.info(f"Using weights for balanced scoring: {weights}")

        # Convert workload_profile to traffic_profile format for config_finder
        from planner.shared.schemas import TrafficProfile

        traffic_profile = TrafficProfile(
            prompt_tokens=workload_profile.prompt_tokens,
            output_tokens=workload_profile.output_tokens,
            expected_qps=workload_profile.expected_qps,
        )

        all_configs, estimation_warnings = self.config_finder.plan_all_capacities(
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
            intent=intent,
            include_near_miss=include_near_miss,
            weights=weights,
            cluster_gpu_types=cluster_gpu_types,
            preferred_models=intent.preferred_models if intent.preferred_models else None,
            enable_estimated=enable_estimated,
        )

        if not all_configs:
            logger.warning("No viable configurations found")
            return RankedRecommendations(
                min_quality_threshold=min_quality,
                max_cost_ceiling=max_cost,
                include_near_miss=include_near_miss,
                specification=spec,
                total_configs_evaluated=0,
                configs_after_filters=0,
                warnings=estimation_warnings,
            )

        # Generate ranked lists (top 10 solutions per criterion)
        # Pass use_case for task-specific bonuses on Balanced card
        analyzer = Analyzer()
        ranked_lists = analyzer.generate_ranked_lists(
            configurations=all_configs,
            min_quality=min_quality,
            max_cost=max_cost,
            top_n=10,  # Top 10 quality models only
            weights=weights,
            use_case=intent.use_case,  # Task bonuses for Balanced
            preferred_models=intent.preferred_models if intent.preferred_models else None,
        )

        # Count configs after filtering
        configs_after_filters = analyzer.get_unique_configs_count(ranked_lists)

        logger.info(
            f"Generated ranked recommendations from spec: {len(all_configs)} total configs, "
            f"{configs_after_filters} after filters"
        )

        return RankedRecommendations(
            min_quality_threshold=min_quality,
            max_cost_ceiling=max_cost,
            include_near_miss=include_near_miss,
            specification=spec,
            best_quality=ranked_lists["best_quality"],
            lowest_cost=ranked_lists["lowest_cost"],
            lowest_latency=ranked_lists["lowest_latency"],
            balanced=ranked_lists["balanced"],
            total_configs_evaluated=len(all_configs),
            configs_after_filters=configs_after_filters,
            warnings=estimation_warnings,
        )
