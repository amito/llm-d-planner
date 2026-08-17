"""Workflow orchestration for end-to-end recommendation flow."""

import json
import logging
from pathlib import Path

from planner.cluster.gpu_detector import detect_cluster_gpus
from planner.recommendation.analyzer import Analyzer
from planner.recommendation.config_finder import ConfigFinder
from planner.shared.schemas import RankedRecommendations

logger = logging.getLogger(__name__)


class RecommendationWorkflow:
    """Orchestrate the recommendation workflow."""

    def __init__(
        self,
        config_finder: ConfigFinder | None = None,
    ):
        self.config_finder = config_finder or ConfigFinder()

    def _create_workload_profile(self, traffic_profile, intent):
        """Create WorkloadProfile from TrafficProfile."""
        from planner.shared.schemas import WorkloadProfile

        return WorkloadProfile(
            prompt_tokens=traffic_profile.prompt_tokens,
            output_tokens=traffic_profile.output_tokens,
            expected_qps=traffic_profile.expected_qps or 0.0,
        )

    def _load_quality_weights(self, use_case: str):
        """Load QualityWeights from quality_weights.json for the use case."""
        from planner.shared.schemas import QualityWeights

        repo_root = Path(__file__).parent.parent.parent.parent
        weights_path = repo_root / "data" / "configuration" / "quality_weights.json"

        if not weights_path.is_file():
            logger.warning("Quality weights file not found: %s. Using fallback.", weights_path)
            return QualityWeights(categories={"overall": 10})

        with open(weights_path) as f:
            data = json.load(f)

        # Filter out metadata keys (starting with "_")
        weights_by_use_case = {k: v for k, v in data.items() if not k.startswith("_")}

        if use_case not in weights_by_use_case:
            logger.warning("Use case %r not found in quality weights. Using fallback.", use_case)
            return QualityWeights(categories={"overall": 10})

        categories = weights_by_use_case[use_case].get("categories", {})
        return QualityWeights(categories=categories)

    def _load_priorities(self, intent):
        """Load Priorities from priority_weights.json using intent priority levels."""
        from planner.shared.schemas import Priorities, PriorityEntry

        repo_root = Path(__file__).parent.parent.parent.parent
        weights_path = repo_root / "data" / "configuration" / "priority_weights.json"

        if not weights_path.is_file():
            logger.warning("Priority weights file not found: %s. Using fallback.", weights_path)
            weight_map = {"low": 1, "medium": 4, "high": 8}
        else:
            with open(weights_path) as f:
                data = json.load(f)
            weight_map_data = data.get("priority_weights", {})
            quality_weights = weight_map_data.get("quality", {"low": 2, "medium": 4, "high": 8})
            cost_weights = weight_map_data.get("cost", {"low": 2, "medium": 4, "high": 8})
            latency_weights = weight_map_data.get("latency", {"low": 0, "medium": 1, "high": 2})

            return Priorities(
                quality=PriorityEntry(
                    priority=intent.quality_priority,
                    weight=quality_weights[intent.quality_priority],
                ),
                cost=PriorityEntry(
                    priority=intent.cost_priority,
                    weight=cost_weights[intent.cost_priority],
                ),
                latency=PriorityEntry(
                    priority=intent.latency_priority,
                    weight=latency_weights[intent.latency_priority],
                ),
            )

        # Fallback if file not found
        return Priorities(
            quality=PriorityEntry(
                priority=intent.quality_priority,
                weight=weight_map[intent.quality_priority],
            ),
            cost=PriorityEntry(
                priority=intent.cost_priority,
                weight=weight_map[intent.cost_priority],
            ),
            latency=PriorityEntry(
                priority=intent.latency_priority,
                weight=weight_map[intent.latency_priority],
            ),
        )

    def generate_recommendations(
        self,
        specifications: dict,
        min_quality: float | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
        weights: dict[str, int] | None = None,
        enable_estimated: bool = True,
    ) -> RankedRecommendations:
        """
        Generate ranked recommendation lists from pre-built specifications.

        This bypasses intent extraction and uses the provided specs directly.
        Used when UI has already extracted and potentially edited the specs.

        Args:
            specifications: Dict with keys: intent, traffic_profile, slo_targets
            min_quality: Minimum quality score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            include_near_miss: Whether to include near-SLO configurations
            weights: Optional custom weights for balanced score (0-10 scale)
                     Keys: quality, price, latency

        Returns:
            RankedRecommendations with 4 ranked lists
        """
        from planner.shared.schemas import (
            DeploymentIntent,
            DeploymentSpecification,
            SLOTargets,
            TrafficProfile,
        )

        logger.info("Starting ranked recommendation workflow from specifications")

        intent_data = specifications["intent"]

        # Parse specifications into schema objects
        intent = DeploymentIntent(**intent_data)
        traffic_profile = TrafficProfile(**specifications["traffic_profile"])
        slo_targets = SLOTargets(**specifications["slo_targets"])

        # Create new required fields for DeploymentSpecification
        workload_profile = self._create_workload_profile(traffic_profile, intent)
        quality_weights = self._load_quality_weights(intent.use_case)
        priorities = self._load_priorities(intent)

        specification = DeploymentSpecification(
            intent=intent,
            slo_targets=slo_targets,
            workload_profile=workload_profile,
            quality_weights=quality_weights,
            priorities=priorities,
        )

        logger.info(
            f"Specs: {intent.use_case}, {intent.user_count} users, "
            f"{traffic_profile.expected_qps} QPS, "
            f"TTFT target={slo_targets.ttft_target_ms}ms (p95)"
        )

        # Get ALL configurations with scores
        # Detect cluster GPUs for filtering
        cluster_gpu_types = detect_cluster_gpus()
        logger.info("Planning capacity for all model/GPU combinations")
        logger.info(f"Using weights for balanced scoring: {weights}")
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
                specification=specification,
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
            specification=specification,
            best_quality=ranked_lists["best_quality"],
            lowest_cost=ranked_lists["lowest_cost"],
            lowest_latency=ranked_lists["lowest_latency"],
            balanced=ranked_lists["balanced"],
            total_configs_evaluated=len(all_configs),
            configs_after_filters=configs_after_filters,
            warnings=estimation_warnings,
        )
