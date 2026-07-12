"""Ranking service for multi-criteria recommendation sorting.

Each card (Best Accuracy, Lowest Cost, Best Latency, Balanced)
ranks ALL filtered configurations independently by its own criterion.
"""

import logging

from planner.shared.schemas import DeploymentRecommendation

logger = logging.getLogger(__name__)


class Analyzer:
    """Generate ranked recommendation lists from scored configurations."""

    def generate_ranked_lists(
        self,
        configurations: list[DeploymentRecommendation],
        min_quality: float | None = None,
        max_cost: float | None = None,
        top_n: int = 5,
        weights: dict[str, int] | None = None,
        use_case: str | None = None,
        preferred_models: list[str] | None = None,
    ) -> dict[str, list[DeploymentRecommendation]]:
        """
        Generate 4 ranked lists, each sorted independently by its criterion.

        Args:
            configurations: List of scored DeploymentRecommendations
            min_quality: Minimum quality score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            top_n: Number of top configurations to return per list
            weights: Optional custom weights for balanced score (0-10 scale)
                     Keys: quality, price, latency
            use_case: Use case identifier (unused, kept for API compatibility)
            preferred_models: User-specified models that bypass min_quality filter

        Returns:
            Dict with keys: best_quality, lowest_cost, lowest_latency, balanced
        """
        filtered = self._apply_filters(configurations, min_quality, max_cost, preferred_models)

        if not filtered:
            logger.warning("No configurations remain after filtering")
            return {
                "best_quality": [],
                "lowest_cost": [],
                "lowest_latency": [],
                "balanced": [],
            }

        def get_quality(x):
            return x.scores.quality_score if x.scores else 0

        def get_cost_inverted(x):
            cost = x.cost_per_month_usd or float("inf")
            return -cost

        def get_latency(x):
            return x.scores.latency_score if x.scores else 0

        def get_balanced(x):
            return x.scores.balanced_score if x.scores else 0.0

        def deduplicate_by_model(
            configs: list[DeploymentRecommendation], n: int
        ) -> list[DeploymentRecommendation]:
            seen: set[str] = set()
            result: list[DeploymentRecommendation] = []
            for config in configs:
                model = config.model_id or config.model_name or "Unknown"
                if model not in seen:
                    seen.add(model)
                    result.append(config)
                    if len(result) >= n:
                        break
            return result

        sorted_by_quality = sorted(
            filtered,
            key=lambda x: (get_quality(x), get_cost_inverted(x), get_latency(x)),
            reverse=True,
        )
        sorted_by_cost = sorted(
            filtered,
            key=lambda x: (get_cost_inverted(x), get_quality(x), get_latency(x)),
            reverse=True,
        )
        sorted_by_balanced = sorted(
            filtered,
            key=lambda x: (get_balanced(x), get_quality(x), get_cost_inverted(x)),
            reverse=True,
        )

        ranked_lists = {
            "best_quality": deduplicate_by_model(sorted_by_quality, top_n),
            "lowest_cost": deduplicate_by_model(sorted_by_cost, top_n),
            "lowest_latency": deduplicate_by_model(
                sorted(
                    filtered,
                    key=lambda x: (get_latency(x), get_quality(x), get_cost_inverted(x)),
                    reverse=True,
                ),
                top_n,
            ),
            "balanced": deduplicate_by_model(sorted_by_balanced, top_n),
        }

        logger.info(
            f"Generated ranked lists: {len(filtered)} filtered configs, top {top_n} per criterion"
        )

        return ranked_lists

    def _apply_filters(
        self,
        configs: list[DeploymentRecommendation],
        min_quality: float | None,
        max_cost: float | None,
        preferred_models: list[str] | None = None,
    ) -> list[DeploymentRecommendation]:
        """
        Apply quality and cost filters to configurations.

        Args:
            configs: List of configurations to filter
            min_quality: Minimum quality score (0-100), None = no filter
            max_cost: Maximum monthly cost (USD), None = no filter
            preferred_models: User-specified models that bypass min_quality filter

        Returns:
            Filtered list of configurations
        """
        filtered = configs

        # Filter by minimum quality — exempt user-specified preferred models
        if min_quality is not None and min_quality > 0:
            preferred_set = {m.lower() for m in preferred_models} if preferred_models else set()
            filtered = [
                c
                for c in filtered
                if (c.scores and c.scores.quality_score >= min_quality)
                or (c.model_id and c.model_id.lower() in preferred_set)
            ]
            logger.debug(f"After min_quality={min_quality} filter: {len(filtered)} configs")

        # Filter by maximum cost
        if max_cost is not None and max_cost > 0:
            filtered = [
                c
                for c in filtered
                if c.cost_per_month_usd is not None and c.cost_per_month_usd <= max_cost
            ]
            logger.debug(f"After max_cost=${max_cost} filter: {len(filtered)} configs")

        return filtered

    def get_unique_configs_count(
        self, ranked_lists: dict[str, list[DeploymentRecommendation]]
    ) -> int:
        """
        Count unique configurations across all ranked lists.

        Since the same configuration may appear in multiple lists
        (e.g., best quality AND lowest cost), this counts unique ones.

        Args:
            ranked_lists: Dict of ranked lists

        Returns:
            Count of unique configurations
        """
        seen = set()
        for configs in ranked_lists.values():
            for config in configs:
                # Use model_id + gpu_config as unique key
                if config.gpu_config:
                    key = (
                        config.model_id,
                        config.gpu_config.gpu_type,
                        config.gpu_config.gpu_count,
                        config.gpu_config.tensor_parallel,
                        config.gpu_config.replicas,
                    )
                    seen.add(key)
        return len(seen)
