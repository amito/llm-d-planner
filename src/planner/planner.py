"""Public API facade for llm-d-planner.

Provides zero-config access to the recommendation engine without
FastAPI, PostgreSQL, Ollama, or Kubernetes.

Usage::

    from planner.planner import Planner

    p = Planner()
    result = p.recommend(use_case="chatbot_conversational")
    for rec in result.best_accuracy:
        print(rec.model_name, rec.gpu_config, rec.cost_per_month_usd)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from planner.data._resolver import data_path
from planner.knowledge_base.local_benchmarks import LocalBenchmarkRepository
from planner.knowledge_base.model_catalog import ModelCatalog
from planner.recommendation.analyzer import Analyzer
from planner.recommendation.config_finder import ConfigFinder
from planner.recommendation.quality.usecase_scorer import UseCaseQualityScorer
from planner.shared.schemas import (
    DeploymentIntent,
    DeploymentRecommendation,
    DeploymentSpecification,
    RankedRecommendationsResponse,
)
from planner.specification.traffic_profile import TrafficProfileGenerator

logger = logging.getLogger(__name__)

_ExperienceClass = Literal["instant", "conversational", "interactive", "deferred", "batch"]

# use_case -> experience_class mapping (mirrors workflow.py:363-384)
_EXPERIENCE_CLASS: dict[str, _ExperienceClass] = {
    "code_completion": "instant",
    "chatbot_conversational": "conversational",
    "code_generation_detailed": "conversational",
    "translation": "conversational",
    "content_generation": "conversational",
    "summarization_short": "conversational",
    "document_analysis_rag": "interactive",
    "long_document_summarization": "deferred",
    "research_legal_analysis": "batch",
}

_DEFAULT_BLIS = "benchmarks/performance/benchmarks_BLIS.json"


class Planner:
    """Zero-config facade for LLM deployment recommendations.

    Loads bundled benchmark data, model catalog, and SLO templates
    automatically. Call :meth:`recommend` to get ranked GPU
    configurations for a given use case.
    """

    def __init__(
        self,
        benchmark_repo: LocalBenchmarkRepository | None = None,
        benchmark_paths: list[str | Path] | None = None,
    ) -> None:
        if benchmark_repo is not None:
            repo = benchmark_repo
        elif benchmark_paths is not None:
            repo = LocalBenchmarkRepository.from_files([Path(p) for p in benchmark_paths])
        else:
            repo = LocalBenchmarkRepository.from_files([data_path(_DEFAULT_BLIS)])

        catalog = ModelCatalog()
        quality_scorer = UseCaseQualityScorer()

        self._config_finder = ConfigFinder(
            benchmark_repo=repo,  # type: ignore[arg-type]  # duck-type compatible
            catalog=catalog,
            quality_scorer=quality_scorer,
        )
        self._traffic_gen = TrafficProfileGenerator()

    def recommend(
        self,
        use_case: Literal[
            "chatbot_conversational",
            "code_completion",
            "code_generation_detailed",
            "translation",
            "content_generation",
            "summarization_short",
            "document_analysis_rag",
            "long_document_summarization",
            "research_legal_analysis",
        ],
        model: str | None = None,
        user_count: int = 100,
        accuracy_priority: Literal["low", "medium", "high"] = "medium",
        cost_priority: Literal["low", "medium", "high"] = "medium",
        latency_priority: Literal["low", "medium", "high"] = "medium",
        gpu_types: list[str] | None = None,
        include_near_miss: bool = True,
        min_accuracy: int | None = None,
        max_cost: float | None = None,
    ) -> RankedRecommendationsResponse:
        """Find ranked GPU configurations for a use case.

        Args:
            use_case: One of the 9 supported use cases (e.g.
                ``"chatbot_conversational"``, ``"code_completion"``).
            model: Optional HuggingFace model ID to prefer.
            user_count: Expected number of concurrent users.
            accuracy_priority: ``"low"``, ``"medium"``, or ``"high"``.
            cost_priority: ``"low"``, ``"medium"``, or ``"high"``.
            latency_priority: ``"low"``, ``"medium"``, or ``"high"``.
            gpu_types: Optional list of GPU types to consider.
            include_near_miss: Include configs near but not meeting SLO.
            min_accuracy: Minimum accuracy score filter (0-100).
            max_cost: Maximum monthly cost filter (USD).

        Returns:
            Ranked recommendations with 4 views: best_accuracy,
            lowest_cost, lowest_latency, balanced.
        """
        experience_class: _ExperienceClass = _EXPERIENCE_CLASS.get(use_case, "conversational")

        intent = DeploymentIntent(
            use_case=use_case,
            experience_class=experience_class,
            user_count=user_count,
            accuracy_priority=accuracy_priority,
            cost_priority=cost_priority,
            latency_priority=latency_priority,
            preferred_models=[model] if model else [],
            preferred_gpu_types=gpu_types or [],
        )

        traffic_profile = self._traffic_gen.generate_profile(intent)
        slo_targets = self._traffic_gen.generate_slo_targets(intent)

        specification = DeploymentSpecification(
            intent=intent,
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
        )

        all_configs, estimation_warnings = self._config_finder.plan_all_capacities(
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
            intent=intent,
            include_near_miss=include_near_miss,
            preferred_models=intent.preferred_models if intent.preferred_models else None,
        )

        if not all_configs:
            return RankedRecommendationsResponse(
                min_accuracy_threshold=min_accuracy,
                max_cost_ceiling=max_cost,
                include_near_miss=include_near_miss,
                specification=specification,
                total_configs_evaluated=0,
                configs_after_filters=0,
                warnings=estimation_warnings,
            )

        analyzer = Analyzer()
        ranked_lists = analyzer.generate_ranked_lists(
            configurations=all_configs,
            min_accuracy=min_accuracy,
            max_cost=max_cost,
            top_n=10,
            use_case=intent.use_case,
            preferred_models=intent.preferred_models if intent.preferred_models else None,
        )

        configs_after_filters = analyzer.get_unique_configs_count(ranked_lists)

        return RankedRecommendationsResponse(
            min_accuracy_threshold=min_accuracy,
            max_cost_ceiling=max_cost,
            include_near_miss=include_near_miss,
            specification=specification,
            best_accuracy=ranked_lists["best_accuracy"],
            lowest_cost=ranked_lists["lowest_cost"],
            lowest_latency=ranked_lists["lowest_latency"],
            balanced=ranked_lists["balanced"],
            total_configs_evaluated=len(all_configs),
            configs_after_filters=configs_after_filters,
            warnings=estimation_warnings,
        )

    def generate_config(
        self,
        recommendation: DeploymentRecommendation,
        output_dir: str | None = None,
        namespace: str = "default",
    ) -> dict[str, str]:
        """Generate KServe/vLLM deployment YAML from a recommendation.

        Args:
            recommendation: A single ``DeploymentRecommendation`` from
                the ``recommend()`` response lists.
            output_dir: Directory to write YAML files. Uses a temp
                directory if not specified.
            namespace: Kubernetes namespace for the deployment.

        Returns:
            Dict mapping config type (``"inferenceservice"``,
            ``"autoscaling"``, ``"servicemonitor"``) to rendered YAML.
        """
        from planner.configuration.generator import DeploymentGenerator

        generator = DeploymentGenerator(output_dir=output_dir)
        result = generator.generate_all(recommendation, namespace=namespace)
        return result["contents"]  # type: ignore[no-any-return]
