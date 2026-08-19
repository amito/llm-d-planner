"""Test ConfigFinder uses injected quality scorer."""

from unittest.mock import MagicMock, patch

import pytest

from planner.knowledge_base.benchmarks import BenchmarkData
from planner.recommendation.config_finder import ConfigFinder
from planner.shared.schemas import DeploymentIntent, SLOTargets, TrafficProfile


def _make_bench(
    model: str = "RedHatAI/test-model",
    hardware: str = "H100",
    hw_count: int = 1,
) -> BenchmarkData:
    return BenchmarkData(
        {
            "model_hf_repo": model,
            "hardware": hardware,
            "hardware_count": hw_count,
            "framework": "vllm",
            "framework_version": "0.8.4",
            "prompt_tokens": 512,
            "output_tokens": 256,
            "mean_input_tokens": 512,
            "mean_output_tokens": 256,
            "ttft_mean": 50,
            "ttft_p90": 70,
            "ttft_p95": 80,
            "ttft_p99": 100,
            "itl_mean": 10,
            "itl_p90": 15,
            "itl_p95": 20,
            "itl_p99": 25,
            "e2e_mean": 3000,
            "e2e_p90": 4000,
            "e2e_p95": 5000,
            "e2e_p99": 6000,
            "tps_mean": 1000,
            "tps_p90": 900,
            "tps_p95": 800,
            "tps_p99": 700,
            "tokens_per_second": 1000,
            "requests_per_second": 7,
        }
    )


def _make_intent(use_case: str = "chatbot_conversational") -> DeploymentIntent:
    return DeploymentIntent(
        use_case=use_case,  # type: ignore[arg-type]
        user_count=100,
    )


def _make_traffic() -> TrafficProfile:
    return TrafficProfile(
        prompt_tokens=512,
        output_tokens=256,
        expected_qps=5.0,
    )


def _make_slo() -> SLOTargets:
    return SLOTargets(
        ttft_target_ms=200,
        itl_target_ms=50,
        e2e_target_ms=7000,
    )


@pytest.mark.unit
def test_config_finder_accepts_quality_scorer():
    """ConfigFinder should accept an optional ScoringEngine parameter."""
    mock_source = MagicMock()
    mock_catalog = MagicMock()
    mock_engine = MagicMock()
    mock_engine.get_scores.return_value = None  # No scorecard available

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        engine=mock_engine,
        quality_weights={},
    )
    assert finder._engine is mock_engine


@pytest.mark.unit
def test_config_finder_quality_scorer_defaults_to_none():
    """When no ScoringEngine provided, _engine should be None."""
    mock_source = MagicMock()
    mock_catalog = MagicMock()

    finder = ConfigFinder(benchmark_repo=mock_source, catalog=mock_catalog)
    assert finder._engine is None


@pytest.mark.unit
def test_injected_scorer_used_in_plan_all_capacities():
    """When ScoringEngine is injected, plan_all_capacities uses it to compute quality scores."""
    from quality_scoring.models import (
        CompositeScore,
        MatchType,
        ModelScorecard,
        NormalizedScore,
    )

    mock_source = MagicMock()
    bench = _make_bench()
    mock_source.find_configurations_meeting_slo.return_value = [bench]

    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = []
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    mock_engine = MagicMock()
    # Return a scorecard with 82.5 percentile
    scorecard = ModelScorecard(
        model_name="test-model",
        arena_name="test-model",
        aa_name=None,
        arena_match_type=MatchType.EXACT,
        aa_match_type=MatchType.NONE,
        overall=CompositeScore(
            category="overall",
            percentile=82.5,
            arena_score=NormalizedScore(
                raw_score=1400.0,
                percentile=82.5,
                tied_rank=1,
                population_size=100,
                source="arena",
            ),
            aa_score=None,
        ),
        categories={},
    )
    mock_engine.get_scores.return_value = scorecard

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        engine=mock_engine,
        quality_weights={"chatbot": {"categories": {}}},
    )

    results, _warnings = finder.plan_all_capacities(
        traffic_profile=_make_traffic(),
        slo_targets=_make_slo(),
        intent=_make_intent(),
    )

    # Verify the injected engine was called
    mock_engine.get_scores.assert_called()
    # Should have been called with the model name (bench.model_hf_repo)
    call_args = mock_engine.get_scores.call_args_list[0]
    assert call_args[0][0] == "RedHatAI/test-model"

    # Verify we got results with quality score from the injected engine
    assert len(results) >= 1
    assert results[0].scores is not None
    assert results[0].scores.quality_score == 82.5  # exact float match


@pytest.mark.unit
def test_default_scorer_used_when_none_injected():
    """When no ScoringEngine injected, plan_all_capacities falls back to size-based scoring."""
    mock_source = MagicMock()
    bench = _make_bench()
    mock_source.find_configurations_meeting_slo.return_value = [bench]

    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = []
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        # No engine -- uses fallback
    )

    results, _warnings = finder.plan_all_capacities(
        traffic_profile=_make_traffic(),
        slo_targets=_make_slo(),
        intent=_make_intent(),
    )

    # Without an engine, should fall back to size-based scoring
    assert len(results) >= 1
    assert results[0].scores is not None
    # Size-based scoring should give a reasonable score (model_hf_repo doesn't have size, so minimum)
    assert results[0].scores.quality_score > 0


@pytest.mark.unit
def test_injected_scorer_uses_hf_repo():
    """ScoringEngine is always called with bench.model_hf_repo to preserve quantization info."""
    from quality_scoring.models import (
        CompositeScore,
        MatchType,
        ModelScorecard,
        NormalizedScore,
    )

    mock_source = MagicMock()
    bench = _make_bench(model="RedHatAI/some-model-quantized.w4a16")
    mock_source.find_configurations_meeting_slo.return_value = [bench]

    mock_model = MagicMock()
    mock_model.model_id = "redhatai/some-model-quantized.w4a16"
    mock_model.name = "Some Model Display Name"
    mock_model.size_parameters = "8B"
    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = [mock_model]
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    mock_engine = MagicMock()
    scorecard = ModelScorecard(
        model_name="test-model",
        arena_name="test-model",
        aa_name=None,
        arena_match_type=MatchType.EXACT,
        aa_match_type=MatchType.NONE,
        overall=CompositeScore(
            category="overall",
            percentile=70.0,
            arena_score=NormalizedScore(
                raw_score=1400.0,
                percentile=70.0,
                tied_rank=1,
                population_size=100,
                source="arena",
            ),
            aa_score=None,
        ),
        categories={},
    )
    mock_engine.get_scores.return_value = scorecard

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        engine=mock_engine,
        quality_weights={"chatbot": {"categories": {}}},
    )

    results, _warnings = finder.plan_all_capacities(
        traffic_profile=_make_traffic(),
        slo_targets=_make_slo(),
        intent=_make_intent(),
    )

    # Engine called once with the HF repo name (not display name)
    assert mock_engine.get_scores.call_count == 1
    call_args = mock_engine.get_scores.call_args[0]
    assert call_args[0] == "RedHatAI/some-model-quantized.w4a16"
    assert len(results) >= 1
    assert results[0].scores is not None
    assert results[0].scores.quality_score >= 70
