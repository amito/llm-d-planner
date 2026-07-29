"""Unit tests for the Planner public API facade."""

import pytest

from planner.data._resolver import data_path
from planner.knowledge_base.local_benchmarks import LocalBenchmarkRepository

_BLIS_PATH = data_path("benchmarks/performance/benchmarks_BLIS.json")


@pytest.mark.unit
class TestPlannerRecommend:
    @pytest.mark.skipif(not _BLIS_PATH.exists(), reason="BLIS benchmark file not found")
    def test_recommend_chatbot_returns_ranked_results(self):
        from planner.planner import Planner

        planner = Planner()
        result = planner.recommend(use_case="chatbot_conversational")

        assert result.total_configs_evaluated > 0
        assert len(result.best_accuracy) > 0 or len(result.balanced) > 0

    @pytest.mark.skipif(not _BLIS_PATH.exists(), reason="BLIS benchmark file not found")
    def test_recommend_with_model_preference(self):
        from planner.planner import Planner

        planner = Planner()
        result = planner.recommend(
            use_case="chatbot_conversational",
            model="ibm-granite/granite-3.1-8b-instruct",
        )

        assert result.total_configs_evaluated > 0

    @pytest.mark.skipif(not _BLIS_PATH.exists(), reason="BLIS benchmark file not found")
    def test_recommend_unknown_use_case_raises(self):
        from pydantic import ValidationError

        from planner.planner import Planner

        planner = Planner()
        with pytest.raises(ValidationError):
            planner.recommend(use_case="bogus")

    @pytest.mark.skipif(not _BLIS_PATH.exists(), reason="BLIS benchmark file not found")
    def test_custom_benchmark_repo_injection(self):
        from planner.planner import Planner

        repo = LocalBenchmarkRepository.from_files([_BLIS_PATH])
        planner = Planner(benchmark_repo=repo)
        result = planner.recommend(use_case="code_completion", user_count=50)

        assert result.total_configs_evaluated > 0

    @pytest.mark.skipif(not _BLIS_PATH.exists(), reason="BLIS benchmark file not found")
    def test_recommend_with_priorities(self):
        from planner.planner import Planner

        planner = Planner()
        result = planner.recommend(
            use_case="chatbot_conversational",
            accuracy_priority="high",
            cost_priority="low",
            latency_priority="high",
        )

        assert result.total_configs_evaluated > 0
