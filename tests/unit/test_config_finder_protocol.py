"""Verify ConfigFinder works with any BenchmarkSource implementation."""

import pytest
from unittest.mock import MagicMock

from neuralnav.knowledge_base.benchmark_source import BenchmarkSource
from neuralnav.knowledge_base.benchmarks import BenchmarkData
from neuralnav.recommendation.config_finder import ConfigFinder


def _make_bench(
    model: str = "test/model", hardware: str = "H100", hw_count: int = 1
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


@pytest.mark.unit
def test_config_finder_accepts_benchmark_source():
    """ConfigFinder should accept any BenchmarkSource, not just BenchmarkRepository."""
    mock_source = MagicMock(spec=BenchmarkSource)
    mock_source.find_configurations_meeting_slo.return_value = [_make_bench()]

    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = []
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    finder = ConfigFinder(benchmark_repo=mock_source, catalog=mock_catalog)
    assert finder.benchmark_repo is mock_source
