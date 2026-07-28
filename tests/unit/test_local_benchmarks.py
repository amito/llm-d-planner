"""Unit tests for LocalBenchmarkRepository."""

import pytest

from planner.knowledge_base.local_benchmarks import LocalBenchmarkRepository

# Minimal benchmark dict matching the BLIS JSON format
SAMPLE_BENCHMARK = {
    "model_hf_repo": "ibm-granite/granite-3.1-8b-instruct",
    "hardware": "H100",
    "hardware_count": 1,
    "framework": "vllm",
    "framework_version": "v0.8.4",
    "prompt_tokens": 512,
    "output_tokens": 256,
    "mean_input_tokens": 512,
    "mean_output_tokens": 256,
    "ttft_mean": 21.951,
    "ttft_p90": 25.583,
    "ttft_p95": 26.297,
    "ttft_p99": 27.042,
    "itl_mean": 10.764,
    "itl_p90": 10.748,
    "itl_p95": 10.748,
    "itl_p99": 10.749,
    "e2e_mean": 2720.175,
    "e2e_p90": 3729.216,
    "e2e_p95": 3856.112,
    "e2e_p99": 4208.735,
    "requests_per_second": 1.0,
    "tokens_per_second": 247.464,
}


@pytest.mark.unit
class TestLocalBenchmarkRepositoryInit:
    def test_empty_repo_creates_successfully(self):
        repo = LocalBenchmarkRepository()
        assert repo is not None

    def test_add_benchmarks_returns_count(self):
        repo = LocalBenchmarkRepository()
        inserted = repo.add_benchmarks([SAMPLE_BENCHMARK])
        assert inserted == 1

    def test_add_benchmarks_skips_duplicates(self):
        repo = LocalBenchmarkRepository()
        repo.add_benchmarks([SAMPLE_BENCHMARK])
        inserted = repo.add_benchmarks([SAMPLE_BENCHMARK])
        assert inserted == 0

    def test_add_benchmarks_empty_list(self):
        repo = LocalBenchmarkRepository()
        inserted = repo.add_benchmarks([])
        assert inserted == 0

    def test_add_benchmarks_normalizes_field_names(self):
        """Benchmarks using model_id instead of model_hf_repo should work."""
        benchmark = {
            **SAMPLE_BENCHMARK,
            "model_id": "test/model",
        }
        del benchmark["model_hf_repo"]
        repo = LocalBenchmarkRepository()
        inserted = repo.add_benchmarks([benchmark])
        assert inserted == 1
