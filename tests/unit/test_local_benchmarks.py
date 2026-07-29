"""Unit tests for LocalBenchmarkRepository."""

import json
from pathlib import Path

import pytest

from planner.data._resolver import data_path
from planner.knowledge_base.benchmarks import BenchmarkData
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


@pytest.mark.unit
class TestFindConfigurationsMeetingSlo:
    def _make_repo(self, benchmarks=None):
        repo = LocalBenchmarkRepository()
        repo.add_benchmarks(benchmarks or [SAMPLE_BENCHMARK])
        return repo

    def test_returns_matching_config(self):
        repo = self._make_repo()
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,  # 26.297 < 100
            itl_p95_max_ms=50,  # 10.748 < 50
            e2e_p95_max_ms=5000,  # 3856.112 < 5000
        )
        assert len(results) == 1
        assert results[0].model_hf_repo == "ibm-granite/granite-3.1-8b-instruct"
        assert results[0].hardware == "H100"

    def test_returns_empty_when_slo_too_tight(self):
        repo = self._make_repo()
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=1,  # 26.297 > 1 — too tight
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 0

    def test_returns_empty_for_no_data(self):
        repo = LocalBenchmarkRepository()
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 0

    def test_filters_by_gpu_type(self):
        benchmarks = [
            {**SAMPLE_BENCHMARK, "hardware": "H100"},
            {**SAMPLE_BENCHMARK, "hardware": "A100", "requests_per_second": 0.8},
        ]
        repo = self._make_repo(benchmarks)
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
            gpu_types=["A100"],
        )
        assert len(results) == 1
        assert results[0].hardware == "A100"

    def test_window_function_dedup(self):
        """Multiple QPS rates for same model/hardware — returns highest QPS meeting SLO."""
        low_qps = {**SAMPLE_BENCHMARK, "requests_per_second": 1.0}
        high_qps = {**SAMPLE_BENCHMARK, "requests_per_second": 5.0}
        repo = self._make_repo([low_qps, high_qps])
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 1
        assert results[0].requests_per_second == 5.0

    def test_percentile_parameter(self):
        """Using p99 percentile should filter against p99 columns."""
        repo = self._make_repo()
        # ttft_p99 is 27.042, so threshold of 25 should exclude it
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=25,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
            percentile="p99",
        )
        assert len(results) == 0

    def test_exclude_estimated(self):
        repo = self._make_repo()
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
            exclude_estimated=True,
        )
        # Default confidence_level is 'estimated', so all rows excluded
        assert len(results) == 0

    def test_result_is_benchmark_data(self):
        repo = self._make_repo()
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert isinstance(results[0], BenchmarkData)
        assert results[0].ttft_p95 == 26.297
        assert results[0].itl_p95 == 10.748
        assert results[0].tokens_per_second == 247.464


@pytest.mark.unit
class TestFromFiles:
    def test_loads_json_file(self, tmp_path):
        json_file = tmp_path / "benchmarks.json"
        json_file.write_text(
            json.dumps(
                {
                    "_metadata": {"source": "test", "confidence_level": "benchmarked"},
                    "benchmarks": [SAMPLE_BENCHMARK],
                }
            )
        )
        repo = LocalBenchmarkRepository.from_files([json_file])
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 1
        assert results[0].source == "test"
        assert results[0].confidence_level == "benchmarked"

    def test_loads_multiple_files(self, tmp_path):
        for i, hw in enumerate(["H100", "A100"]):
            f = tmp_path / f"bench_{i}.json"
            bench = {**SAMPLE_BENCHMARK, "hardware": hw, "requests_per_second": float(i + 1)}
            f.write_text(json.dumps({"benchmarks": [bench]}))
        repo = LocalBenchmarkRepository.from_files(
            [
                tmp_path / "bench_0.json",
                tmp_path / "bench_1.json",
            ]
        )
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 2

    def test_uses_default_metadata_when_absent(self, tmp_path):
        json_file = tmp_path / "benchmarks.json"
        json_file.write_text(json.dumps({"benchmarks": [SAMPLE_BENCHMARK]}))
        repo = LocalBenchmarkRepository.from_files([json_file])
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 1
        assert results[0].source == "local"


@pytest.mark.unit
class TestFromBenchmarks:
    def test_loads_dicts(self):
        repo = LocalBenchmarkRepository.from_benchmarks([SAMPLE_BENCHMARK])
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 1


@pytest.mark.unit
class TestReplaceBenchmarks:
    def test_replaces_source(self):
        repo = LocalBenchmarkRepository()
        repo.add_benchmarks(
            [SAMPLE_BENCHMARK],
            source="model_catalog",
        )
        new_bench = {**SAMPLE_BENCHMARK, "hardware": "A100", "requests_per_second": 2.0}
        replaced = repo.replace_benchmarks("model_catalog", [new_bench])
        assert replaced == 1
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 1
        assert results[0].hardware == "A100"

    def test_preserves_other_sources(self):
        repo = LocalBenchmarkRepository()
        repo.add_benchmarks([SAMPLE_BENCHMARK], source="blis")
        catalog_bench = {**SAMPLE_BENCHMARK, "hardware": "A100", "requests_per_second": 2.0}
        repo.add_benchmarks([catalog_bench], source="model_catalog")
        # Replace only model_catalog rows
        repo.replace_benchmarks("model_catalog", [])
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 1
        assert results[0].source == "blis"


@pytest.mark.unit
class TestSaveBenchmarks:
    def test_saves_benchmark_data_objects(self):
        repo = LocalBenchmarkRepository()
        bd = BenchmarkData(SAMPLE_BENCHMARK)
        repo.save_benchmarks([bd])
        results = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=100,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=5000,
        )
        assert len(results) == 1


# Path to real benchmark data via importlib.resources
_BLIS_PATH = data_path("benchmarks/performance/benchmarks_BLIS.json")


@pytest.mark.unit
class TestConfigFinderIntegration:
    """Wire LocalBenchmarkRepository into ConfigFinder and produce recommendations."""

    @pytest.mark.skipif(not _BLIS_PATH.exists(), reason="BLIS benchmark file not found")
    def test_end_to_end_recommendations(self):
        from planner.recommendation.config_finder import ConfigFinder
        from planner.shared.schemas.intent import DeploymentIntent
        from planner.specification.traffic_profile import TrafficProfileGenerator

        repo = LocalBenchmarkRepository.from_files([_BLIS_PATH])
        finder = ConfigFinder(benchmark_repo=repo)  # type: ignore[arg-type]

        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            experience_class="conversational",
            user_count=100,
            accuracy_priority="medium",
            cost_priority="medium",
            latency_priority="medium",
        )

        gen = TrafficProfileGenerator()
        traffic = gen.generate_profile(intent)
        slo = gen.generate_slo_targets(intent)

        results = finder.plan_all_capacities(
            traffic_profile=traffic,
            slo_targets=slo,
            intent=intent,
        )
        # plan_all_capacities returns (ranked_configs, warnings)
        ranked_configs, warnings = results

        # Should produce at least one recommendation from BLIS data
        total = (
            sum(len(v) for v in ranked_configs.values())
            if isinstance(ranked_configs, dict)
            else len(ranked_configs)
        )
        assert total > 0, f"Expected recommendations but got none. Warnings: {warnings}"
