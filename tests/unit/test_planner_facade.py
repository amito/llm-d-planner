"""Tests for Planner facade class."""

import pytest

from planner import Planner, PlannerError
from planner.shared.schemas import DeploymentIntent


class TestPlannerInit:
    """Test Planner initialization."""

    def test_planner_creates_with_defaults(self):
        """Planner() creates with default configuration."""
        planner = Planner()
        assert planner is not None

    def test_planner_creates_with_custom_data_dir(self, tmp_path):
        """Planner(data_dir=valid_dir) accepts custom data directory."""
        # Create minimal required data structure
        config_dir = tmp_path / "configuration"
        config_dir.mkdir(parents=True)

        # Create minimal required files
        (config_dir / "slo_templates.json").write_text('{"use_cases":{}}')
        (config_dir / "model_catalog.json").write_text('{"models":[]}')
        (config_dir / "gpu_catalog.json").write_text('{"gpu_types":[]}')
        (config_dir / "quality_weights.json").write_text("{}")
        (config_dir / "usecase_slo_workload.json").write_text('{"use_case_slo_workload":{}}')

        # Quality data dir
        quality_dir = tmp_path / "quality"
        quality_dir.mkdir(parents=True)
        (quality_dir / "arena_models.json").write_text('{"rows":[],"fetched_at":"2024-01-01"}')
        (quality_dir / "aa_models.json").write_text('{"models":[],"fetched_at":"2024-01-01"}')

        planner = Planner(data_dir=tmp_path)
        assert planner is not None


class TestGenerateSpecification:
    """Test specification generation."""

    def test_generate_specification_returns_valid_spec(self):
        """generate_specification() returns a DeploymentSpecification."""
        planner = Planner()
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=100,
        )
        spec = planner.generate_specification(intent)

        assert spec is not None
        assert spec.intent == intent
        assert spec.slo_targets is not None
        assert spec.workload_profile is not None


class TestGenerateRecommendations:
    """Test recommendation generation."""

    def test_generate_recommendations_raises_without_benchmarks(self):
        """generate_recommendations() raises PlannerError without benchmarks loaded."""
        planner = Planner()
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=100,
        )
        spec = planner.generate_specification(intent)

        with pytest.raises(PlannerError, match="No benchmarks loaded"):
            planner.generate_recommendations(spec)

    def test_generate_recommendations_works_after_load(self):
        """generate_recommendations() works after load_bundled_benchmarks()."""
        planner = Planner()
        planner.load_bundled_benchmarks()

        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=100,
        )
        spec = planner.generate_specification(intent)

        # Should not raise
        result = planner.generate_recommendations(spec)
        assert result is not None


class TestGenerateDeployment:
    """Test deployment generation."""

    def test_generate_deployment_returns_valid_bundle(self):
        """generate_deployment() returns a DeploymentBundle."""
        from planner.shared.schemas import DeploymentConfiguration, GPUConfig

        planner = Planner()
        config = DeploymentConfiguration(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            model_name="Llama 3.3 70B Instruct",
            model_uri=None,
            gpu_config=GPUConfig(
                gpu_type="NVIDIA-L4",
                gpu_count=2,
                tensor_parallel=2,
                replicas=1,
            ),
            use_case="chatbot_conversational",
            expected_qps=1.0,
            prompt_tokens=512,
            output_tokens=256,
            e2e_target_ms=7000,
        )

        bundle = planner.generate_deployment(config)

        assert bundle is not None
        assert bundle.deployment_id is not None
        assert bundle.namespace == "default"
        assert bundle.stack == "vllm"
        assert bundle.configuration == config
        assert isinstance(bundle.files, dict)
        assert len(bundle.files) > 0

    def test_generate_deployment_supports_llmd_stack(self):
        """generate_deployment() supports stack='llm-d'."""
        from planner.shared.schemas import DeploymentConfiguration, GPUConfig

        planner = Planner()
        config = DeploymentConfiguration(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            model_name="Llama 3.3 70B Instruct",
            model_uri=None,
            gpu_config=GPUConfig(
                gpu_type="NVIDIA-L4",
                gpu_count=2,
                tensor_parallel=2,
                replicas=1,
            ),
            use_case="chatbot_conversational",
            expected_qps=1.0,
            prompt_tokens=512,
            output_tokens=256,
            e2e_target_ms=7000,
        )

        bundle = planner.generate_deployment(config, stack="llm-d")

        assert bundle is not None
        assert bundle.stack == "llm-d"


class TestExtractIntent:
    """Test intent extraction."""

    def test_extract_intent_raises_without_llm_configured(self):
        """extract_intent() raises PlannerError without LLM provider configured."""
        planner = Planner()

        with pytest.raises(PlannerError, match="No LLM provider configured"):
            planner.extract_intent("I need a chatbot for customer support")


class TestLoadBenchmarks:
    """Test benchmark loading."""

    def test_load_bundled_benchmarks_succeeds(self):
        """load_bundled_benchmarks() loads without error."""
        planner = Planner()

        planner.load_bundled_benchmarks()
        assert planner._benchmark_repo.get_stats()["total_benchmarks"] > 0

    def test_load_benchmarks_from_file(self, tmp_path):
        """load_benchmarks(path) loads custom benchmark file."""
        # Create a test benchmark file
        bench_file = tmp_path / "test_benchmarks.json"
        bench_file.write_text("""{
            "benchmarks": [
                {
                    "model_hf_repo": "test/model",
                    "hardware": "NVIDIA-L4",
                    "hardware_count": 1,
                    "prompt_tokens": 512,
                    "output_tokens": 256,
                    "mean_input_tokens": 512,
                    "mean_output_tokens": 256,
                    "ttft_mean": 100,
                    "ttft_p90": 150,
                    "ttft_p95": 180,
                    "ttft_p99": 250,
                    "itl_mean": 10,
                    "itl_p90": 15,
                    "itl_p95": 18,
                    "itl_p99": 25,
                    "e2e_mean": 5000,
                    "e2e_p90": 6000,
                    "e2e_p95": 6500,
                    "e2e_p99": 7000,
                    "tokens_per_second": 50,
                    "requests_per_second": 2.0
                }
            ]
        }""")

        planner = Planner()
        planner.load_benchmarks(bench_file)

        assert planner._benchmark_repo.get_stats()["total_benchmarks"] > 0
