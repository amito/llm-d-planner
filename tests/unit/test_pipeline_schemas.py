"""Unit tests for pipeline schema objects."""

import pytest

from planner.shared.schemas.intent import DeploymentIntent, GpuPreference
from planner.shared.schemas.recommendation import DeploymentBundle
from planner.shared.schemas.specification import (
    DeploymentSpecification,
    Priorities,
    PriorityEntry,
    QualityWeights,
    SLORange,
    SLOTargets,
    WorkloadProfile,
)


@pytest.mark.unit
class TestGpuPreference:
    def test_plain_string_in_preferred_gpu_types(self):
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=1000,
            preferred_gpu_types=["H100", "L4"],
        )
        assert intent.preferred_gpu_types == ["H100", "L4"]

    def test_gpu_preference_with_max_count(self):
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=1000,
            preferred_gpu_types=[
                "L4",
                GpuPreference(gpu_type="H100", max_count=4),
            ],
        )
        assert len(intent.preferred_gpu_types) == 2
        assert intent.preferred_gpu_types[0] == "L4"
        pref = intent.preferred_gpu_types[1]
        assert isinstance(pref, GpuPreference)
        assert pref.gpu_type == "H100"
        assert pref.max_count == 4

    def test_gpu_preference_from_dict(self):
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=1000,
            preferred_gpu_types=[
                GpuPreference(gpu_type="H100", max_count=2),
            ],
        )
        pref = intent.preferred_gpu_types[0]
        assert isinstance(pref, GpuPreference)
        assert pref.max_count == 2

    def test_gpu_preference_no_max_count(self):
        pref = GpuPreference(gpu_type="H100")
        assert pref.max_count is None


@pytest.mark.unit
class TestSLOTargets:
    def test_basic_construction(self):
        slo = SLOTargets(ttft_target_ms=200, itl_target_ms=25, e2e_target_ms=7000)
        assert slo.percentile == "p95"
        assert slo.ttft_range is None

    def test_with_ranges(self):
        slo = SLOTargets(
            ttft_target_ms=200,
            itl_target_ms=25,
            e2e_target_ms=7000,
            ttft_range=SLORange(min=100, max=500),
            itl_range=SLORange(min=15, max=50),
            e2e_range=SLORange(min=3940, max=13300),
        )
        assert slo.ttft_range is not None
        assert slo.ttft_range.min == 100
        assert slo.ttft_range.max == 500

    def test_percentile_validation(self):
        slo = SLOTargets(ttft_target_ms=200, itl_target_ms=25, e2e_target_ms=7000, percentile="p99")
        assert slo.percentile == "p99"


@pytest.mark.unit
class TestWorkloadProfile:
    def test_basic_construction(self):
        wp = WorkloadProfile(prompt_tokens=512, output_tokens=256, expected_qps=0.87)
        assert wp.prompt_tokens == 512
        assert wp.expected_qps == 0.87


@pytest.mark.unit
class TestQualityWeights:
    def test_basic_construction(self):
        qw = QualityWeights(categories={"overall": 4, "coding": 3})
        assert qw.categories["overall"] == 4


@pytest.mark.unit
class TestPriorities:
    def test_basic_construction(self):
        p = Priorities(
            quality=PriorityEntry(priority="medium", weight=4),
            cost=PriorityEntry(priority="medium", weight=4),
            latency=PriorityEntry(priority="high", weight=2),
        )
        assert p.quality.weight == 4
        assert p.latency.priority == "high"


@pytest.mark.unit
class TestDeploymentSpecification:
    def test_full_construction(self):
        spec = DeploymentSpecification(
            intent=DeploymentIntent(use_case="chatbot_conversational", user_count=1000),
            slo_targets=SLOTargets(ttft_target_ms=200, itl_target_ms=25, e2e_target_ms=7000),
            workload_profile=WorkloadProfile(
                prompt_tokens=512, output_tokens=256, expected_qps=0.87
            ),
            quality_weights=QualityWeights(categories={"overall": 4, "instruction_following": 3}),
            priorities=Priorities(
                quality=PriorityEntry(priority="medium", weight=4),
                cost=PriorityEntry(priority="medium", weight=4),
                latency=PriorityEntry(priority="high", weight=2),
            ),
        )
        assert spec.intent.use_case == "chatbot_conversational"
        assert spec.workload_profile.expected_qps == 0.87
        assert spec.quality_weights is not None
        assert spec.quality_weights.categories["overall"] == 4
        assert spec.priorities.latency.weight == 2

    def test_quality_weights_optional(self):
        """Test that quality_weights can be omitted."""
        spec = DeploymentSpecification(
            intent=DeploymentIntent(use_case="chatbot_conversational", user_count=1000),
            slo_targets=SLOTargets(ttft_target_ms=200, itl_target_ms=25, e2e_target_ms=7000),
            workload_profile=WorkloadProfile(
                prompt_tokens=512, output_tokens=256, expected_qps=0.87
            ),
            priorities=Priorities(
                quality=PriorityEntry(priority="medium", weight=4),
                cost=PriorityEntry(priority="medium", weight=4),
                latency=PriorityEntry(priority="medium", weight=1),
            ),
        )
        assert spec.quality_weights is None

    def test_serialization_roundtrip(self):
        spec = DeploymentSpecification(
            intent=DeploymentIntent(use_case="chatbot_conversational", user_count=1000),
            slo_targets=SLOTargets(ttft_target_ms=200, itl_target_ms=25, e2e_target_ms=7000),
            workload_profile=WorkloadProfile(
                prompt_tokens=512, output_tokens=256, expected_qps=0.87
            ),
            quality_weights=QualityWeights(categories={"overall": 4}),
            priorities=Priorities(
                quality=PriorityEntry(priority="medium", weight=4),
                cost=PriorityEntry(priority="medium", weight=4),
                latency=PriorityEntry(priority="medium", weight=1),
            ),
        )
        data = spec.model_dump()
        spec2 = DeploymentSpecification(**data)
        assert spec2.workload_profile.expected_qps == 0.87


@pytest.mark.unit
class TestDeploymentBundle:
    def test_basic_construction(self):
        from planner.shared.schemas.recommendation import DeploymentConfiguration, GPUConfig

        config = DeploymentConfiguration(
            model_id="test-model",
            model_name="Test Model",
            model_uri="test-uri",
            gpu_config=GPUConfig(gpu_type="H100", gpu_count=1, tensor_parallel=1, replicas=1),
            use_case="chatbot_conversational",
            expected_qps=0.87,
            prompt_tokens=512,
            output_tokens=256,
            e2e_target_ms=7000,
        )
        bundle = DeploymentBundle(
            deployment_id="deploy-abc123",
            namespace="default",
            stack="vllm",
            configuration=config,
            files={"inferenceservice": "apiVersion: ..."},
        )
        assert bundle.deployment_id == "deploy-abc123"
        assert "inferenceservice" in bundle.files
